"""E3 (cross-channel) and E4 (cross-corpus email) transfer for DistilBERT, EE6405 proposal v2.

E3 (4.4): binary fraud detection on label_fraud (1 = fraud, 0 = legitimate; marketing
spam and insurance/sales calls are NA and dropped). Channels: sms = mendeley,
email = email_fraud, phone = bothbosu. Training sources: the 3 single channels and the
3 pairs; every model is evaluated on the test set of every channel (same channel =
in-domain, other channels = transfer).
E4 (4.4): binary spam detection on label_spam, CEAS_08 -> SpamAssassin and the reverse,
once on the raw text and once on text_stripped (header/format traces removed). When
CEAS_08 is the target on split_seed0, results are also reported without the CNN
"Daily Top 10" campaign, which appears only in that test set.
E3M (size-matched control): each two-channel E3 run is repeated with its training set
randomly subsampled (random_state = split seed) to the size of the larger of its two
channels, to separate the effect of a second channel from the effect of more data.
E5 (4.4-4.5): the single-channel E3 models are trained again in the same way and scored on
their own channel's test set, clean and after each perturbation of perturb.py (P1, P2, P3 and
their sub-forms), each also after the D1 defence normalise() (src/normalise.py). The changed
texts are saved as e5_texts_<channel>_seed<k>.csv.gz so other models can use the same ones.
All experiments call remove_overlap (same logic as src/splits.py) to drop source rows
that duplicate any target test set, train on the source train split only (source val
is just recorded), and reuse the Mendeley E2 setting (lr 3e-5, 3 epochs, class weights,
max 512 tokens). A failing run does not stop the others; tracebacks go to errors.txt.

Inputs: mendeley / email_fraud / bothbosu / ceas_08 / spamassassin as .csv or .csv.gz in
any folder under DATA_DIR (default /kaggle/input).

Usage (Kaggle: attach the processed datasets, GPU on, then Save & Run All):
    python e3_e4_transfer.py
    EXPERIMENTS=E4 SEEDS=0 python e3_e4_transfer.py    # one experiment, one split seed
    EXPERIMENTS=E3M python e3_e4_transfer.py          # E3 size-matched control only
    EXPERIMENTS=E5 python e3_e4_transfer.py           # E5 robustness (needs perturb.py, normalise.py)

Outputs: results_e3_e4/ (e3_e4_summary.csv, e3_e4_mean_std.csv, metrics_<run>.json,
pred_<run>_<eval set>.csv, e5_texts_*.csv.gz for E5, env.json, errors.txt if any).
Models are not saved.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one T4 only: batch 16 is the real batch

import glob
import json
import platform
import re
import time
import traceback
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input")
OUT = os.environ.get("OUT", "/kaggle/working")
DRY_RUN = int(os.environ.get("DRY_RUN", "0"))   # > 0: use a few rows per split to test the code
MODEL = os.environ.get("MODEL", "distilbert/distilbert-base-uncased")
EXPERIMENTS = os.environ.get("EXPERIMENTS", "E3,E4").split(",")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
LR, EPOCHS, BATCH, SEED, MAX_LEN = 3e-5, 3, 16, 42, 512
if DRY_RUN:
    EPOCHS = 1
FP16 = torch.cuda.is_available()

RES = f"{OUT}/results_e3_e4"
os.makedirs(RES, exist_ok=True)
tok = AutoTokenizer.from_pretrained(MODEL)
print("device:", torch.cuda.get_device_name(0) if FP16 else "cpu", flush=True)


def dump(obj: Any, path: str) -> None:
    """Write obj as indented JSON, converting numpy scalars to Python numbers."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load(name: str) -> pd.DataFrame:
    """Load <name>.csv(.gz) from DATA_DIR with nullable integer labels and no NaN text."""
    hits = [p for p in glob.glob(f"{DATA_DIR}/**/{name}.csv*", recursive=True) if os.path.isfile(p)]
    if not hits:
        raise FileNotFoundError(f"{name}.csv / {name}.csv.gz not found under {DATA_DIR}")
    df = pd.read_csv(hits[0], dtype={"label_fraud": "Int64", "label_spam": "Int64"})
    df["text"] = df["text"].fillna("")
    if "text_stripped" in df:
        df["text_stripped"] = df["text_stripped"].fillna("")
    df["template"] = df["template"].fillna("")
    return df


# Same de-duplication logic as the group's src/data.py and src/splits.py
SPACES_RE = re.compile(r"\s+")


def dedup_key(text: str) -> str:
    """Lower-cased text with whitespace collapsed, used to find exact duplicates."""
    return SPACES_RE.sub(" ", text).strip().lower()


def remove_overlap(source: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop source rows whose text or template also appears in target (4.4, leakage control).

    Returns the filtered source and the counts removed by exact text, by template and in total.
    """
    exact = source["text"].map(dedup_key).isin(set(target["text"].map(dedup_key)))
    template = source["template"].isin(set(target["template"]))
    counts = {"exact_text": int(exact.sum()), "template": int(template.sum()),
              "removed": int((exact | template).sum())}
    return source[~(exact | template)], counts


def to_ds(d: pd.DataFrame, text_col: str, label_col: str) -> Dataset:
    """Tokenize text_col (truncated to MAX_LEN tokens) into a HF Dataset with integer labels."""
    ds = Dataset.from_dict({"text": d[text_col].tolist(), "label": d[label_col].astype(int).tolist()})
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN),
                  batched=True, remove_columns=["text"])


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class WeightedTrainer(Trainer):
    """HF Trainer with class-weighted cross-entropy (4.2: handles class imbalance)."""

    def __init__(self, *a: Any, class_weights: torch.Tensor | None = None, **k: Any) -> None:
        super().__init__(*a, **k)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Cross-entropy on the logits, weighted when class_weights is set."""
        labels = inputs.pop("labels")
        out = model(**inputs)
        w = None if self.class_weights is None else self.class_weights.to(out.logits.device)
        loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=w)
        return (loss, out) if return_outputs else loss


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def metrics(y: np.ndarray, prob: np.ndarray, labels: list[str]) -> dict[str, Any]:
    """Binary metrics from proposal 4.5 (class 1 is positive): accuracy, macro-F1,
    per-class scores, positive recall / false-positive rate / PR-AUC."""
    pred = prob.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    neg = y != 1
    return {
        "n": int(len(y)), "n_positive": int((y == 1).sum()),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0),
        "per_class": {labels[i]: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
                      for i in range(2)},
        "positive_recall": r[1],
        "positive_false_positive_rate": float(((pred == 1) & neg).sum() / max(neg.sum(), 1)),
        "positive_pr_auc": average_precision_score((y == 1).astype(int), prob[:, 1])
                           if 0 < (y == 1).sum() < len(y) else None,
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


# --------------------------------------------------------------------------
# Training and evaluation of one run
# --------------------------------------------------------------------------
def train_and_eval(exp: str, tag: str, train_d: pd.DataFrame, val_d: pd.DataFrame,
                   tests: dict[str, pd.DataFrame], labels: list[str], text_col: str,
                   label_col: str, extra: dict[str, Any]) -> list[dict[str, Any]]:
    """Train once on train_d and evaluate on source val and every test set in tests
    (name -> DataFrame). Writes metrics and predictions; returns the summary rows."""
    if DRY_RUN:
        train_d = train_d.groupby(label_col, group_keys=False).head(DRY_RUN // 2)
        val_d = val_d.head(DRY_RUN)
        tests = {k: v.groupby(label_col, group_keys=False).head(DRY_RUN // 2) for k, v in tests.items()}
    set_seed(SEED)
    tr = to_ds(train_d, text_col, label_col)
    counts = np.bincount(tr["label"], minlength=2)
    w = torch.tensor(len(tr) / (2 * np.maximum(counts, 1)), dtype=torch.float)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)},
        ignore_mismatched_sizes=bool(DRY_RUN))
    args = TrainingArguments(output_dir=f"{OUT}/tmp", num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=64,
        learning_rate=LR, weight_decay=0.01, fp16=FP16, seed=SEED, report_to="none",
        eval_strategy="no", save_strategy="no", logging_strategy="epoch")
    trainer = WeightedTrainer(model=model, args=args, train_dataset=tr,
                              data_collator=DataCollatorWithPadding(tok), class_weights=w)
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0

    detail = {"experiment": exp, "run": tag, **extra, "train_seconds": round(train_s, 1),
              "train_counts": dict(zip(labels, counts.tolist())), "class_weights": w.tolist(),
              "eval": {}}
    rows = []
    for name, d in {"source_val": val_d, **tests}.items():
        if len(d) == 0:
            continue
        p = trainer.predict(to_ds(d, text_col, label_col))
        prob = torch.softmax(torch.tensor(p.predictions, dtype=torch.float32), -1).numpy()
        y = p.label_ids
        m = metrics(y, prob, labels)
        detail["eval"][name] = m
        pd.DataFrame({"sample_id": d["id"].values, "eval_set": name,
                      "true_label": [labels[i] for i in y],
                      "pred_label": [labels[i] for i in prob.argmax(1)],
                      f"score_{labels[1]}": prob[:, 1]}
                     ).to_csv(f"{RES}/pred_{tag}_{name}.csv", index=False)
        rows.append({"experiment": exp, "run": tag, **{k: v for k, v in extra.items() if not isinstance(v, dict)},
                     "eval_set": name, "n": m["n"], "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                     "pos_recall": m["positive_recall"], "pos_fpr": m["positive_false_positive_rate"],
                     "pos_pr_auc": m["positive_pr_auc"], "train_seconds": round(train_s, 1)})
        # CEAS_08 split_seed0: also report results without the CNN Daily Top 10 campaign
        if "ceas_08" in name and "seed0" in tag:
            keep = ~d["template"].str.startswith("thedailytop").values
            m2 = metrics(y[keep], prob[keep], labels)
            detail["eval"][name + "_without_dailytop"] = m2
            rows.append({**rows[-1], "eval_set": name + "_without_dailytop", "n": m2["n"],
                         "macro_f1": m2["macro_f1"], "accuracy": m2["accuracy"],
                         "pos_recall": m2["positive_recall"], "pos_fpr": m2["positive_false_positive_rate"],
                         "pos_pr_auc": m2["positive_pr_auc"]})
    dump(detail, f"{RES}/metrics_{tag}.json")
    del trainer.model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


summary: list[dict[str, Any]] = []


def record(rows: list[dict[str, Any]]) -> None:
    """Append result rows to the summary CSV (rewritten after every run) and print them."""
    summary.extend(rows)
    pd.DataFrame(summary).to_csv(f"{RES}/e3_e4_summary.csv", index=False)
    for r in rows:
        print("[result]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)


def safe(label: str, fn: Callable[[], None]) -> None:
    """Run fn; on error, log the traceback to errors.txt and continue with the next run."""
    try:
        fn()
    except Exception:
        msg = f"### {label}\n{traceback.format_exc()}\n"
        print(msg, flush=True)
        with open(f"{RES}/errors.txt", "a") as f:
            f.write(msg)


# --------------------------------------------------------------------------
# E3: cross-channel transfer
# --------------------------------------------------------------------------
if "E3" in EXPERIMENTS or "E3M" in EXPERIMENTS or "E5" in EXPERIMENTS:
    CH = {"sms": "mendeley", "email": "email_fraud", "phone": "bothbosu"}
    data = {}
    for ch, name in CH.items():
        d = load(name)
        data[ch] = d[d["label_fraud"].notna()].reset_index(drop=True)
        print(f"E3 {ch} ({name}):", data[ch]["label_fraud"].value_counts().to_dict(), flush=True)
    SOURCES = [("sms",), ("email",), ("phone",), ("sms", "email"), ("sms", "phone"), ("email", "phone")]
    # E3M (size-matched control, 5 Experiments table): the two-channel runs again, with the
    # training set randomly subsampled to the size of the larger of its two channels.
    RUNS = ([("E3", s) for s in SOURCES] if "E3" in EXPERIMENTS else []) + \
           ([("E3M", s) for s in SOURCES if len(s) == 2] if "E3M" in EXPERIMENTS else []) + \
           ([("E5", s) for s in SOURCES if len(s) == 1] if "E5" in EXPERIMENTS else [])
    # E5 (robustness, 4.4-4.5): the single-channel models of E3, trained the same way, scored on
    # their own channel's test set: clean, every perturbation kind, and each again after D1.
    if "E5" in EXPERIMENTS:
        try:
            from perturb import e5_test_sets          # Kaggle: perturb.py next to this script
        except ImportError:                           # group repo: src/perturb.py
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
            from src.perturb import e5_test_sets
    for k in SEEDS:
        sc = f"split_seed{k}"
        tests = {f"test_{ch}": d[d[sc] == "test"] for ch, d in data.items()}
        for exp, src in RUNS:
            def run(exp=exp, src=src, k=k, sc=sc, tests=tests):
                """Train on the source channel(s) of this split seed and test on all channels."""
                tag = f"{exp}_{'+'.join(src)}_seed{k}"
                tr_parts, va_parts, overlap = [], [], {}
                for ch in src:
                    d = data[ch]
                    tr, va = d[d[sc] == "train"], d[d[sc] == "val"]
                    for tch, td in tests.items():          # drop source rows that duplicate any test set
                        if tch == f"test_{ch}":
                            continue                        # same-channel split is already template-grouped
                        tr, c1 = remove_overlap(tr, td)
                        va, c2 = remove_overlap(va, td)
                        overlap[f"{ch}->{tch}"] = {"train": c1, "val": c2}
                    tr_parts.append(tr)
                    va_parts.append(va)
                train = pd.concat(tr_parts)
                extra = {"source": "+".join(src), "seed": k, "overlap_removed": overlap}
                if exp == "E3M":
                    n = max(len(t) for t in tr_parts)
                    train = pd.concat(tr_parts, keys=list(src)).sample(n=n, random_state=k)
                    kept = train.index.get_level_values(0).value_counts()
                    extra["n_train"] = n
                    extra["size_matched"] = {"n_train": n, "per_channel": {ch: int(kept.get(ch, 0)) for ch in src}}
                eval_sets = tests
                if exp == "E5":
                    ch = src[0]
                    variants, log = e5_test_sets(tests[f"test_{ch}"], seed=0)
                    log.to_csv(f"{RES}/e5_texts_{ch}_seed{k}.csv.gz", index=False)
                    eval_sets = {f"test_{ch}_{name}": d for name, d in variants.items()}
                record(train_and_eval(exp, tag, train, pd.concat(va_parts), eval_sets,
                                      ["legitimate", "fraud"], "text", "label_fraud", extra))
            safe(f"{exp} {src} seed{k}", run)

# --------------------------------------------------------------------------
# E4: cross-corpus email transfer (CEAS_08 <-> SpamAssassin, raw vs stripped)
# --------------------------------------------------------------------------
if "E4" in EXPERIMENTS:
    corp = {n: load(n) for n in ["ceas_08", "spamassassin"]}
    for n, d in corp.items():
        corp[n] = d[d["label_spam"].notna()].reset_index(drop=True)
        print(f"E4 {n}:", corp[n]["label_spam"].value_counts().to_dict(), flush=True)
    for k in SEEDS:
        sc = f"split_seed{k}"
        for src, tgt in [("spamassassin", "ceas_08"), ("ceas_08", "spamassassin")]:
            for text_col in ["text", "text_stripped"]:
                def run(src=src, tgt=tgt, text_col=text_col, k=k, sc=sc):
                    """Train on src and test on both the src and tgt test sets."""
                    s, t = corp[src], corp[tgt]
                    tgt_test = t[t[sc] == "test"]
                    tr, c1 = remove_overlap(s[s[sc] == "train"], tgt_test)
                    va, c2 = remove_overlap(s[s[sc] == "val"], tgt_test)
                    tests = {f"test_{src}": s[s[sc] == "test"], f"test_{tgt}": tgt_test}
                    variant = "stripped" if text_col == "text_stripped" else "raw"
                    tag = f"E4_{src}_to_{tgt}_{variant}_seed{k}"
                    extra = {"source": src, "target": tgt, "text": variant, "seed": k,
                             "overlap_removed": {"train": c1, "val": c2}}
                    record(train_and_eval("E4", tag, tr, va, tests, ["legitimate", "spam"],
                                          text_col, "label_spam", extra))
                safe(f"E4 {src}->{tgt} {text_col} seed{k}", run)

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
s = pd.DataFrame(summary)
if len(s):
    keys = [c for c in ["experiment", "source", "target", "text", "eval_set"] if c in s]
    agg = s.groupby(keys, dropna=False)[["macro_f1", "pos_recall", "pos_fpr", "pos_pr_auc"]] \
           .agg(["mean", "std", "count"]).round(4)
    agg.to_csv(f"{RES}/e3_e4_mean_std.csv")
    print(agg.to_string())
dump({"lr": LR, "epochs": EPOCHS, "batch_size": BATCH, "seed": SEED, "max_length": MAX_LEN,
      "fp16": FP16, "seeds": SEEDS, "python": platform.python_version(), "torch": torch.__version__,
      "transformers": transformers.__version__,
      "device": torch.cuda.get_device_name(0) if FP16 else "cpu"}, f"{RES}/env.json")
print("All done")
