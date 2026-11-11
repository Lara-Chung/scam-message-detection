"""E1 for DistilBERT: one model per dataset on the original labels (label_orig), EE6405 proposal v2.

Settings reuse the Mendeley E2 result (4.3): lr 3e-5, 3 epochs, batch 16, class weights
computed on the training set only (4.2). Each dataset is trained on split_seed0/1/2;
validation and test metrics and per-message predictions are saved, and the split_seed0
model is kept.
uci_replication follows the original paper's protocol: first 30% of the file for
training, the remaining 70% for testing (no validation set), with 3 training seeds.
For CEAS_08 split_seed0, the test set is also reported without the "THE DAILY TOP 10
from CNN" campaign (template starting with "thedailytop"), which appears only in that test set.
A failing dataset does not stop the others; its traceback goes to results_e1/errors.txt.

Usage (Kaggle: attach the processed datasets, GPU on, then Save & Run All):
    python e1_all_datasets.py
    ONLY=uci,bothbosu python e1_all_datasets.py    # run only some datasets
    DRY_RUN=40 python e1_all_datasets.py           # quick code check on a few rows

Outputs: results_e1/ (e1_summary.csv, e1_mean_std.csv, metrics_<dataset>_<run>.json,
pred_<dataset>_<run>_<val|test>.csv, env.json, errors.txt if any) and
models/distilbert_<dataset>_v1/ (split_seed0 model with task_config.json).
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one T4 only: batch 16 is the real batch

import glob
import json
import platform
import time
import traceback
from typing import Any, Optional

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
DRY_RUN = int(os.environ.get("DRY_RUN", "0"))   # > 0: use this many rows per split to test the code
MODEL = os.environ.get("MODEL", "distilbert/distilbert-base-uncased")
LR, EPOCHS, BATCH, SEED = 3e-5, 3, 16, 42
if DRY_RUN:
    EPOCHS = 1
FP16 = torch.cuda.is_available()

# Datasets: label order (index 0 is the negative class), risk-positive class, its meaning, max_length
DATASETS = [
    # name,            labels,                      positive,  meaning,  max_len
    ("uci",             ["ham", "spam"],             "spam",    "spam",   256),
    ("uci_replication", ["ham", "spam"],             "spam",    "spam",   256),
    ("bothbosu",        ["non-scam", "scam"],        "scam",    "fraud",  512),
    ("email_fraud",     ["legitimate", "fraud"],     "fraud",   "fraud",  512),
    ("spamassassin",    ["legitimate", "spam"],      "spam",    "spam",   512),
    ("ceas_08",         ["legitimate", "spam"],      "spam",    "spam",   512),
]
ONLY = os.environ.get("ONLY")  # e.g. ONLY=uci,bothbosu runs only these
if ONLY:
    DATASETS = [d for d in DATASETS if d[0] in ONLY.split(",")]

RES, MOD = f"{OUT}/results_e1", f"{OUT}/models"
os.makedirs(RES, exist_ok=True)
os.makedirs(MOD, exist_ok=True)
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
def find(name: str) -> str:
    """Return the path of <name>.csv or <name>.csv.gz anywhere under DATA_DIR."""
    hits = [p for p in glob.glob(f"{DATA_DIR}/**/{name}.csv*", recursive=True) if os.path.isfile(p)]
    if not hits:
        raise FileNotFoundError(f"{name}.csv / {name}.csv.gz not found under {DATA_DIR}")
    return hits[0]


def to_ds(d: pd.DataFrame, label2id: dict[str, int], max_len: int) -> Dataset:
    """Tokenize the rows of d (truncated to max_len tokens) into a HF Dataset with labels."""
    ds = Dataset.from_dict({"text": d["text"].fillna("").tolist(),
                            "label": d["label_orig"].map(label2id).tolist()})
    return ds.map(lambda b: tok(b["text"], truncation=True, max_length=max_len),
                  batched=True, remove_columns=["text"])


def trunc_rate(d: pd.DataFrame, max_len: int) -> float:
    """Share of texts longer than max_len tokens, i.e. cut by truncation."""
    lens = [len(x) for x in tok(d["text"].fillna("").tolist(), truncation=False)["input_ids"]]
    return float(np.mean(np.array(lens) > max_len))


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
def metrics(y: np.ndarray, prob: np.ndarray, labels: list[str], pos: int) -> dict[str, Any]:
    """Metrics from proposal 4.5: accuracy, macro-F1, per-class scores and the positive
    class's recall / false-positive rate / PR-AUC (argmax prediction)."""
    pred = prob.argmax(1)
    k = len(labels)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(k)), zero_division=0)
    neg = y != pos
    return {
        "n": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "per_class": {labels[i]: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
                      for i in range(k)},
        "positive_label": labels[pos],
        "positive_recall": r[pos],
        "positive_false_positive_rate": float(((pred == pos) & neg).sum() / max(neg.sum(), 1)),
        "positive_pr_auc": average_precision_score((y == pos).astype(int), prob[:, pos])
                           if 0 < (y == pos).sum() < len(y) else None,
        "confusion_matrix": {"labels": labels,
                             "rows_true_cols_pred": confusion_matrix(y, pred, labels=list(range(k))).tolist()},
    }


# --------------------------------------------------------------------------
# Training and evaluation of one run
# --------------------------------------------------------------------------
def run_one(name: str, labels: list[str], pos_label: str, meaning: str, max_len: int,
            df: pd.DataFrame, train_d: pd.DataFrame, val_d: Optional[pd.DataFrame],
            test_d: pd.DataFrame, tag: str, train_seed: int,
            save_dir: Optional[str] = None) -> dict[str, Any]:
    """Fine-tune DistilBERT on train_d, evaluate on val_d / test_d and write metrics and
    predictions (E1). Saves the model to save_dir when given."""
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    pos = label2id[pos_label]
    if DRY_RUN:
        train_d, test_d = train_d.head(DRY_RUN), test_d.head(DRY_RUN)
        val_d = val_d.head(DRY_RUN) if val_d is not None else None
    set_seed(train_seed)
    tr = to_ds(train_d, label2id, max_len)
    counts = np.bincount(tr["label"], minlength=len(labels))
    w = torch.tensor(len(tr) / (len(labels) * np.maximum(counts, 1)), dtype=torch.float)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(labels), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=bool(DRY_RUN))
    args = TrainingArguments(output_dir=f"{OUT}/tmp", num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=64,
        learning_rate=LR, weight_decay=0.01, fp16=FP16, seed=train_seed, report_to="none",
        eval_strategy="no", save_strategy="no", logging_strategy="epoch")
    trainer = WeightedTrainer(model=model, args=args, train_dataset=tr,
                              data_collator=DataCollatorWithPadding(tok), class_weights=w)
    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0

    out = {"dataset": name, "run": tag, "train_seed": train_seed, "max_length": max_len,
           "train_seconds": round(train_s, 1), "train_counts": dict(zip(labels, counts.tolist())),
           "class_weights": w.tolist(), "train_truncated_rate": trunc_rate(train_d, max_len)}
    for split, d in [("val", val_d), ("test", test_d)]:
        if d is None:
            continue
        t0 = time.time()
        p = trainer.predict(to_ds(d, label2id, max_len))
        secs = time.time() - t0
        prob = torch.softmax(torch.tensor(p.predictions, dtype=torch.float32), -1).numpy()
        y = p.label_ids
        out[split] = metrics(y, prob, labels, pos)
        out[split]["truncated_rate"] = trunc_rate(d, max_len)
        out[split]["predict_ms_per_sample"] = round(secs * 1000 / len(d), 2)
        pd.DataFrame({"sample_id": d["id"].values, "split": split,
                      "true_label": [labels[i] for i in y],
                      "pred_label": [labels[i] for i in prob.argmax(1)],
                      **{f"score_{l}": prob[:, i] for i, l in enumerate(labels)}}
                     ).to_csv(f"{RES}/pred_{name}_{tag}_{split}.csv", index=False)
        if name == "ceas_08" and split == "test" and tag == "split_seed0":
            keep = ~d["template"].fillna("").str.startswith("thedailytop").values
            out["test_without_dailytop"] = metrics(y[keep], prob[keep], labels, pos)
            out["test_without_dailytop"]["removed"] = int((~keep).sum())

    if save_dir:
        trainer.save_model(save_dir)
        tok.save_pretrained(save_dir)
        dump({"model_id": os.path.basename(save_dir), "task_id": f"{name}_{'_'.join(labels)}",
              "labels": labels, "positive_labels": [pos_label],
              "positive_meaning": meaning, "default_threshold": None,
              "max_length": max_len, "base_model": MODEL,
              "train_config": {"lr": LR, "epochs": EPOCHS, "batch_size": BATCH, "class_weights": True,
                               "class_weight_values": w.tolist(), "seed": train_seed, "split": tag,
                               "fp16": FP16}}, f"{save_dir}/task_config.json")
    dump(out, f"{RES}/metrics_{name}_{tag}.json")
    del trainer.model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# Main: all datasets x 3 runs
# --------------------------------------------------------------------------
summary: list[dict[str, Any]] = []
for name, labels, pos_label, meaning, max_len in DATASETS:
    try:
        df = pd.read_csv(find(name), dtype={"label_fraud": "Int64", "label_spam": "Int64"})
        df = df[df["label_orig"].isin(labels)].reset_index(drop=True)
        print(f"== {name}: {len(df)} rows", df["label_orig"].value_counts().to_dict(), flush=True)
        if name == "uci_replication":
            cut = int(len(df) * 0.3)   # paper protocol: first 30% train, the rest test
            runs = [(df.iloc[:cut], None, df.iloc[cut:], f"first30_seed{s}", SEED + s) for s in range(3)]
        else:
            runs = [(df[df[f"split_seed{k}"] == "train"], df[df[f"split_seed{k}"] == "val"],
                     df[df[f"split_seed{k}"] == "test"], f"split_seed{k}", SEED) for k in range(3)]
        for i, (tr, va, te, tag, ts) in enumerate(runs):
            save = f"{MOD}/distilbert_{name}_v1" if i == 0 else None
            o = run_one(name, labels, pos_label, meaning, max_len, df, tr, va, te, tag, ts, save)
            row = {"dataset": name, "run": tag, "max_length": max_len,
                   "val_macro_f1": o.get("val", {}).get("macro_f1"),
                   "test_macro_f1": o["test"]["macro_f1"], "test_accuracy": o["test"]["accuracy"],
                   "positive": pos_label, "test_pos_recall": o["test"]["positive_recall"],
                   "test_pos_fpr": o["test"]["positive_false_positive_rate"],
                   "test_pos_pr_auc": o["test"]["positive_pr_auc"],
                   "test_truncated_rate": o["test"]["truncated_rate"],
                   "train_seconds": o["train_seconds"]}
            if "test_without_dailytop" in o:
                row["test_macro_f1_without_dailytop"] = o["test_without_dailytop"]["macro_f1"]
            summary.append(row)
            pd.DataFrame(summary).to_csv(f"{RES}/e1_summary.csv", index=False)  # rewritten after every run
            print("[E1]", row, flush=True)
    except Exception:
        msg = f"### {name}\n{traceback.format_exc()}\n"
        print(msg, flush=True)
        with open(f"{RES}/errors.txt", "a") as f:
            f.write(msg)

s = pd.DataFrame(summary)
if len(s):
    num = s.select_dtypes("number").columns
    agg = s.groupby("dataset")[list(num)].agg(["mean", "std"]).round(4)
    agg.to_csv(f"{RES}/e1_mean_std.csv")
    print(agg.to_string())
dump({"lr": LR, "epochs": EPOCHS, "batch_size": BATCH, "seed": SEED, "fp16": FP16,
      "python": platform.python_version(), "torch": torch.__version__,
      "transformers": transformers.__version__,
      "device": torch.cuda.get_device_name(0) if FP16 else "cpu"}, f"{RES}/env.json")
print("All done")
