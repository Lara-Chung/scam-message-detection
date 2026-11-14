"""BiLSTM scam classifier (Keras, trainable embedding), following the EE6405 proposal v2.

Runs E2 (tuning on Mendeley), E1 (every dataset), E3 (cross-channel), E4
(cross-corpus), E3M (E3 two-channel runs with the training set subsampled to the size of
the larger channel) and E5 (robustness to the perturbations of perturb.py, with and without
the D1 defence normalise()) with the shared splits (split_seed0/1/2), so results are
directly comparable with the DistilBERT runs.

Model (4.2): string input -> TextVectorization (adapted on the training set only,
20,000 words) -> Embedding(128, mask_zero) -> Bidirectional LSTM -> Dropout -> softmax.
Training: Adam, batch 64, up to 10 epochs with early stopping on validation loss
(patience 2, best weights restored), class weights from the training set (4.2).
Tuning (4.3): lr {1e-3, 3e-4} x hidden {64, 128} x dropout {0.2, 0.5} on Mendeley
split_seed0, chosen by validation macro-F1, plus one run without class weights.
E3/E4 call remove_overlap (same logic as src/splits.py) before training.
Sequence length: 100 words for SMS, 400 for email and phone (longer texts are cut).

Usage (Kaggle: attach the processed datasets, GPU on, then Save & Run All):
    python bilstm_experiments.py
    EXPERIMENTS=E1,E3 BEST=1e-3,128,0.5 python bilstm_experiments.py   # skip tuning
    EXPERIMENTS=E3M BEST=1e-3,64,0.2 python bilstm_experiments.py      # E3 size-matched control only
    EXPERIMENTS=E5 BEST=1e-3,64,0.2 python bilstm_experiments.py       # E5 robustness (needs perturb.py)

Outputs: results_bilstm/ (bilstm_summary.csv, bilstm_mean_std.csv, e2_tuning_val.csv,
metrics_<run>.json, pred_<run>_<eval set>.csv) and models_bilstm/ (Mendeley-tuned
E1 models for seed 0 and the single-channel fraud models from E3, kept for E5).
"""
from __future__ import annotations

import glob
import json
import os
import platform
import re
import time
import traceback
from typing import Callable, Optional

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input")
OUT = os.environ.get("OUT", "/kaggle/working")
DRY_RUN = int(os.environ.get("DRY_RUN", "0"))        # > 0: use a few rows per split to test the code
EXPERIMENTS = os.environ.get("EXPERIMENTS", "E2,E1,E3,E4").split(",")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]

VOCAB, EMB, BATCH, MAX_EPOCHS, SEED = 20000, 128, 64, 10, 42
if DRY_RUN:
    MAX_EPOCHS = 1
SEQ_LEN = {"sms": 100, "email": 400, "phone": 400}
CHANNEL_OF = {"mendeley": "sms", "uci": "sms", "uci_replication": "sms", "email_fraud": "email",
              "spamassassin": "email", "ceas_08": "email", "bothbosu": "phone"}
E2_GRID = [{"lr": lr, "hidden": h, "dropout": d, "class_weights": True}
           for lr in [1e-3, 3e-4] for h in [64, 128] for d in [0.2, 0.5]]

# Replaced by the E2 result; set BEST=lr,hidden,dropout when E2 is skipped.
BEST = {"lr": 1e-3, "hidden": 64, "dropout": 0.5, "class_weights": True}
if os.environ.get("BEST"):
    _lr, _h, _d = os.environ["BEST"].split(",")
    BEST = {"lr": float(_lr), "hidden": int(_h), "dropout": float(_d), "class_weights": True}

MENDELEY_LABELS = ["ham", "spam", "smishing"]
E1_SETS = [("mendeley", MENDELEY_LABELS, "smishing"), ("uci", ["ham", "spam"], "spam"),
           ("uci_replication", ["ham", "spam"], "spam"), ("bothbosu", ["non-scam", "scam"], "scam"),
           ("email_fraud", ["legitimate", "fraud"], "fraud"),
           ("spamassassin", ["legitimate", "spam"], "spam"), ("ceas_08", ["legitimate", "spam"], "spam")]
E3_CHANNELS = {"sms": "mendeley", "email": "email_fraud", "phone": "bothbosu"}
E3_SOURCES = [("sms",), ("email",), ("phone",), ("sms", "email"), ("sms", "phone"), ("email", "phone")]
# Start token added to every text: an empty or punctuation-only text would otherwise become a
# fully masked sequence, which the cuDNN LSTM kernel on GPU rejects.
PREFIX = "xxbos "

RES, MOD = f"{OUT}/results_bilstm", f"{OUT}/models_bilstm"
os.makedirs(RES, exist_ok=True)
os.makedirs(MOD, exist_ok=True)


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
_cache: dict[str, pd.DataFrame] = {}
SPACES_RE = re.compile(r"\s+")


def load(name: str) -> pd.DataFrame:
    """Load one processed dataset (<name>.csv or .csv.gz anywhere under DATA_DIR)."""
    if name not in _cache:
        hits = [p for p in glob.glob(f"{DATA_DIR}/**/{name}.csv*", recursive=True) if os.path.isfile(p)]
        if not hits:
            raise FileNotFoundError(f"{name}.csv(.gz) not found under {DATA_DIR}")
        df = pd.read_csv(hits[0], dtype={"label_fraud": "Int64", "label_spam": "Int64"})
        for c in ["text", "text_stripped", "template"]:
            if c in df:
                df[c] = df[c].fillna("")
        _cache[name] = df
    return _cache[name]


def dedup_key(text: str) -> str:
    """Lower-case and collapse white space (same as src/data.py)."""
    return SPACES_RE.sub(" ", text).strip().lower()


def remove_overlap(source: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop source rows whose text or template appears in the target test set (same as src/splits.py)."""
    exact = source["text"].map(dedup_key).isin(set(target["text"].map(dedup_key)))
    template = source["template"].isin(set(target["template"]))
    counts = {"exact_text": int(exact.sum()), "template": int(template.sum()),
              "removed": int((exact | template).sum())}
    return source[~(exact | template)], counts


def label_mapper(labels: list[str]) -> Callable[[pd.Series], pd.Series]:
    """Map label names to 0..k-1 in the given order."""
    return lambda s: s.map({l: i for i, l in enumerate(labels)})


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_model(train_texts: list[str], n_classes: int, seq_len: int, cfg: dict,
                seed: int = SEED) -> keras.Model:
    """BiLSTM with a trainable embedding; the vectoriser is part of the model, so it takes raw strings."""
    keras.utils.set_random_seed(seed)
    vec = layers.TextVectorization(max_tokens=VOCAB, output_sequence_length=seq_len)
    vec.adapt(tf.data.Dataset.from_tensor_slices([PREFIX + t for t in train_texts]).batch(256))
    inp = keras.Input(shape=(), dtype="string")
    x = vec(inp)
    x = layers.Embedding(VOCAB, EMB, mask_zero=True)(x)
    x = layers.Bidirectional(layers.LSTM(cfg["hidden"]))(x)
    x = layers.Dropout(cfg["dropout"])(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(cfg["lr"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def as_input(texts: pd.Series) -> tf.Tensor:
    """Raw strings (with the start token) as a 1-D string tensor."""
    return tf.constant([PREFIX + t for t in texts])


def truncated_rate(texts: pd.Series, seq_len: int) -> float:
    """Share of texts with more whitespace-separated words than seq_len (approximate)."""
    return float(np.mean([len(t.split()) > seq_len for t in texts]))


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def metrics(y: np.ndarray, prob: np.ndarray, labels: list[str], pos: int) -> dict:
    """Accuracy, macro-F1, per-class scores, and recall / false-alarm rate / PR-AUC of the positive class (5.3)."""
    pred = prob.argmax(1)
    k = len(labels)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(k)), zero_division=0)
    neg = y != pos
    return {
        "n": int(len(y)),
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=list(range(k)), zero_division=0),
        "per_class": {labels[i]: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
                      for i in range(k)},
        "positive_label": labels[pos],
        "positive_recall": r[pos],
        "positive_false_positive_rate": float(((pred == pos) & neg).sum() / max(neg.sum(), 1)),
        "positive_pr_auc": average_precision_score((y == pos).astype(int), prob[:, pos])
                           if 0 < (y == pos).sum() < len(y) else None,
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(k))).tolist(),
    }


def dump(obj: dict, path: str) -> None:
    """Write JSON, converting numpy scalars."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


def train_eval(exp: str, tag: str, train_d: pd.DataFrame, val_d: Optional[pd.DataFrame],
               tests: dict[str, pd.DataFrame], labels: list[str], pos_label: str, y_col: str,
               text_col: str, seq_len: int, cfg: dict, extra: Optional[dict] = None,
               save_dir: Optional[str] = None, y_map: Optional[Callable] = None,
               seed: int = SEED) -> list[dict]:
    """Train one model, evaluate it on val and every test set, save predictions; return summary rows."""
    extra = dict(extra or {})
    y_map = y_map or (lambda s: s.astype(int))
    if DRY_RUN:
        train_d = train_d.groupby(y_col, group_keys=False).head(DRY_RUN)
        val_d = val_d.head(DRY_RUN) if val_d is not None else None
        tests = {k: v.head(DRY_RUN) for k, v in tests.items()}
    if val_d is None or len(val_d) == 0:   # uci_replication has no val split: last 10% of train for early stopping
        cut = int(len(train_d) * 0.9)
        train_d, val_d = train_d.iloc[:cut], train_d.iloc[cut:]
        extra["early_stop_val"] = "last 10% of the training part"
    pos = labels.index(pos_label)
    ytr, yva = y_map(train_d[y_col]).values, y_map(val_d[y_col]).values

    model = build_model(train_d[text_col].tolist(), len(labels), seq_len, cfg, seed)
    counts = np.bincount(ytr, minlength=len(labels))
    cw = ({i: len(ytr) / (len(labels) * max(c, 1)) for i, c in enumerate(counts)}
          if cfg["class_weights"] else None)
    t0 = time.time()
    hist = model.fit(as_input(train_d[text_col]), ytr,
                     validation_data=(as_input(val_d[text_col]), yva),
                     epochs=MAX_EPOCHS, batch_size=BATCH, class_weight=cw, verbose=2,
                     callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                                              restore_best_weights=True)])
    train_s = time.time() - t0
    detail = {"experiment": exp, "run": tag, **extra, "config": cfg, "seq_len": seq_len, "seed": seed,
              "epochs_run": len(hist.history["loss"]),
              "best_epoch": int(np.argmin(hist.history["val_loss"])) + 1,
              "train_seconds": round(train_s, 1), "train_counts": dict(zip(labels, counts.tolist())),
              "class_weights": cw, "eval": {}}

    rows = []
    for name, d in {"val": val_d, **tests}.items():
        if d is None or len(d) == 0:
            continue
        t0 = time.time()
        prob = model.predict(as_input(d[text_col]), batch_size=256, verbose=0)
        ms = (time.time() - t0) * 1000 / len(d)
        y = y_map(d[y_col]).values
        m = metrics(y, prob, labels, pos)
        m["truncated_rate"] = truncated_rate(d[text_col], seq_len)
        m["predict_ms_per_sample"] = round(ms, 3)
        detail["eval"][name] = m
        pd.DataFrame({"sample_id": d["id"].values, "eval_set": name,
                      "true_label": [labels[i] for i in y],
                      "pred_label": [labels[i] for i in prob.argmax(1)],
                      **{f"score_{l}": prob[:, i] for i, l in enumerate(labels)}}
                     ).to_csv(f"{RES}/pred_{tag}_{name}.csv", index=False)
        row = {"experiment": exp, "run": tag, **{k: v for k, v in extra.items() if not isinstance(v, dict)},
               "eval_set": name, "n": m["n"], "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
               "positive": pos_label, "pos_recall": m["positive_recall"],
               "pos_fpr": m["positive_false_positive_rate"], "pos_pr_auc": m["positive_pr_auc"],
               "truncated_rate": m["truncated_rate"], "train_seconds": round(train_s, 1),
               "best_epoch": detail["best_epoch"]}
        rows.append(row)
        # CEAS_08 seed 0: also report without the "THE DAILY TOP 10 from CNN" campaign (README)
        if "ceas_08" in name and "seed0" in tag:
            keep = ~d["template"].str.startswith("thedailytop").values
            m2 = metrics(y[keep], prob[keep], labels, pos)
            detail["eval"][name + "_without_dailytop"] = m2
            rows.append({**row, "eval_set": name + "_without_dailytop", "n": m2["n"],
                         "macro_f1": m2["macro_f1"], "accuracy": m2["accuracy"],
                         "pos_recall": m2["positive_recall"], "pos_fpr": m2["positive_false_positive_rate"],
                         "pos_pr_auc": m2["positive_pr_auc"]})

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model.save(f"{save_dir}/model.keras")
        dump({"labels": labels, "positive_labels": [pos_label], "seq_len": seq_len, "config": cfg,
              "text_column": text_col, "input_prefix": PREFIX, "run": tag}, f"{save_dir}/task_config.json")
    dump(detail, f"{RES}/metrics_{tag}.json")
    keras.backend.clear_session()
    return rows


summary: list[dict] = []


def record(rows: list[dict]) -> None:
    """Append rows to the running summary and rewrite bilstm_summary.csv."""
    summary.extend(rows)
    pd.DataFrame(summary).to_csv(f"{RES}/bilstm_summary.csv", index=False)
    for r in rows:
        print("[result]", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)


def safe(label: str, fn: Callable[[], None]) -> None:
    """Run one experiment; log any error to errors.txt and carry on with the rest."""
    try:
        fn()
    except Exception:
        msg = f"### {label}\n{traceback.format_exc()}\n"
        print(msg, flush=True)
        with open(f"{RES}/errors.txt", "a") as f:
            f.write(msg)


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def run_e2() -> dict:
    """E2: tune on Mendeley split_seed0 (validation macro-F1), then one run without class weights."""
    grid: list[dict] = []

    def one(cfg: dict) -> None:
        df, sc = load("mendeley"), "split_seed0"
        tag = f"E2_lr{cfg['lr']}_h{cfg['hidden']}_d{cfg['dropout']}_cw{int(cfg['class_weights'])}"
        rows = train_eval("E2", tag, df[df[sc] == "train"], df[df[sc] == "val"], {}, MENDELEY_LABELS,
                          "smishing", "label_orig", "text", SEQ_LEN["sms"], cfg,
                          extra={"dataset": "mendeley", "seed": 0}, y_map=label_mapper(MENDELEY_LABELS))
        record(rows)
        grid.append({**cfg, "val_macro_f1": rows[0]["macro_f1"]})
        pd.DataFrame(grid).to_csv(f"{RES}/e2_tuning_val.csv", index=False)

    for cfg in E2_GRID:
        safe(f"E2 {cfg}", lambda cfg=cfg: one(cfg))
    if not grid:
        return BEST
    b = max(grid, key=lambda g: g["val_macro_f1"])
    safe("E2 no class weights", lambda: one({**{k: b[k] for k in ["lr", "hidden", "dropout"]},
                                             "class_weights": False}))
    b = max(grid, key=lambda g: g["val_macro_f1"])
    return {k: b[k] for k in ["lr", "hidden", "dropout", "class_weights"]}


def run_e1() -> None:
    """E1: every dataset with its original labels; uci_replication uses the published first-30% protocol."""
    for name, labels, pos in E1_SETS:
        for k in SEEDS:
            def one(name=name, labels=labels, pos=pos, k=k) -> None:
                df = load(name)
                df = df[df["label_orig"].isin(labels)]
                if name == "uci_replication":
                    cut = int(len(df) * 0.3)
                    tr, va, te = df.iloc[:cut], None, df.iloc[cut:]
                    tag, seed = f"E1_{name}_first30_seed{k}", SEED + k
                else:
                    sc = f"split_seed{k}"
                    tr, va, te = df[df[sc] == "train"], df[df[sc] == "val"], df[df[sc] == "test"]
                    tag, seed = f"E1_{name}_seed{k}", SEED
                record(train_eval("E1", tag, tr, va, {"test": te}, labels, pos, "label_orig", "text",
                                  SEQ_LEN[CHANNEL_OF[name]], BEST, extra={"dataset": name, "seed": k},
                                  y_map=label_mapper(labels), seed=seed,
                                  save_dir=f"{MOD}/bilstm_{name}_seed0" if k == 0 else None))
            safe(f"E1 {name} seed{k}", one)


def run_e3(size_matched: bool = False) -> None:
    """E3: fraud vs legitimate across channels; single-channel models are saved for E5.

    size_matched=True runs the E3M control instead: only the two-channel sources, with the
    training set randomly subsampled to the size of the larger of its two channels.
    """
    exp = "E3M" if size_matched else "E3"
    data = {ch: load(n)[load(n)["label_fraud"].notna()].reset_index(drop=True)
            for ch, n in E3_CHANNELS.items()}
    for k in SEEDS:
        sc = f"split_seed{k}"
        tests = {f"test_{ch}": d[d[sc] == "test"] for ch, d in data.items()}
        for src in [s for s in E3_SOURCES if len(s) == 2] if size_matched else E3_SOURCES:
            def one(src=src, k=k, sc=sc, tests=tests) -> None:
                tr_parts, va_parts, overlap = [], [], {}
                for ch in src:
                    d = data[ch]
                    tr, va = d[d[sc] == "train"], d[d[sc] == "val"]
                    for tch, td in tests.items():
                        if tch == f"test_{ch}":
                            continue   # the in-channel split is already template-grouped
                        tr, c1 = remove_overlap(tr, td)
                        va, c2 = remove_overlap(va, td)
                        overlap[f"{ch}->{tch}"] = {"train": c1, "val": c2}
                    tr_parts.append(tr)
                    va_parts.append(va)
                save = f"{MOD}/bilstm_fraud_{src[0]}_seed{k}" if len(src) == 1 else None
                extra = {"source": "+".join(src), "seed": k, "overlap_removed": overlap}
                if size_matched:
                    n = max(len(t) for t in tr_parts)
                    train = pd.concat(tr_parts, keys=list(src)).sample(n=n, random_state=k)
                    kept = train.index.get_level_values(0).value_counts()
                    extra["n_train"] = n
                    extra["size_matched"] = {"n_train": n, "per_channel": {ch: int(kept.get(ch, 0)) for ch in src}}
                else:
                    train = pd.concat(tr_parts).sample(frac=1, random_state=k)
                record(train_eval(exp, f"{exp}_{'+'.join(src)}_seed{k}", train, pd.concat(va_parts),
                                  tests, ["legitimate", "fraud"], "fraud", "label_fraud", "text",
                                  max(SEQ_LEN[ch] for ch in src), BEST, extra=extra, save_dir=save))
            safe(f"{exp} {src} seed{k}", one)


def run_e5() -> None:
    """E5 (4.4-4.5): single-channel fraud models (trained as in E3) scored on their own channel's
    test set: clean, every perturbation of perturb.py, and each again after D1 normalise()."""
    try:
        from perturb import e5_test_sets          # Kaggle: perturb.py next to this script
    except ImportError:                           # group repo: src/perturb.py
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from src.perturb import e5_test_sets
    data = {ch: load(n)[load(n)["label_fraud"].notna()].reset_index(drop=True)
            for ch, n in E3_CHANNELS.items()}
    for k in SEEDS:
        sc = f"split_seed{k}"
        tests = {f"test_{ch}": d[d[sc] == "test"] for ch, d in data.items()}
        for ch in E3_CHANNELS:
            def one(ch=ch, k=k, sc=sc, tests=tests) -> None:
                d = data[ch]
                tr, va = d[d[sc] == "train"], d[d[sc] == "val"]
                overlap = {}
                for tch, td in tests.items():   # same training data as the E3 single-channel run
                    if tch == f"test_{ch}":
                        continue
                    tr, c1 = remove_overlap(tr, td)
                    va, c2 = remove_overlap(va, td)
                    overlap[f"{ch}->{tch}"] = {"train": c1, "val": c2}
                variants, log = e5_test_sets(tests[f"test_{ch}"], seed=0)
                log.to_csv(f"{RES}/e5_texts_{ch}_seed{k}.csv.gz", index=False)
                record(train_eval("E5", f"E5_{ch}_seed{k}", tr.sample(frac=1, random_state=k), va,
                                  {f"test_{ch}_{name}": v for name, v in variants.items()},
                                  ["legitimate", "fraud"], "fraud", "label_fraud", "text", SEQ_LEN[ch], BEST,
                                  extra={"source": ch, "seed": k, "overlap_removed": overlap}))
            safe(f"E5 {ch} seed{k}", one)


def run_e4() -> None:
    """E4: CEAS_08 <-> SpamAssassin, raw text and text_stripped (3.4)."""
    corp = {n: load(n)[load(n)["label_spam"].notna()].reset_index(drop=True)
            for n in ["ceas_08", "spamassassin"]}
    for k in SEEDS:
        sc = f"split_seed{k}"
        for src, tgt in [("spamassassin", "ceas_08"), ("ceas_08", "spamassassin")]:
            for text_col in ["text", "text_stripped"]:
                def one(src=src, tgt=tgt, text_col=text_col, k=k, sc=sc) -> None:
                    s, t = corp[src], corp[tgt]
                    tgt_test = t[t[sc] == "test"]
                    tr, c1 = remove_overlap(s[s[sc] == "train"], tgt_test)
                    va, c2 = remove_overlap(s[s[sc] == "val"], tgt_test)
                    variant = "stripped" if text_col == "text_stripped" else "raw"
                    record(train_eval("E4", f"E4_{src}_to_{tgt}_{variant}_seed{k}", tr, va,
                                      {f"test_{src}": s[s[sc] == "test"], f"test_{tgt}": tgt_test},
                                      ["legitimate", "spam"], "spam", "label_spam", text_col,
                                      SEQ_LEN["email"], BEST,
                                      extra={"source": src, "target": tgt, "text": variant, "seed": k,
                                             "overlap_removed": {"train": c1, "val": c2}}))
                safe(f"E4 {src}->{tgt} {text_col} seed{k}", one)


def write_summary() -> None:
    """Mean and std over seeds for every experiment / dataset / eval set, plus the environment."""
    s = pd.DataFrame(summary)
    if len(s):
        keys = [c for c in ["experiment", "dataset", "source", "target", "text", "eval_set"] if c in s]
        s[keys] = s[keys].fillna("-")
        agg = (s.groupby(keys)[["macro_f1", "pos_recall", "pos_fpr", "pos_pr_auc"]]
                .agg(["mean", "std", "count"]).round(4))
        agg.to_csv(f"{RES}/bilstm_mean_std.csv")
        print(agg.to_string())
    dump({"best": BEST, "vocab": VOCAB, "emb": EMB, "batch": BATCH, "max_epochs": MAX_EPOCHS,
          "seed": SEED, "seq_len": SEQ_LEN, "python": platform.python_version(),
          "tensorflow": tf.__version__, "keras": keras.__version__,
          "gpu": str(tf.config.list_physical_devices("GPU"))}, f"{RES}/env.json")


if __name__ == "__main__":
    print("TF", tf.__version__, "| GPU:", tf.config.list_physical_devices("GPU"), flush=True)
    if "E2" in EXPERIMENTS:
        BEST = run_e2()
        dump(BEST, f"{RES}/e2_selected.json")
        print("[E2] selected:", BEST, flush=True)
    if "E1" in EXPERIMENTS:
        run_e1()
    if "E3" in EXPERIMENTS:
        run_e3()
    if "E4" in EXPERIMENTS:
        run_e4()
    if "E3M" in EXPERIMENTS:
        run_e3(size_matched=True)
    if "E5" in EXPERIMENTS:
        run_e5()
    write_summary()
    print("All done")
