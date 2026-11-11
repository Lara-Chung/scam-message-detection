"""DistilBERT on Mendeley SMS (3 classes: ham / spam / smishing), following the EE6405 proposal v2.

Runs in one go (Kaggle, background commit):
  Stage 1, E2 tuning (4.3), validation set only: learning rate {2e-5, 3e-5, 5e-5} x
  epochs {2, 3, 4}, all with class weights; then the best setting once more without
  class weights as a control.
  Stage 2, final evaluation: the setting chosen in stage 1 is trained once on each of
  split_seed0/1/2; validation and test metrics and per-message predictions are saved.
  The split_seed0 model is exported for the frontend (prediction protocol v0.1).

Model (4.2): distilbert-base-uncased with a 3-way classification head, max 256 tokens,
batch 16, weight decay 0.01, fp16 on GPU, weighted cross-entropy (weights from the
training set only). Only "smishing" is the fraud positive class; "spam" is marketing.

Usage (Kaggle: attach the Mendeley dataset, GPU on, then Save & Run All):
    python mendeley_full_run.py
    DRY_RUN=40 python mendeley_full_run.py     # quick code check on a few rows

Outputs: results/ (e2_tuning_val.csv, metrics_split_seed<k>.json,
pred_split_seed<k>_<val|test>.csv, final_3seeds.csv, final_summary.json) and
models/distilbert_mendeley_3class_v1/ (weights, tokenizer, task_config.json).
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # one T4 only: batch 16 is the real batch, no gather warning

import glob
import json
import platform
import time
from typing import Any

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
DATA = os.environ.get("DATA") or glob.glob("/kaggle/input/**/mendeley.csv", recursive=True)[0]
OUT = os.environ.get("OUT", "/kaggle/working")
DRY_RUN = int(os.environ.get("DRY_RUN", "0"))       # > 0: use a few rows per split to test the code
MODEL = "distilbert/distilbert-base-uncased"
LABELS = ["ham", "spam", "smishing"]
POS = "smishing"                                     # only smishing is the fraud class; spam is not
label2id = {l: i for i, l in enumerate(LABELS)}
id2label = {i: l for l, i in label2id.items()}
MAX_LEN, BATCH, SEED = 256, 16, 42
LRS, EPOCHS = [2e-5, 3e-5, 5e-5], [2, 3, 4]
if DRY_RUN:
    LRS, EPOCHS = [2e-5, 5e-5], [1]
FP16 = torch.cuda.is_available()

RES, MOD = f"{OUT}/results", f"{OUT}/models"
os.makedirs(RES, exist_ok=True)
os.makedirs(MOD, exist_ok=True)


def dump(obj: Any, path: str) -> None:
    """Write obj as indented JSON, converting numpy scalars to Python numbers."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
df = pd.read_csv(DATA, dtype={"label_fraud": "Int64", "label_spam": "Int64"})
tok = AutoTokenizer.from_pretrained(MODEL)
print("data:", DATA, df.shape, "| device:", torch.cuda.get_device_name(0) if FP16 else "cpu", flush=True)

_cache: dict[tuple[str, str], tuple[Dataset, pd.DataFrame]] = {}


def get_split(split_col: str, split: str) -> tuple[Dataset, pd.DataFrame]:
    """Return (tokenized Dataset, matching raw rows) for one split of one split seed (cached)."""
    key = (split_col, split)
    if key not in _cache:
        d = df[df[split_col] == split].reset_index(drop=True)
        if DRY_RUN:
            d = d.sample(min(DRY_RUN, len(d)), random_state=0).reset_index(drop=True)
        ds = Dataset.from_dict({"text": d["text"].tolist(),
                                "label": d["label_orig"].map(label2id).tolist()})
        ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=MAX_LEN),
                    batched=True, remove_columns=["text"])
        _cache[key] = (ds, d)
    return _cache[key]


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


def train(split_col: str, lr: float, epochs: int,
          use_weights: bool) -> tuple[Trainer, float, np.ndarray, torch.Tensor]:
    """Fine-tune DistilBERT on the training part of split_col (4.2).

    Returns the trainer, training seconds, training class counts and the class weights.
    """
    set_seed(SEED)
    tr, _ = get_split(split_col, "train")
    counts = np.bincount(tr["label"], minlength=len(LABELS))
    w = torch.tensor(len(tr) / (len(LABELS) * np.maximum(counts, 1)), dtype=torch.float)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(LABELS), id2label=id2label, label2id=label2id)
    args = TrainingArguments(output_dir=f"{OUT}/tmp", num_train_epochs=epochs,
        per_device_train_batch_size=BATCH, per_device_eval_batch_size=64,
        learning_rate=lr, weight_decay=0.01, fp16=FP16, seed=SEED, report_to="none",
        eval_strategy="no", save_strategy="no", logging_strategy="epoch")
    trainer = WeightedTrainer(model=model, args=args, train_dataset=tr,
                              data_collator=DataCollatorWithPadding(tok),
                              class_weights=w if use_weights else None)
    t0 = time.time()
    trainer.train()
    secs = time.time() - t0
    return trainer, secs, counts, w


def predict(trainer: Trainer, split_col: str, split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return (raw rows, true label ids, softmax probabilities) for one split."""
    ds, d = get_split(split_col, split)
    p = trainer.predict(ds)
    prob = torch.softmax(torch.tensor(p.predictions, dtype=torch.float32), -1).numpy()
    return d, p.label_ids, prob


def free(trainer: Trainer) -> None:
    """Release the model and GPU memory between runs."""
    del trainer.model
    del trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def metrics(y: np.ndarray, prob: np.ndarray) -> dict[str, Any]:
    """Metrics from proposal 4.5: accuracy, macro-F1, per-class scores and fraud-class
    recall / false-positive rate / PR-AUC (smishing one-vs-rest, argmax prediction)."""
    pred = prob.argmax(1)
    pos = label2id[POS]
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(len(LABELS))), zero_division=0)
    neg = y != pos
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "per_class": {LABELS[i]: {"precision": p[i], "recall": r[i], "f1": f[i], "support": int(s[i])}
                      for i in range(len(LABELS))},
        "smishing_recall": r[pos],
        "smishing_false_positive_rate": float(((pred == pos) & neg).sum() / max(neg.sum(), 1)),
        "smishing_pr_auc": average_precision_score((y == pos).astype(int), prob[:, pos]),
        "confusion_matrix": {"labels": LABELS,
                             "rows_true_cols_pred": confusion_matrix(y, pred, labels=list(range(len(LABELS)))).tolist()},
    }


# --------------------------------------------------------------------------
# Stage 1: E2 tuning (split_seed0, validation set only)
# --------------------------------------------------------------------------
grid: list[dict[str, Any]] = []


def grid_run(lr: float, ep: int, use_w: bool) -> None:
    """Train one E2 setting, score it on validation and append it to e2_tuning_val.csv."""
    trainer, secs, _, _ = train("split_seed0", lr, ep, use_w)
    _, y, prob = predict(trainer, "split_seed0", "val")
    m = metrics(y, prob)
    row = {"lr": lr, "epochs": ep, "class_weights": use_w, "val_macro_f1": m["macro_f1"],
           "val_accuracy": m["accuracy"], "val_smishing_recall": m["smishing_recall"],
           "val_spam_f1": m["per_class"]["spam"]["f1"], "train_seconds": round(secs, 1)}
    grid.append(row)
    pd.DataFrame(grid).to_csv(f"{RES}/e2_tuning_val.csv", index=False)  # rewritten after every run
    print("[E2]", row, flush=True)
    free(trainer)


for lr in LRS:
    for ep in EPOCHS:
        grid_run(lr, ep, True)
best_w = max(grid, key=lambda g: g["val_macro_f1"])
grid_run(best_w["lr"], best_w["epochs"], False)       # control run without class weights
best = max(grid, key=lambda g: g["val_macro_f1"])     # final setting: highest validation macro-F1
print("[E2] selected setting:", best, flush=True)

# --------------------------------------------------------------------------
# Stage 2: train the selected setting on the 3 split seeds and evaluate
# --------------------------------------------------------------------------
summary: list[dict[str, Any]] = []
for k in [0, 1, 2]:
    sc = f"split_seed{k}"
    trainer, secs, counts, w = train(sc, best["lr"], best["epochs"], best["class_weights"])
    out: dict[str, Any] = {"split": sc, "train_seconds": round(secs, 1)}
    for split in ["val", "test"]:
        d, y, prob = predict(trainer, sc, split)
        m = metrics(y, prob)
        out[split] = m
        pd.DataFrame({
            "sample_id": d["id"], "split": split, "true_label": [LABELS[i] for i in y],
            "pred_label": [LABELS[i] for i in prob.argmax(1)],
            **{f"score_{l}": prob[:, i] for i, l in enumerate(LABELS)},
        }).to_csv(f"{RES}/pred_{sc}_{split}.csv", index=False)
    dump(out, f"{RES}/metrics_{sc}.json")
    summary.append({"split": sc, "val_macro_f1": out["val"]["macro_f1"],
                    "test_macro_f1": out["test"]["macro_f1"], "test_accuracy": out["test"]["accuracy"],
                    "test_smishing_recall": out["test"]["smishing_recall"],
                    "test_smishing_fpr": out["test"]["smishing_false_positive_rate"],
                    "test_smishing_pr_auc": out["test"]["smishing_pr_auc"],
                    "test_spam_f1": out["test"]["per_class"]["spam"]["f1"], "train_seconds": out["train_seconds"]})
    print("[final]", summary[-1], flush=True)

    if k == 0:  # export the model used by the frontend / agent (prediction protocol v0.1)
        save = f"{MOD}/distilbert_mendeley_3class_v1"
        trainer.save_model(save)
        tok.save_pretrained(save)
        dump({
            "model_id": "distilbert_mendeley_3class_v1",
            "task_id": "mendeley_sms_3class",
            "labels": LABELS,
            "positive_labels": [POS],
            "positive_meaning": "fraud (smishing); spam means marketing spam, not fraud",
            "default_threshold": None,   # set later from the validation set
            "max_length": MAX_LEN, "base_model": MODEL,
            "train_config": {"lr": best["lr"], "epochs": best["epochs"], "batch_size": BATCH,
                             "class_weights": best["class_weights"], "class_weight_values": w.tolist(),
                             "train_counts": dict(zip(LABELS, counts.tolist())),
                             "weight_decay": 0.01, "fp16": FP16, "seed": SEED, "split": sc},
            "data": os.path.basename(DATA),
        }, f"{save}/task_config.json")
    free(trainer)

s = pd.DataFrame(summary)
s.to_csv(f"{RES}/final_3seeds.csv", index=False)
dump({"selected": best, "mean": s.drop(columns="split").mean().to_dict(),
      "std": s.drop(columns="split").std().to_dict(),
      "env": {"python": platform.python_version(), "torch": torch.__version__,
              "transformers": transformers.__version__,
              "device": torch.cuda.get_device_name(0) if FP16 else "cpu"}},
     f"{RES}/final_summary.json")
print(s.to_string(index=False))
print("All done")
