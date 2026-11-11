"""Reload check: the exported model must reproduce the split_seed0 test predictions saved during training.

Guards the hand-over to the frontend (proposal 4.6): if the reloaded model gives the
same labels (>= 99.5%) and near-identical scores (max difference < 0.01), the export
and predict.py are correct.

Usage (from the project root):
    python inference/check_reload.py

Needs:
    data/mendeley.csv.gz
    results/pred_split_seed0_test.csv
    models/distilbert_mendeley_3class_v1/

Outputs: printed label agreement, maximum score difference, recomputed test accuracy
and PASS / MISMATCH.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
from predict import DEVICE, predict_batch  # noqa: E402

MODEL_ID = "distilbert_mendeley_3class_v1"

# --------------------------------------------------------------------------
# Data: saved predictions joined with the original test texts
# --------------------------------------------------------------------------
df = pd.read_csv(ROOT / "data" / "mendeley.csv.gz")
old = pd.read_csv(ROOT / "results" / "pred_split_seed0_test.csv")
test = df[df["split_seed0"] == "test"][["id", "text"]]
m = old.merge(test, left_on="sample_id", right_on="id", how="left")
assert m["text"].notna().all(), "some sample_id values are missing from the data; data version may differ"

# --------------------------------------------------------------------------
# Re-predict and compare
# --------------------------------------------------------------------------
print(f"device: {DEVICE}, {len(m)} test messages, predicting...")
t0 = time.time()
res = predict_batch(m["text"].tolist(), MODEL_ID)
print(f"took {time.time() - t0:.1f} s")

bad = [r for r in res if r["status"] != "ok"]
assert not bad, f"{len(bad)} results returned errors: {bad[:3]}"
new_label = [r["predicted_label"] for r in res]
same = (m["pred_label"] == pd.Series(new_label)).mean()
diff = max(abs(r["class_scores"][l] - m.loc[i, f"score_{l}"])
           for i, r in enumerate(res) for l in r["class_scores"])
acc = (pd.Series(new_label) == m["true_label"]).mean()
print(f"label agreement: {same:.4f}")
print(f"max class-score difference: {diff:.6f}")
print(f"recomputed test accuracy: {acc:.4f} (0.9716 at training time)")
print("PASS" if same > 0.995 and diff < 0.01 else "MISMATCH: please send this output")
