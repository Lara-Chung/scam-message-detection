"""E5 robustness summary for DistilBERT and BiLSTM (EE6405 proposal v2, 4.4, 4.5 and 5.3).

Reads the E5 prediction files written by e3_e4_transfer.py (DistilBERT) and
bilstm_experiments.py (BiLSTM) and reports, per model, channel, perturbation and defence:
  - fraud recall and false alarm rate (FPR) on the changed text, next to the clean text;
  - flip rate: the share of scams detected on the clean text that the change turns into
    "legitimate" (proposal 5.3, robustness metric);
  - macro-F1.
Rows with "+D1" use the normalisation defence; "clean+D1" checks that D1 does not create
false alarms on clean text. Means and standard deviations are over the 3 split seeds.
It also draws a random sample of 50 changed test messages for the hand check of 4.4
(readable, meaning kept).

Usage:
    python e5_analysis.py                     # RESULTS_DIR defaults to ./results

Inputs (inside RESULTS_DIR): results_e3_e4/pred_E5_*.csv and results_bilstm/pred_E5_*.csv,
and e5_texts_*.csv.gz from either folder.
Outputs: RESULTS_DIR/results_e5/ with e5_runs.csv (one row per model x channel x seed x
eval set), e5_summary.csv (mean and std over seeds) and e5_hand_check_sample.csv.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))
OUT = RESULTS_DIR / "results_e5"
FOLDERS = {"distilbert": "results_e3_e4", "bilstm": "results_bilstm"}
NAME_RE = re.compile(r"pred_E5_(sms|email|phone)_seed(\d)_test_\1_(.+)\.csv")
POS, NEG = "fraud", "legitimate"
N_HAND, HAND_SEED = 50, 0


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_runs(res: Path) -> pd.DataFrame:
    """All E5 predictions as one long table: model, channel, seed, eval set and the labels."""
    parts = []
    for model, folder in FOLDERS.items():
        # Current exports group robustness predictions by model. Retain support
        # for older bundles that placed them beside E1/E3 prediction files.
        exported = res / "results_e5" / "predictions" / model
        directory = exported if exported.is_dir() else res / folder
        for f in sorted(directory.glob("pred_E5_*.csv")):
            m = NAME_RE.fullmatch(f.name)
            if not m:
                continue
            d = pd.read_csv(f, usecols=["sample_id", "true_label", "pred_label"])
            d["model"], d["channel"], d["seed"], d["eval_set"] = model, m.group(1), int(m.group(2)), m.group(3)
            parts.append(d)
    if not parts:
        raise FileNotFoundError(f"no pred_E5_*.csv files under {res}")
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def score(d: pd.DataFrame, clean: pd.DataFrame) -> dict[str, float]:
    """Recall, FPR, macro-F1 and flip rate of one eval set against the clean predictions."""
    y, p = d["true_label"].to_numpy(), d["pred_label"].to_numpy()
    pos, neg = y == POS, y == NEG
    caught = clean.set_index("sample_id")["pred_label"].eq(POS) & clean.set_index("sample_id")["true_label"].eq(POS)
    caught_ids = caught[caught].index
    now = d.set_index("sample_id").loc[caught_ids, "pred_label"]
    return {"n": len(d), "fraud_recall": float((p[pos] == POS).mean()) if pos.any() else np.nan,
            "false_alarm_rate": float((p[neg] == POS).mean()) if neg.any() else np.nan,
            "macro_f1": f1_score(y, p, labels=[NEG, POS], average="macro", zero_division=0),
            "n_caught_clean": len(caught_ids),
            "flip_rate": float((now != POS).mean()) if len(caught_ids) else np.nan}


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    """Score every eval set of every model x channel x seed; flips are measured against "clean"."""
    rows = []
    for (model, ch, k), g in df.groupby(["model", "channel", "seed"]):
        clean = g[g["eval_set"] == "clean"]
        for ev, d in g.groupby("eval_set"):
            kind, _, defence = ev.partition("+")
            rows.append({"model": model, "channel": ch, "seed": k, "eval_set": ev, "perturbation": kind,
                         "defence": defence or "none", **score(d, clean)})
    return pd.DataFrame(rows)


def hand_check(res: Path) -> pd.DataFrame:
    """50 random changed test messages (scam and legitimate; all channels and kinds, seed 0) for the hand check."""
    logs = []
    for folder in FOLDERS.values():
        for f in sorted((res / folder).glob("e5_texts_*_seed0.csv.gz")):
            ch = f.name.split("_")[2]
            d = pd.read_csv(f)
            d["channel"] = ch
            logs.append(d)
        if logs:
            break        # both models use the same changed texts; one copy is enough
    if not logs:
        return pd.DataFrame()
    d = pd.concat(logs, ignore_index=True)
    d = d[d["n_changed"] > 0]
    sample = d.sample(min(N_HAND, len(d)), random_state=HAND_SEED)
    return sample.assign(readable="", meaning_kept="", note="")[
        ["channel", "kind", "sample_id", "n_eligible", "n_changed", "original", "text", "readable",
         "meaning_kept", "note"]]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    """Write the per-run table, the summary over seeds and the hand-check sample."""
    runs = per_run(load_runs(RESULTS_DIR))
    OUT.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUT / "e5_runs.csv", index=False)
    keys = ["model", "channel", "perturbation", "defence"]
    cols = ["fraud_recall", "false_alarm_rate", "flip_rate", "macro_f1"]
    summary = runs.groupby(keys)[cols].agg(["mean", "std"]).round(4)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary["n_seeds"] = runs.groupby(keys).size()
    summary = summary.reset_index()
    summary.to_csv(OUT / "e5_summary.csv", index=False)
    hc = hand_check(RESULTS_DIR)
    hc.to_csv(OUT / "e5_hand_check_sample.csv", index=False)

    pd.set_option("display.width", 220)
    view = summary.pivot_table(index=["model", "channel", "perturbation"], columns="defence",
                               values="fraud_recall_mean")[["none", "D1"]].round(3)
    print("Fraud recall (mean over seeds); columns: no defence vs D1")
    print(view.to_string())
    print(f"\nwrote {len(runs)} rows to {OUT}; hand-check sample: {len(hc)} messages")


if __name__ == "__main__":
    main()
