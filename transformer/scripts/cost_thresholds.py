"""Cost-based decision thresholds for DistilBERT and BiLSTM (EE6405 proposal v2, section 5.5).

Reuses the saved validation and test predictions of E1 (every dataset, original labels)
and E3 (cross-channel, fraud labels); no training is needed and it runs on a CPU.

For an assumed scam rate p and error costs C_FN (missed scam) and C_FP = 1 (false alarm),
the expected cost per message at threshold t is
    Cost(t) = C_FN * p * FNR(t) + C_FP * (1 - p) * FPR(t).
Because the cost is computed from FNR and FPR, the scam rate of our test sets does not
matter; p is the assumed rate in real traffic. Grid: C_FN in {1, 10, 50}, p in {1%, 10%}.
For every run the threshold is chosen on the validation predictions (lowest cost; ties go
to the threshold closest to 0.5) and then applied unchanged to the test predictions.
For E3 the validation set is the source channel's, so a transfer row shows what happens
when a threshold tuned on one channel is used on another.
The score is the predicted probability of the positive class: smishing for the Mendeley
3-class task, fraud for E3, and the dataset's risk class (spam / scam / fraud) for E1.
Reported per setting: FNR, FPR, recall, precision recomputed for p
(recall * p / (recall * p + FPR * (1 - p))), expected cost, and per 10,000 messages the
missed scams (10,000 * p * FNR) and false alarms (10,000 * (1 - p) * FPR), each also at the
default threshold 0.5 for comparison.
DistilBERT uci_replication has no validation set (paper protocol) and is skipped.

Usage:
    python cost_thresholds.py                       # RESULTS_DIR defaults to ./results
    RESULTS_DIR=/path/to/results python cost_thresholds.py

Inputs (inside RESULTS_DIR): results_mendeley/, results_e1/, results_e3_e4/, results_bilstm/.
Outputs: RESULTS_DIR/results_thresholds/threshold_runs.csv (one row per run x eval set x
setting) and threshold_summary.csv (mean and std over the 3 seeds).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "results"))
OUT = RESULTS_DIR / "results_thresholds"
C_FP = 1.0
C_FN_GRID = [1.0, 10.0, 50.0]
P_GRID = [0.01, 0.10]
DEFAULT_T = 0.5
PER = 10_000

# Positive (risk) class of each E1 dataset, as in e1_all_datasets.py / bilstm_experiments.py
E1_POSITIVE = {"mendeley": "smishing", "uci": "spam", "uci_replication": "spam", "bothbosu": "scam",
               "email_fraud": "fraud", "spamassassin": "spam", "ceas_08": "spam"}


@dataclass
class Run:
    """One trained model: its validation predictions and the test sets it was scored on."""
    model: str
    experiment: str
    source: str
    seed: int
    positive: str
    val: Path
    tests: dict[str, Path]


# --------------------------------------------------------------------------
# Data: find the prediction files
# --------------------------------------------------------------------------
def find_runs(res: Path) -> list[Run]:
    """Collect all E1 and E3 runs of both models from the prediction file names."""
    runs: list[Run] = []
    # DistilBERT, Mendeley 3-class (mendeley_full_run.py)
    for k in range(3):
        d = res / "results_mendeley"
        if (d / f"pred_split_seed{k}_val.csv").exists():
            runs.append(Run("distilbert", "E1", "mendeley", k, "smishing", d / f"pred_split_seed{k}_val.csv",
                            {"test": d / f"pred_split_seed{k}_test.csv"}))
    # DistilBERT, other E1 datasets (e1_all_datasets.py)
    for f in sorted((res / "results_e1").glob("pred_*_split_seed*_val.csv")):
        m = re.fullmatch(r"pred_(.+)_split_seed(\d)_val\.csv", f.name)
        name, k = m.group(1), int(m.group(2))
        runs.append(Run("distilbert", "E1", name, k, E1_POSITIVE[name], f,
                        {"test": f.with_name(f.name.replace("_val.csv", "_test.csv"))}))
    # BiLSTM, E1 (bilstm_experiments.py); uci_replication runs are named first30_seed<k>
    for f in sorted((res / "results_bilstm").glob("pred_E1_*_val.csv")):
        m = re.fullmatch(r"pred_E1_(.+?)(?:_first30)?_seed(\d)_val\.csv", f.name)
        name, k = m.group(1), int(m.group(2))
        runs.append(Run("bilstm", "E1", name, k, E1_POSITIVE[name], f,
                        {"test": f.with_name(f.name.replace("_val.csv", "_test.csv"))}))
    # E3 for both models: validation = source channel(s), tests = every channel
    for model, folder, val_suffix in [("distilbert", "results_e3_e4", "source_val"),
                                      ("bilstm", "results_bilstm", "val")]:
        for f in sorted((res / folder).glob(f"pred_E3_*_seed*_{val_suffix}.csv")):
            m = re.fullmatch(rf"pred_E3_(.+)_seed(\d)_{val_suffix}\.csv", f.name)
            src, k = m.group(1), int(m.group(2))
            tests = {f"test_{ch}": f.with_name(f"pred_E3_{src}_seed{k}_test_{ch}.csv")
                     for ch in ["sms", "email", "phone"]}
            runs.append(Run(model, "E3", src, k, "fraud", f, tests))
    return runs


def load_scores(path: Path, positive: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (is_positive 0/1, score of the positive class) from one prediction file."""
    d = pd.read_csv(path)
    return (d["true_label"] == positive).to_numpy(int), d[f"score_{positive}"].to_numpy(float)


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def rates(y: np.ndarray, s: np.ndarray, t: float) -> tuple[float, float]:
    """FNR and FPR when messages with score >= t are flagged."""
    flag = s >= t
    pos, neg = y == 1, y == 0
    fnr = float((~flag & pos).sum() / max(pos.sum(), 1))
    fpr = float((flag & neg).sum() / max(neg.sum(), 1))
    return fnr, fpr


def cost(fnr: float, fpr: float, c_fn: float, p: float) -> float:
    """Expected cost per message (proposal 5.5)."""
    return c_fn * p * fnr + C_FP * (1 - p) * fpr


def choose_threshold(y: np.ndarray, s: np.ndarray, c_fn: float, p: float) -> float:
    """Threshold with the lowest validation cost; ties go to the one closest to 0.5.

    Candidates are the distinct validation scores plus +inf (flag nothing).
    """
    cands = np.append(np.unique(s), np.inf)
    costs = np.array([cost(*rates(y, s, t), c_fn, p) for t in cands])
    best = np.flatnonzero(np.isclose(costs, costs.min(), rtol=0, atol=1e-12))
    return float(cands[best[np.argmin(np.abs(np.minimum(cands[best], 1.0) - DEFAULT_T))]])


def describe(y: np.ndarray, s: np.ndarray, t: float, c_fn: float, p: float, tag: str) -> dict[str, float]:
    """Test-set numbers at threshold t for one cost setting (keys prefixed with tag)."""
    fnr, fpr = rates(y, s, t)
    recall = 1 - fnr
    denom = recall * p + fpr * (1 - p)
    return {f"{tag}fnr": fnr, f"{tag}fpr": fpr, f"{tag}recall": recall,
            f"{tag}precision_at_p": recall * p / denom if denom > 0 else np.nan,
            f"{tag}cost": cost(fnr, fpr, c_fn, p),
            f"{tag}missed_per_10k": PER * p * fnr, f"{tag}false_alarms_per_10k": PER * (1 - p) * fpr}


def analyse(run: Run) -> list[dict[str, object]]:
    """All cost settings for one run: choose t on validation, report it on every test set."""
    yv, sv = load_scores(run.val, run.positive)
    rows = []
    for c_fn in C_FN_GRID:
        for p in P_GRID:
            t = choose_threshold(yv, sv, c_fn, p)
            for eval_set, path in run.tests.items():
                if not path.exists():
                    continue
                yt, st = load_scores(path, run.positive)
                rows.append({"model": run.model, "experiment": run.experiment, "source": run.source,
                             "eval_set": eval_set, "seed": run.seed, "positive": run.positive,
                             "c_fn": c_fn, "p": p, "threshold": t if np.isfinite(t) else np.nan,
                             "flags_nothing": not np.isfinite(t), "n_test": len(yt),
                             **describe(yt, st, t, c_fn, p, ""),
                             **describe(yt, st, DEFAULT_T, c_fn, p, "default_")})
    return rows


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    """Run every E1/E3 run through the cost grid and write the two CSV files."""
    runs = find_runs(RESULTS_DIR)
    print(f"{len(runs)} runs found")
    df = pd.DataFrame([r for run in runs for r in analyse(run)])
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "threshold_runs.csv", index=False)
    keys = ["model", "experiment", "source", "eval_set", "positive", "c_fn", "p"]
    cols = ["threshold", "fnr", "fpr", "recall", "precision_at_p", "cost", "missed_per_10k",
            "false_alarms_per_10k", "default_cost", "default_missed_per_10k", "default_false_alarms_per_10k"]
    summary = df.groupby(keys)[cols].agg(["mean", "std"]).round(4)
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary["n_seeds"] = df.groupby(keys).size()
    summary.reset_index().to_csv(OUT / "threshold_summary.csv", index=False)
    print(f"wrote {len(df)} rows to {OUT}")
    # Headline check: Mendeley test, both models, every cost setting (means over seeds)
    s = summary.reset_index()
    show = s[(s["source"] == "mendeley") & (s["eval_set"] == "test")]
    print(show[["model", "c_fn", "p", "threshold_mean", "recall_mean", "fpr_mean", "missed_per_10k_mean",
                "false_alarms_per_10k_mean", "cost_mean", "default_cost_mean"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
