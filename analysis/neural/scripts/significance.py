"""Significance tests for DistilBERT vs BiLSTM and the UCI MCC target (EE6405 proposal v2, 5.1 and 5.3).

Reuses the saved test predictions of E1, E3 and E4; no training is needed and it runs on a CPU.

5.3: whenever the two models are scored on the same test set (same split seed, same
messages), we compare them with
  - McNemar's test on their paired correct / wrong outcomes (exact binomial test when
    fewer than 25 messages are discordant, chi-square with continuity correction otherwise);
  - 95% bootstrap confidence intervals (1,000 resamples of the test messages, seed 0) for
    macro-F1 and positive-class recall of each model, and for the paired difference
    DistilBERT minus BiLSTM.
A model is called better on a run only when McNemar gives p < 0.05 and the 95% interval
of the macro-F1 difference excludes 0.
The positive class is smishing (Mendeley), the dataset's risk class for the other E1
sets, fraud for E3 and spam for E4. Predictions are the argmax labels saved by training.

5.1: on UCI with the published 30/70 split (uci_replication), Almeida et al. (2011) report
MCC 0.893 for their best model. We report MCC, spam caught (SC% = spam recall), blocked
hams (BH% = share of ham predicted spam) and accuracy for both models and every seed.

Usage:
    python significance.py                      # RESULTS_DIR defaults to ./results
    RESULTS_DIR=/path/to/results python significance.py

Inputs (inside RESULTS_DIR): results_mendeley/, results_e1/, results_e3_e4/, results_bilstm/.
Outputs: RESULTS_DIR/results_significance/ with significance_runs.csv (one row per paired
run), significance_summary.csv (per comparison over seeds) and uci_mcc_target.csv.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import f1_score, matthews_corrcoef

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parents[1] / "results"))
OUT = RESULTS_DIR / "results_significance"
N_BOOT, BOOT_SEED, ALPHA = 1000, 0, 0.05
MCC_TARGET = 0.893
SEEDS = [0, 1, 2]

E1_POSITIVE = {"mendeley": "smishing", "uci": "spam", "uci_replication": "spam", "bothbosu": "scam",
               "email_fraud": "fraud", "spamassassin": "spam", "ceas_08": "spam"}
E3_SOURCES = ["sms", "email", "phone", "sms+email", "sms+phone", "email+phone"]
E4_RUNS = [(s, t, v) for s, t in [("spamassassin", "ceas_08"), ("ceas_08", "spamassassin")]
           for v in ["raw", "stripped"]]


# --------------------------------------------------------------------------
# Data: pairs of prediction files on the same test set
# --------------------------------------------------------------------------
def find_pairs(res: Path) -> list[dict[str, object]]:
    """List (DistilBERT file, BiLSTM file) pairs scored on the same test set."""
    d_e1, d_e34, bl = res / "results_e1", res / "results_e3_e4", res / "results_bilstm"
    pairs = []
    for k in SEEDS:
        for name, pos in E1_POSITIVE.items():
            if name == "mendeley":
                df = res / "results_mendeley" / f"pred_split_seed{k}_test.csv"
            elif name == "uci_replication":
                df = d_e1 / f"pred_uci_replication_first30_seed{k}_test.csv"
            else:
                df = d_e1 / f"pred_{name}_split_seed{k}_test.csv"
            bf = bl / (f"pred_E1_uci_replication_first30_seed{k}_test.csv" if name == "uci_replication"
                       else f"pred_E1_{name}_seed{k}_test.csv")
            pairs.append({"experiment": "E1", "source": name, "eval_set": "test", "seed": k,
                          "positive": pos, "distilbert": df, "bilstm": bf})
        for src in E3_SOURCES:
            for ch in ["sms", "email", "phone"]:
                f = f"pred_E3_{src}_seed{k}_test_{ch}.csv"
                pairs.append({"experiment": "E3", "source": src, "eval_set": f"test_{ch}", "seed": k,
                              "positive": "fraud", "distilbert": d_e34 / f, "bilstm": bl / f})
        for s, t, v in E4_RUNS:
            for ev in [s, t]:
                f = f"pred_E4_{s}_to_{t}_{v}_seed{k}_test_{ev}.csv"
                pairs.append({"experiment": "E4", "source": f"{s}->{t} ({v})", "eval_set": f"test_{ev}",
                              "seed": k, "positive": "spam", "distilbert": d_e34 / f, "bilstm": bl / f})
    return [p for p in pairs if Path(p["distilbert"]).exists() and Path(p["bilstm"]).exists()]


def load_pair(f_a: Path, f_b: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (true labels, DistilBERT labels, BiLSTM labels) aligned on sample_id."""
    a = pd.read_csv(f_a, usecols=["sample_id", "true_label", "pred_label"])
    b = pd.read_csv(f_b, usecols=["sample_id", "true_label", "pred_label"])
    m = a.merge(b, on="sample_id", suffixes=("_a", "_b"), validate="one_to_one")
    if len(m) != len(a) or len(m) != len(b) or (m["true_label_a"] != m["true_label_b"]).any():
        raise ValueError(f"test sets differ: {f_a.name} vs {f_b.name}")
    return m["true_label_a"].to_numpy(), m["pred_label_a"].to_numpy(), m["pred_label_b"].to_numpy()


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def mcnemar(y: np.ndarray, pa: np.ndarray, pb: np.ndarray) -> tuple[int, int, float]:
    """McNemar's test on paired outcomes; returns (only A correct, only B correct, p-value)."""
    ca, cb = pa == y, pb == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    if b + c == 0:
        return b, c, 1.0
    if b + c < 25:
        return b, c, float(binomtest(b, b + c, 0.5).pvalue)
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return b, c, float(chi2.sf(stat, 1))


def macro_f1(y: np.ndarray, p: np.ndarray, labels: np.ndarray) -> float:
    """Macro-F1 over the given label set."""
    return f1_score(y, p, labels=labels, average="macro", zero_division=0)


def recall(y: np.ndarray, p: np.ndarray, pos: str) -> float:
    """Recall of the positive class (nan if the test set has none)."""
    m = y == pos
    return float((p[m] == pos).mean()) if m.any() else np.nan


def boot_scores(codes: np.ndarray, idx: np.ndarray, k: int, eval_ids: np.ndarray,
                pos_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Macro-F1 and positive recall for every bootstrap resample at once.

    codes = true_id * k + pred_id per message; idx holds one resample per row. Each row's
    confusion matrix is built with one bincount, which is far faster than calling
    sklearn 1,000 times. Same definition as sklearn f1_score(average="macro",
    labels=<true labels of the full test set>, zero_division=0).
    """
    b = idx.shape[0]
    flat = (np.arange(b)[:, None] * k * k + codes[idx]).ravel()
    cm = np.bincount(flat, minlength=b * k * k).reshape(b, k, k).astype(float)
    tp = np.diagonal(cm, axis1=1, axis2=2)
    row, col = cm.sum(2), cm.sum(1)
    denom = row + col                                   # 2tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    rec = np.divide(tp[:, pos_id], row[:, pos_id], out=np.full(b, np.nan), where=row[:, pos_id] > 0)
    return f1[:, eval_ids].mean(1), rec


def bootstrap(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, pos: str) -> dict[str, float]:
    """95% percentile intervals for each model's macro-F1 / recall and for their difference
    (paired: both models are scored on the same resampled messages)."""
    names = np.unique(np.concatenate([y, pa, pb, [pos]]))
    k = len(names)
    yi, ai, bi = (np.searchsorted(names, v) for v in (y, pa, pb))
    eval_ids = np.searchsorted(names, np.unique(y))
    pos_id = int(np.searchsorted(names, pos))
    idx = np.random.default_rng(BOOT_SEED).integers(0, len(y), (N_BOOT, len(y)))
    f_a, r_a = boot_scores(yi * k + ai, idx, k, eval_ids, pos_id)
    f_b, r_b = boot_scores(yi * k + bi, idx, k, eval_ids, pos_id)
    stats = np.column_stack([f_a, f_b, r_a, r_b])
    lo, hi = 100 * ALPHA / 2, 100 * (1 - ALPHA / 2)
    diff_f1 = stats[:, 0] - stats[:, 1]
    diff_rec = stats[:, 2] - stats[:, 3]
    out = {}
    for name, col in [("distilbert_macro_f1", stats[:, 0]), ("bilstm_macro_f1", stats[:, 1]),
                      ("distilbert_recall", stats[:, 2]), ("bilstm_recall", stats[:, 3]),
                      ("diff_macro_f1", diff_f1), ("diff_recall", diff_rec)]:
        out[f"{name}_ci_low"] = float(np.nanpercentile(col, lo))
        out[f"{name}_ci_high"] = float(np.nanpercentile(col, hi))
    return out


def compare(pair: dict[str, object]) -> dict[str, object]:
    """All 5.3 numbers for one paired run."""
    y, pa, pb = load_pair(Path(pair["distilbert"]), Path(pair["bilstm"]))
    pos, labels = str(pair["positive"]), np.unique(y)
    only_a, only_b, p = mcnemar(y, pa, pb)
    ci = bootstrap(y, pa, pb, pos)
    f_a, f_b = macro_f1(y, pa, labels), macro_f1(y, pb, labels)
    if p < ALPHA and ci["diff_macro_f1_ci_low"] > 0:
        verdict = "distilbert better"
    elif p < ALPHA and ci["diff_macro_f1_ci_high"] < 0:
        verdict = "bilstm better"
    else:
        verdict = "no significant difference"
    return {k: v for k, v in pair.items() if k not in ("distilbert", "bilstm")} | {
        "n_test": len(y), "distilbert_macro_f1": f_a, "bilstm_macro_f1": f_b,
        "distilbert_recall": recall(y, pa, pos), "bilstm_recall": recall(y, pb, pos),
        "only_distilbert_correct": only_a, "only_bilstm_correct": only_b, "mcnemar_p": p,
        **ci, "verdict": verdict}


def uci_target(res: Path) -> pd.DataFrame:
    """5.1: MCC, SC%, BH% and accuracy on the published UCI 30/70 split for both models."""
    rows = []
    for model, pattern in [("distilbert", "results_e1/pred_uci_replication_first30_seed{k}_test.csv"),
                           ("bilstm", "results_bilstm/pred_E1_uci_replication_first30_seed{k}_test.csv")]:
        for k in SEEDS:
            f = res / pattern.format(k=k)
            if not f.exists():
                continue
            d = pd.read_csv(f)
            y, p = (d["true_label"] == "spam").to_numpy(int), (d["pred_label"] == "spam").to_numpy(int)
            mcc = matthews_corrcoef(y, p)
            rows.append({"model": model, "seed": k, "n_test": len(d), "mcc": mcc,
                         "spam_caught_pct": 100 * p[y == 1].mean(), "blocked_ham_pct": 100 * p[y == 0].mean(),
                         "accuracy_pct": 100 * (y == p).mean(), "target_mcc": MCC_TARGET,
                         "meets_target": mcc > MCC_TARGET})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    """Run every paired comparison and the UCI target check, then write the CSV files."""
    pairs = find_pairs(RESULTS_DIR)
    print(f"{len(pairs)} paired runs found")
    runs = pd.DataFrame([compare(p) for p in pairs])
    OUT.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUT / "significance_runs.csv", index=False)

    keys = ["experiment", "source", "eval_set"]
    g = runs.groupby(keys, sort=False)
    summary = g[["distilbert_macro_f1", "bilstm_macro_f1"]].mean().round(4)
    summary["mean_diff_macro_f1"] = (summary["distilbert_macro_f1"] - summary["bilstm_macro_f1"]).round(4)
    for v in ["distilbert better", "bilstm better", "no significant difference"]:
        summary[f"seeds_{v.replace(' ', '_')}"] = g["verdict"].apply(lambda s, v=v: int((s == v).sum()))
    summary["max_mcnemar_p"] = g["mcnemar_p"].max()
    summary = summary.reset_index()
    summary.to_csv(OUT / "significance_summary.csv", index=False)

    uci = uci_target(RESULTS_DIR)
    uci.to_csv(OUT / "uci_mcc_target.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nE1, DistilBERT vs BiLSTM (seeds with each verdict):")
    print(summary[summary["experiment"] == "E1"].drop(columns=["experiment", "eval_set"]).to_string(index=False))
    print("\nUCI 30/70 split, target MCC > 0.893:")
    print(uci.round(3).to_string(index=False))
    print(f"\nwrote {len(runs)} paired runs to {OUT}")


if __name__ == "__main__":
    main()
