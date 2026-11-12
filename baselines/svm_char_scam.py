"""Linear SVM scam classifier (character TF-IDF, defence D2), following the EE6405 proposal v2.

Same protocol as nb_scam.py and lr_scam.py so the models are directly comparable:
template-grouped 80/20 splits (5.4) or an official train/test split, TF-IDF +
LinearSVC inside a Pipeline, class_weight="balanced" (4.2), GridSearchCV with 5-fold
CV over C in {0.1, 1, 10} and character n-gram range {(2,4), (3,5)} (4.3), metrics with
bootstrap CIs (5.3), efficiency numbers, and optional cost-based thresholds (5.5).
Assumes the CSV is already preprocessed.

Note: a linear SVM has no probabilities. Predictions use the highest decision score,
PR-AUC is computed from decision scores (it only needs a ranking), and the cost-based
thresholds are in decision-score units (0 = the SVM's default boundary), not 0-1.

Usage (random grouped splits):
    python3 baselines/svm_char_scam.py --data my_data.csv --label-col label_orig --positive-label scam --splits 3 --compare-weights

Usage (official split):
    python3 baselines/svm_char_scam.py --data train.csv --test-data test.csv --label-col label_orig

Outputs (in --out-dir, default outputs_svm_char): saved models, svm_char_results.json,
results_table.csv, grid_*.csv, sensitivity_*.png, weight_comparison.csv.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_fscore_support)
from sklearn.model_selection import (GridSearchCV, GroupShuffleSplit,
                                     StratifiedGroupKFold, cross_val_predict)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

# --------------------------------------------------------------------------
# Config (the proposal keeps these in config.yaml; edit here or use CLI flags)
# --------------------------------------------------------------------------
PARAM_GRID = {
    "tfidf__ngram_range": [(2, 4), (3, 5)],
    "clf__C": [0.1, 1, 10],
}
URL_RE = re.compile(r"https?://\S+|www\.\S+")
DIGITS_RE = re.compile(r"\d+")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def normalise_key(text: str) -> str:
    """Lower-case and collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def template_key(text: str) -> str:
    """Group key: replace URLs and digit runs so near-identical templates match."""
    t = URL_RE.sub(" URL ", normalise_key(text))
    return DIGITS_RE.sub("0", t)


def load_data(path: str, text_col: str, label_col: str) -> pd.DataFrame:
    """Load an already-preprocessed CSV (no cleaning or de-duplication here).
    Only adds the template 'group' column needed for grouped splitting/CV."""
    df = pd.read_csv(path, usecols=[text_col, label_col]).dropna()
    df = df.rename(columns={text_col: "text", label_col: "label"})
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(str)
    df["group"] = df["text"].map(template_key)
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {path}.")
    print(df["label"].value_counts().to_string(), "\n")
    return df


def grouped_split(df: pd.DataFrame, seed: int, test_size: float = 0.2):
    """80/20 split that never puts one template on both sides.

    GroupShuffleSplit is not stratified, so we try a few nearby seeds and keep the
    first split where every class appears in both train and test.
    """
    for offset in range(50):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + offset * 1000)
        tr, te = next(gss.split(df, df["label"], groups=df["group"]))
        if set(df.iloc[tr]["label"]) == set(df.iloc[te]["label"]) == set(df["label"]):
            return tr, te
    raise RuntimeError("Could not find a split containing all classes on both sides.")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
def build_pipeline(class_weighted: bool, seed: int = 42) -> Pipeline:
    """Character TF-IDF + linear SVM. The vectoriser sits inside the Pipeline so it never
    sees validation folds. class_weight="balanced" is native to LinearSVC.
    (The course notebook uses SVC with an RBF kernel; the proposal switches it to
    linear. LinearSVC is the fast linear-kernel implementation.)"""
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", lowercase=True, sublinear_tf=True, min_df=1)),
        ("clf", LinearSVC(class_weight="balanced" if class_weighted else None,
                          max_iter=10000, random_state=seed)),
    ])


def class_scores(model: Pipeline, texts) -> np.ndarray:
    """Decision scores as an (n_samples, n_classes) array. Binary decision_function
    returns one column (for classes_[1]), so we add its negation for classes_[0]."""
    d = model.decision_function(texts)
    return np.column_stack([-d, d]) if d.ndim == 1 else d


def tune(train: pd.DataFrame, class_weighted: bool, seed: int) -> GridSearchCV:
    """GridSearchCV, 5-fold, stratified and grouped by template."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    gs = GridSearchCV(build_pipeline(class_weighted, seed), PARAM_GRID, cv=cv,
                      scoring="f1_macro", n_jobs=-1)
    gs.fit(train["text"], train["label"], groups=train["group"])
    return gs


def grid_table(gs: GridSearchCV) -> pd.DataFrame:
    """Full grid-search results as a tidy table (one row per setting)."""
    r = gs.cv_results_
    return pd.DataFrame({
        "ngram_range": [str(v) for v in r["param_tfidf__ngram_range"]],
        "C": [float(v) for v in r["param_clf__C"]],
        "mean_cv_macro_f1": r["mean_test_score"],
        "std_cv_macro_f1": r["std_test_score"],
    })


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------
def pr_auc(y_true, scores: np.ndarray, classes: list[str]) -> float:
    """Binary: PR-AUC of the second class. Multiclass: one-vs-rest, macro-averaged.
    Works on decision scores because PR-AUC only needs a ranking."""
    if len(classes) == 2:
        return float(average_precision_score(y_true == classes[1], scores[:, 1]))
    y_bin = label_binarize(y_true, classes=classes)
    return float(average_precision_score(y_bin, scores, average="macro"))


def bootstrap_ci(y_true, y_pred, n: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap CI for macro-F1 (1,000 resamples)."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def evaluate(pipe: Pipeline, test: pd.DataFrame, seed: int) -> dict:
    classes = list(pipe.classes_)
    t0 = time.perf_counter()
    scores = class_scores(pipe, test["text"])
    infer_ms = (time.perf_counter() - t0) / len(test) * 1000
    pred = np.array(classes)[scores.argmax(axis=1)]
    y = test["label"].to_numpy()

    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    lo, hi = bootstrap_ci(y, pred, seed=seed)
    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "macro_f1_ci95": [lo, hi],
        "pr_auc": pr_auc(y, scores, classes),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": confusion_matrix(y, pred, labels=classes).tolist(),
        "classes": classes,
        "infer_ms_per_msg": infer_ms,
    }


def summarise(runs: list[dict]) -> dict:
    """Mean and std over splits for the headline numbers and per-class metrics."""
    def ms(vals):
        return float(np.mean(vals)), float(np.std(vals))

    out: dict = {"n_splits": len(runs)}
    for key in ["macro_f1", "pr_auc", "infer_ms_per_msg", "train_seconds",
                "tune_seconds", "model_size_mb"]:
        out[key + "_mean"], out[key + "_std"] = ms([r[key] for r in runs])
    for c in runs[0]["classes"]:
        for m in ["precision", "recall", "f1"]:
            out[f"{m}_{c}_mean"], out[f"{m}_{c}_std"] = ms([r["per_class"][c][m] for r in runs])
    return out


# --------------------------------------------------------------------------
# Cost-based threshold (section 5.5) - scam-vs-rest decision score
# --------------------------------------------------------------------------
def expected_cost(y_pos: np.ndarray, score: np.ndarray, t: float,
                  c_fn: float, c_fp: float, p: float) -> float:
    """Cost(t) = C_FN * p * FNR(t) + C_FP * (1-p) * FPR(t), for assumed scam rate p."""
    flagged = score >= t
    fnr = float(((~flagged) & y_pos).sum() / max(y_pos.sum(), 1))
    fpr = float((flagged & ~y_pos).sum() / max((~y_pos).sum(), 1))
    return c_fn * p * fnr + c_fp * (1 - p) * fpr


def threshold_study(val_y_pos, val_score, test_y_pos, test_score,
                    c_fns=(1, 10, 50), ps=(0.01, 0.10), c_fp: float = 1.0) -> list[dict]:
    """Choose t on validation scores, apply to test; report counts per 10,000 messages.
    Candidate thresholds are quantiles of the validation decision scores."""
    grid = np.unique(np.quantile(val_score, np.linspace(0, 1, 201)))
    rows = []
    for p in ps:
        for c_fn in c_fns:
            t = float(grid[np.argmin([expected_cost(val_y_pos, val_score, g, c_fn, c_fp, p)
                                      for g in grid])])
            flagged = test_score >= t
            fnr = float(((~flagged) & test_y_pos).sum() / max(test_y_pos.sum(), 1))
            fpr = float((flagged & ~test_y_pos).sum() / max((~test_y_pos).sum(), 1))
            recall = 1 - fnr
            denom = recall * p + fpr * (1 - p)
            rows.append({
                "scam_rate": p, "C_FN": c_fn, "C_FP": c_fp, "threshold_score": t,
                "recall": recall, "FPR": fpr,
                "precision_at_p": float(recall * p / denom) if denom > 0 else float("nan"),
                "missed_per_10k": 10_000 * p * fnr,
                "false_alarms_per_10k": 10_000 * (1 - p) * fpr,
            })
    return rows


def out_of_fold_scores(best_params: dict, train: pd.DataFrame, weighted: bool, seed: int) -> np.ndarray:
    """Cross-validated decision scores on the training data (for threshold choice)."""
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    pipe = build_pipeline(weighted, seed).set_params(**best_params)
    d = cross_val_predict(pipe, train["text"], train["label"], groups=train["group"],
                          cv=cv, method="decision_function")
    return np.column_stack([-d, d]) if d.ndim == 1 else d


# --------------------------------------------------------------------------
# Reporting: tables and plots
# --------------------------------------------------------------------------
def save_sensitivity_plot(grids: list[pd.DataFrame], path: Path, title: str) -> None:
    """Macro-F1 versus C for each n-gram range, averaged over splits."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping sensitivity plot (pip install matplotlib).")
        return
    g = pd.concat(grids).groupby(["ngram_range", "C"], as_index=False).agg(
        f1=("mean_cv_macro_f1", "mean"), sd=("mean_cv_macro_f1", "std"))
    fig, ax = plt.subplots(figsize=(6, 4))
    for ngram, sub in g.groupby("ngram_range"):
        sub = sub.sort_values("C")
        ax.errorbar(sub["C"], sub["f1"], yerr=sub["sd"].fillna(0), marker="o",
                    capsize=3, label=f"ngram {ngram}")
    ax.set_xscale("log")
    ax.set_xlabel("C (inverse regularisation strength)")
    ax.set_ylabel("CV macro-F1 (mean over splits)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def results_table(summaries: dict[str, dict]) -> pd.DataFrame:
    """One row per setting (weighted / unweighted) with mean and std columns."""
    return pd.DataFrame([{"setting": tag, **s} for tag, s in summaries.items()])


def weight_comparison(summaries: dict[str, dict], classes: list[str]) -> pd.DataFrame:
    """Per-class recall and macro-F1, weighted versus unweighted (E2)."""
    rows = []
    for c in classes:
        w = summaries["weighted"][f"recall_{c}_mean"]
        u = summaries["unweighted"][f"recall_{c}_mean"]
        rows.append({"metric": f"recall_{c}", "weighted": w, "unweighted": u, "difference": w - u})
    w, u = summaries["weighted"]["macro_f1_mean"], summaries["unweighted"]["macro_f1_mean"]
    rows.append({"metric": "macro_f1", "weighted": w, "unweighted": u, "difference": w - u})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = load_data(args.data, args.text_col, args.label_col)
    fixed_test = load_data(args.test_data, args.text_col, args.label_col) if args.test_data else None
    n_splits = 1 if fixed_test is not None else args.splits
    if fixed_test is not None:
        print("Using the official train/test files: 1 run, no random splits.\n")

    weighted_options = [True, False] if args.compare_weights else [True]
    all_results: dict[str, list[dict]] = {}
    grids: dict[str, list[pd.DataFrame]] = {}

    for weighted in weighted_options:
        tag = "weighted" if weighted else "unweighted"
        all_results[tag], grids[tag] = [], []
        for i in range(n_splits):
            seed = args.seed + i
            if fixed_test is not None:
                train, test = df, fixed_test
            else:
                tr_idx, te_idx = grouped_split(df, seed)
                train = df.iloc[tr_idx].reset_index(drop=True)
                test = df.iloc[te_idx].reset_index(drop=True)

            t0 = time.perf_counter()
            gs = tune(train, weighted, seed)
            tune_s = time.perf_counter() - t0

            grid = grid_table(gs)
            grid.insert(0, "split", i)
            grid.to_csv(out / f"grid_{tag}_split{i}.csv", index=False)
            grids[tag].append(grid)

            t0 = time.perf_counter()
            model = build_pipeline(weighted, seed).set_params(**gs.best_params_)
            model.fit(train["text"], train["label"])
            train_s = time.perf_counter() - t0

            res = evaluate(model, test, seed)
            res.update({"split_seed": seed, "best_params": {k: str(v) for k, v in gs.best_params_.items()},
                        "tune_seconds": tune_s, "train_seconds": train_s})

            model_path = out / f"svm_char_{tag}_split{i}.joblib"
            joblib.dump(model, model_path)  # reuse later, no retraining
            res["model_size_mb"] = model_path.stat().st_size / 1e6

            # Optional cost-based threshold analysis (scam-vs-rest decision score)
            if args.positive_label:
                pos = args.positive_label
                if pos not in model.classes_:
                    raise ValueError(f"--positive-label {pos!r} not in classes {list(model.classes_)}")
                col = list(model.classes_).index(pos)
                oof = out_of_fold_scores(gs.best_params_, train, weighted, seed)
                res["thresholds"] = threshold_study(
                    (train["label"] == pos).to_numpy(), oof[:, col],
                    (test["label"] == pos).to_numpy(), class_scores(model, test["text"])[:, col])

            all_results[tag].append(res)
            print(f"[{tag}] split {i}: macro-F1={res['macro_f1']:.4f} "
                  f"CI95={res['macro_f1_ci95'][0]:.3f}-{res['macro_f1_ci95'][1]:.3f} "
                  f"PR-AUC={res['pr_auc']:.4f} best={gs.best_params_}")
            print(classification_report(test["label"], model.predict(test["text"]), zero_division=0))

    # ---- Summaries, tables, plots -------------------------------------------
    summaries = {tag: summarise(runs) for tag, runs in all_results.items()}
    classes = all_results[next(iter(all_results))][0]["classes"]

    print("\n=== Summary (mean +/- std over splits) ===")
    for tag, s in summaries.items():
        print(f"{tag:11s} macro-F1 {s['macro_f1_mean']:.4f} +/- {s['macro_f1_std']:.4f} | "
              f"PR-AUC {s['pr_auc_mean']:.4f} +/- {s['pr_auc_std']:.4f} | "
              f"infer {s['infer_ms_per_msg_mean']:.3f} ms/msg | size {s['model_size_mb_mean']:.2f} MB")

    results_table(summaries).to_csv(out / "results_table.csv", index=False)

    for tag, g in grids.items():
        save_sensitivity_plot(g, out / f"sensitivity_{tag}.png", f"Linear SVM (char TF-IDF) sensitivity ({tag})")

    if args.compare_weights:
        wc = weight_comparison(summaries, classes)
        wc.to_csv(out / "weight_comparison.csv", index=False)
        print("\n=== Class weights: weighted vs unweighted (mean over splits) ===")
        print(wc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    (out / "svm_char_results.json").write_text(json.dumps({"runs": all_results, "summary": summaries}, indent=2))
    print(f"\nSaved models, tables, plots and svm_char_results.json to {out}/")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="CSV file (or the train CSV if --test-data is given)")
    ap.add_argument("--test-data", default=None,
                    help="Optional test CSV; uses the official split instead of random grouped splits")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--positive-label", default=None,
                    help="Scam class name; enables the cost-based threshold study")
    ap.add_argument("--splits", type=int, default=3, help="Number of random splits (proposal: 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--compare-weights", action="store_true",
                    help="Also run without class weights (E2 comparison)")
    ap.add_argument("--out-dir", default="outputs_svm_char")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())