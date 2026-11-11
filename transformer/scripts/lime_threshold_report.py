"""LIME error analysis (proposal 5.7) and app thresholds (6.1 cost slider), from app_lime_results.zip.

Groups the 50 LIME-explained failures by failure type, draws the figures for the report and
computes cost-optimal thresholds (5.5 formula, chosen on validation) for the app models.

Usage: python lime_threshold_report.py path/to/app_lime_results   (folder with results_app/, results_bilstm/,
results_lime/; cost_thresholds.py must sit next to this script, as in transformer/scripts/)
Outputs (inside that folder, results_report/): lime_categorised.csv, lime_category_summary.csv,
lime_examples.html, fig_lime_categories.png, fig_lime_words.png, app_thresholds.csv / .json, fig_thresholds.png
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cost_thresholds as ct  # noqa: E402

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "app_lime_results")
OUT = ROOT / "results_report"
OUT.mkdir(exist_ok=True)

# Reference palette (dataviz skill): categorical slots 1-2, ink and gridlines
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
                     "xtick.color": MUTED, "ytick.color": MUTED, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False})

FORMAT = {"caller", "receiver", "1234567890"}         # phone transcript format and placeholder numbers
CORPUS = {"jose", "monkey", "org", "ENRON_DEVELOPMENT"}  # dataset artefacts (Nazario recipient, Enron)
CATS = {"A": "Uninformative: score saturated (LIME weights ~0)",
        "B": "Corpus artefact (jose@monkey.org recipient)",
        "C": "Channel format (caller / receiver tags, placeholder numbers)",
        "D": "Perturbed or split words lose their scam signal",
        "E": "Unfamiliar vocabulary of another channel",
        "F": "Other"}


def parse(s: str) -> list[tuple[str, float]]:
    """'word +0.12; word2 -0.05' -> [(word, 0.12), (word2, -0.05)]."""
    return [(a.rsplit(" ", 1)[0], float(a.rsplit(" ", 1)[1])) for a in s.split("; ")]


def category(r) -> str:
    """Rule-based failure type from the top LIME words (checked by hand on all 50 rows)."""
    top = parse(r.lime_top_words)
    if max(abs(v) for _, v in top) < 0.02:
        return "A"
    if any(w in CORPUS and abs(v) >= 0.02 for w, v in top):
        return "B"
    if any(w in FORMAT and v > 0.1 for w, v in top):
        return "C"
    changed = [w for w, _ in top if (re.search(r"\d|@", w) and re.search(r"[a-zA-Z]", w))
               or re.search(r"[^\x00-\x7f]", w) or len(w) <= 2]
    if r.experiment == "E5" and any(abs(v) >= 0.02 for w, v in top if w in changed):
        return "D"
    return "E" if r.experiment == "E3" else "F"


# --------------------------------------------------------------------------
# LIME: categories, figures, highlighted examples
# --------------------------------------------------------------------------
lime = pd.read_csv(ROOT / "results_lime" / "lime_failures.csv")
lime["category"] = [category(r) for r in lime.itertuples()]
lime["category_name"] = lime["category"].map(CATS)
lime.to_csv(OUT / "lime_categorised.csv", index=False)
summary = lime.groupby(["category", "category_name", "experiment"]).size().unstack(fill_value=0)
summary.to_csv(OUT / "lime_category_summary.csv")
print(summary.to_string(), "\n")

# Figure 1: failures per category, E3 vs E5 (stacked horizontal bars, 2px surface gap)
s = summary.droplevel("category_name").reindex(list(CATS)).fillna(0)
s = s[(s.get("E3", 0) + s.get("E5", 0)) > 0].iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 3.6))
left = np.zeros(len(s))
for exp, col in [("E3", BLUE), ("E5", ORANGE)]:
    v = s.get(exp, pd.Series(0, index=s.index)).to_numpy()
    ax.barh(range(len(s)), v, left=left, color=col, height=0.6, edgecolor=SURFACE, linewidth=2,
            label={"E3": "E3 cross-channel", "E5": "E5 perturbation"}[exp])
    left += v
for i, t in enumerate(left):
    ax.text(t + 0.3, i, f"{int(t)}", va="center", color=INK2)
ax.set_yticks(range(len(s)), [CATS[c] for c in s.index], color=INK)
ax.set_xlabel("failures explained with LIME (n = 50)")
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.set_xlim(0, left.max() + 2)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
fig.suptitle("What drove the DistilBERT failures (LIME, 5.7)", x=0.02, ha="left", color=INK)
fig.tight_layout()
fig.savefig(OUT / "fig_lime_categories.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
plt.close(fig)

# Figure 2: words that most often decide a failure (summed |weight| over the 50 explanations)
w = pd.read_csv(ROOT / "results_lime" / "lime_words.csv")
w = w[w["word"].str.len() > 2]                       # drop single letters left by P2
agg = w.assign(a=w.weight_fraud.abs()).groupby("word")["a"].sum().sort_values(ascending=False).head(15)
is_art = agg.index.isin(FORMAT | CORPUS)
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(range(len(agg))[::-1], agg.values, color=[ORANGE if a else BLUE for a in is_art], height=0.6)
ax.set_yticks(range(len(agg))[::-1], agg.index, color=INK)
ax.set_xlabel("summed |LIME weight| over the 50 failures")
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
          labels=["dataset / format artefact", "message content"], frameon=False, loc="lower right")
ax.set_title("Words that most often decided a failure", loc="left", color=INK)
fig.tight_layout()
fig.savefig(OUT / "fig_lime_words.png", dpi=200)
plt.close(fig)


def highlight(text: str, weights: dict[str, float]) -> str:
    """HTML with each LIME word shaded: orange pushes toward fraud, blue toward legitimate."""
    m = max([abs(v) for v in weights.values()] + [1e-9])
    out = []
    for tok in re.split(r"(\W+)", text):
        v = weights.get(tok)
        if v is None:
            out.append(html.escape(tok))
        else:
            rgb = (235, 104, 52) if v > 0 else (42, 120, 214)
            out.append(f'<span style="background:rgba({rgb[0]},{rgb[1]},{rgb[2]},{0.15 + 0.6 * abs(v) / m:.2f});'
                       f'border-radius:3px;padding:0 2px" title="{v:+.3f}">{html.escape(tok)}</span>')
    return "".join(out)


blocks = []
for c in ["B", "C", "D", "E"]:
    for r in lime[lime.category == c].sort_values("lime_r2", ascending=False).head(2).itertuples():
        wts = dict(parse(r.lime_top_words))
        blocks.append(f"<h3>{html.escape(CATS[c])}</h3><p class=meta>{r.experiment} · {r.model} model on "
                      f"{r.eval_set} · {r.failure} · P(fraud) = {r.score_fraud:.3f} · LIME R² = {r.lime_r2:.2f}</p>"
                      f"<div class=msg>{highlight(str(r.text)[:1500], wts)}</div>")
(OUT / "lime_examples.html").write_text(
    "<!doctype html><meta charset=utf-8><title>LIME examples</title><style>body{font:14px/1.5 system-ui;"
    "max-width:900px;margin:24px auto;padding:0 16px;color:#0b0b0b;background:#fcfcfb}.meta{color:#52514e}"
    ".msg{white-space:pre-wrap;border:1px solid #e1e0d9;border-radius:6px;padding:12px}</style>"
    "<h1>LIME examples (5.7)</h1><p>Orange words push toward <b>fraud</b>, blue toward <b>legitimate</b>; "
    "stronger shade = larger weight (hover for the value). Texts cut at 1,500 characters.</p>" + "".join(blocks))

# --------------------------------------------------------------------------
# App thresholds (5.5 formula on validation predictions) for the cost slider
# --------------------------------------------------------------------------
def load(p: Path) -> tuple[np.ndarray, np.ndarray]:
    d = pd.read_csv(p)
    return (d.true_label == "fraud").astype(int).to_numpy(), d.score_fraud.to_numpy()


A, B = ROOT / "results_app", ROOT / "results_bilstm"
MODELS = {"distilbert_fraud_allchannels_v1": (A / "pred_distilbert_allchannels_val.csv",
                                              {c: A / f"pred_distilbert_allchannels_test_{c}.csv" for c in ["sms", "email", "phone"]}),
          "bilstm_fraud_allchannels_seed0": (B / "pred_APP_sms+email+phone_seed0_val.csv",
                                             {c: B / f"pred_APP_sms+email+phone_seed0_test_{c}.csv" for c in ["sms", "email", "phone"]}),
          **{f"distilbert_fraud_{c}_v1": (A / f"pred_distilbert_{c}_val.csv", {}) for c in ["sms", "email", "phone"]}}
CFN, P = [1, 2, 5, 10, 20, 50, 100], [0.01, 0.10]
rows, table = [], {}
for m, (val, tests) in MODELS.items():
    y, sc = load(val)
    table[m] = {}
    for p in P:
        for c in CFN:
            t = min(ct.choose_threshold(y, sc, c, p), 1.0)
            table[m][f"p={p}|C_FN={c}"] = round(t, 4)
            row = {"model": m, "p": p, "C_FN": c, "threshold": round(t, 4)}
            for ch, tp in tests.items():
                yt, st = load(tp)
                fnr, fpr = ct.rates(yt, st, t)
                row[f"{ch}_missed_per10k"] = round(1e4 * p * fnr, 1)
                row[f"{ch}_false_alarms_per10k"] = round(1e4 * (1 - p) * fpr, 1)
            rows.append(row)
th = pd.DataFrame(rows)
th.to_csv(OUT / "app_thresholds.csv", index=False)
json.dump({"note": "Cost-optimal thresholds on P(fraud), chosen on split_seed0 validation predictions "
                   "(5.5: Cost = C_FN*p*FNR + (1-p)*FPR). p = assumed scam rate, C_FN = cost of a missed "
                   "scam relative to one false alarm.", "thresholds": table},
          open(OUT / "app_thresholds.json", "w"), indent=1)

# Figure 3: threshold vs cost ratio, one panel per scam rate, the two all-channel models
fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
for ax, p in zip(axes, P):
    for m, col, name in [("distilbert_fraud_allchannels_v1", BLUE, "DistilBERT"),
                         ("bilstm_fraud_allchannels_seed0", ORANGE, "BiLSTM")]:
        d = th[(th.model == m) & (th.p == p)]
        ax.step(d.C_FN, d.threshold, where="post", color=col, linewidth=2, label=name)
        ax.plot(d.C_FN, d.threshold, "o", color=col, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.set_xscale("log")
    ax.set_xticks(CFN, [str(c) for c in CFN])
    ax.set_xlabel("C_FN (cost of a missed scam / cost of a false alarm)")
    ax.set_title(f"assumed scam rate p = {p:.0%}", loc="left", color=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
axes[0].set_ylabel("cost-optimal threshold on P(fraud)")
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig_thresholds.png", dpi=200)
plt.close(fig)

print(th[th.model.str.contains("allchannels")].to_string(index=False))
print(f"\nwrote {OUT}")
