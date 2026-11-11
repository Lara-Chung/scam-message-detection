# Transformer (DistilBERT) and BiLSTM

Owner: Daniel. This folder holds the DistilBERT and BiLSTM parts of the project: experiments E1–E5,
the E3 size-matched control, the 5.1 UCI target, the 5.3 significance tests, the 5.5 cost-based
thresholds and the prediction interface used by the Streamlit app.
`src/perturb.py` (E5 perturbations, proposal 4.4) and `tests/test_perturb.py` sit in the shared
`src/` and `tests/` folders of the repository.

## Layout

```
transformer/
├── README.md
├── INTERFACE_for_frontend.md      how the app calls predict()
├── scripts/                       training and analysis scripts
├── inference/                     predict.py (prediction protocol v0.1), check_reload.py
├── notebooks/                     the Kaggle notebooks that produced the results, with their outputs
└── results/                       metrics, summaries and per-message predictions of every run
src/perturb.py                     P1-P3 perturbations (E5)
tests/test_perturb.py              pytest checks for perturb.py
```

### scripts/

| Script | What it runs | Where |
|---|---|---|
| `mendeley_full_run.py` | DistilBERT E2 tuning on Mendeley (lr × epochs, class weights) and the final 3-class model, 3 split seeds | Kaggle GPU |
| `e1_all_datasets.py` | DistilBERT E1 on uci, uci_replication (published 30/70 split), bothbosu, email_fraud, spamassassin, ceas_08 | Kaggle GPU |
| `e3_e4_transfer.py` | DistilBERT E3 (cross-channel, fraud labels), E3M (size-matched control), E4 (CEAS_08 ↔ SpamAssassin, raw vs stripped), E5 (robustness) | Kaggle GPU |
| `bilstm_experiments.py` | BiLSTM (Keras) E2, E1, E3, E3M, E4, E5 | Kaggle GPU |
| `cost_thresholds.py` | 5.5 cost-based thresholds from the saved E1/E3 predictions | CPU |
| `significance.py` | 5.3 McNemar tests and bootstrap 95% CIs (DistilBERT vs BiLSTM), 5.1 UCI MCC target | CPU |
| `e5_analysis.py` | E5 summary (fraud recall, false alarm rate, flip rate, with and without D1) and the 50-message hand-check sample | CPU |

All runs use the shared splits `split_seed0/1/2` from `src/splits.py` and report the mean and
standard deviation over the 3 seeds. The training scripts read the processed data
(`*.csv.gz`) from `DATA_DIR` (default `/kaggle/input`); every script's docstring lists its
settings, usage and outputs.

### notebooks/

The Kaggle notebooks (GPU T4) that produced `results/`, kept with their outputs as run records.

| Notebook | Results folder |
|---|---|
| `01_distilbert_mendeley_e2.ipynb` | `results/results_mendeley/` |
| `02_distilbert_e1.ipynb` | `results/results_e1/` |
| `03_distilbert_e3_e4.ipynb` | `results/results_e3_e4/` |
| `04_bilstm_e2_e1_e3_e4.ipynb` | `results/results_bilstm/` |
| `05_e3m_size_matched.ipynb` | `results/results_e3m/` |
| `06_e5_robustness.ipynb` | `results/results_e5/` (with the E5 predictions) |

Notebooks 01–03 were first run with Chinese comments in the code. Their code cells now hold the
English-commented version; the program logic is identical (checked by comparing syntax trees),
and the outputs are the original ones, so a few printed status lines are in Chinese.
Notebooks 04–05 contain the script versions of their run date; the copies in `scripts/` add
later options (E3M, E5) without changing the earlier experiments.

### results/

| Folder | Content |
|---|---|
| `results_mendeley/` | DistilBERT Mendeley 3-class: E2 grid (`e2_tuning_val.csv`), 3-seed results, predictions |
| `results_e1/` | DistilBERT E1: `e1_summary.csv`, `e1_mean_std.csv`, metrics and predictions per run |
| `results_e3_e4/` | DistilBERT E3/E4: `e3_e4_summary.csv`, `e3_e4_mean_std.csv`, metrics and predictions |
| `results_bilstm/` | BiLSTM E2/E1/E3/E4: `bilstm_summary.csv`, `bilstm_mean_std.csv`, metrics and predictions |
| `results_e3m/` | E3 size-matched control for both models; `e3m_comparison.csv` compares full and size-matched training |
| `results_significance/` | 5.3 / 5.1: `significance_summary.csv`, `significance_runs.csv`, `uci_mcc_target.csv` |
| `results_thresholds/` | 5.5: `threshold_summary.csv`, `threshold_runs.csv` |
| `results_e5/` | E5: `e5_summary.csv`, `e5_runs.csv`, `e5_hand_check_sample.csv`; `predictions/{distilbert,bilstm}/` holds the E5 metrics and predictions (the perturbed texts `e5_texts_*.csv.gz`, 160 MB, are not included: `src/perturb.py` regenerates them with the same seed) |

To rerun the CPU analyses from the saved predictions:

```
cd transformer
python scripts/cost_thresholds.py      # RESULTS_DIR defaults to ./results
python scripts/significance.py
python scripts/e5_analysis.py
```

## Models

Trained weights are not in git (`.gitignore` excludes `models/`). The DistilBERT Mendeley
model used by the app (`distilbert_mendeley_3class_v1`, about 270 MB) and the BiLSTM models
(about 490 MB) are shared separately; put them under `transformer/models/` or set `MODELS_DIR`.
`inference/check_reload.py` checks that a downloaded model reproduces the saved test predictions.
BiLSTM inputs need the start token `xxbos ` (stored as `input_prefix` in each model's
`task_config.json`).

## Main results (test macro-F1, mean of 3 seeds)

| Dataset (E1) | DistilBERT | BiLSTM |
|---|---|---|
| mendeley (3-class) | 0.887 | 0.849 |
| uci | 0.978 | 0.942 |
| uci_replication (30/70) | 0.976 | 0.946 |
| email_fraud | 0.992 | 0.991 |
| spamassassin | 0.989 | 0.981 |
| ceas_08 | 0.856 | 0.851 |
| bothbosu (synthetic) | 1.000 | 1.000 |

- ceas_08: split_seed0 has a campaign (CNN "Daily Top 10") that appears only in its test set;
  without it seed 0 scores 0.995 for DistilBERT. The other seeds score 0.99+.
- 5.1: on the UCI 30/70 split DistilBERT reaches MCC ≈ 0.95 (target 0.893); BiLSTM 0.887–0.897.
- 5.3: DistilBERT is significantly better than BiLSTM on uci, uci_replication and ceas_08 in all
  3 seeds; on email_fraud, spamassassin and bothbosu the difference is not significant.
- E3: in-channel ≈ 0.99, cross-channel 0.33–0.66. Training on sms+email lifts the phone score to
  0.75 (DistilBERT); the size-matched control (E3M) gives the same result, so the gain is not
  just more data.
- E4: SpamAssassin → CEAS_08 ≈ 0.77, CEAS_08 → SpamAssassin ≈ 0.73 (DistilBERT); stripping
  corpus artefacts does not help.
- 5.5: the cost-optimal threshold moves strongly with the cost ratio (Mendeley, p = 1%:
  C_FN 1 → t ≈ 0.99, C_FN 10 → t ≈ 0.86, C_FN 50 → t ≈ 0.25), and thresholds tuned on one channel
  do not carry over to another.
- E5 (fraud recall, mean of 3 seeds; flip rate = detected scams turned into "legitimate"):
  BiLSTM breaks on P2_space (`v e r i f y`): recall SMS 0.968 → 0.564, email 0.990 → 0.428
  (flip rate 0.42 / 0.57); D1 normalisation restores it (0.965 / 0.990). DistilBERT loses at most
  about 5.5 points (worst: email P2_dot 0.991 → 0.936, flip rate 0.056); D1 fixes P2_space but, as
  designed, not P2_dot or P1. D1 does not change the clean results. The phone channel (bothbosu,
  synthetic) stays at 1.000 in every setting, so it does not separate the conditions.
  Hand check (proposal 4.4) of 50 random perturbed test messages: 50/50 readable, 47/50 meaning kept;
  the 3 exceptions changed an identifier (a bare domain, a login ID, host names in a mailbox file),
  which perturb.py does not protect (it protects http links and email addresses only).
