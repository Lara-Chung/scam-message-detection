# Project directory guide

The code is grouped by the model or workflow it implements. Shared data,
frontend code and comparisons have their own directories.

```text
traditional/                 TF-IDF + LR, NB, word SVM and character SVM
transformer/
  scripts/                   DistilBERT training and transfer experiments
  notebooks/                 Executed DistilBERT notebooks
  inference/                 Hugging Face checkpoint prediction
recurrent/
  scripts/                   BiLSTM training and experiments
  notebooks/                 Executed BiLSTM notebook
  inference/                 Keras checkpoint prediction
agent/
  runtime/                   Agent service, adapters, tools, provider and schemas
  scripts/                   Agent preparation, comparisons and reporting
analysis/neural/
  scripts/                   Shared thresholds, significance, robustness and LIME
  notebooks/                 Shared E3M, E5 and app/LIME run records
  results/                   Original neural metrics and per-message predictions
app/                         Shared Streamlit interface for all four methods
src/                         Shared data, split, defence and path utilities
configs/                     Runtime configuration
artifacts/                   Saved conventional-model reports
notebooks/                   Shared data-summary notebook
scripts/                     Shared demo commands and Agent command namespace
baselines/                   Compatibility entry points for traditional training
src/agent/                   Compatibility namespace for the Agent Python API
tests/                       Automated component and integration tests
docs/                        Integration, validation and experiment guides
```

## Entry points

| Component | Code | Guide |
|---|---|---|
| Traditional | `traditional/lr_scam.py`, `nb_scam.py`, `svm_word_scam.py`, `svm_char_scam.py` | [Traditional models](../traditional/README.md) |
| Transformer | `transformer/scripts/`, `transformer/inference/predict.py` | [DistilBERT](../transformer/README.md) |
| Recurrent | `recurrent/scripts/bilstm_experiments.py`, `recurrent/inference/predict.py` | [BiLSTM](../recurrent/README.md) |
| Agent | `agent/runtime/`, `agent/scripts/`; CLI `python -m agent` | [Agent](../agent/README.md) |
| Shared neural analysis | `analysis/neural/scripts/` | [Comparison guide](../analysis/neural/README.md) |
| Frontend | `app/streamlit_app.py` | [Demo setup](../START_HERE.md) |

## Compatibility and preserved results

- Existing `src.agent.*` imports and `python -m src.agent` load the code in
  `agent/runtime/`. Existing `python -m scripts.<agent_command>` commands load
  `agent/scripts/`. The compatibility namespaces contain no duplicate implementation.
- The older `baselines/*.py` commands and root `lr_scam.py` forward to
  `traditional/`. The duplicate LR implementations had identical program logic;
  their usage docstrings differed.
- Neural results moved together from `transformer/results/` to
  `analysis/neural/results/`. Their original model/experiment subdirectories,
  file contents, labels, predictions and metrics are preserved.
- Shared E3M, E5 and app/LIME notebooks are under `analysis/neural/notebooks/`.
  The original notebook contents and executed outputs are preserved.
- BiLSTM checkpoints can remain in `transformer/models_bilstm/` or be placed in
  `recurrent/models/`. Explicit `BILSTM_MODELS_DIR` and `MODELS_DIR` settings retain
  precedence. Runtime configuration and ignored Agent model locations are unchanged.
- Historical execution audits retain the file paths and hashes recorded at the
  time. New Agent protocols snapshot the canonical implementation locations.

This change organises code and extracts the existing BiLSTM prediction logic
into its component. It does not retrain neural models or change experiment results.
