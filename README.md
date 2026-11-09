# Scam Message Detection

Robust scam message detection across SMS, email and phone dialogue.

## Setup

```
conda create -n scam-detection python=3.11 -y && conda run -n scam-detection pip install -r requirements.txt
```

Using venv instead also works: create an environment with Python 3.11 and run `pip install -r requirements.txt`.

## Layout

| Path | Content |
|---|---|
| `data/raw/` | Downloaded datasets (not committed) |
| `data/processed/` | Cleaned CSVs and `splits.csv` |
| `src/` | Shared modules |
| `notebooks/` | One notebook per experiment (E1 to E6) and `00_data_summary.ipynb` |
| `tests/` | pytest tests |
| `app/` | Streamlit app |

## Using the splits

`data/processed/splits.csv` gives each row a split (train / val / test) for seeds 0, 1 and 2.

| Split | Use for | Do not use for |
|---|---|---|
| train | Fitting models, vectorisers (e.g. TF-IDF) and any other learned step | — |
| val | Hyperparameter selection and threshold tuning | Reporting final results |
| test | Final evaluation only | Tuning, model selection or threshold choice |

Cross-dataset tests (E3, E4): tune on the source dataset's val split only. Do not use target labels to choose a model or threshold. Before training, call `remove_overlap(source, target)` from `src/splits.py` to drop source rows that also appear in the target test set.

CEAS_08: one spam campaign ("THE DAILY TOP 10 from CNN", 2,930 emails) is in the test split for seed 0, where it makes up about 39% of it, and in the train split for seeds 1 and 2. Report CEAS_08 seed 0 test results with and without this campaign; its rows are those whose `template` starts with `thedailytop`.

If a model is retrained on train + val after tuning, the threshold chosen on val may no longer fit; re-check it on a held-out part of the training data.

## Data

Raw data sources. Download each dataset into `data/raw/`:

| Dataset | Source | Save as |
|---|---|---|
| SMS Phishing Dataset (Mendeley) | https://doi.org/10.17632/f45bkkt8pr.1 (`Dataset_5971.zip`) | `data/raw/mendeley/Dataset_5971.csv` |
| UCI SMS Spam Collection | https://archive.ics.uci.edu/dataset/228/sms+spam+collection | `data/raw/uci/SMSSpamCollection` |
| Nazario_5, Nigerian_5, CEAS_08, SpamAssassin | https://doi.org/10.5281/zenodo.8339691 | `data/raw/zenodo/<file name>.csv` |
| BothBosu/scam-dialogue | Downloaded automatically by `load_bothbosu()` from Hugging Face (pinned revision) | `data/raw/hf_cache/` |

To build the processed data from the raw files, run:

```
python -m src.splits && python -m src.export
```

`src.splits` writes `data/processed/splits.csv` and checks for leakage; `src.export` writes the processed files to `data/processed/export/`.

## Agent backend

The UI-independent entry point is `src.agent.service.ScamAgentService`.
Use the separate `requirements-agent.txt` environment and existing delivered
splits; do not combine its sklearn pin with the original training requirements.
See [frontend setup and JSON contract](docs/agent_frontend_handoff.md),
[artifact/data audit](docs/agent_integration_audit.md), and
[executed validation and pending live checks](docs/agent_validation.md).

```sh
.venv/bin/python -m src.agent.cli --text 'Please verify your account immediately.' --channel sms --mode classifier_only
```

The `agent`, `direct_llm` and `fixed_workflow` modes require explicitly configured
`LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL`; otherwise they return a structured
`provider_not_configured` result. Local inputs, artifacts and evaluation traces
remain ignored by Git.

A subsequent bounded [real DeepSeek validation](docs/agent_live_validation.md)
recorded 5 calls and 6,228 tokens on three validation messages: one complete
Agent answer and two local budget stops. The ignored local macOS profile uses
Keychain; no API key is stored in source or YAML.

The [budget repair and current acceptance](docs/agent_budget_fix.md) recorded
three real final responses in 9,164 tokens. SMS/phone passed directly; email
retains a visible review flag after deterministic correction of a unique exact
quote's coordinates. Full offline suite: 121 passed, 7 skipped. Run
`python -m scripts.prepare_agent_tokenizer` to install the pinned official
counter data before configured live inference.
