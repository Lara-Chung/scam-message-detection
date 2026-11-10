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

The current default is an **adaptive investigation with a free classifier path**.
It normalises a bounded source window and calls a VAL-selected saved classifier.
Ordinary complete messages make no API request. Long, near-boundary or incomplete
inputs escalate to one Agent, which chooses source sections to inspect, optional
cross-checks/TRAIN retrieval, clarification or finalization. Exact evidence IDs,
source coordinates, coverage, budget stops and review decisions are validated.

| Current mode | Behavior |
|---|---|
| `document_classifier` | Deterministic first substantive window + fixed classifier; no API |
| `document_fixed` | Same gate/tools, fixed reading schedule, one LLM finalization on escalation |
| `adaptive_agent` | Same gate, LLM chooses next actions from actual observations |

See [implementation, real failures/repair and handoff](docs/adaptive_agent_handoff.md).
This is a tested backend extension; **a fresh held-out effectiveness comparison
is still pending**. Historical experiments below do not establish an Agent gain.

For a fresh checkout, use Python 3.12 in the isolated Agent environment, place the
supplied archive in ignored `local_inputs/`, then run:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-agent.txt
.venv/bin/python -m scripts.prepare_agent_inputs
.venv/bin/python -m src.agent.baseline
.venv/bin/python -m src.agent.candidates
.venv/bin/python -m scripts.prepare_document_triage
.venv/bin/python -m scripts.frontend_agent_example
.venv/bin/python -m pytest -q
```

Preparation/training refuses existing artifact directories. Do not repeat it on
an already prepared checkout. Before live escalation, install the pinned counter
with `python -m scripts.prepare_agent_tokenizer` and configure the provider using
the frontend handoff. Keys, artifacts and traces are never committed. The CLI
defaults to the configured adaptive mode; use `--mode document_classifier` for
an explicitly offline path. No frontend framework or server was added.

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

A [six-case paired comparison](docs/agent_comparison_pilot.md) has now compared
classifier, direct LLM, fixed workflow and Agent on the same balanced TEST sample.
It found no Agent classification gain over the classifier/fixed workflow, with
higher latency and API cost. The report includes actual cache-aware cost ranges,
coverage, review flags, failure analysis and the small-sample limits.

The [classifier router extension](docs/classifier_router_report.md) adds
`fixed_model`, `rule_router` and `agent_router` modes. Twelve candidates were
trained on identical canonical TRAIN memberships, with fixed/rule choices from
VAL only. A paired 12-case TEST pilot found 12/12 correct for all three policies;
the Agent used 24 actual requests and 45,839 tokens without improving accuracy.
Candidate artifacts and runtime catalogs remain ignored by Git.

An [independent stress/disagreement challenge](docs/router_stress_report.md)
used a VAL-locked design and excluded 105 previously used TEST IDs. On 12 views
from nine new bases, all three routing policies scored 11/12 (4/5 on selected
disagreement inputs). Agent routing did not rescue a zero-width-character SMS
miss despite correct alternative candidates. The report separates challenge
strata, clean/obfuscated pairs, costs and post-hoc diagnostics.

The [normalised boundary pilot](docs/normalised_boundary_report.md) now gives all
three policies the same normalised input and enforces a VAL-calibrated second
classifier near each primary's native decision boundary. Use
`configs/agent.normalised.yaml` for these version-1.2 routing modes. On 12 inputs
from nine new TEST bases, every policy scored 10/12; the two misses are paired
views of one very long email whose common prefix contains mostly addresses.
Rule/Agent guards made eight/six secondary calls and each flagged one correct
decision for review. Agent routing made 24 requests / 55,177 tokens without an
accuracy gain. The report includes the input-clipping limitation and real costs.
No API request is made by `fixed_model` or `rule_router`.
