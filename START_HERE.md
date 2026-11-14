# Signal Streamlit Demo

Run from the project directory with Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-app.txt
python -m streamlit run app/streamlit_app.py
```

On Windows PowerShell, activate with `..venv\Scripts\Activate.ps1` in place of the `source` command. Open http://localhost:8501 in the browser. If port 8501 is busy, Streamlit picks the next free port and prints it in the terminal.

`.streamlit/config.toml` turns Streamlit's file watcher off (`fileWatcherType = "none"`): with transformers installed, the watcher scans hundreds of lazily loaded modules on every rerun, which floods the terminal with `torchvision` errors and freezes the page. Restart Streamlit after editing code. `headless = true` skips the first-run email prompt; open the URL yourself. Leave the terminal running and use Ctrl+C to stop the app.

## Demo layout

Choose a method, model or workflow, and channel in the sidebar. Traditional modelling puts message prediction, robustness testing and evaluation on one page. Use the sidebar score threshold to adjust review flags for probability-based models. The other methods use the workspace above the content.

| Method | Description |
| --- | --- |
| Traditional modelling | Classical classifiers trained on word or character TF-IDF features. |
| RNN | BiLSTM with a trainable embedding layer. |
| Transformer | DistilBERT fine-tuned with Hugging Face Trainer. |
| AI Agent | Routing and investigation workflows for fraud review. |

The canonical TF-IDF models use seed 0 TRAIN artifacts and show the corresponding validation results. The UCI spam models retain the original demo's grouped holdout evaluation and automatic LIME explanations after prediction. Phone training data are synthetic. Keep the bundled files in `models/` and `data/processed/export/` for local inference and provenance checks.

## Neural checkpoints

The Evaluation workspace reads saved team results independently of the live checkpoints. To enable inference, copy the original exports into the following directories.

| Model | Default directory | Required files |
| --- | --- | --- |
| BiLSTM on fraud across channels | `recurrent/models/bilstm_fraud_allchannels_seed0/` | `model.keras` and `task_config.json` |
| BiLSTM on Mendeley SMS | `recurrent/models/bilstm_mendeley_seed0/` | `model.keras` and `task_config.json` |
| DistilBERT on fraud across channels | `transformer/models/distilbert_fraud_allchannels_v1/` | Hugging Face weights, configuration, tokenizer and `task_config.json` |
| DistilBERT on Mendeley SMS | `transformer/models/distilbert_mendeley_3class_v1/` | Hugging Face weights, configuration, tokenizer and `task_config.json` |
| DistilBERT per channel (optional) | `transformer/models/distilbert_fraud_{sms,email,phone}_v1/` | Same as above; listed automatically when present |

The weights are not in Git (`models/` and `*.keras` are ignored). Ask the Transformer owner for `app_models.zip` (all-channel and per-channel models) and `mendeley_models.zip`, unzip them, copy the `models/` contents into `transformer/models/`, and copy the
`models_bilstm/` contents into `recurrent/models/`. Existing
`transformer/models_bilstm/` installations also remain supported.

Set `BILSTM_MODELS_DIR` or `MODELS_DIR` before startup to use another checkpoint parent directory. Install the corresponding runtime in the same virtual environment.

```bash
python -m pip install tensorflow==2.21.0 keras==3.13.2
python -m pip install torch transformers==5.16.1
```

On the RNN and Transformer pages:

- **Prediction settings** offers three review thresholds: *Cost-based* (default; reads `analysis/neural/results/results_app_lime/results_report/app_thresholds.json`, chosen on seed 0 validation predictions to minimise C_FN·p·FNR + (1−p)·FPR, proposal 5.5), *Model default* (0.50) and *Custom*. Changing the threshold re-flags the current result without rerunning the model.
- After a prediction, the message is shown with candidate indicators highlighted, and **Explain with LIME** runs a 500-sample LIME explanation (proposal 5.7). It takes about a second for BiLSTM and a few seconds for DistilBERT on CPU.
- The first prediction with each model loads it from disk and takes a few seconds; later predictions reuse the loaded model.

Evaluation separates per-run test or validation results, E1 benchmark summaries, E5 robustness experiments and available cost thresholds. DistilBERT also includes the saved cross-channel LIME failure analysis.

## Agent setup

Open **AI Agent → Adaptive investigation → Adaptive investigation Agent → SMS** and click **Analyse message** on the preloaded everyday message to demonstrate local screening. See **Execution details** for the provider call count. Messages that need further investigation are marked for review in offline mode.

Without a key, the LLM router and tool-calling Agent show recorded runs, with a note at the top of the page. **Execution details** lists the run summary, classifier outputs and the steps the Agent took as tables; the raw backend response is behind the *Show raw response (debug)* toggle. Direct LLM and Fixed tool workflow show their saved comparisons under **Evaluation**.

For optional live requests, set `LLM_API_KEY` in the terminal that starts Streamlit. The defaults are the official DeepSeek endpoint and `deepseek-flash`. `LLM_BASE_URL` and `LLM_MODEL` are optional overrides. Select **Allow DeepSeek to process this input** before clicking **Analyse message**. Live requests may incur API charges. See `docs/agent_frontend_handoff.md` for the backend API.

## Rebuild local artifacts

Use a fresh model directory and place the original `processed_data_export.zip` under `local_inputs/` before rebuilding. Existing output directories are protected by the training scripts.

```bash
python -m scripts.prepare_agent_inputs
python -m src.agent.candidates
python -m src.agent.baseline
python -m scripts.prepare_document_triage
python -m scripts.prepare_boundary_router validation
python -m scripts.prepare_agent_tokenizer
```

See `README.md`, `recurrent/README.md` and `analysis/neural/README.md` for the full experiment commands.
