# Transformer models: DistilBERT

Owner: Daniel. This component contains the DistilBERT experiments and inference
interface used by the Streamlit frontend.

| Directory | Content |
|---|---|
| `scripts/mendeley_full_run.py` | E2 tuning and Mendeley three-class model |
| `scripts/e1_all_datasets.py` | E1 dataset benchmarks |
| `scripts/e3_e4_transfer.py` | E3 transfer, E3M size control, E4 corpus transfer and E5 robustness |
| `notebooks/01_distilbert_mendeley_e2.ipynb` | Original executed Mendeley run |
| `notebooks/02_distilbert_e1.ipynb` | Original executed E1 run |
| `notebooks/03_distilbert_e3_e4.ipynb` | Original executed transfer run |
| `inference/` | Prediction API and checkpoint-reload check |
| `models/` | Local Hugging Face checkpoints, ignored by Git |

The scripts retain their Kaggle configuration, split rules, model settings and
outputs. Set `DATA_DIR` and other options as documented in each script. Original
notebook outputs are retained.

For local inference, put exported weights and task metadata under
`transformer/models/<model_id>/`, or set `MODELS_DIR`:

```python
from transformer.inference.predict import predict, predict_batch

result = predict('Please verify your account.', 'distilbert_mendeley_3class_v1')
```

See the [prediction interface](INTERFACE_for_frontend.md) for dependencies and
returned fields. [START_HERE.md](../START_HERE.md) describes the shared frontend.

Saved experiments are grouped under
[`analysis/neural/results/`](../analysis/neural/results), including
`results_mendeley/`, `results_e1/` and `results_e3_e4/`.
The [shared neural analysis guide](../analysis/neural/README.md) records all
DistilBERT/BiLSTM comparisons, reported metrics and CPU analysis commands.
BiLSTM code is in [`recurrent/`](../recurrent).
