# Recurrent models: BiLSTM

Owner: Daniel. This component groups the BiLSTM training record and reusable
inference interface separately from DistilBERT.

| Directory | Content |
|---|---|
| `scripts/bilstm_experiments.py` | Existing Keras experiments E1, E2, E3, E3M, E4 and E5 |
| `notebooks/04_bilstm_e2_e1_e3_e4.ipynb` | Original executed Kaggle notebook, including its recorded outputs |
| `inference/predict.py` | Native checkpoint loading and batched prediction used by the frontend |
| `models/` | Local BiLSTM checkpoints, ignored by Git |

Put exported checkpoints in `recurrent/models/<model_id>/`, with `model.keras`
and `task_config.json` in each model directory. The previous
`transformer/models_bilstm/<model_id>/` location is also supported, including
mixed old/new installations. `BILSTM_MODELS_DIR` explicitly selects another root.

The inference code retains the checkpoint's class order, positive labels,
`xxbos ` prefix, sequence length and review threshold:

```python
from recurrent.inference.predict import predict, predict_batch

result = predict('Please verify your account.', 'bilstm_fraud_allchannels_seed0')
rows = predict_batch(['message one', 'message two'], 'bilstm_fraud_allchannels_seed0')
```

Install the TensorFlow/Keras versions in [START_HERE.md](../START_HERE.md) to load
real checkpoints. The prediction module imports TensorFlow when a model is loaded.

Saved BiLSTM results are in
[`analysis/neural/results/results_bilstm/`](../analysis/neural/results/results_bilstm),
with app-model exports under
[`analysis/neural/results/results_app_lime/results_bilstm/`](../analysis/neural/results/results_app_lime/results_bilstm).
Shared robustness/LIME notebooks, thresholds and comparisons are in
[`analysis/neural/`](../analysis/neural). The executed notebook and saved results
were moved without changing their contents.
