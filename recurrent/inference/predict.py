"""Load and score the team's BiLSTM checkpoints; no training or Streamlit imports."""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from src.project_paths import neural_checkpoint_dir


@lru_cache(maxsize=8)
def load_model(directory: str):
    """Load native Keras weights and task metadata, then cache the traced graph."""
    import tensorflow as tf

    folder = Path(directory)
    config = json.loads((folder / 'task_config.json').read_text(encoding='utf-8'))
    model = tf.keras.models.load_model(folder / 'model.keras', compile=False, safe_mode=True)
    compiled = tf.function(lambda inputs: model(inputs, training=False), reduce_retracing=True)
    return tf, compiled, config


def predict_batch(texts: list[str], model_id: str, threshold: float | None = None,
                  models_dir: str | Path | None = None) -> list[dict]:
    """Return the existing frontend prediction fields using native checkpoint labels."""
    if not texts:
        return []
    folder = Path(models_dir) / model_id if models_dir is not None else neural_checkpoint_dir('RNN', model_id)
    tf, model, config = load_model(str(folder))
    labels, positives = config['labels'], config['positive_labels']
    start = time.perf_counter()
    inputs = [config['input_prefix'] + text for text in texts]
    scores = np.concatenate([
        np.asarray(model(tf.constant(inputs[index:index + 64], dtype=tf.string)), dtype=float)
        for index in range(0, len(inputs), 64)
    ])
    elapsed = (time.perf_counter() - start) * 1000 / len(texts)
    if scores.shape != (len(texts), len(labels)) or not np.isfinite(scores).all():
        raise ValueError('BiLSTM scores do not match the saved class metadata.')
    limit = config['seq_len']
    threshold = config.get('default_threshold', .5) if threshold is None else threshold
    results = []
    for text, row in zip(inputs, scores, strict=True):
        by_label = dict(zip(labels, row.tolist(), strict=True))
        risk = sum(by_label[label] for label in positives)
        results.append({
            'model_id': model_id, 'label': max(by_label, key=by_label.get), 'scores': by_label,
            'score': risk, 'score_name': f"{' + '.join(positives)} class score",
            'flagged': risk >= threshold, 'threshold': threshold,
            'truncated': None, 'possible_truncation': len(text.split()) > limit,
            'inference_ms': elapsed,
        })
    return results


def predict(text: str, model_id: str, threshold: float | None = None,
            models_dir: str | Path | None = None) -> dict:
    return predict_batch([text], model_id, threshold, models_dir)[0]
