"""Load the team's saved models through their native inference interfaces."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import streamlit as st

from app.core import ROOT

CHANNELS = {'sms': 'SMS', 'email': 'Email', 'phone': 'Phone transcript'}
TRADITIONAL = {
    'word_nb': 'Multinomial Naive Bayes on Word TF-IDF',
    'word_lr': 'Logistic Regression on Word TF-IDF',
    'word_svm': 'Linear SVM on Word TF-IDF',
    'char_svm': 'Linear SVM on Character TF-IDF (D2)',
}
DATASET_CHANNEL = {
    'mendeley_3class': 'sms', 'mendeley': 'sms', 'uci': 'sms', 'uci_replication': 'sms',
    'email_fraud': 'email', 'spamassassin': 'email', 'ceas_08': 'email', 'bothbosu': 'phone',
}
NEURAL = {
    'RNN': ('bilstm_fraud_allchannels_seed0', 'bilstm_mendeley_seed0'),
    'Transformer': ('distilbert_fraud_allchannels_v1', 'distilbert_mendeley_3class_v1'),
}
METRICS = {
    'bilstm_fraud_allchannels_seed0': 'results_app_lime/results_bilstm/metrics_APP_sms+email+phone_seed0.json',
    'distilbert_fraud_allchannels_v1': 'results_app_lime/results_app/metrics_distilbert_allchannels.json',
    'bilstm_mendeley_seed0': 'results_bilstm/metrics_E1_mendeley_seed0.json',
    'distilbert_mendeley_3class_v1': 'results_mendeley/metrics_split_seed0.json',
}


def model_root(family: str) -> Path:
    """Find the configured checkpoint directory."""
    directory = 'models_bilstm' if family == 'RNN' else 'models'
    env = 'BILSTM_MODELS_DIR' if family == 'RNN' else 'MODELS_DIR'
    return Path(os.getenv(env, ROOT / 'transformer' / directory)).resolve()


def neural_domain(model_id: str) -> str | None:
    """Identify the input channel for a checkpoint."""
    if model_id.startswith('bilstm_'):
        variant = re.sub(r'_seed\d+$', '', model_id.removeprefix('bilstm_'))
    else:
        variant = model_id.removeprefix('distilbert_').removesuffix('_v1')
    if variant == 'fraud_allchannels':
        return 'all'
    return DATASET_CHANNEL.get(variant) or next(
        (channel for channel in CHANNELS if variant in (f'fraud_{channel}', f'{channel}')), None
    )


def neural_models(family: str) -> list[str]:
    """List documented runs and compatible local checkpoints."""
    featured = NEURAL[family]
    prefix = 'bilstm_' if family == 'RNN' else 'distilbert_'
    extras = sorted(path.parent.name for path in model_root(family).glob('*/task_config.json')
                    if path.parent.name.startswith(prefix) and neural_domain(path.parent.name))
    return list(dict.fromkeys((*featured, *extras)))


def neural_requirements(family: str, model_id: str) -> list[str]:
    """Find missing checkpoint files."""
    folder = model_root(family) / model_id
    required = ['task_config.json', 'model.keras'] if family == 'RNN' else [
        'task_config.json', 'config.json', 'tokenizer_config.json'
    ]
    missing = [name for name in required if not (folder / name).is_file()]
    if family == 'Transformer' and not any(
        (folder / name).is_file() for name in ('model.safetensors', 'model.safetensors.index.json')
    ):
        missing.append('model.safetensors')
    return missing


@st.cache_resource
def router():
    """Load the local classifier catalog."""
    from src.agent.routing import RouterEngine
    from src.agent.service import ScamAgentService

    service = ScamAgentService.from_config(ROOT / 'configs/agent.yaml', enable_provider=False)
    return RouterEngine(service.config, service.limits)


def traditional_batch(model_id: str, channel: str, texts: list[str], defence: bool = False) -> list[dict]:
    """Score messages using a saved TF-IDF classifier."""
    engine = router()
    model = engine.adapter(channel, model_id)
    from src.normalise import normalise

    results = []
    for text in texts:
        limit = engine.limits['max_input_chars']
        used = text[:limit]
        start = time.perf_counter()
        value = model.classify(normalise(used) if defence else used, 'normalised' if defence else 'original')
        results.append({
            'model_id': value['model_id'],
            'label': value['decision'],
            'scores': {model.label_map[label]: score for label, score in value['scores'].items()},
            'score': value['fraud_score'],
            'score_name': 'Fraud decision margin' if value['score_kind'] == 'decision_margin'
            else 'Fraud class score',
            'flagged': value['decision'] == 'fraud',
            'truncated': len(text) > limit,
            'inference_ms': (time.perf_counter() - start) * 1000,
        })
    return results


@st.cache_resource
def load_bilstm(directory: str):
    """Load a Keras checkpoint and its task metadata."""
    import tensorflow as tf

    folder = Path(directory)
    config = json.loads((folder / 'task_config.json').read_text(encoding='utf-8'))
    model = tf.keras.models.load_model(folder / 'model.keras', compile=False, safe_mode=True)
    # Calling a Keras model eagerly costs ~0.3-1.4 s per call on CPU; a traced graph costs ~10 ms.
    # reduce_retracing lets one graph serve every batch size.
    compiled = tf.function(lambda inputs: model(inputs, training=False), reduce_retracing=True)
    return tf, compiled, config


def neural_batch(family: str, model_id: str, texts: list[str], threshold: float | None) -> list[dict]:
    """Score messages using the selected neural checkpoint."""
    if family == 'Transformer':
        from transformer.inference.predict import predict_batch

        outputs = predict_batch(texts, model_id, threshold=threshold)
        for output in outputs:
            if output['status'] != 'ok':
                raise ValueError(output['message'])
        return [{
            'model_id': output['model_id'], 'label': output['predicted_label'],
            'scores': output['class_scores'], 'score': output['risk_score'],
            'score_name': f"{' + '.join(output['positive_labels'])} class score",
            'flagged': output['flagged'], 'truncated': output['truncated'],
            'threshold': output['threshold'], 'inference_ms': output['inference_ms'],
        } for output in outputs]

    tf, model, config = load_bilstm(str(model_root(family) / model_id))
    labels, positives = config['labels'], config['positive_labels']
    start = time.perf_counter()
    inputs = [config['input_prefix'] + text for text in texts]
    scores = np.concatenate([
        np.asarray(model(tf.constant(inputs[start:start + 64], dtype=tf.string)), dtype=float)
        for start in range(0, len(inputs), 64)
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


class NeuralProbabilities:
    """Expose a neural checkpoint as classes_ + predict_proba so LIME can explain it (proposal 5.7)."""

    def __init__(self, family: str, model_id: str) -> None:
        """Read the class order from the checkpoint's task metadata."""
        self.family, self.model_id = family, model_id
        config = json.loads((model_root(family) / model_id / 'task_config.json').read_text(encoding='utf-8'))
        self.classes_ = np.asarray(config['labels'])
        self.positive_label = config['positive_labels'][0]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return class probabilities in classes_ order; empty texts score as the first class."""
        texts = list(texts)
        output = np.zeros((len(texts), len(self.classes_)))
        output[:, 0] = 1.0
        keep = [index for index, text in enumerate(texts) if text.strip()]
        if keep:
            rows = neural_batch(self.family, self.model_id, [texts[index] for index in keep], None)
            for index, row in zip(keep, rows, strict=True):
                output[index] = [row['scores'][label] for label in self.classes_]
        return output
