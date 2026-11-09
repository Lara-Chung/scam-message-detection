"""Cached saved-model inference, explicit class orientation and provenance checks."""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from src.agent.data import CHANNEL_DATASETS, EXPORT, LABEL_DEFINITION, eligible, file_hash, load_channel, membership_hash


class AgentError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class ModelAdapter:
    def __init__(self, model, model_id: str, label_map: dict, provenance: str,
                 preprocessing: str = 'Saved pipeline; no fit at inference'):
        self.model, self.model_id = model, model_id
        self.classes = [str(c) for c in model.classes_]
        self.label_map = {str(k): v for k, v in label_map.items()}
        if set(self.classes) != set(self.label_map) or not set(self.label_map.values()) <= {'fraud', 'legitimate', 'abstain'}:
            raise AgentError('label_mapping_missing', 'Every original class needs an explicit task mapping.')
        self.provenance, self.preprocessing = provenance, preprocessing
        self.training_rows = None

    @classmethod
    def load(cls, spec: dict, channel: str, export_dir: Path = EXPORT):
        path = Path(spec['path'])
        if not path.is_file():
            raise AgentError('artifact_missing', f'Model artifact unavailable for {channel}.')
        verified = spec.get('provenance') == 'canonical_train_verified'
        rows = None
        if verified:
            try:
                meta = json.loads(path.with_suffix('.metadata.json').read_text())
                if meta['versions']['scikit-learn'] != sklearn.__version__:
                    raise ValueError('scikit-learn version mismatch')
                # Baselines require the exact numerical runtime recorded at training.
                import scipy
                for name, version in [('numpy', np.__version__), ('scipy', scipy.__version__), ('joblib', joblib.__version__)]:
                    if meta['versions'][name] != version:
                        raise ValueError(f'{name} version mismatch')
                if {str(k):v for k,v in spec['label_map'].items()} != {'0':'legitimate','1':'fraud'}:
                    raise ValueError('canonical binary label mapping mismatch')
                if (meta['artifact_sha256'] != file_hash(path) or meta['channel'] != channel or
                    meta['split_key'] != 'split_seed0' or meta['label_definition'] != LABEL_DEFINITION):
                    raise ValueError('artifact metadata mismatch')
                dataset = CHANNEL_DATASETS[channel]
                if meta['dataset'] != dataset or meta['dataset_sha256'] != file_hash(export_dir/f'{dataset}.csv.gz'):
                    raise ValueError('export hash mismatch')
                df = eligible(load_channel(channel, export_dir))
                rows = df.loc[df.split_seed0 == 'train'].sort_values('id').reset_index(drop=True)
                if (meta['train_ids'] != rows.id.tolist() or meta['train_ids_sha256'] != membership_hash(rows.id)
                    or meta['training_text_sha256'] != {r.id: hashlib.sha256(r.text.encode()).hexdigest() for r in rows.itertuples()}):
                    raise ValueError('training membership mismatch')
            except (OSError, KeyError, ValueError) as exc:
                raise AgentError('provenance_invalid', f'Canonical provenance check failed: {exc}') from exc
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', InconsistentVersionWarning)
                model = joblib.load(path)
        except Exception as exc:
            raise AgentError('artifact_incompatible', 'Saved model cannot load in this runtime; use the documented environment.') from exc
        adapter = cls(model, spec.get('model_id', path.stem), spec['label_map'],
                      'canonical_train_verified' if verified else 'unverified_integration_only')
        if verified and [str(c) for c in meta['classes']] != adapter.classes:
            raise AgentError('provenance_invalid', 'Saved class order differs from metadata.')
        adapter.training_rows = rows
        return adapter

    def classify(self, text: str, view: str) -> dict:
        try:
            prediction = str(self.model.predict([text])[0])
            fraud_classes = [c for c in self.classes if self.label_map[c] == 'fraud']
            if hasattr(self.model, 'predict_proba'):
                values = np.asarray(self.model.predict_proba([text])[0], dtype=float)
                score_kind = 'model_probability_uncalibrated'
                fraud_score = float(sum(values[self.classes.index(c)] for c in fraud_classes)) if fraud_classes else None
                scores = dict(zip(self.classes, values.tolist()))
            elif hasattr(self.model, 'decision_function'):
                raw = np.asarray(self.model.decision_function([text]), dtype=float)
                score_kind = 'decision_margin'
                if len(self.classes) == 2:
                    margin = float(raw.reshape(-1)[0])
                    scores = {self.classes[0]: -margin, self.classes[1]: margin}
                else:
                    scores = dict(zip(self.classes, raw[0].tolist()))
                fraud_score = scores[fraud_classes[0]] if len(fraud_classes) == 1 else None
            else:
                scores, fraud_score, score_kind = {}, None, 'label_only'
            if any(not np.isfinite(v) for v in scores.values()):
                raise ValueError('non-finite score')
        except Exception as exc:
            raise AgentError('inference_failed', 'Saved-model inference failed.') from exc
        return {'model_id': self.model_id, 'provenance': self.provenance, 'original_classes': self.model.classes_.tolist(),
                'predicted_label': prediction, 'decision': self.label_map[prediction], 'scores': scores,
                'fraud_score': fraud_score, 'score_kind': score_kind, 'view': view,
                'preprocessing': self.preprocessing}

    def retrieve(self, text: str, k: int = 3) -> list[dict]:
        if self.training_rows is None or self.provenance != 'canonical_train_verified':
            raise AgentError('retrieval_unavailable', 'Retrieval requires verified canonical training membership.')
        rows = self.training_rows
        vectorizer = self.model.named_steps['tfidf']
        # Reuse the TRAIN-fitted vectorizer, and cache the TRAIN-only matrix.
        if not hasattr(self, '_train_matrix'):
            self._train_matrix = vectorizer.transform(rows.text)
        scores = (self._train_matrix @ vectorizer.transform([text]).T).toarray().ravel()
        top = np.argsort(-scores, kind='stable')[:k]
        return [{'case_id': rows.iloc[i].id, 'excerpt': rows.iloc[i].text[:400],
                 'similarity': float(scores[i]),
                 'training_label': 'fraud' if int(rows.iloc[i].label_fraud) else 'legitimate',
                 'partition': 'canonical_seed0_train'} for i in top if scores[i] > 0]
