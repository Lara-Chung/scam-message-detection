"""Minimal canonical TRAIN-only baselines; original experiments remain untouched."""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.agent.data import CHANNEL_DATASETS, EXPORT, LABEL_DEFINITION, eligible, file_hash, load_channel, membership_hash
from src.data import REPO_ROOT


def train(output: Path, export_dir: Path = EXPORT):
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory; artifacts are never overwritten.")
    for channel, dataset in CHANNEL_DATASETS.items():
        df = eligible(load_channel(channel, export_dir))
        rows = df.loc[df.split_seed0 == 'train'].sort_values('id')
        model = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=1,
                                                   sublinear_tf=True, max_features=50000)),
                          ('clf', LogisticRegression(C=1.0, class_weight='balanced',
                                                     random_state=0, max_iter=1000))])
        model.fit(rows.text, rows.label_fraud.astype(int))
        path = output / f'{channel}.joblib'
        joblib.dump(model, path)
        meta = {'provenance': 'canonical_train_verified', 'channel': channel, 'dataset': dataset,
                'dataset_sha256': file_hash(export_dir / f'{dataset}.csv.gz'),
                'artifact_sha256': file_hash(path), 'split_key': 'split_seed0', 'seed': 0,
                'label_definition': LABEL_DEFINITION, 'classes': model.classes_.tolist(),
                'train_ids': rows.id.tolist(), 'train_ids_sha256': membership_hash(rows.id),
                'training_text_sha256': {r.id: __import__('hashlib').sha256(r.text.encode()).hexdigest()
                                         for r in rows.itertuples()},
                'versions': {'python': platform.python_version(), 'scikit-learn': sklearn.__version__,
                             'pandas': pd.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
                             'joblib': joblib.__version__},
                'hyperparameters': {'tfidf_ngram_range': [1, 2], 'tfidf_max_features': 50000,
                    'tfidf_sublinear_tf': True, 'C': 1.0, 'class_weight': 'balanced',
                    'max_iter': 1000, 'random_state': 0},
                'selection_policy': 'Fixed a priori; no tuning on validation or test.',
                'input_policy': 'Raw training text. Runtime uses disclosed common prefix clipping.'}
        path.with_suffix('.metadata.json').write_text(json.dumps(meta, indent=2))
        print(f'{channel}: trained {len(rows)} canonical seed0 TRAIN rows -> {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPO_ROOT/'models/agent_seed0')
    args = parser.parse_args()
    train(args.output)

if __name__ == '__main__':
    main()
