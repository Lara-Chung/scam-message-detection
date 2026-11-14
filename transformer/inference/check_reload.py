"""Check a local DistilBERT export against its saved canonical test predictions.

Run from the repository root:
    python transformer/inference/check_reload.py --help
    python transformer/inference/check_reload.py

Data and predictions default to the shared export and neural results directory.
The pass criterion is unchanged: label agreement > 99.5% and max score difference
< 0.01. Model weights are loaded only when the check actually runs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.project_paths import NEURAL_RESULTS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=REPO_ROOT / 'data/processed/export/mendeley.csv.gz')
    parser.add_argument('--predictions', type=Path, default=NEURAL_RESULTS / 'results_mendeley/pred_split_seed0_test.csv')
    parser.add_argument('--model-id', default='distilbert_mendeley_3class_v1')
    args = parser.parse_args(argv)

    import pandas as pd
    from transformer.inference.predict import DEVICE, predict_batch

    df = pd.read_csv(args.data)
    old = pd.read_csv(args.predictions)
    test = df[df['split_seed0'] == 'test'][['id', 'text']]
    merged = old.merge(test, left_on='sample_id', right_on='id', how='left')
    assert merged['text'].notna().all(), 'Some sample IDs are missing; confirm the data version.'

    print(f'device: {DEVICE}, {len(merged)} test messages, predicting...')
    started = time.time()
    results = predict_batch(merged['text'].tolist(), args.model_id)
    print(f'took {time.time() - started:.1f} s')
    bad = [r for r in results if r['status'] != 'ok']
    assert not bad, f'{len(bad)} results returned errors: {bad[:3]}'
    labels = [r['predicted_label'] for r in results]
    same = (merged['pred_label'] == pd.Series(labels)).mean()
    diff = max(abs(r['class_scores'][label] - merged.loc[i, f'score_{label}'])
               for i, r in enumerate(results) for label in r['class_scores'])
    accuracy = (pd.Series(labels) == merged['true_label']).mean()
    print(f'label agreement: {same:.4f}')
    print(f'max class-score difference: {diff:.6f}')
    print(f'recomputed test accuracy: {accuracy:.4f} (0.9716 at training time)')
    passed = same > .995 and diff < .01
    print('PASS' if passed else 'MISMATCH: confirm checkpoint, data and export versions.')
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
