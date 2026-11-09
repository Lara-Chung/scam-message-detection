"""Read-only local audit, including reconstructed legacy split membership."""
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.agent.data import CHANNEL_DATASETS, EXPORT, eligible, file_hash, load_channel
from src.splits import leakage_counts
from lr_scam import grouped_split, template_key


def main():
    root = Path('models/teammate_legacy')
    report = {'runtime_sklearn': sklearn.__version__, 'legacy_artifacts': len(list(root.rglob('*.joblib'))),
              'channels': {}, 'other_exports': {}}
    saved = pd.read_csv('data/processed/splits.csv')
    for channel, dataset in CHANNEL_DATASETS.items():
        df = load_channel(channel)
        task = eligible(df)
        split_alignment = {}
        for seed in range(3):
            current = saved[(saved.dataset == dataset) & (saved.seed == seed)].set_index('id')['split']
            split_alignment[str(seed)] = bool(set(current.index) == set(df.id) and
                (df.id.map(current) == df[f'split_seed{seed}']).all())
        name = 'email' if channel == 'email' else dataset
        meta_path = next(root.rglob(f'outputs_{name}_lr/lr_results.json'))
        meta = json.loads(meta_path.read_text())['runs']['weighted'][0]
        legacy = df[['text', 'label_orig']].rename(columns={'label_orig': 'label'}).copy()
        legacy['group'] = legacy.text.map(template_key)
        train_idx, _ = grouped_split(legacy, meta['split_seed'])
        canonical = set(task.loc[task.split_seed0 == 'test', 'id'])
        reconstructed = set(df.iloc[train_idx].id)
        model_info = []
        for kind, filename in [('lr', 'lr_weighted_split0.joblib'), ('svm_word', 'svm_word_weighted_split0.joblib')]:
            path = next(root.rglob(f'outputs_{name}_{kind}/{filename}'))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = joblib.load(path)
            model_info.append({'path': str(path), 'sha256': file_hash(path), 'classes': model.classes_.tolist(),
                               'steps': list(model.named_steps), 'classifier': type(model[-1]).__name__,
                               'random_state': model[-1].random_state,
                               'warnings': [str(w.message) for w in caught],
                               'smoke_prediction': model.predict(['Please verify your account now.']).tolist()})
        report['channels'][channel] = {'dataset': dataset, 'rows': len(df), 'eligible': len(task),
            'sha256': file_hash(EXPORT / f'{dataset}.csv.gz'),
            'labels_including_NA': {str(k): int(v) for k, v in df.label_fraud.value_counts(dropna=False).items()},
            'task_splits': task.split_seed0.value_counts().to_dict(), 'saved_split_alignment': split_alignment,
            'leakage_each_seed': {str(s): leakage_counts(df, df[f'split_seed{s}']) for s in range(3)},
            'legacy_seed': meta['split_seed'], 'legacy_reported_test_n': sum(map(sum,meta['confusion_matrix'])),
            'reconstructed_train_overlap_test_eligible': len(canonical & reconstructed),
            'canonical_test_eligible': len(canonical), 'legacy_model_inspection': model_info}
    for dataset in ['uci', 'uci_replication', 'ceas_08', 'spamassassin']:
        df = pd.read_csv(EXPORT/f'{dataset}.csv.gz', dtype={'label_fraud':'Int64'})
        report['other_exports'][dataset] = {'rows':len(df), 'fraud_eligible':int(df.label_fraud.notna().sum())}
    out = Path('outputs/agent'); out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    main()
