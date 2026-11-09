"""Frozen canonical evaluation manifests and auditable A/B/C/D comparisons."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score

from src.agent.data import CHANNEL_DATASETS, EXPORT, eligible, file_hash, load_channel
from src.agent.service import PROMPT_VERSION, SYSTEM_PROMPT, ScamAgentService
from src.agent.contract import FINAL_SCHEMA
from src.agent.tools import definitions
from src.agent.budget import PromptBudget

MODES = ['classifier_only', 'direct_llm', 'agent', 'fixed_workflow']


def prompt_hash():
    return hashlib.sha256((SYSTEM_PROMPT+json.dumps({'final_schema':FINAL_SCHEMA, 'tools':definitions(True)}, sort_keys=True)).encode()).hexdigest()


def create_manifest(path: Path, split='val', per_channel=3, sample_seed=0, max_chars=8000):
    if path.exists():
        raise FileExistsError('Manifest already exists; use a new filename instead of overwriting.')
    if per_channel < 1 or max_chars < 1 or split not in ['val', 'test']:
        raise ValueError('Manifest requires val/test, positive per-channel count and max chars.')
    manifest = {'protocol': 'in_domain_canonical_seed0_v1', 'partition': split, 'sample_seed': sample_seed,
                'prompt_version': PROMPT_VERSION, 'prompt_sha256': prompt_hash(),
                'input_policy': {'name': 'prefix_chars_v1', 'max_chars': max_chars},
                'dataset_sha256': {}, 'cases': [], 'support': {}}
    by_channel = {}
    for channel, dataset in CHANNEL_DATASETS.items():
        df = eligible(load_channel(channel))
        df = df.loc[df.split_seed0 == split].copy()
        df['sample_order'] = df.id.map(lambda i: hashlib.sha256(f'{sample_seed}:{i}'.encode()).hexdigest())
        rows = df.sort_values(['sample_order', 'id']).head(per_channel)
        manifest['dataset_sha256'][channel] = file_hash(EXPORT/f'{dataset}.csv.gz')
        manifest['support'][channel] = {'selected': len(rows), 'fraud': int(rows.label_fraud.sum()),
                                        'legitimate': int((rows.label_fraud == 0).sum()),
                                        'synthetic': channel == 'phone'}
        by_channel[channel] = [{'case_id': r.id, 'channel': channel, 'label': int(r.label_fraud),
                                'text_sha256': hashlib.sha256(r.text.encode()).hexdigest()} for r in rows.itertuples()]
    # Round-robin allows a bounded three-message smoke to include all channels.
    for i in range(per_channel):
        for channel in CHANNEL_DATASETS:
            if i < len(by_channel[channel]):
                manifest['cases'].append(by_channel[channel][i])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def aggregate(records):
    """No failed, pending or abstaining case disappears from the denominator."""
    n = len(records)
    answered = [r for r in records if r.get('result') and r['result']['status'] != 'error'
                and r['result']['decision'] in ['fraud', 'legitimate']]
    ys = [r['label'] for r in answered]
    ps = [int(r['result']['decision'] == 'fraud') for r in answered]
    cm = confusion_matrix(ys, ps, labels=[0, 1]).tolist() if answered else [[0, 0], [0, 0]]
    correct = sum(a == b for a, b in zip(ys, ps))
    failures = sum(bool(r.get('result') and r['result']['status'] == 'error') for r in records)
    pending = sum(r.get('result') is None for r in records)
    abstain = sum(bool(r.get('result') and r['result']['status'] != 'error' and r['result']['decision'] == 'abstain') for r in records)
    review = sum(bool(r.get('result') and r['result']['needs_review']) for r in records)
    latency = [r['result']['elapsed_ms'] for r in records if r.get('result')]
    out = {'total_cases': n, 'class_support': {'legitimate': sum(r['label']==0 for r in records),
                                            'fraud': sum(r['label']==1 for r in records)},
           'answered_cases': len(answered), 'answered_coverage': len(answered)/n if n else 0,
           'abstentions': abstain, 'technical_failures': failures, 'pending_cases': pending,
           'technical_failure_rate': failures/n if n else 0, 'review_cases': review,
           'review_coverage': review/n if n else 0, 'conservative_full_denominator_success': correct/n if n else 0,
           'confusion_matrix_answered_label_order_0_1': cm,
           'macro_f1_answered': float(f1_score(ys, ps, labels=[0, 1], average='macro', zero_division=0)) if answered else None,
           'fraud_precision_answered': float(precision_score(ys, ps, zero_division=0)) if answered else None,
           'fraud_recall_answered': float(recall_score(ys, ps, zero_division=0)) if answered else None,
           'false_positive_rate_answered': cm[0][1]/sum(cm[0]) if sum(cm[0]) else None,
           'mean_elapsed_ms': statistics.mean(latency) if latency else None,
           'p95_elapsed_ms': sorted(latency)[min(len(latency)-1, int(len(latency)*.95))] if latency else None,
           'tool_calls': sum(len(r['result']['tool_trace']) for r in records if r.get('result')),
           'llm_calls': sum(r['result']['llm_calls'] for r in records if r.get('result')),
           'usage': {key: sum(r['result']['usage'][key] for r in records if r.get('result'))
                     for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']},
           'accounting_complete': all(r['result']['usage']['accounting_complete'] for r in records if r.get('result')),
           'average_precision_fraud': None}
    continuous = []
    for r in answered:
        result = r['result']
        if result['mode'] == 'classifier_only' and result['classifier_results']:
            score = result['classifier_results'][0]
            if score['score_kind'] in ['model_probability_uncalibrated', 'decision_margin'] and score['fraud_score'] is not None:
                continuous.append((r['label'], score['fraud_score']))
    if len(continuous) == len(answered) and continuous and len(set(ys)) == 2:
        out['average_precision_fraud'] = float(average_precision_score([a for a,b in continuous], [b for a,b in continuous]))
    return out


def run(manifest_path: Path, output: Path, service, modes, max_cases=None, token_budget=12000, max_live_messages=3, live_test=False):
    if output.exists():
        raise FileExistsError('Run output already exists; use a new directory.')
    if not modes or len(set(modes)) != len(modes) or not set(modes) <= set(MODES):
        raise ValueError('Unsupported or repeated comparison mode.')
    if token_budget < 1 or max_live_messages < 1 or (max_cases is not None and max_cases < 1):
        raise ValueError('Budgets must be positive.')
    manifest = json.loads(manifest_path.read_text())
    if (manifest['protocol'] != 'in_domain_canonical_seed0_v1' or manifest['partition'] not in ['val', 'test']
        or manifest['prompt_sha256'] != prompt_hash()
        or manifest['input_policy'] != {'name': 'prefix_chars_v1', 'max_chars': service.limits['max_input_chars']}):
        raise ValueError('Manifest prompt/input protocol mismatch; freeze a new manifest.')
    if len({(c['channel'], c['case_id']) for c in manifest['cases']}) != len(manifest['cases']):
        raise ValueError('Repeated manifest case.')
    if service.provider is not None and any(m != 'classifier_only' for m in modes) and manifest['partition'] == 'test' and not live_test:
        raise ValueError('Live TEST evaluation requires the explicit --live-test flag; paid tests are not run by default.')
    lookup = {}
    for channel, dataset in CHANNEL_DATASETS.items():
        if manifest['dataset_sha256'][channel] != file_hash(Path(service.config['export_dir'])/f'{dataset}.csv.gz'):
            raise ValueError('Manifest export hash mismatch.')
        df = eligible(load_channel(channel, Path(service.config['export_dir'])))
        lookup[channel] = df.loc[df.split_seed0 == manifest['partition']].set_index('id')
        if any(m != 'direct_llm' for m in modes):
            if service.adapter(channel).provenance != 'canonical_train_verified':
                raise ValueError('Held-out evaluation rejects unverified classifiers/retrieval.')
    selected = manifest['cases'][:max_cases] if max_cases else manifest['cases']
    # Validate the entire selected membership before any provider request.
    for case in selected:
        row = lookup[case['channel']].loc[case['case_id']]
        if int(row.label_fraud) != case['label'] or hashlib.sha256(row.text.encode()).hexdigest() != case['text_sha256']:
            raise ValueError('Manifest case membership/label/text mismatch.')
    output.mkdir(parents=True)
    run_meta = {'manifest_sha256': file_hash(manifest_path), 'manifest': manifest,
                'selected_cases': selected, 'modes': modes, 'global_token_budget': token_budget,
                'max_live_messages': max_live_messages, 'live_test': live_test, 'limits': service.limits, 'prompt_sha256': prompt_hash(),
                'provider': None if service.provider is None else {'kind':service.provider.kind,
                    'model':service.provider.model, 'mock':service.provider.is_mock},
                'model_policy': service.config.get('models'),
                'budget_policy': getattr(service.provider, 'prompt_budget', PromptBudget()).describe(),
                'scope': 'in-domain; phone is synthetic; review flags require separate human assessment'}
    (output/'run.json').write_text(json.dumps(run_meta, indent=2))
    records, remaining = [], token_budget
    live_attempts = 0
    with (output/'predictions.jsonl').open('w') as stream:
        for case in selected:
            text = lookup[case['channel']].loc[case['case_id'], 'text']
            for mode in modes:
                record = {**case, 'mode': mode, 'result': None, 'execution_status': 'pending_budget'}
                if mode == 'classifier_only' or remaining > 0 or service.provider is None:
                    # Default paid scope is <=3 validation messages, including comparison modes.
                    if service.provider is not None and mode != 'classifier_only':
                        live_attempts += 1
                        if live_attempts > max_live_messages:
                            records.append(record)
                            stream.write(json.dumps(record)+'\n'); stream.flush()
                            continue
                    result = service.analyze_message(text, case['channel'], mode, token_budget=max(1, remaining))
                    record.update(result=result, execution_status='executed')
                    remaining -= result['usage']['total_tokens']
                    if not result['usage']['accounting_complete']:
                        remaining = 0
                records.append(record)
                # Persist each actual result/trace before aggregate calculation.
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+'\n')
                stream.flush()
    summary = {}
    for mode in modes:
        group = [r for r in records if r['mode'] == mode]
        summary[mode] = {'overall': aggregate(group), 'per_channel': {
            c: {**aggregate([r for r in group if r['channel'] == c]), 'synthetic': c == 'phone'} for c in CHANNEL_DATASETS}}
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    make = sub.add_parser('manifest')
    make.add_argument('--output', required=True, type=Path)
    make.add_argument('--split', choices=['val', 'test'], default='val')
    make.add_argument('--per-channel', type=int, default=3)
    make.add_argument('--sample-seed', type=int, default=0)
    make.add_argument('--max-chars', type=int, default=8000)
    compare = sub.add_parser('run')
    compare.add_argument('--manifest', required=True, type=Path)
    compare.add_argument('--output', required=True, type=Path)
    compare.add_argument('--config', default='configs/agent.yaml')
    compare.add_argument('--modes', nargs='+', choices=MODES, default=['classifier_only'])
    compare.add_argument('--max-cases', type=int)
    compare.add_argument('--token-budget', type=int, default=12000)
    compare.add_argument('--max-live-messages', type=int, default=3)
    compare.add_argument('--live-test', action='store_true', help='Explicitly enable paid test runs after planning their budget.')
    args = parser.parse_args()
    if args.command == 'manifest':
        manifest = create_manifest(args.output, args.split, args.per_channel, args.sample_seed, args.max_chars)
        print(json.dumps(manifest['support'], indent=2))
    else:
        summary = run(args.manifest, args.output, ScamAgentService.from_config(args.config), args.modes,
                      args.max_cases, args.token_budget, args.max_live_messages, args.live_test)
        print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
