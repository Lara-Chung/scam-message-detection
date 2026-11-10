"""Frozen six-case A/B/D/C diagnostic comparison with explicit live accounting."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from src.agent.data import load_channel
from src.agent.evaluate import create_manifest, run
from src.agent.live_meter import LiveMeter
from src.agent.service import ScamAgentService

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/agent/comparison_pilot_v1'
MODES=['classifier_only','direct_llm','fixed_workflow','agent']
PRICING={
    'source':'https://api-docs.deepseek.com/quick_start/pricing/',
    'checked_date_singapore':'2026-11-10', 'model':'deepseek-flash',
    'currency':'USD', 'per_million_tokens':{
        'off_peak':{'cache_hit':0.003,'cache_miss':0.15,'output':0.6},
        'peak':{'cache_hit':0.006,'cache_miss':0.3,'output':1.2}},
    'note':'Estimated list-price range, not an account-balance or invoice measurement.'}


def prepare():
    manifest=BASE/'manifest.json'
    if BASE.exists():
        raise FileExistsError('Pilot directory exists; inspect the frozen protocol instead of replacing it.')
    create_manifest(manifest,split='test',per_channel=2,sample_seed=20261110,balanced=True)
    config=ScamAgentService.from_config(ROOT/'configs/agent.local.yaml').config
    # Snapshot resolved paths/hyperparameters but never serialize the provider/key.
    config['project_root']=str(ROOT)
    config['pricing']={'input_usd_per_million':0.3,'output_usd_per_million':1.2}
    (BASE/'run_config.yaml').write_text(__import__('yaml').safe_dump(config,sort_keys=False))
    (BASE/'run_config.yaml').chmod(0o600)
    files=['src/agent/service.py','src/agent/contract.py','src/agent/tools.py','src/agent/provider.py',
           'src/agent/evaluate.py','src/agent/budget.py','src/agent/live_meter.py','scripts/run_agent_comparison.py']
    protocol={'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'design':'paired, balanced diagnostic canonical seed0 TEST sample; not representative prevalence',
        'cases':6,'per_channel':2,'per_channel_class_support':{'legitimate':1,'fraud':1},
        'modes':MODES,'mode_order_policy':'rotate_llm','repeats':1,
        'provider':'deepseek_chat','model':'deepseek-flash','temperature':0,'thinking':'disabled',
        'global_token_budget':60000,'global_generation_call_limit':36,'live_message_limit':18,
        'output_tokens_per_call':700,'per_message_turn_limit':4,'per_message_tool_limit':6,
        'input_policy':{'name':'prefix_chars_v1','max_chars':8000},
        'decision_rule':'unchanged production final validator including visible unique-quote repairs',
        'primary_metrics':['conservative_full_denominator_success','macro_f1_answered','answered_coverage',
                           'fraud_precision_answered','fraud_recall_answered','false_positive_rate_answered',
                           'technical_failure_rate','review_coverage','latency','token_usage','estimated_cost'],
        'latency_policy':'preload all classifiers and TRAIN retrieval matrices; report warm-service timings',
        'failure_policy':'no retries, no prompt fixes after results, all errors/abstentions/pending cases retained',
        'pricing':PRICING,'code_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files}}
    (BASE/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps({'status':'frozen','directory':str(BASE),
                      'support':json.loads(manifest.read_text())['support']},indent=2))


def execute():
    protocol=json.loads((BASE/'protocol.json').read_text())
    for name,expected in protocol['code_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Frozen comparison source changed; no live request was made.')
    service=ScamAgentService.from_config(BASE/'run_config.yaml')
    if service.provider is None or service.provider.is_mock or service.provider.model!='deepseek-flash':
        raise ValueError('Required real provider/model not configured.')
    if service.provider.base_url!='https://api.deepseek.com':
        raise ValueError('Pilot is authorized only for the official DeepSeek endpoint.')
    if (BASE/'results').exists():
        raise FileExistsError('Comparison results exist; paid experiments are never automatically repeated.')
    # All preprocessing is already fitted on canonical TRAIN. Warm only transforms.
    for channel in ['sms','email','phone']:
        service.adapter(channel).retrieve('warmup',1)
    provider=LiveMeter(service.provider,BASE/'api_audit',token_limit=60000,call_limit=36)
    service.provider=provider
    summary=run(BASE/'manifest.json',BASE/'results',service,MODES,
                token_budget=60000,max_live_messages=18,live_test=True,mode_order_policy='rotate_llm')
    print(json.dumps({'status':'comparison_recorded','reported_tokens':provider.state['reported_tokens'],
                     'actual_calls':provider.state['actual_calls'],
                     'accounting_complete':provider.state['accounting_complete'],
                     'summary':{mode:data['overall'] for mode,data in summary.items()}},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run'])
    args=parser.parse_args()
    prepare() if args.command=='prepare' else execute()

if __name__=='__main__':
    main()
