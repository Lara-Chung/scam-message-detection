"""Freeze and execute a 12-case classifier-routing pilot, with one paid mode."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from src.agent.candidates import CANDIDATES
from src.agent.data import load_channel,file_hash
from src.agent.evaluate import create_manifest,aggregate
from src.agent.live_meter import LiveMeter
from src.agent.routing import RouterEngine,ROUTER_PROMPT,FINAL,TOOLS
from src.agent.service import ScamAgentService
from scripts.run_agent_comparison import PRICING

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/agent/router_comparison_v1'
MODES=['fixed_model','rule_router','agent_router']
FILES=['src/agent/candidates.py','src/agent/routing.py','src/agent/models.py','src/agent/service.py',
       'src/agent/contract.py','src/agent/schema.json','src/agent/budget.py','src/agent/provider.py',
       'src/agent/live_meter.py','scripts/run_router_comparison.py']


def prepare():
    if BASE.exists():raise FileExistsError('Routing pilot already exists; never overwrite a frozen experiment.')
    manifest=create_manifest(BASE/'manifest.json',split='test',per_channel=4,sample_seed=6405,balanced=True)
    service=ScamAgentService.from_config(ROOT/'configs/agent.local.yaml')
    config=service.config
    config['project_root']=str(ROOT)
    config['limits'].update(llm_turns=3,tool_calls=2,max_output_tokens=350,max_message_tokens=12000)
    config['pricing']={'input_usd_per_million':.3,'output_usd_per_million':1.2}
    (BASE/'run_config.yaml').write_text(yaml.safe_dump(config,sort_keys=False));(BASE/'run_config.yaml').chmod(0o600)
    catalog_path=Path(config['routing_catalog_path'])
    engine=RouterEngine(config,config['limits'])
    for channel in ['sms','email','phone']:
        for model in CANDIDATES:engine.adapter(channel,model)
    protocol={'design':'12 paired balanced canonical seed0 TEST cases; two cases per class per channel',
        'modes':MODES,'case_selection':'hash seed 6405, class-stratified, no result-driven selection',
        'input_policy':manifest['input_policy'],'candidate_order':CANDIDATES,
        'fixed_policy':'best global VAL macro-F1 per channel; fixed candidate order breaks ties',
        'rule_policy':engine.catalog['bucket_policy'],'validation_selection_only':True,
        'candidate_catalog_sha256':file_hash(catalog_path),'router_prompt':ROUTER_PROMPT,'final_schema':FINAL,'tools':TOOLS,
        'decision_policy':'Only selected, actually called classifier predicts the label. Conflicting outputs force review.',
        'global_token_budget':60000,'global_call_limit':36,'paid_message_limit':12,
        'max_output_tokens_per_call':350,'turns_per_message':3,'classifier_calls_per_message':2,
        'provider':'deepseek_chat','model':'deepseek-flash','temperature':0,'thinking':'disabled',
        'pricing':PRICING,'repeats':1,'model_warmup':'preload all 12 artifacts before timing; inference only, no fitting',
        'stop_policy':'Unknown accounting stops further calls. No retries or post-result rule/prompt changes.',
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),
        'code_sha256':{p:file_hash(ROOT/p) for p in FILES}}
    (BASE/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps({'status':'frozen','support':manifest['support'],'directory':str(BASE)},indent=2))


def execute():
    protocol=json.loads((BASE/'protocol.json').read_text())
    if any(file_hash(ROOT/p)!=h for p,h in protocol['code_sha256'].items()):raise ValueError('Frozen routing code changed.')
    config=yaml.safe_load((BASE/'run_config.yaml').read_text())
    if file_hash(Path(config['routing_catalog_path']))!=protocol['candidate_catalog_sha256']:
        raise ValueError('Frozen candidate catalog changed.')
    if (BASE/'results').exists():raise FileExistsError('Results exist; do not repeat paid calls.')
    service=ScamAgentService.from_config(BASE/'run_config.yaml')
    if service.provider is None or service.provider.is_mock or service.provider.model!='deepseek-flash' or service.provider.base_url!='https://api.deepseek.com':
        raise ValueError('This pilot requires the explicit real official DeepSeek model/profile.')
    engine=RouterEngine(service.config,service.limits)
    for channel in ['sms','email','phone']:
        for name in CANDIDATES:engine.adapter(channel,name)
    service._router_engine=engine
    manifest=json.loads((BASE/'manifest.json').read_text())
    lookups={c:load_channel(c).set_index('id') for c in ['sms','email','phone']}
    for case in manifest['cases']:
        row=lookups[case['channel']].loc[case['case_id']]
        if row.split_seed0!='test' or int(row.label_fraud)!=case['label'] or hashlib.sha256(row.text.encode()).hexdigest()!=case['text_sha256']:
            raise ValueError('Test membership/text/label mismatch before live calls.')
    meter=LiveMeter(service.provider,BASE/'api_audit',token_limit=60000,call_limit=36)
    service.provider=meter
    output=BASE/'results';output.mkdir()
    (output/'run.json').write_text(json.dumps({'protocol_sha256':file_hash(BASE/'protocol.json'),
        'manifest_sha256':file_hash(BASE/'manifest.json'),'provider':{'kind':meter.kind,'model':meter.model,'mock':False}},indent=2))
    records=[];remaining=60000
    with (output/'predictions.jsonl').open('x') as stream:
        for case in manifest['cases']:
            text=lookups[case['channel']].loc[case['case_id'],'text']
            for mode in MODES:
                record={**case,'mode':mode,'result':None,'execution_status':'pending_budget'}
                if mode!='agent_router' or remaining>0:
                    result=service.analyze_message(text,case['channel'],mode,token_budget=max(1,remaining))
                    record.update(result=result,execution_status='executed')
                    remaining-=result['usage']['total_tokens']
                    if not result['usage']['accounting_complete']:remaining=0
                records.append(record)
                stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
    summary={m:{'overall':aggregate([r for r in records if r['mode']==m]),
                'per_channel':{c:aggregate([r for r in records if r['mode']==m and r['channel']==c]) for c in lookups}}
             for m in MODES}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    # Post-run oracle is diagnostic only; predictions are never given to the LLM.
    with (output/'candidate_test_predictions.jsonl').open('x') as stream:
        for case in manifest['cases']:
            text=lookups[case['channel']].loc[case['case_id'],'text'][:service.limits['max_input_chars']]
            predictions={name:engine.adapter(case['channel'],name).classify(text,'original')['decision'] for name in CANDIDATES}
            stream.write(json.dumps({**case,'candidate_decisions':predictions})+'\n')
    print(json.dumps({'status':'recorded','actual_calls':meter.state['actual_calls'],
        'reported_tokens':meter.state['reported_tokens'],'accounting_complete':meter.state['accounting_complete'],
        'summary':{m:d['overall'] for m,d in summary.items()}},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','run'])
    args=parser.parse_args();prepare() if args.command=='prepare' else execute()

if __name__=='__main__':main()
