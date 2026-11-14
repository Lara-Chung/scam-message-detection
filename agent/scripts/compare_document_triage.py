"""Bounded three-mode comparison; offline baseline and explicitly enabled live modes.

Prepare manifests separately. Never auto-select a new case or retry paid failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.agent.contract import validate_result
from src.agent.data import file_hash,load_channel
from src.agent.evaluate import aggregate
from src.agent.live_meter import LiveMeter
from src.agent.service import ScamAgentService
from src.agent.triage import MODES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',default='configs/agent.yaml')
    parser.add_argument('--modes',nargs='+',choices=MODES,default=MODES)
    parser.add_argument('--token-budget',type=int,default=12000)
    parser.add_argument('--max-paid-messages',type=int,default=3)
    parser.add_argument('--allow-live',action='store_true')
    parser.add_argument('--allow-live-test',action='store_true')
    args=parser.parse_args()
    if min(args.token_budget,args.max_paid_messages)<=0:parser.error('Positive global bounds required.')
    manifest=json.loads(args.manifest.read_text())
    if manifest['partition'] not in ['val','test']:raise ValueError('Explicit canonical partition required.')
    if args.output.exists():raise FileExistsError('Output exists; no automatic paid repeat.')
    service=ScamAgentService.from_config(args.config)
    paid=bool(set(args.modes)&{'adaptive_agent','document_fixed'})
    if paid and not args.allow_live:raise ValueError('Pass --allow-live for explicitly configured real execution.')
    if paid and manifest['partition']=='test' and not args.allow_live_test:raise ValueError('Held-out paid TEST requires --allow-live-test.')
    if paid and (service.provider is None or service.provider.is_mock):raise ValueError('Explicit real provider required, no fixtures/fallback.')
    root=Path(__file__).resolve().parents[2]
    for path,digest in manifest['source_code_sha256'].items():
        if file_hash(root/path)!=digest:raise ValueError('Frozen source changed before evaluation.')
    catalog=Path(service.config['triage_catalog_path'])
    if file_hash(catalog)!=manifest['catalog_sha256']:raise ValueError('VAL policy changed.')
    lookups={ch:load_channel(ch).set_index('id') for ch in ['sms','email','phone']}
    cases=[]
    for case in manifest['cases']:
        row=lookups[case['channel']].loc[case['source_id']]
        if row.split_seed0!=manifest['partition'] or hashlib.sha256(row.text.encode()).hexdigest()!=case['source_sha256']:
            raise ValueError('Canonical source mismatch.')
        cases.append((case,row.text,int(row.label_fraud)))
    args.output.mkdir(parents=True)
    meter=LiveMeter(service.provider,args.output/'api_audit',token_limit=args.token_budget,call_limit=12) if paid else None
    if meter:service.provider=meter
    # Preserve the exact code underlying this execution, without local profiles/keys.
    for path in manifest['source_code_sha256']:
        target=args.output/'frozen_source'/path;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((root/path).read_bytes())
    (args.output/'protocol.json').write_text(json.dumps({'manifest_sha256':file_hash(args.manifest),'modes':args.modes,
        'global_token_limit':args.token_budget,'max_paid_messages':args.max_paid_messages,'max_api_calls':12,
        'retries':0,'partition':manifest['partition'],'model':getattr(service.provider,'model',None),
        'interpretation':'Engineering validation, not held-out effectiveness' if manifest['partition']=='val' else 'Independent TEST comparison',
        'limits':service.limits,'catalog_sha256':file_hash(catalog),'source_code_sha256':manifest['source_code_sha256']},indent=2))
    records=[];remaining=args.token_budget;used_paid=0
    with (args.output/'predictions.jsonl').open('x') as stream:
        for case,text,label in cases:
            for mode in args.modes:
                record={**case,'label':label,'mode':mode,'result':None,'execution_status':'pending_budget'}
                if mode=='document_classifier' or (remaining>0 and used_paid<args.max_paid_messages):
                    result=service.analyze_message(text,case['channel'],mode,token_budget=max(1,remaining))
                    validate_result(result);record.update(result=result,execution_status='executed')
                    if result['llm_calls']:used_paid+=1
                    remaining-=result['usage']['total_tokens']
                    if not result['usage']['accounting_complete']:remaining=0
                records.append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
    summary={m:{'overall':aggregate([r for r in records if r['mode']==m]),
        'per_channel':{ch:aggregate([r for r in records if r['mode']==m and r['channel']==ch]) for ch in lookups}} for m in args.modes}
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'actual_calls':meter.state['actual_calls'] if meter else 0,
        'reported_tokens':meter.state['reported_tokens'] if meter else 0,
        'accounting_complete':meter.state['accounting_complete'] if meter else True,'summary':summary},indent=2))

if __name__=='__main__':main()
