"""Execute the locked common-normalisation and mandatory-boundary-check pilot."""
import hashlib
import json

from src.agent.data import load_channel,file_hash
from src.agent.evaluate import aggregate
from src.agent.live_meter import LiveMeter
from src.agent.routing import RouterEngine
from src.agent.service import ScamAgentService
from src.agent.stress import BASE_CHARS,perturb
from scripts.prepare_boundary_router import BASE,CATALOG,ROOT

MODES=['fixed_model','rule_router','agent_router']


def main():
    p=json.loads((BASE/'protocol.json').read_text())
    for path,digest in p['source_code_sha256'].items():
        if file_hash(ROOT/path)!=digest or file_hash(BASE/'frozen_source'/path)!=digest:raise ValueError('Frozen source changed.')
    for path,key in [(BASE/'validation_design.json','design_sha256'),(CATALOG,'catalog_sha256'),
                     (BASE/'manifest.json','manifest_sha256'),(BASE/'run_config.yaml','run_config_sha256')]:
        if file_hash(path)!=p[key]:raise ValueError('Frozen design/input/config changed.')
    if (BASE/'results').exists():raise FileExistsError('Results exist; no automatic paid repeat.')
    service=ScamAgentService.from_config(BASE/'run_config.yaml')
    if service.provider is None or service.provider.is_mock or service.provider.model!='deepseek-flash' or service.provider.base_url!='https://api.deepseek.com':
        raise ValueError('Expected explicit real official provider and model.')
    engine=RouterEngine(service.config,service.limits)
    for ch in ['sms','email','phone']:
        for n in engine.catalog['candidate_order']:engine.adapter(ch,n)
    service._router_engine=engine
    manifest=json.loads((BASE/'manifest.json').read_text());lookup={ch:load_channel(ch).set_index('id') for ch in ['sms','email','phone']}
    for case in manifest['cases']:
        row=lookup[case['channel']].loc[case['source_id']]
        if row.split_seed0!='test' or int(row.label_fraud)!=case['label'] or hashlib.sha256(row.text.encode()).hexdigest()!=case['source_sha256']:
            raise ValueError('Independent TEST source mismatch.')
        text=row.text[:BASE_CHARS]
        if case['transform']!='none':
            f,s=case['transform'].split(':');text,edits=perturb(text,f,float(s),f'boundary-test:{case["source_id"]}')
            if edits!=case['edits']:raise ValueError('Transform differs from frozen provenance.')
        if text!=case['text'] or hashlib.sha256(text.encode()).hexdigest()!=case['input_sha256']:raise ValueError('Input mismatch.')
    meter=LiveMeter(service.provider,BASE/'api_audit',token_limit=75000,call_limit=36);service.provider=meter
    out=BASE/'results';out.mkdir();rows=[];remaining=75000
    (out/'run.json').write_text(json.dumps({'protocol_sha256':file_hash(BASE/'protocol.json'),
        'provider':{'kind':meter.kind,'model':meter.model,'mock':False}},indent=2))
    with (out/'predictions.jsonl').open('x') as stream:
        for case in manifest['cases']:
            for mode in MODES:
                row={k:v for k,v in case.items() if k not in ['text','edits']}
                row.update(mode=mode,result=None,execution_status='pending_budget')
                if mode!='agent_router' or remaining>0:
                    result=service.analyze_message(case['text'],case['channel'],mode,token_budget=max(1,remaining))
                    row.update(result=result,execution_status='executed');remaining-=result['usage']['total_tokens']
                    if not result['usage']['accounting_complete']:remaining=0
                rows.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
    strata=sorted({r['stratum'] for r in rows})
    summary={m:{'overall_conditioned':aggregate([r for r in rows if r['mode']==m]),
                'per_stratum':{s:aggregate([r for r in rows if r['mode']==m and r['stratum']==s]) for s in strata},
                'per_channel':{ch:aggregate([r for r in rows if r['mode']==m and r['channel']==ch]) for ch in lookup}}
             for m in MODES}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'actual_calls':meter.state['actual_calls'],'reported_tokens':meter.state['reported_tokens'],
        'accounting_complete':meter.state['accounting_complete'],'summary':{m:d['overall_conditioned'] for m,d in summary.items()}},indent=2))

if __name__=='__main__':main()
