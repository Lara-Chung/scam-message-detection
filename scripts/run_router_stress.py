"""Execute one locked independent stress/disagreement challenge, never auto retry."""
import json
from pathlib import Path

from src.agent.data import load_channel,file_hash
from src.agent.evaluate import aggregate
from src.agent.live_meter import LiveMeter
from src.agent.routing import RouterEngine
from src.agent.service import ScamAgentService
from src.agent.stress import BASE_CHARS,perturb
from src.agent.candidates import CANDIDATES
from scripts.prepare_router_stress import BASE,ROOT

MODES=['fixed_model','rule_router','agent_router']


def main():
    protocol=json.loads((BASE/'protocol.json').read_text())
    for path,digest in protocol['source_code_sha256'].items():
        if file_hash(ROOT/path)!=digest:raise ValueError('Frozen source changed before execution.')
    for file,key in [('catalog.json','catalog_sha256'),('validation_design.json','design_sha256'),('manifest.json','manifest_sha256')]:
        if file_hash(BASE/file)!=protocol[key]:raise ValueError('Frozen design/catalog/manifest changed.')
    if (BASE/'results').exists():raise FileExistsError('Results exist; no repeat of paid calls.')
    service=ScamAgentService.from_config(BASE/'run_config.yaml')
    if service.provider is None or service.provider.is_mock or service.provider.model!='deepseek-flash' or service.provider.base_url!='https://api.deepseek.com':
        raise ValueError('Required real official provider/model is not configured.')
    engine=RouterEngine(service.config,service.limits)
    for channel in ['sms','email','phone']:
        for name in CANDIDATES:engine.adapter(channel,name)
    service._router_engine=engine
    manifest=json.loads((BASE/'manifest.json').read_text());lookups={ch:load_channel(ch).set_index('id') for ch in ['sms','email','phone']}
    import hashlib
    for case in manifest['cases']:
        source=lookups[case['channel']].loc[case['source_id']]
        if source.split_seed0!='test' or int(source.label_fraud)!=case['label'] or hashlib.sha256(source.text.encode()).hexdigest()!=case['source_sha256']:
            raise ValueError('Independent TEST source identity/hash mismatch.')
        text=source.text[:BASE_CHARS]
        if case['transform']!='none':
            family,severity=case['transform'].split(':');text,edits=perturb(text,family,float(severity),f'test:{case["source_id"]}')
            if edits!=case['edits']:raise ValueError('Transform provenance differs.')
        if text!=case['text'] or len(text)>=8000:raise ValueError('Prepared challenge input changed.')
    meter=LiveMeter(service.provider,BASE/'api_audit',token_limit=75000,call_limit=36);service.provider=meter
    out=BASE/'results';out.mkdir()
    (out/'run.json').write_text(json.dumps({'protocol_sha256':file_hash(BASE/'protocol.json'),'provider':{'kind':meter.kind,'model':meter.model,'mock':False}},indent=2))
    rows=[];remaining=75000
    with (out/'predictions.jsonl').open('x') as stream:
        for case in manifest['cases']:
            for mode in MODES:
                record={k:v for k,v in case.items() if k not in ['text','edits']}
                record.update(mode=mode,result=None,execution_status='pending_budget')
                if mode!='agent_router' or remaining>0:
                    result=service.analyze_message(case['text'],case['channel'],mode,token_budget=max(1,remaining))
                    record.update(result=result,execution_status='executed');remaining-=result['usage']['total_tokens']
                    if not result['usage']['accounting_complete']:remaining=0
                rows.append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
    strata=sorted({r['stratum'] for r in rows})
    summary={m:{'overall_challenge_only':aggregate([r for r in rows if r['mode']==m]),
                'per_stratum':{s:aggregate([r for r in rows if r['mode']==m and r['stratum']==s]) for s in strata},
                'per_channel':{c:aggregate([r for r in rows if r['mode']==m and r['channel']==c]) for c in lookups}}
             for m in MODES}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    with (out/'candidate_diagnostics.jsonl').open('x') as stream:
        for case in manifest['cases']:
            predictions={n:engine.adapter(case['channel'],n).classify(case['text'],'original')['decision'] for n in CANDIDATES}
            stream.write(json.dumps({'case_id':case['case_id'],'channel':case['channel'],'label':case['label'],
                'stratum':case['stratum'],'predictions':predictions})+'\n')
    print(json.dumps({'actual_calls':meter.state['actual_calls'],'reported_tokens':meter.state['reported_tokens'],
        'accounting_complete':meter.state['accounting_complete'],'summary':{m:d['overall_challenge_only'] for m,d in summary.items()}},indent=2))

if __name__=='__main__':main()
