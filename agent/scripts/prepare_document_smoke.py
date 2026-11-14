"""Freeze three label-blind canonical VAL sources for engineering verification."""
import hashlib
import json
from pathlib import Path

import yaml

from src.agent.boundary import distance
from src.agent.data import eligible,file_hash,load_channel
from src.agent.documents import Document
from src.agent.service import ScamAgentService
from src.agent.triage import TriageEngine

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/agent/document_triage_validation_v1'


def main():
    if BASE.exists():raise FileExistsError('Frozen smoke exists; no overwrite.')
    local=yaml.safe_load((ROOT/'configs/agent.local.yaml').read_text())
    public=yaml.safe_load((ROOT/'configs/agent.yaml').read_text())
    for key in ['default_mode','triage_catalog_path','triage']:local[key]=public[key]
    local['project_root']=str(ROOT)
    local['limits'].update(llm_turns=3,tool_calls=4,max_output_tokens=300,max_message_tokens=8000)
    service=ScamAgentService.from_config(ROOT/'configs/agent.yaml')
    engine=TriageEngine(service);cases=[]
    # Select by observable coverage / classifier distance, never current-row labels.
    # Email is investigated by both paid modes; SMS/phone exercise the free path.
    for channel in ['sms','email','phone']:
        df=eligible(load_channel(channel));pool=df.loc[df.split_seed0=='val',['id','text']].copy()
        pool['order']=pool.id.map(lambda i:hashlib.sha256(f'document-smoke-v1:{i}'.encode()).hexdigest())
        policy=engine.models.catalog['channels'][channel]['triage_policy']
        adapter=engine.models.adapter(channel,policy['primary_model'])
        chosen=None
        for row in pool.sort_values('order').itertuples():
            if channel=='email' and not 2000<len(row.text)<=3800:continue
            if channel!='email' and len(row.text)>500:continue
            d=Document(row.text);reading=d.read(d.initial_id)
            pred=adapter.classify(reading['normalised'],reading['normalised_view'])
            if channel=='email':
                unseen=[s for s in d.sections if s['section_id']!=d.initial_id]
                if pred['decision']!='legitimate' or not any(s['request_phrase_count'] for s in unseen):continue
            elif distance(pred)<=policy['boundary_threshold'] or not d.by_id[d.initial_id]['substantive']:continue
            chosen={'source_id':row.id,'case_id':row.id,'channel':channel,
                    'source_sha256':hashlib.sha256(row.text.encode()).hexdigest(),
                    'source_chars':len(row.text),'selection':'long unread request-bearing window' if channel=='email' else 'short non-boundary fast path'}
            break
        if chosen is None:raise ValueError(f'No qualifying VAL case for {channel}; no silent case substitution.')
        cases.append(chosen)
    # Execute long email first; free paths remain runnable even after paid budget stop.
    cases.sort(key=lambda c: c['channel']!='email')
    files=['src/agent/__init__.py','scripts/__init__.py','agent/runtime/documents.py','agent/runtime/triage.py','agent/runtime/service.py','agent/runtime/schema.json',
           'agent/runtime/contract.py','agent/runtime/models.py','agent/runtime/routing.py','agent/runtime/provider.py',
           'agent/runtime/budget.py','agent/runtime/live_meter.py','agent/scripts/compare_document_triage.py',
           'agent/scripts/prepare_document_triage.py','agent/scripts/prepare_document_smoke.py']
    manifest={'partition':'val','interpretation':'Selected engineering coverage; not an effectiveness benchmark.',
              'catalog_sha256':file_hash(Path(service.config['triage_catalog_path'])),
              'source_code_sha256':{p:file_hash(ROOT/p) for p in files},'cases':cases}
    BASE.mkdir(parents=True)
    (BASE/'manifest.json').write_text(json.dumps(manifest,indent=2))
    profile=BASE/'run_config.yaml';profile.write_text(yaml.safe_dump(local,sort_keys=False));profile.chmod(0o600)
    print(json.dumps({'manifest':str(BASE/'manifest.json'),'cases':cases},indent=2))

if __name__=='__main__':main()
