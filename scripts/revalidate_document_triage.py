"""Finish saved real actions after protocol repair, within the ORIGINAL token cap.

Not a fresh end-to-end benchmark. Original failed records remain immutable.
Two final decisions and one constrained-budget execution are new live generations.
No fake tool calls or substituted predictions.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from src.agent.contract import validate_result
from src.agent.data import file_hash,load_channel
from src.agent.live_meter import LiveMeter
from src.agent.service import ScamAgentService
from src.agent.triage import MODES

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/agent/document_triage_validation_v1'


class SavedActionsThenLive:
    is_mock=False
    kind='deepseek_saved_actions_with_live_final'

    def __init__(self,meter,responses):
        self.meter=meter;self.responses=copy.deepcopy(responses)
        self.model=meter.model;self.prompt_budget=meter.prompt_budget

    def complete(self,messages,tools,max_tokens,timeout):
        if self.responses:
            response=self.responses.pop(0)
            allowed={t['function']['name'] for t in tools}
            if not all(c['function']['name'] in allowed for c in response['message']['tool_calls']):
                raise ValueError('Saved real action is no longer allowed; do not substitute.')
            return response
        if tools:raise ValueError('Repair permits live finalization only, no new exploratory action.')
        return self.meter.complete(messages,tools,max_tokens,timeout)


def main():
    original=BASE/'run';target=BASE/'repair_v3'
    if target.exists():raise FileExistsError('Repair exists; no automatic paid repeat.')
    previous=json.loads((original/'api_audit/budget_ledger.json').read_text())
    if not previous['accounting_complete'] or previous['unresolved_reserved_tokens']:
        raise ValueError('Original accounting must be complete before any continuation.')
    repair_tokens = 0
    for path in BASE.glob('repair_*/api_audit/budget_ledger.json'):
        state=json.loads(path.read_text())
        if not state['accounting_complete'] or state['unresolved_reserved_tokens']:
            raise ValueError('Earlier repair accounting is unresolved.')
        repair_tokens+=state['reported_tokens']
    remaining=12000-previous['reported_tokens']-repair_tokens
    if remaining<=0:raise ValueError('Original 12000-token ceiling exhausted.')
    rows=[json.loads(s) for s in (original/'predictions.jsonl').read_text().splitlines()]
    email=next(r for r in rows if r['channel']=='email' and r['mode']=='adaptive_agent')
    raw=load_channel('email').set_index('id').loc[email['source_id']].text
    exchanges=[json.loads(p.read_text()) for p in sorted((original/'api_audit').glob('exchange_*.json'))]
    saved=[exchanges[0]['response']]
    if not all(x['message'].get('tool_calls') for x in saved):raise ValueError('Expected actual saved actions only.')
    service=ScamAgentService.from_config(BASE/'run_config.yaml')
    target.mkdir()
    meter=LiveMeter(service.provider,target/'api_audit',token_limit=remaining,call_limit=3)
    protocol={'execution_kind':'saved_real_actions_reexecuted_with_new_live_finalization_not_fresh_end_to_end',
        'original_reported_tokens':previous['reported_tokens'],'original_global_ceiling':12000,
        'additional_token_cap':remaining,'additional_call_cap':3,'saved_real_action_exchanges':[1],
        'prior_repair_tokens':repair_tokens,
        'original_results_sha256':file_hash(original/'predictions.jsonl'),
        'source_code_sha256':{p:file_hash(ROOT/p) for p in ['src/agent/documents.py','src/agent/triage.py','src/agent/schema.json','src/agent/service.py','scripts/revalidate_document_triage.py']}}
    for path in protocol['source_code_sha256']:
        p=target/'frozen_source'/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/path).read_bytes())
    (target/'protocol.json').write_text(json.dumps(protocol,indent=2))
    repaired=[]
    for mode in ['adaptive_agent','document_fixed']:
        service.provider=SavedActionsThenLive(meter,saved if mode=='adaptive_agent' else [])
        result=service.analyze_message(raw,'email',mode,token_budget=8000)
        validate_result(result)
        prior=next(r for r in rows if r['channel']=='email' and r['mode']==mode)
        repaired.append({**{k:v for k,v in email.items() if k not in ['result','mode']},'mode':mode,'result':result,
                         'execution_kind':protocol['execution_kind']})
        (target/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in repaired))
        assert all(v in prior['result']['classifier_results'] for v in result['classifier_results']),'Reexecuted inference changed.'
        assert result['status']!='error',result['errors']
    # A fresh constrained-budget end-to-end execution also exercises early
    # finalization/partial-document abstention; it is not a classification gain.
    service.provider=meter
    fresh=service.analyze_message(raw,'email','adaptive_agent',token_budget=remaining-meter.state['reported_tokens'])
    repaired.append({**{k:v for k,v in email.items() if k not in ['result','mode']},'mode':'adaptive_agent',
        'result':fresh,'execution_kind':'fresh_end_to_end_remaining_budget_check'})
    (target/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in repaired))
    assert fresh['status']!='error',fresh['errors']
    total=previous['reported_tokens']+repair_tokens+meter.state['reported_tokens']
    assert total<=12000
    print(json.dumps({'additional_calls':meter.state['actual_calls'],'additional_tokens':meter.state['reported_tokens'],
        'combined_tokens_all_attempts':total,'accounting_complete':meter.state['accounting_complete'],
        'outcomes':[{'mode':r['mode'],'execution_kind':r['execution_kind'],
                     **{k:r['result'][k] for k in ['decision','status','stop_reason']}} for r in repaired]},indent=2))

if __name__=='__main__':main()
