"""Revalidate saved real responses after a deterministic validator repair; no API.

Preserves original results and raw exchanges. Derived records explicitly identify
postprocessing; no prediction, tool result or token usage is invented/replaced.
"""
import argparse
import copy
import json
import time
from pathlib import Path

from src.agent.contract import apply_final_answer, validate_result
from src.agent.evaluate import aggregate


def revalidate(run_dir, audit_dir):
    destination=run_dir/'revalidated_predictions.jsonl'
    if destination.exists():
        raise FileExistsError('Derived revalidation already exists; refusing overwrite.')
    finals={}
    for path in sorted(audit_dir.glob('exchange_*.json')):
        exchange=json.loads(path.read_text())
        message=exchange['response']['message']
        if not message.get('tool_calls'):
            payload=json.loads(exchange['messages'][1]['content'])
            finals[(payload['channel'],payload['untrusted_message'])]=(json.loads(message['content']),path.name)
    records=[]
    for line in (run_dir/'predictions.jsonl').read_text().splitlines():
        record=json.loads(line)
        result=copy.deepcopy(record['result'])
        if result['stop_reason']!='final_answer':
            raise ValueError('Cannot revalidate an unfinished provider conversation as a success.')
        final,filename=finals[(record['channel'],result['views']['original'])]
        before=time.perf_counter()
        result['evidence'], result['unsupported_claims']=[],[]
        result['errors']=[e for e in result['errors'] if e['source']=='tool']
        apply_final_answer(result,final)
        result['needs_review'] |= result['input_policy']['clipped']
        result['status']='needs_review' if result['needs_review'] else 'ok'
        validate_result(result)
        record['result']=result
        record['postprocessing']={'kind':'deterministic_revalidation_of_saved_live_final',
            'source_exchange':filename,'additional_api_calls':0,
            'elapsed_ms':round((time.perf_counter()-before)*1000,3)}
        records.append(record)
    with destination.open('x') as stream:
        for record in records:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    summary={'agent':{'overall':aggregate(records),'per_channel':{
        c:aggregate([r for r in records if r['channel']==c]) for c in ['sms','email','phone']}},
        'execution_kind':'deterministic_revalidation_of_saved_real_responses','additional_api_calls':0}
    (run_dir/'revalidated_summary.json').write_text(json.dumps(summary,indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=Path('outputs/agent/live_agent_smoke_v2'))
    parser.add_argument('--audit-dir',type=Path,default=Path('outputs/agent/live_agent_smoke_v2_api_audit'))
    args=parser.parse_args()
    print(json.dumps(revalidate(args.run_dir,args.audit_dir),indent=2))

if __name__=='__main__':
    main()
