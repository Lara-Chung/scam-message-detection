"""Run at most three frozen validation messages with a durable 12,000-token cap."""
import argparse
import json
from pathlib import Path

import httpx

from src.agent.evaluate import run
from src.agent.live_meter import LiveMeter
from src.agent.service import ScamAgentService

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/agent.local.yaml')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/agent/live_agent_smoke_v2')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Live output already exists; inspect it instead of automatically rerunning.')
    service=ScamAgentService.from_config(args.config)
    if service.provider is None:
        raise ValueError('Local DeepSeek credential/profile is not configured.')
    if service.provider.base_url != 'https://api.deepseek.com':
        raise ValueError('This authorized smoke targets only the official DeepSeek endpoint.')
    # Auth/model discovery is a non-generative GET; no extra validation message.
    with httpx.Client(timeout=15,follow_redirects=False) as client:
        response=client.get(service.provider.base_url+'/models',
                            headers={'Authorization':'Bearer '+service.provider._key})
    if response.status_code != 200:
        print(json.dumps({'status':'preflight_failed','http_status':response.status_code,
                          'llm_calls':0,'message':'DeepSeek authentication/model listing failed; no generation was attempted.'}))
        return 2
    models=response.json()
    if service.provider.model not in [item['id'] for item in models.get('data',[])]:
        raise ValueError('The explicitly configured model is unavailable; no model fallback was attempted.')
    # The wrapper creates only its audit child directory; the evaluation output
    # itself must remain absent until the runner has validated the frozen cases.
    audit=args.output.parent/(args.output.name+'_api_audit')
    provider=LiveMeter(service.provider,audit,token_limit=12000,call_limit=12)
    (audit/'model_preflight.json').write_text(json.dumps({'status':'ok','http_status':200,
        'selected_model':provider.model,'available_model_ids':[m['id'] for m in models['data']],
        'generation_calls':0},indent=2))
    service.provider=provider
    summary=run(ROOT/'outputs/agent/validation_manifest_v1.json',args.output,service,
                ['agent'],max_cases=3,token_budget=12000,max_live_messages=3)
    print(json.dumps({'status':'run_recorded','output':str(args.output),'audit':str(audit),
        'actual_provider_calls':provider.state['actual_calls'],'reported_tokens':provider.state['reported_tokens'],
        'accounting_complete':provider.state['accounting_complete'],
        'summary':summary['agent']['overall']},indent=2))
    passed = all(summary['agent']['per_channel'][c]['answered_cases'] == 1
                 and summary['agent']['per_channel'][c]['technical_failures'] == 0 for c in ['sms','email','phone'])
    return 0 if passed and provider.state['accounting_complete'] else 2

if __name__=='__main__':
    raise SystemExit(main())
