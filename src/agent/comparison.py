"""Cost attribution and paired outcomes for a saved, already executed pilot."""
import json
import statistics
from pathlib import Path


def cost_range(usage, rates):
    """Use reported cache counts; list-price estimates are not billing receipts."""
    prompt,output=usage['prompt_tokens'],usage['completion_tokens']
    hit=usage.get('prompt_cache_hit_tokens',usage.get('prompt_tokens_details',{}).get('cached_tokens'))
    miss=usage.get('prompt_cache_miss_tokens')
    if miss is None and type(hit) is int:
        miss=prompt-hit
    known=type(hit) is int and type(miss) is int and min(hit,miss)>=0 and hit+miss==prompt
    if known:
        low=(hit*rates['off_peak']['cache_hit']+miss*rates['off_peak']['cache_miss']+
             output*rates['off_peak']['output'])/1_000_000
        high=(hit*rates['peak']['cache_hit']+miss*rates['peak']['cache_miss']+
              output*rates['peak']['output'])/1_000_000
    else:
        low=(prompt*rates['off_peak']['cache_hit']+output*rates['off_peak']['output'])/1_000_000
        high=(prompt*rates['peak']['cache_miss']+output*rates['peak']['output'])/1_000_000
    return {'usd_low':low,'usd_high':high,'cache_counts_reported':known,
            'cache_hit_tokens':hit if known else None,'cache_miss_tokens':miss if known else None}


def correct(record):
    result=record.get('result')
    if not result or result['status']=='error' or result['decision']=='abstain':
        return False
    return int(result['decision']=='fraud')==record['label']


def analyze_pilot(base: Path):
    """Read results/filtered wire data only. Never invoke a provider or alter records."""
    protocol=json.loads((base/'protocol.json').read_text())
    rates=protocol['pricing']['per_million_tokens']
    records=[json.loads(line) for line in (base/'results/predictions.jsonl').read_text().splitlines()]
    ledger=json.loads((base/'api_audit/budget_ledger.json').read_text())
    modes=protocol['modes']
    lookup={(r['channel'],r['result']['views']['original'],r['mode']):r for r in records if r.get('result')}
    costs={mode:{'usd_low':0.,'usd_high':0.,'actual_calls':0,'prompt_tokens':0,'completion_tokens':0,
                 'cache_hit_tokens':0,'cache_miss_tokens':0,'cache_accounting_complete':True} for mode in modes}
    events=[]
    for path in sorted((base/'api_audit').glob('exchange_*.json')):
        exchange=json.loads(path.read_text())
        payload=json.loads(exchange['messages'][1]['content'])
        mode='agent' if exchange['tools'] else ('fixed_workflow' if 'fixed_tool_results' in payload else 'direct_llm')
        record=lookup[(payload['channel'],payload['untrusted_message'],mode)]
        usage=exchange['response']['usage']
        if usage is None:
            continue
        charge=cost_range(usage,rates)
        target=costs[mode]
        target['actual_calls']+=1
        for key in ['prompt_tokens','completion_tokens']:
            target[key]+=usage[key]
        for key in ['usd_low','usd_high']:
            target[key]+=charge[key]
        for key in ['cache_hit_tokens','cache_miss_tokens']:
            target[key]+=charge[key] or 0
        target['cache_accounting_complete'] &= charge['cache_counts_reported']
        events.append({'index':exchange['index'],'case_id':record['case_id'],'channel':record['channel'],
                       'mode':mode,'usage':usage,'list_price_estimate':charge})
    accounted=sum(c['prompt_tokens']+c['completion_tokens'] for c in costs.values())
    if accounted!=ledger['reported_tokens']:
        raise ValueError('Filtered exchanges do not reconcile with ledger usage; do not publish cost totals.')
    accounting_complete=(ledger['accounting_complete'] and ledger['unresolved_reserved_tokens']==0
                         and sum(c['actual_calls'] for c in costs.values())==ledger['actual_calls'])
    summary=json.loads((base/'results/summary.json').read_text())
    for mode in modes:
        group=[r for r in records if r['mode']==mode]
        metrics=summary[mode]['overall']
        if metrics['usage']['total_tokens']!=costs[mode]['prompt_tokens']+costs[mode]['completion_tokens']:
            raise ValueError('Mode-level result tokens do not match the actual wire usage.')
        latencies=[r['result']['elapsed_ms'] for r in group if r.get('result')]
        metrics['median_elapsed_ms']=statistics.median(latencies) if latencies else None
        metrics['correct_cases_full_denominator']=sum(map(correct,group))
        metrics['evidence_coordinate_repairs']=sum(
            sum(e['code']=='evidence_span_repaired' for e in r['result']['errors']) for r in group if r.get('result'))
        metrics['unsupported_claim_events']=sum(len(r['result']['unsupported_claims']) for r in group if r.get('result'))
        metrics['estimated_list_price']=costs[mode] if accounting_complete else None
    paired={}
    indexed={(r['case_id'],r['mode']):r for r in records}
    ids=list(dict.fromkeys(r['case_id'] for r in records))
    for baseline in ['classifier_only','direct_llm','fixed_workflow']:
        improved,harmed=[],[]
        for ident in ids:
            a,b=correct(indexed[(ident,'agent')]),correct(indexed[(ident,baseline)])
            if a and not b:improved.append(ident)
            if b and not a:harmed.append(ident)
        paired[baseline]={'agent_improved_cases':improved,'agent_harmed_cases':harmed}
    result={'execution_kind':'actual_live_paired_test_pilot','protocol':protocol,'summary':summary,
            'cost_events':events,'accounting_complete':accounting_complete,'actual_calls':ledger['actual_calls'],
            'reported_tokens':accounted,'estimated_usd_low':sum(c['usd_low'] for c in costs.values()) if accounting_complete else None,
            'estimated_usd_high':sum(c['usd_high'] for c in costs.values()) if accounting_complete else None,
            'paired_agent_outcomes':paired}
    target=base/'analysis.json'
    if target.exists():
        raise FileExistsError('Pilot analysis exists; do not overwrite the recorded analysis.')
    target.write_text(json.dumps(result,indent=2))
    return result
