"""Verify and report saved normalised/boundary pilot evidence without API calls."""
from __future__ import annotations

import collections
import json

from scripts.prepare_boundary_router import BASE,CATALOG,ROOT
from scripts.prepare_router_stress import exclude_previous
from src.agent.comparison import correct,cost_range
from src.agent.contract import validate_result
from src.agent.data import file_hash
from src.agent.evaluate import aggregate
from src.normalise import normalise_with_changes

MODES=['fixed_model','rule_router','agent_router']
NAMES={'fixed_model':'Normalisation + fixed model','rule_router':'Rule router + boundary guard','agent_router':'Agent router + boundary guard'}


def analyse():
    protocol=json.loads((BASE/'protocol.json').read_text());design=json.loads((BASE/'validation_design.json').read_text())
    manifest=json.loads((BASE/'manifest.json').read_text());catalog=json.loads(CATALOG.read_text())
    for path,digest in protocol['source_code_sha256'].items():
        assert file_hash(ROOT/path)==digest==file_hash(BASE/'frozen_source'/path)
    for path,key in [(BASE/'validation_design.json','design_sha256'),(CATALOG,'catalog_sha256'),
                     (BASE/'manifest.json','manifest_sha256'),(BASE/'run_config.yaml','run_config_sha256')]:assert file_hash(path)==protocol[key]
    excluded,_=exclude_previous(BASE)
    assert len(excluded)==manifest['excluded_previous_base_ids']
    assert all((c['channel'],c['source_id']) not in excluded for c in manifest['cases'])
    rows=[json.loads(s) for s in (BASE/'results/predictions.jsonl').read_text().splitlines()]
    expected={(c['case_id'],m) for c in manifest['cases'] for m in MODES}
    assert len(rows)==len(expected) and {(r['case_id'],r['mode']) for r in rows}==expected
    cases={c['case_id']:c for c in manifest['cases']};indexed={(r['case_id'],r['mode']):r for r in rows}
    performance={};case_rows=[]
    for r in rows:
        result=r['result']
        if result is None:continue
        validate_result(result);norm,changes=normalise_with_changes(cases[r['case_id']]['text'])
        assert result['views']['normalised']==norm
        assert all(v['view']=='normalised' for v in result['classifier_results'])
        trace=[t for t in result['tool_trace'] if t.get('result') and t['tool']=='classify_message']
        assert len(trace)<=2 and len(trace)==len(result['classifier_results'])
        checks=result['routing']['boundary_checks']
        if result['status']!='error':
            assert len(checks)==1
            check=checks[0]
            assert check['secondary_executed']==(check['near_boundary'] and r['mode']!='fixed_model')
            assert sum(t['source']=='boundary_policy' for t in trace)==int(check['secondary_executed'])
            if r['mode']=='agent_router':assert sum(t['source']=='model_tool_call' for t in trace)==1
            if check['secondary_executed']:
                p,s=result['classifier_results']
                if p['decision']!=s['decision']:
                    rules=catalog['channels'][r['channel']]['boundary_policies'][check['primary_model']]
                    winner=rules['conflict_by_primary_prediction'][str(int(p['decision']=='fraud'))]['selected_model']
                    assert result['needs_review'] and check['resolution']==winner
                    assert (result['decision']=='abstain')==(winner=='abstain')
                    if winner!='abstain':assert result['routing']['selected_model_id']==winner
            if result['decision']!='abstain':
                selected=result['routing']['selected_model_id'];assert selected in result['routing']['candidates_called']
                chosen=trace[result['routing']['candidates_called'].index(selected)]['result']
                assert result['decision']==chosen['decision']
    for mode in MODES:
        group=[r for r in rows if r['mode']==mode];metrics=aggregate(group)
        checks=[r['result']['routing']['boundary_checks'][0] for r in group if r['result'] and r['result']['routing']['boundary_checks']]
        conflicts=[r for r in group if r['result'] and len(r['result']['classifier_results'])==2 and
                   len({v['decision'] for v in r['result']['classifier_results']})==2]
        performance[mode]={'metrics':metrics,'correct':sum(map(correct,group)),
            'near_boundary_primaries':sum(c['near_boundary'] for c in checks),'second_classifiers':sum(c['secondary_executed'] for c in checks),
            'conflicts':len(conflicts),'primary_errors_exposed_by_conflict':sum(
                int(r['result']['classifier_results'][0]['decision']=='fraud')!=r['label'] for r in conflicts),
            'unreviewed_wrong':sum(not correct(r) and r['result'] is not None and not r['result']['needs_review'] for r in group),
            'fraud_recall_full_denominator':sum(correct(r) and r['label']==1 for r in group)/sum(r['label']==1 for r in group) if any(r['label']==1 for r in group) else None,
            'selected_models':dict(collections.Counter(r['result']['routing']['selected_model_id'] if r['result'] else None for r in group)),
            'per_stratum':{s:aggregate([r for r in group if r['stratum']==s]) for s in sorted({r['stratum'] for r in group})},
            'per_channel':{ch:aggregate([r for r in group if r['channel']==ch]) for ch in ['sms','email','phone']}}
    for case in manifest['cases']:
        entry={k:case[k] for k in ['case_id','source_id','channel','stratum','label']};entry['modes']={}
        for mode in MODES:
            r=indexed[case['case_id'],mode];v=r['result']
            entry['modes'][mode]=None if v is None else {'correct':correct(r),'decision':v['decision'],'review':v['needs_review'],
                'status':v['status'],'primary':v['routing']['boundary_checks'][0]['primary_model'] if v['routing']['boundary_checks'] else None,
                'called':v['routing']['candidates_called'],'boundary_checks':v['routing']['boundary_checks'],'errors':v['errors']}
        case_rows.append(entry)
    ledger=json.loads((BASE/'api_audit/budget_ledger.json').read_text());events=[];prompt=output=hit=miss=0;low=high=0.
    for path in sorted((BASE/'api_audit').glob('exchange_*.json')):
        x=json.loads(path.read_text());usage=x['response']['usage'];c=cost_range(usage,design['pricing']['per_million_tokens'])
        assert c['cache_counts_reported'];prompt+=usage['prompt_tokens'];output+=usage['completion_tokens']
        hit+=c['cache_hit_tokens'];miss+=c['cache_miss_tokens'];low+=c['usd_low'];high+=c['usd_high']
        events.append({'index':x['index'],'usage':usage,'price_estimate':c})
    assert ledger['accounting_complete'] and ledger['unresolved_reserved_tokens']==0
    assert len(events)==ledger['actual_calls']<=protocol['global_call_limit']
    assert prompt+output==ledger['reported_tokens']<=protocol['global_token_budget']
    assert sum(r['result']['usage']['total_tokens'] for r in rows if r['result'])==prompt+output
    paired={m:{'agent_improved':[c for c in cases if correct(indexed[c,'agent_router']) and not correct(indexed[c,m])],
               'agent_harmed':[c for c in cases if correct(indexed[c,m]) and not correct(indexed[c,'agent_router'])]}
            for m in ['fixed_model','rule_router']}
    analysis={'performance':performance,'cases':case_rows,'paired_agent_outcomes':paired,'actual_calls':len(events),
        'prompt_tokens':prompt,'completion_tokens':output,'reported_tokens':prompt+output,'cache_hit_tokens':hit,'cache_miss_tokens':miss,
        'estimated_usd_low':low,'estimated_usd_high':high,'accounting_complete':True,'cost_events':events}
    (BASE/'analysis.json').write_text(json.dumps(analysis,indent=2))
    return analysis,manifest,design,rows


def report(a,manifest,design,rows):
    lines=['# Normalised routing with mandatory boundary cross-checks','',
        'Date: 10 November 2026, Asia/Singapore. This is a small, deliberately boundary-conditioned independent TEST challenge.','',
        '**No incremental classification gain was observed. All three policies scored 10/12 (83.3%), answered macro-F1 0.8125, fraud recall 0.60, and zero false positives.** '
        'All 36 analyses completed, without technical failures or abstentions. The rule and Agent guards made eight and six secondary calls respectively. '
        'Each routed one disagreement for review, retaining the already correct primary result. Neither exposed the observed false negatives. '
        'The Agent used 24 paid requests and 55,177 tokens; the local rule router reached the same labels without API calls.','',
        '## What was implemented','',
        'All three policies run the same existing English normaliser before every classifier. The fixed baseline runs one VAL-selected model. '
        'The rule and Agent routers must run a distinct, VAL-selected second classifier whenever their primary is within its own native-score boundary threshold. '
        'The Agent makes a genuine function call to choose exactly one primary and receives every actual classifier result in one tool reply. '
        'The backend, rather than an optional LLM decision, enforces the second call and conflict policy. '
        'At most two classifiers and three LLM turns are permitted. This hybrid Agent delegates primary selection; it is not an unconstrained model ensemble.','',
        'Schema 1.2 records the original/normalised views, preprocessing changes, primary distance/threshold, actual secondary execution, '
        'proposed versus enforced selection and trace source `boundary_policy`. A mandatory local call is never presented as a model-generated call. '
        'Unsupported conflicts abstain and request review; supported winners still require review. Classifier outputs remain the source of final labels.','',
        '## VAL-only calibration and TEST isolation','',
        'The same 12 eligible saved candidates use identical canonical seed0 TRAIN memberships within each channel. No new model/vectorizer fitting occurred. '
        'Calibration used all 2,330 VAL bases, each clean plus four character perturbation families at two severities (20,970 correlated views). '
        'Original source prefixes are capped at 4,000 characters for every method, then perturbed where specified and normalised. '
        'Stress edits preserve digits and protected URL/email/amount spans. Inherited labels assume intended-meaning preservation.','',
        'Fixed models maximize mean per-condition VAL macro-F1. Profile rules require at least 30 distinct VAL bases and three of each class. '
        'For each primary, choose a threshold quantile from 5%, 10%, 20%, 30% and a secondary using '
        '`(correct - 2*wrong - .25*abstain - .02*second_calls)/N`. Distances are abs(SVM margin) or abs(uncalibrated fraud probability minus .5); '
        'their numeric scales are model-specific, not calibrated risk and not compared between models. '
        'A conflict direction requires 10 distinct base IDs and at least 60% winner share; otherwise abstain. '
        'Correlated variants from one source count as one conflict vote. These constants and candidate quantiles were locked before TEST. '
        'Apparent VAL results are selection statistics, not an independent gain estimate.','',
        '| Channel | Normalised fixed | VAL mean macro-F1 | Primary boundary threshold | Secondary | VAL quantile |',
        '|---|---|---:|---:|---|---:|']
    for ch,d in design['diagnostics'].items():
        fixed=d['fixed_model'];p=d['boundary_policies'][fixed]
        lines.append(f"| {ch} | {fixed} | {d['mean_normalised_condition_f1'][fixed]:.6f} | {p['threshold']:.6f} | {p['secondary_model']} | {p['quantile']:.0%} |")
    lines+=['',f"Exclude all {manifest['excluded_previous_base_ids']} previously used TEST base IDs. The selector receives only IDs/text and normalised native distances, never true labels. "
        'Select one clean near-boundary input, one clean/perturbed near-boundary pair and one random control per channel, '
        'by deterministic hash seed 6407. Choose the perturbation condition on VAL candidate-disagreement rate after normalisation; '
        'no harder condition was chosen on TEST. Only after source IDs were fixed were labels joined for scoring. '
        'All channels had eligible near-boundary candidates, with no fallback. Freeze design, catalog, manifest, config and source hashes before paid calls.','',
        f"The resulting 12 inputs have nine independent source messages, {sum(c['label']==0 for c in manifest['cases'])} legitimate and "
        f"{sum(c['label']==1 for c in manifest['cases'])} fraud views. Four phone views are synthetic. "
        f"The base-prefix policy clipped {sum(c['base_was_clipped'] for c in manifest['cases'])} selected observations. "
        'Paired views are correlated and boundary selection deliberately changes the distribution. This is not general population accuracy.','',
        '## Measured outcomes','',
        '| Policy | Correct / total | Coverage | Answered macro-F1 | Fraud recall, all fraud inputs | FPR, answered | Review | Abstain | Technical failure |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda n:'n/a' if n is None else f'{n:.4f}'
    for mode,p in a['performance'].items():
        m=p['metrics'];lines.append(f"| {NAMES[mode]} | {p['correct']}/{m['total_cases']} | {m['answered_coverage']:.1%} | {fmt(m['macro_f1_answered'])} | "
            f"{fmt(p['fraud_recall_full_denominator'])} | {fmt(m['false_positive_rate_answered'])} | {m['review_cases']} | {m['abstentions']} | {m['technical_failures']} |")
    lines+=['','Correct/total keeps abstentions, technical failures and pending inputs in the denominator. Answered-case F1/FPR must be read with coverage; '
        'abstention is not counted as correct. Confusion matrices, precision/recall, per-channel and per-stratum metrics are retained in analysis.json.','',
        '| Stratum | n | Fixed correct | Rule correct | Agent correct |','|---|---:|---:|---:|---:|']
    for s in sorted({r['stratum'] for r in rows}):
        group=[r for r in rows if r['stratum']==s];n=sum(r['mode']=='fixed_model' for r in group)
        lines.append(f'| {s} | {n} | '+' | '.join(str(sum(correct(r) for r in group if r['mode']==m)) for m in MODES)+' |')
    lines+=['','| Policy | Near primaries | Second classifiers | Conflicts | Primary errors exposed | Unreviewed wrong | Mean latency | API requests / tokens |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for mode,p in a['performance'].items():
        m=p['metrics'];lines.append(f"| {NAMES[mode]} | {p['near_boundary_primaries']} | {p['second_classifiers']} | {p['conflicts']} | "
            f"{p['primary_errors_exposed_by_conflict']} | {p['unreviewed_wrong']} | {m['mean_elapsed_ms']/1000:.4f}s | {m['llm_calls']} / {m['usage']['total_tokens']:,} |")
    lines+=['','Near-boundary status alone is not an error label. Agreement between two classifiers does not establish correctness. '
        'The fixed baseline records distance for audit but deliberately does not cross-check. '
        'Every completed rule/Agent analysis with a near-boundary primary was verified to execute its mandatory second classifier.','',
        '| Input | Truth | Fixed | Rule | Agent | Agent actual calls |','|---|---|---|---|---|---|']
    for c in a['cases']:
        v=c['modes'];decision=lambda m:'pending' if v[m] is None else v[m]['decision']+(' (review)' if v[m]['review'] else '')
        lines.append(f"| {c['case_id']} | {'fraud' if c['label'] else 'legitimate'} | "+' | '.join(decision(m) for m in MODES)+
            ' | '+(', '.join(v['agent_router']['called']) if v['agent_router'] else 'none')+' |')
    lines+=['','## Failure and complementary-error analysis','',
        'The two incorrect observations are the clean and perturbed views of the same fraud-labelled email, `nazario_5_2033`; '
        'they are one independent source failure. Its source has 1,025,016 characters. Under the common, predeclared 4,000-character prefix policy, '
        'the observed input is dominated by a long recipient address list. Later content is unavailable to every policy. '
        'This clipping limitation is a plausible cause of the shared failure; the experiment does not establish the full-message outcome. '
        'A post-run local diagnostic ran all four normalised candidates on the same bounded inputs: every candidate predicted legitimate '
        'on both views. Thus this candidate set and input policy offer no routing rescue for that source. '
        'The Agent actually cross-checked char SVM with word SVM on both views, obtained agreement, and did not request review. '
        'The rule router used word NB alone on the clean view and both SVMs on the perturbed view, with the same missed-fraud label.','',
        'The only observed disagreement is `mendeley_4596`: char SVM correctly predicts fraud while word NB predicts legitimate. '
        'The locked VAL conflict policy retains char SVM and requests review. Post-run local diagnostics show word LR and word SVM also predict legitimate; '
        'a majority vote would be wrong. The guard adds review work on a correct fixed-model decision, not a classification improvement. '
        'These post-run diagnostics did not change policies, prompts, samples or paid results. They are saved separately in candidate_diagnostics.jsonl.','',
        'Per-channel correct counts are SMS 4/4, email 2/4 and synthetic phone 4/4 for all three methods. '
        'All three clean/perturbed pairs have identical outcomes within each policy. '
        'This new set differs from earlier experiments; 10/12 here versus 11/12 previously is not a measured regression caused by normalisation.','',
        '## Cost and evidence','',
        f"Actual API usage: {a['actual_calls']} requests, {a['prompt_tokens']:,} prompt + {a['completion_tokens']:,} completion = "
        f"{a['reported_tokens']:,} tokens. Cache accounting: {a['cache_hit_tokens']:,} hit / {a['cache_miss_tokens']:,} miss. "
        f"Estimated total API price: **${a['estimated_usd_low']:.8f}–${a['estimated_usd_high']:.8f} USD**, using "
        '[official off-peak/peak list rates](https://api-docs.deepseek.com/quick_start/pricing/) checked 10 November 2026. '
        'This is not a billing receipt. Local CPU, training and storage costs are excluded. Fixed/rule policies make zero API requests. '
        'Warm-service timings include routing, local inference and provider round trips.','',
        'The frozen ceiling was 75,000 tokens, 36 requests, 12 paid analyses, 350 output tokens/request, no retries or post-result policy changes. '
        'All filtered wire usage reconciles with per-result usage and the global ledger; outstanding reservations are zero and accounting is complete. '
        'Keys remain in macOS Keychain and are excluded from outputs. Analysis/verification sends no additional API requests.','',
        'Local ignored evidence: outputs/agent/normalised_boundary_v1/{validation_design.json,validation_predictions.jsonl,manifest.json,protocol.json,frozen_source/,run_config.yaml,results/,api_audit/,analysis.json}. '
        'The runtime catalog is models/router_normalised_seed0/catalog.json; original candidate models and teammate artifacts are preserved.','',
        '```sh','.venv/bin/python -m scripts.prepare_boundary_router validation','.venv/bin/python -m scripts.prepare_boundary_router test',
        '.venv/bin/python -m scripts.run_boundary_router','.venv/bin/python -m scripts.report_boundary_router','```','',
        'Preparation and paid execution refuse existing experiment/results paths. The report command only reads saved results and refreshes the derived report. '
        'Reproduction needs the ignored canonical export, candidate artifacts and explicit provider credentials for the paid step.','',
        '## Interpretation','',
        'Compare the observed counts above without treating review/abstention as a correct fraud decision. '
        'All three methods share normalisation, so this experiment measures incremental cross-check/routing behavior; it cannot isolate normalisation gain against a raw-input baseline. '
        'Nine independent source messages, one provider run, synthetic phone data and deliberate boundary conditioning cannot establish statistical superiority or real-traffic recall. '
        'The Agent has less freedom than earlier experiments: its primary choice is LLM-generated, but secondary execution and conflict handling are deterministic. '
        'Any gain shared by the rule router is not evidence that an LLM is needed. '
        'The present evidence supports the normalised fixed model as the economical baseline and the local rule guard as an optional review mechanism; '
        'it does not support paying for an LLM on every message.','',
        '## Verification','',
        'The full offline suite passed: 154 tests, seven original raw-data-dependent skips. Dependency consistency and compilation passed. '
        'Every recorded result validates against schema.json. Source snapshots, config/catalog/input hashes, prior-source exclusions, '
        'actual normalised views, mandatory secondary execution, selected-model grounding and exact usage reconciliation were checked. '
        'Offline fixtures exercise unsupported-conflict abstention and prevention of an LLM overriding the locked winner; '
        'no unsupported-conflict abstention occurred in this live sample. No extra paid verification or retries were sent.']
    content='\n'.join(lines)+'\n';(ROOT/'docs/normalised_boundary_report.md').write_text(content);(BASE/'report.md').write_text(content)


def main():
    a,manifest,design,rows=analyse();report(a,manifest,design,rows)
    print(json.dumps({k:a[k] for k in ['actual_calls','reported_tokens','estimated_usd_low','estimated_usd_high','paired_agent_outcomes']},indent=2))

if __name__=='__main__':main()
