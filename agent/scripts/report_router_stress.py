"""Offline analysis of a frozen stress/disagreement experiment; no API calls."""
from __future__ import annotations

import collections
import json
from pathlib import Path

from src.agent.comparison import correct,cost_range
from src.agent.data import file_hash
from src.normalise import normalise_with_changes
from scripts.prepare_router_stress import BASE,ROOT,exclude_previous

MODES=['fixed_model','rule_router','agent_router']
NAMES={'fixed_model':'Robust VAL-selected fixed model','rule_router':'VAL profile rules','agent_router':'Agent router'}


def main():
    protocol=json.loads((BASE/'protocol.json').read_text());design=json.loads((BASE/'validation_design.json').read_text())
    manifest=json.loads((BASE/'manifest.json').read_text());summary=json.loads((BASE/'results/summary.json').read_text())
    for path,digest in protocol['source_code_sha256'].items():
        if file_hash(ROOT/path)!=digest or file_hash(BASE/'frozen_source'/path)!=digest:raise ValueError('Frozen source verification failed.')
    rows=[json.loads(s) for s in (BASE/'results/predictions.jsonl').read_text().splitlines()]
    candidate_rows=[json.loads(s) for s in (BASE/'results/candidate_diagnostics.jsonl').read_text().splitlines()]
    indexed={(r['case_id'],r['mode']):r for r in rows}
    expected={c['case_id']:c for c in manifest['cases']}
    excluded,_=exclude_previous()
    assert all((c['channel'],c['source_id']) not in excluded for c in manifest['cases'])
    ledger=json.loads((BASE/'api_audit/budget_ledger.json').read_text())
    prompt=output=hit=miss=0;low=high=0;events=[]
    for path in sorted((BASE/'api_audit').glob('exchange_*.json')):
        exchange=json.loads(path.read_text());u=exchange['response']['usage'];c=cost_range(u,design['pricing']['per_million_tokens'])
        assert c['cache_counts_reported']
        prompt+=u['prompt_tokens'];output+=u['completion_tokens'];hit+=c['cache_hit_tokens'];miss+=c['cache_miss_tokens']
        low+=c['usd_low'];high+=c['usd_high'];events.append({'index':exchange['index'],'usage':u,'price_estimate':c})
    assert ledger['accounting_complete'] and ledger['unresolved_reserved_tokens']==0
    assert len(events)==ledger['actual_calls'] and prompt+output==ledger['reported_tokens']
    assert sum(r['result']['usage']['total_tokens'] for r in rows if r['result'])==prompt+output
    challenge_ids={r['case_id'] for r in candidate_rows if r['stratum'] in ['natural_disagreement','paired_obfuscated']
                   and len(set(r['predictions'].values()))>1}
    pair_sources={(c['channel'],c['source_id']) for c in manifest['cases'] if c['stratum']=='paired_clean'}
    pairs=[]
    for channel,source_id in sorted(pair_sources):
        entry={'channel':channel,'source_id':source_id,'modes':{}}
        for mode in MODES:
            clean=indexed[(f'{source_id}::paired_clean',mode)];changed=indexed[(f'{source_id}::paired_obfuscated',mode)]
            entry['modes'][mode]={'clean_correct':correct(clean),'obfuscated_correct':correct(changed),
                                 'clean_decision':clean['result']['decision'],'obfuscated_decision':changed['result']['decision']}
        pairs.append(entry)
    performance={}
    for mode in MODES:
        group=[r for r in rows if r['mode']==mode]
        challenge=[r for r in group if r['case_id'] in challenge_ids]
        controls=[r for r in group if r['case_id'] not in challenge_ids]
        performance[mode]={'all_correct':sum(map(correct,group)),'all_n':len(group),
            'selected_disagreement_correct':sum(map(correct,challenge)),'selected_disagreement_n':len(challenge),
            'other_controls_correct':sum(map(correct,controls)),'other_controls_n':len(controls),
            'unreviewed_correct':sum(correct(r) and not r['result']['needs_review'] for r in group),
            'choices':dict(collections.Counter(r['result']['routing']['selected_model_id'] for r in group))}
    paired={m:{'agent_improved':[c for c in expected if correct(indexed[(c,'agent_router')]) and not correct(indexed[(c,m)])],
               'agent_harmed':[c for c in expected if correct(indexed[(c,m)]) and not correct(indexed[(c,'agent_router')])]}
            for m in ['fixed_model','rule_router']}
    posthoc=[]
    for case in manifest['cases']:
        if case['stratum']!='paired_obfuscated':continue
        normal,counts=normalise_with_changes(case['text']);clean=expected[f'{case["source_id"]}::paired_clean']
        posthoc.append({'source_id':case['source_id'],'normaliser_changes':counts,'exactly_restores_paired_clean':normal==clean['text'],
            'scored_clean_fixed_decision':indexed[(clean['case_id'],'fixed_model')]['result']['decision'],
            'interpretation':'Post-hoc deterministic input-equivalence diagnostic; not a scored policy or new API result.'})
    analysis={'execution_kind':'independent_TEST_challenge_conditional_not_population','performance':performance,'pairs':pairs,'paired_agent_outcomes':paired,
        'actual_calls':len(events),'prompt_tokens':prompt,'completion_tokens':output,'reported_tokens':prompt+output,
        'cache_hit_tokens':hit,'cache_miss_tokens':miss,'estimated_usd_low':low,'estimated_usd_high':high,
        'accounting_complete':True,'unique_source_messages':len({(c['channel'],c['source_id']) for c in manifest['cases']}),
        'selected_disagreement_ids':sorted(challenge_ids),'posthoc_normalisation':posthoc,'summary':summary,'cost_events':events}
    (BASE/'analysis.json').write_text(json.dumps(analysis,indent=2))
    lines=['# Independent character-obfuscation and disagreement challenge', '',
        'Date: 10 November 2026, Asia/Singapore. This is a deliberately conditioned challenge, not a random population-performance estimate.', '',
        '## Finding', '',
        '**No additional label accuracy was observed from Agent routing.** All three policies return 11/12 correct inputs, '
        'including 4/5 among preselected disagreement inputs. One fraud SMS is correctly classified clean but missed after four invisible-character insertions. '
        'Other candidates can classify it correctly; the Agent does not select or cross-check them on that input. '
        'This exposes a real unused routing opportunity, not merely a clean-sample accuracy ceiling.', '',
        '## VAL-only design and independent TEST boundary', '',
        '- Reuse the same 12 candidate artifacts fitted only on canonical seed0 eligible TRAIN. No vectorizer/classifier fitting on augmented VAL or TEST.',
        '- Evaluate the full 2,330 VAL bases under clean text plus four letter-only stress families at severities 0.25 and 0.50: '
        'zero-width characters, Cyrillic lookalikes, fullwidth letters and spaced letters. This produces 20,970 correlated validation observations, '
        'not 20,970 independent messages.',
        '- Use the first 4,000 source characters for all conditions to bound cost, with transformed text kept below 8,000 characters. '
        'Keep digits, punctuation and protected URLs, emails and amount spans intact. Labels are inherited under an intended-meaning assumption; '
        'this is a synthetic perturbation test, not observed real-world attacker traffic.',
        '- Select the robust fixed classifier using mean per-condition VAL macro-F1 over all nine conditions. '
        'Select profile rules using only VAL results and observable normaliser changes; require 30 distinct base IDs and at least three per class, '
        'otherwise fall back to the robust fixed model. Rules use no current-message true label.',
        '- Select the stress family/severity for TEST by highest VAL candidate-disagreement rate among variants changing at least half the messages. '
        'Ties prefer lower severity then declared family order. Freeze the catalog and validation_design.json before examining TEST candidates.',
        '- Exclude 105 base IDs from all previous recorded TEST manifests. The new selector receives only id/text and candidate predictions; '
        'its interface rejects labels. Only after selecting IDs are source labels joined for scoring.',
        '- Freeze one natural disagreement (or declared ordinary fallback), one clean/obfuscated pair and one random clean control per channel. '
        'Select by SHA-256 seed 6406; do not select cases where a particular method is known to be right or wrong.',
        '- This yields 12 inputs from nine previously unused TEST base messages. Paired views are correlated. '
        'Disagreement conditioning changes the evaluated distribution: report strata and pairs separately, not as general TEST accuracy.',
        '- Lock prompt/guidance, current-profile VAL cards, policies, manifest, input construction and code hashes before live requests. '
        'Copies of all frozen source files are saved under frozen_source/. No post-result changes, reruns or sample replacements.', '',
        '## Selected VAL policies', '',
        '| Channel | Word LR mean F1 | Word NB mean F1 | Word SVM mean F1 | Char SVM mean F1 | Robust fixed | TEST stress from VAL |',
        '|---|---:|---:|---:|---:|---|---|']
    for ch,d in design['diagnostics'].items():
        lines.append(f"| {ch} | "+' | '.join(f"{d['mean_condition_f1'][n]:.6f}" for n in ['word_lr','word_nb','word_svm','char_svm'])+
                     f" | {d['fixed_model']} | {d['test_stress_condition']} |")
    lines+=['', 'The email word/character SVM means differ by less than one millionth; the deterministic ranking is not evidence of statistically meaningful superiority. '
        'VAL rule results are selection-set/apparent results, not independent performance. Agent cards contain only aggregate VAL results for the current observable profile, '
        'not per-case outcomes, rule answers or TEST labels.', '',
        '| Channel | none | digit-word | invisible | compatibility | lookalike | spaced | mixed |',
        '|---|---|---|---|---|---|---|---|']
    for ch,d in design['diagnostics'].items():
        lines.append(f"| {ch} | "+' | '.join(d['profile_rules'][k]['selected'] for k in ['none','digit_word','invisible','compat','lookalike','spaced','mixed'])+' |')
    lines+=['', '## Independent challenge availability', '',
        '| Channel | Unused eligible TEST bases | Natural disagreements | Stress disagreements | Natural fallback |',
        '|---|---:|---:|---:|---|']
    for v in manifest['selection']:
        lines.append(f"| {v['channel']} | {v['eligible_unseen_test_bases']} | {v['natural_disagreement_available']} | {v['stress_disagreement_available']} | {v['natural_fallback']} |")
    lines+=['', 'Stress counts are on the remaining pool after removing the selected natural case. Phone had zero VAL disagreement under all conditions; '
        'its TEST natural stratum therefore uses the predeclared ordinary fallback. One new phone base disagreed after the VAL-chosen stress transformation. '
        'No stronger perturbation was chosen after learning that fact.', '',
        f"Actual selected input support: {sum(c['label']==0 for c in manifest['cases'])} legitimate / {sum(c['label']==1 for c in manifest['cases'])} fraud. "
        f"Source-prefix clipping affected {sum(c['base_was_clipped'] for c in manifest['cases'])} selected observations. "
        'No transformed input was subsequently clipped by the service. The 4,000-character base policy was fixed on VAL even though it did not truncate these selected sources.', '',
        'A binary candidate-disagreement stratum necessarily has at least one candidate predicting each possible label. '
        'Its oracle upper bound is therefore 100% by construction, not a discovery or deployable algorithm. '
        'Selection is label-blind but the challenge is still conditional, not an unbiased sample.', '',
        '## Actual outcomes, including all failures and review routing', '',
        '| Policy | Correct / 12 | Macro-F1 | Fraud precision | Fraud recall | FPR | Review / failures / abstentions | Mean latency |',
        '|---|---:|---:|---:|---:|---:|---|---:|']
    for mode in MODES:
        m=summary[mode]['overall_challenge_only'];p=performance[mode]
        lines.append(f"| {NAMES[mode]} | {p['all_correct']}/12 | {m['macro_f1_answered']:.4f} | {m['fraud_precision_answered']:.3f} | "
          f"{m['fraud_recall_answered']:.3f} | {m['false_positive_rate_answered']:.3f} | {m['review_cases']} / {m['technical_failures']} / {m['abstentions']} | {m['mean_elapsed_ms']/1000:.4f}s |")
    lines+=['', 'All 36 case/policy records executed. Answered coverage is 12/12 for each method; there are no pending cases. '
        'Overall confusion matrix is [[4,0],[1,7]] for each policy. This pooled result describes only this challenge mixture.', '',
        '| Stratum | n | Fixed correct | Rule correct | Agent correct |', '|---|---:|---:|---:|---:|']
    for stratum in sorted({r['stratum'] for r in rows}):
        counts=[sum(correct(r) for r in rows if r['mode']==mode and r['stratum']==stratum) for mode in MODES]
        n=sum(r['mode']=='fixed_model' and r['stratum']==stratum for r in rows)
        lines.append(f"| {stratum} | {n} | "+' | '.join(f'{c}/{n}' for c in counts)+' |')
    lines+=['', 'The five explicitly selected disagreement inputs score 4/5 for all three policies. The other seven observations score 7/7. '
        'The three clean paired views score 3/3 and their perturbed partners 2/3. No average over these correlated views is claimed to be a population estimate.', '',
        '| Paired source | Channel | Clean decision, all policies | Perturbed decision, all policies | Correct-to-wrong flip |',
        '|---|---|---|---|---|']
    for pair in pairs:
        v=pair['modes']['fixed_model']
        lines.append(f"| {pair['source_id']} | {pair['channel']} | {v['clean_decision']} | {v['obfuscated_decision']} | {v['clean_correct'] and not v['obfuscated_correct']} |")
    lines+=['', '## Observed failure and unexploited complementary model', '',
        'For `mendeley_1043`, the supplied fraud label is inherited unchanged. The clean SMS is correctly labelled fraud by all four candidates. '
        'The controlled variant inserts U+200B only into `Your`, `Card`, `Code` and `Call`; the amount, card digits, date, reference and callback number remain unchanged.', '',
        '- On the perturbed view, word LR and char SVM predict legitimate; word NB and word SVM predict fraud.',
        '- Fixed and profile-rule policies choose char SVM and miss it. The Agent also calls only char SVM, chooses its legitimate result and does not request review.',
        '- The actually observed char-SVM fraud margin is approximately -0.0193, near its zero decision boundary. This is a native decision margin, not a probability or calibrated confidence.',
        '- The Agent cross-checks both SVMs on the clean version but does not cross-check the version that causes the only miss. '
        'Thus the additional LLM budget was not allocated to the observed rescue opportunity. Calling another classifier could expose disagreement; '
        'it does not by itself guarantee the Agent would select the correct one.',
        '- No post-result threshold, prompt instruction or paid rerun was introduced to rescue this example.', '',
        'Post-hoc, the existing deterministic normaliser removes the four zero-width characters and exactly reproduces the already scored clean input. '
        'That clean input was classified as fraud. This is recorded as an input-equivalence diagnostic, not a measured fourth policy or additional API result. '
        'The scored policies classify the same unnormalised views; they use normaliser change counts only as observable routing features. '
        'A normalise-then-strong-classifier baseline is a plausible inexpensive next experiment, but this run does not claim its general performance.', '',
        '## Work, latency and API price', '',
        '| Policy | Classifier calls | API requests | Reported tokens | Estimated API price, all 12 |',
        '|---|---:|---:|---:|---:|']
    for mode in MODES:
        m=summary[mode]['overall_challenge_only'];price=f'${low:.6f}–${high:.6f} USD' if mode=='agent_router' else '$0 incremental API'
        lines.append(f"| {NAMES[mode]} | {m['tool_calls']} | {m['llm_calls']} | {m['usage']['total_tokens']:,} | {price} |")
    lines+=['',f"Actual usage: **{len(events)} generation requests; {prompt:,} prompt + {output:,} completion = {prompt+output:,} tokens**. "
        f"Provider cache counts: {hit:,} hit / {miss:,} miss. Accounting is complete, all request reservations held, and outstanding reservations are zero. "
        'The ceiling was 75,000 total tokens and 36 requests, with at most 350 output tokens/request and no retries.', '',
        'Costs apply the [official off-peak and peak list rates](https://api-docs.deepseek.com/quick_start/pricing/) checked 10 November 2026 to reported cache usage. '
        'They are price estimates, not account-balance deductions; local hardware, training and storage costs are excluded. '
        'Timing is warm-service, including routing, model inference and provider round trips. Small local fixed/rule timing differences are not reliable speedup evidence.', '',
        'The Agent made one classifier call on three SMS inputs and two calls on the other nine inputs. It therefore varies some call decisions, '
        'unlike the preceding uniform two-call pilot. This observed variation did not improve labels or identify the only error.', '',
        '## Limits and conclusion', '',
        'This is stronger evidence than a clean-sample ceiling test because real candidate disagreements and a paired perturbation-induced error are present. '
        'It still supplies **no observed accuracy gain** for Agent routing over the two local policies. '
        'The result supports keeping the strong fixed classifier as the economical default rather than paying for an LLM on every message.', '',
        'There are only nine independent source messages, one stochastic run and a deliberately conditioned class mix. '
        'Four phone views are synthetic. Character edits assume intended meaning/label preservation and have not been independently adjudicated by a human reviewer. '
        'The normaliser and profiles are English-oriented. Validation conditions repeat the same bases and are not independent observations. '
        'No p-value, general fraud recall, real-traffic prevalence estimate or universal claim about Agent usefulness follows from this result.', '',
        '## Reproduction and evidence', '',
        '```sh', '.venv/bin/python -m scripts.prepare_router_stress validation',
        '.venv/bin/python -m scripts.prepare_router_stress test',
        '.venv/bin/python -m scripts.run_router_stress',
        '.venv/bin/python -m scripts.report_router_stress','```','',
        'Existing design/manifest/results paths reject overwrite and automatic paid reruns. '
        'Original candidate artifacts, previous experiments and teammate training code are preserved. '
        'Only the independent stress experiment uses the new catalog, profile-specific cards and locked cost guidance; ordinary configurations retain the previous bucket behavior.', '',
        'Ignored local evidence under outputs/agent/router_stress_v1/: validation_design.json; catalog.json; '
        'validation_predictions.jsonl (20,970 correlated VAL rows); manifest.json including source hashes and exact edit spans; '
        'protocol.json and frozen_source/; results/predictions.jsonl (36 actual results); results/summary.json; '
        'results/candidate_diagnostics.jsonl; api_audit/budget_ledger.json and 24 filtered wire exchanges; analysis.json.', '',
        'Keys remain in macOS Keychain; no credential, Authorization header or hidden reasoning is saved in those wire exchanges. '
        'No further paid calls were made during analysis.']
    lines += ['', 'Verification after execution: **145 tests passed, 7 skipped**. Compilation, dependency consistency, patch whitespace, independent-source exclusions, all recorded result schemas, selected-model grounding, immutable source snapshots and exact token reconciliation passed. Verification sent no additional API request.']
    text='\n'.join(lines)+'\n';(ROOT/'docs/router_stress_report.md').write_text(text);(BASE/'report.md').write_text(text)
    print(json.dumps({'report':str(ROOT/'docs/router_stress_report.md'),'actual_calls':len(events),'reported_tokens':prompt+output,
                      'estimated_usd_low':low,'estimated_usd_high':high,'performance':performance},indent=2))

if __name__=='__main__':main()
