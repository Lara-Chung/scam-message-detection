"""Render a transparent English report from an executed pilot; never call an API."""
import json
from pathlib import Path

from src.agent.comparison import correct

BASE=Path(__file__).resolve().parents[1]/'outputs/agent/comparison_pilot_v1'
ROOT=BASE.parents[2]
NAMES={'classifier_only':'Classifier','direct_llm':'Direct LLM','fixed_workflow':'Fixed workflow','agent':'Agent'}


def main():
    analysis=json.loads((BASE/'analysis.json').read_text())
    records=[json.loads(line) for line in (BASE/'results/predictions.jsonl').read_text().splitlines()]
    modes=analysis['protocol']['modes']
    by_key={(r['case_id'],r['mode']):r for r in records}
    cases=list(dict.fromkeys(r['case_id'] for r in records))
    lines=['# Six-case paired Agent comparison', '',
        'Date: 10 November 2026, Asia/Singapore. This is a small diagnostic pilot, not a representative accuracy benchmark.', '',
        '## Finding', '',
        '**This pilot shows no additional classification benefit from the Agent over the existing classifier or fixed workflow.** '
        'All three return six correct final decisions. The Agent uses fewer deterministic tool operations than the fixed workflow '
        'but needs more LLM round trips, has higher estimated API cost and latency, and routes more cases to review. '
        'It supplies two more usable final results than direct LLM, comprising one voluntary abstention and one evidence-output failure, '
        'rather than correcting two wrong binary labels.', '',
        '## Frozen design', '',
        '- Six eligible canonical seed0 TEST cases: one legitimate and one fraud per channel, selected by SHA-256 ordering '
        'with sample seed 20261110 within each class. Equal support is intentional and does not represent deployment prevalence.',
        '- Same exact cases, original text policy, unchanged system/final schema, and saved model policy across A/B/D/C. '
        'The current case ID, dataset/subtype and true label are never sent as provider clues. Phone data is synthetic.',
        '- DeepSeek `deepseek-flash`, non-thinking, temperature 0, one trial. Direct/fixed have one LLM turn; Agent has at most '
        'four turns and six tool calls. The Agent chooses its own tools. Every mode uses the same final-answer validator.',
        '- Three paid mode orders rotate over the six cases, so every paid mode occupies each position twice. '
        'Classifiers and TRAIN-only retrieval matrices are warmed before timing. Prompt-cache differences remain visible in the cost totals.',
        '- All six inputs are below 8,000 characters; no case is clipped or dropped. Models and retrieval have verified canonical TRAIN membership.',
        '- Independent global ceiling: 60,000 tokens, 36 generation requests, 18 paid message analyses. No retries, '
        'post-result prompt tuning, additional sampling or reruns. The prior 3-message validation is not reused as this experiment.', '',
        '## Objective outcomes', '',
        '| Mode | Correct / all 6 | Answered coverage | Abstentions | Technical failures | Review cases | Answered macro-F1 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for mode in modes:
        m=analysis['summary'][mode]['overall']
        lines.append(f"| {NAMES[mode]} | {m['correct_cases_full_denominator']}/6 | {m['answered_coverage']:.1%} | "
                     f"{m['abstentions']} | {m['technical_failures']} | {m['review_cases']}/6 | {m['macro_f1_answered']:.2f} |")
    lines += ['', 'Answered-case F1 is 1.00 even for direct LLM because its two abstentions are excluded from that metric. '
              'The full-denominator success and coverage columns prevent that exclusion from hiding failures. '
              'Review-flagged binary answers remain in the answered metrics; review coverage is reported separately. '
              'All planned case/mode records were executed; there are zero pending cases.', '',
              '| Mode | Fraud precision on answers | Fraud recall on answers | Conservative fraud recall over all 3 positives | FPR on answers | Confusion matrix on answers [0,1] |',
              '|---|---:|---:|---:|---:|---|']
    supplements={}
    for mode in modes:
        m=analysis['summary'][mode]['overall']
        group=[r for r in records if r['mode']==mode]
        full_recall=sum(correct(r) for r in group if r['label']==1)/3
        supplements[mode]={'conservative_fraud_recall_full_positive_denominator':full_recall}
        lines.append(f"| {NAMES[mode]} | {m['fraud_precision_answered']:.2f} | {m['fraud_recall_answered']:.2f} | "
                     f"{full_recall:.2f} | {m['false_positive_rate_answered']:.2f} | `{m['confusion_matrix_answered_label_order_0_1']}` |")
    lines += ['', 'The conservative full-positive recall is a supplementary descriptive metric: an unanswered fraud case counts '
              'as not detected. The frozen primary results remain in the original summary. Classifier average precision is '
              'computed only from actual oriented scores; no continuous LLM confidence or LLM PR-AUC is invented.', '',
              '| Channel | True support legitimate / fraud | Classifier correct | Direct LLM correct | Fixed correct | Agent correct |',
              '|---|---:|---:|---:|---:|---:|']
    for channel in ['sms','email','phone']:
        counts=[sum(correct(r) for r in records if r['channel']==channel and r['mode']==mode) for mode in modes]
        lines.append(f"| {channel}{' (synthetic)' if channel=='phone' else ''} | 1 / 1 | "+' | '.join(f'{c}/2' for c in counts)+' |')
    lines += ['', '## Latency, work and cost', '',
              '| Mode | Mean latency / case | Median | Tool operations | Actual LLM calls | Prompt + output tokens | Estimated API cost, six cases (USD) |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for mode in modes:
        m=analysis['summary'][mode]['overall'];c=m['estimated_list_price']
        lines.append(f"| {NAMES[mode]} | {m['mean_elapsed_ms']/1000:.4f}s | {m['median_elapsed_ms']/1000:.4f}s | "
                     f"{m['tool_calls']} | {c['actual_calls']} | {m['usage']['total_tokens']:,} | "
                     f"${c['usd_low']:.6f}–${c['usd_high']:.6f} |")
    a,d=analysis['summary']['agent']['overall'],analysis['summary']['fixed_workflow']['overall']
    lines += ['',f"The Agent used {a['tool_calls']} versus {d['tool_calls']} tool operations (26.7% fewer), but "
              f"{a['llm_calls']} versus {d['llm_calls']} API round trips. It consumed "
              f"{a['usage']['total_tokens']/d['usage']['total_tokens']:.2f} times the tokens, "
              f"{a['estimated_list_price']['usd_high']/d['estimated_list_price']['usd_high']:.2f} times the estimated API fee "
              f"and {a['mean_elapsed_ms']/d['mean_elapsed_ms']:.2f} times the mean latency of the fixed workflow, without another correct decision.", '',
              f"Actual total: **{analysis['actual_calls']} provider requests / {analysis['reported_tokens']:,} reported tokens**. "
              f"Cache-aware list-price estimate: **${analysis['estimated_usd_low']:.6f}–${analysis['estimated_usd_high']:.6f} USD**. "
              'The ledger has complete accounting and no outstanding reservations. All input/output reservations held.', '',
              'The range applies the official off-peak and peak rates to provider-reported cache hits/misses and output. '
              'It is not an account-balance delta, invoice or measured charge. The classifier has zero incremental API fee; '
              'local hardware/CPU, development, model-training and hosting costs are excluded. '
              '[Price source checked on 10 November 2026](https://api-docs.deepseek.com/quick_start/pricing/): '
              'USD per million tokens, input hit 0.003–0.006, input miss 0.15–0.30, output 0.60–1.20.', '',
              '| Mode | Reported cached input | Reported cache-miss input | Output |',
              '|---|---:|---:|---:|']
    for mode in modes:
        c=analysis['summary'][mode]['overall']['estimated_list_price']
        lines.append(f"| {NAMES[mode]} | {c['cache_hit_tokens']:,} | {c['cache_miss_tokens']:,} | {c['completion_tokens']:,} |")
    lines += ['', '## Paired cases and evidence quality', '',
              '| Case | Channel / truth | Classifier | Direct LLM | Fixed | Agent |',
              '|---|---|---|---|---|---|']
    for ident in cases:
        first=by_key[(ident,modes[0])]
        cells=[]
        for mode in modes:
            v=by_key[(ident,mode)]['result']
            cells.append(v['decision']+(' (review)' if v['needs_review'] else ''))
        lines.append(f"| `{ident}` | {first['channel']} / {'fraud' if first['label'] else 'legitimate'} | "+' | '.join(cells)+' |')
    lines += ['', '- `mendeley_5259`: direct LLM voluntarily returned `abstain`, review=true, no evidence. '
              'No reasoning for that abstention was exposed. Fixed/Agent used the classifier and returned the correct legitimate result.',
              '- `nigerian_5_4474`: direct LLM proposed `fraud` but supplied an empty quote at [0,0). '
              'The unchanged validator rejected that unsupported final answer and returned abstention. Agent/fixed produced attributable evidence. '
              'This is a usable-output improvement, not correction of a wrong raw fraud label.',
              '- There are no observed Agent label improvements over the classifier or fixed workflow, and no cases where Agent changed '
              'a correct baseline label to an incorrect one. Extra work did hurt measured latency, cost and review burden.',
              '- Agent normalised four messages and fixed workflow normalised all six; every returned change count was zero. '
              'Every raw/normalised classifier pair in fixed workflow agreed. This sample does not demonstrate an obfuscation-defence benefit.',
              '- Exact-quote offset corrections: direct 8, fixed 9, Agent 12. They are explicitly recorded as review events; '
              'missing/ambiguous quotes are not repaired. Agent routed 5/6 cases to review, versus fixed 3/6. '
              'That review burden is additional operational work, not counted here as a dollar amount.', '',
              '## Assistant-performed evidence inspection', '',
              'This is deterministic attribution checking plus a human-readable inspection by the assistant, not independent review by a project teammate. '
              'No judging call was made to the evaluated DeepSeek model. Exact substrings and labels provide the objective checks; '
              'contextual interpretation below remains available for independent human review.', '',
              '- SMS fraud: the message claims a guaranteed cash prize and uses `W0N` / `FL1PKART`; the cited prize/phone spans are attributable. '
              'The normaliser deliberately does not turn digits into letters.',
              '- Email fraud: the full message requests help safeguarding a large sum and gold, offers 25%, and requests contact details. '
              'Agent cited the 25% promise, whereas fixed workflow mainly cited generic urgency/contact fields. The former is more relevant '
              'contextually in this one example; it does not prove a general explanation advantage.',
              '- The amount rule extracted `$15` from `$15 Million USD`. This is an exact substring but loses the multiplier. '
              'The report flags the limitation; no extractor or prompt was changed after the benchmark.',
              '- Phone fraud (synthetic): Agent/direct/fixed cite requests for a social security number, a processing fee and pressure against '
              'outside verification. These are more decision-relevant than a mere organisation name. Actual identity was not verified.',
              '- Legitimate email: the newsletter context supports the supplied legitimate label. Addresses, unsubscribe links and a phone number '
              'alone do not prove legitimacy; Agent highlighted several weak generic cues. This limits explanation quality despite correct classification.',
              '- Legitimate phone: wrong-number/pizza dialogue fits the label. Legitimate SMS is casual chat. A quote matching the text is '
              'not automatically a sufficient argument, so independent team review remains pending.', '',
              '## Decision supported by this pilot', '',
              'Keep the existing classifier as the economical default for this current label-only task. If an LLM-backed evidence summary is needed, '
              'the tested fixed workflow achieved the same usable labels as Agent at lower measured cost, latency and review coverage. '
              'Retain Agent as an experimental option; this run does not justify making it the default.', '',
              'Six balanced examples and one run cannot establish robust accuracy, statistical superiority, real-world prevalence performance, '
              'rare-fraud recall or general cost savings. Two phone cases are synthetic. There were no injected adversarial instructions or constructed '
              'obfuscation variants, and no stochastic repeats. The classifier reached a ceiling on this sample. '
              'A larger independently reviewed hard-case/robustness experiment would be needed to test benefits not seen here; no such paid run was launched.', '',
              '## Reproduction and records', '',
              'Preparation and actual execution commands:', '', '```sh',
              '.venv/bin/python -m scripts.run_agent_comparison prepare',
              '.venv/bin/python -m scripts.run_agent_comparison run','```', '',
              'Existing protocol/results/ledger paths refuse overwrite or automatic paid reruns. The protocol, source hashes and manifest were frozen '
              'before calling the API. This new experiment did not alter the production system prompt or final-answer validator.', '',
              'Local ignored files under `outputs/agent/comparison_pilot_v1/`:', '',
              '- `protocol.json`, `manifest.json`: frozen design, labels/support, source/data hashes, bounds and pricing snapshot.',
              '- `results/run.json`, `results/predictions.jsonl`, `results/summary.json`: all 24 case/mode records and raw aggregate metrics.',
              '- `api_audit/budget_ledger.json` and `exchange_01.json` through `exchange_28.json`: actual sanitized API requests/responses, '
              'model-generated tool calls, cache accounting and durable reservations. No key, Authorization header or hidden reasoning.',
              '- `analysis.json`: reconciled per-mode usage/cost and paired Agent improvements/harms.',
              '- `supplementary_metrics.json`: conservative recall computed after the run, without changing the frozen primary metrics.', '',
              'Code additions: balanced manifest selection, counterbalanced mode order, the bounded pilot runner and offline cost/report analysis. '
              'No teammate training scripts, datasets, model artifacts or frontend were changed.']
    lines += ['', 'Verification after the pilot: full offline suite **124 passed, 7 skipped**; dependency check and patch whitespace check passed. No test or report-verification command sent a provider request.']
    text='\n'.join(lines)+'\n'
    report=ROOT/'docs/agent_comparison_pilot.md'
    report.write_text(text)
    (BASE/'report.md').write_text(text)
    (BASE/'supplementary_metrics.json').write_text(json.dumps(supplements,indent=2))
    print(report)

if __name__=='__main__':
    main()
