"""Audit all saved document-triage attempts and report honest engineering evidence."""
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from src.agent.comparison import cost_range
from src.agent.contract import validate_result
from src.agent.data import file_hash
from scripts.run_agent_comparison import PRICING

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/agent/document_triage_validation_v1'


def main():
    first=BASE/'run';repair=BASE/'repair_v3'
    original=[json.loads(s) for s in (first/'predictions.jsonl').read_text().splitlines()]
    corrected=[json.loads(s) for s in (repair/'predictions.jsonl').read_text().splitlines()]
    old_schema=json.loads((first/'frozen_source/src/agent/schema.json').read_text())
    for r in original:Draft202012Validator(old_schema).validate(r['result'])
    for r in corrected:validate_result(r['result'])
    events=[];tokens=calls=0;low=high=0.
    for directory in [first,*sorted(BASE.glob('repair_*'))]:
        ledger=json.loads((directory/'api_audit/budget_ledger.json').read_text())
        assert ledger['accounting_complete'] and ledger['unresolved_reserved_tokens']==0
        exchanges=sorted((directory/'api_audit').glob('exchange_*.json'))
        assert len(exchanges)==ledger['actual_calls']
        reported=0
        for path in exchanges:
            exchange=json.loads(path.read_text());usage=exchange['response']['usage'];reported+=usage['total_tokens']
            c=cost_range(usage,PRICING['per_million_tokens']);low+=c['usd_low'];high+=c['usd_high']
            events.append({'attempt':directory.name,'exchange':path.name,'usage':usage,'cost_estimate':c})
        assert reported==ledger['reported_tokens'];tokens+=reported;calls+=ledger['actual_calls']
        protocol=json.loads((directory/'protocol.json').read_text())
        for path,digest in protocol['source_code_sha256'].items():
            assert file_hash(directory/'frozen_source'/path)==digest
    assert tokens<=12000 and calls==6
    overview={'execution_kind':'engineering_VAL_not_held_out_benchmark','actual_requests_all_attempts':calls,
        'tokens_all_attempts':tokens,'estimated_usd_low':low,'estimated_usd_high':high,
        'original_failures':[{'channel':r['channel'],'mode':r['mode'],'stop_reason':r['result']['stop_reason']} for r in original if r['result']['status']=='error'],
        'repaired_outcomes':[{'mode':r['mode'],'execution_kind':r['execution_kind'],'status':r['result']['status'],
            'decision':r['result']['decision'],'coverage_complete':r['result']['triage']['coverage_complete'],
            'model_tool_actions':sum(t['source']=='model_tool_call' for t in r['result']['tool_trace'])} for r in corrected],
        'cost_events':events,'accounting_complete':True}
    (BASE/'analysis.json').write_text(json.dumps(overview,indent=2))
    lines=['# Adaptive document investigation: implementation and validation','',
        'Date: 10 November 2026, Asia/Singapore. Engineering validation on three selected canonical VAL messages.','',
        '## Design','',
        'The backend now supports a free classifier path and an escalated single-agent investigation. The Agent chooses which source section to inspect, '
        'whether to cross-check, retrieve same-channel TRAIN cases, request explicit user context or finish. '
        'A deterministic fixed workflow exposes the same source and tool capability for comparison. '
        'Labels remain grounded in actual saved classifiers; evidence and next steps are rendered by the backend. '
        'The validation below covers workflow execution, evidence handling and budget control. Earlier classifier-routing experiments are recorded in the linked comparison reports.','',
        '## Behavior','',
        '- Default `configs/agent.yaml` mode: `adaptive_agent`. Earlier modes remain available for comparison and reproduction.',
        '- `document_classifier`: deterministic first-substantive-window selection, normalisation and one VAL-selected fixed model, no API.',
        '- `document_fixed`: same gate and source index, predetermined section-reading/cross-check schedule, one compact LLM final decision only on escalation.',
        '- `adaptive_agent`: same gate; genuine LLM function calls select the next section/action based on observed tool results. Finalization reserves are held before another exploratory turn.',
        '- Gate reasons: long document, no substantive window, model-specific near-boundary distance or observed cross-check disagreement. Ordinary complete non-boundary messages make no API request.',
        '- Default source limit: 2,000,000 characters, with 2,000-character Unicode windows. Oversized inputs return source_too_large. The full source stays request-local; returned JSON and LLM context contain inspected views and an overview.',
        '- Window ranking uses address/markup density and observed request phrases to suggest relevant sections. Source windows retain their original text and markup. Sender/domain verification requires a separate external service.',
        '- Near-boundary initial assessments automatically execute a distinct secondary before escalation. Later cross-checks are explicit allowlisted actions. Native margins/probabilities remain uncalibrated.',
        '- Final JSON cites backend-generated evidence IDs. The backend copies exact stored spans and checks binary labels against executed assessments. Clarification returns a pending question until the caller supplies an answer.',
        '- Negative partial-document findings become abstention and review. Conflicts and incomplete coverage also request review. Coverage records which source windows were inspected.',
        '- Caller context accepts boolean answers for transaction_initiated, sender_known, independent_verification and missing_body. The frontend collects those answers and passes them to the Python API.',
        '- CLI paid-message caps count actual paid analyses. After the cap, the local path continues processing ordinary messages while paid escalation stops.','',
        '## VAL policy','',
        'The new policy evaluates the existing 12 canonical-TRAIN-only candidates on the deterministic initial normalised window of every canonical VAL message. '
        'Choose the primary by VAL macro-F1 and the distinct secondary by rescued primary VAL errors, then VAL F1/stable candidate order. '
        'The near-boundary distance threshold is the predeclared 10th percentile of primary VAL distances. '
        'All 2,330 VAL cases participate. The existing TRAIN-fitted classifiers and vectorizers are reused. '
        'Catalog and calibration metadata are stored locally under models/triage_seed0/, which Git ignores. These scores describe VAL model selection.','',
        '## Actual live attempts and repair','',
        'Freeze three label-blind VAL engineering sources: a 2,362-character email with an unread request-bearing window, a short SMS and a short synthetic phone dialogue. '
        'After selection, all three source labels are legitimate. '
        'Run three modes on each source (nine records). SMS and phone complete through free fast paths in all modes.','',
        'The initial email Agent actually requested the second section, observed the result and then requested a cross-check. '
        'Its final turn was blocked by the per-message token budget. The initial fixed workflow produced literal/cross-section evidence that failed strict validation. '
        'Both failures remain in original predictions.jsonl and are included in total API cost.','',
        'Repair changes: hold a closing reservation before permitting further exploration; send a deterministic compact summary for finalization; '
        'use backend evidence IDs instead of asking the LLM to copy newlines/offsets. The compact summary retains all executed assessment decisions and exact evidence options. '
        'The final result may request review when that context is insufficient.','',
        'Revalidation reexecutes one saved authentic Agent section-selection action and verifies that its classifier outputs equal recorded original outputs. '
        'The repaired budget guard ends exploration before the original second action. A new real final generation completes that derived Agent result. '
        'The fixed schedule is reexecuted with unchanged actual classifier results and a new live final generation. '
        'The records identify these runs as saved-action revalidations. '
        'A subsequent fresh end-to-end request tests the remaining-budget path: it finalizes early and abstains because some content is unread.','',
        '| Verification record | Actual outcome | Scope |','|---|---|---|']
    for r in corrected:
        v=r['result'];scope='Fresh end-to-end remaining-budget check' if r['execution_kind'].startswith('fresh') else 'Saved real action/schedule + new live finalization'
        lines.append(f"| {r['mode']} | {v['decision']}; {v['status']}; coverage complete={v['triage']['coverage_complete']} | {scope} |")
    lines+=['','An earlier repair preflight was stopped before any provider call (repair_v2 records zero requests). It is retained. '
        'Original predictions, tool traces and responses are preserved. Report generation reads the saved exchanges offline.','',
        f"**All attempts combined: {calls} actual requests, {tokens:,} reported tokens**, below the original shared 12,000-token ceiling. "
        'All ledger usage reconciles with filtered exchanges; zero unresolved reservations. '
        f"Estimated total API list price, including failed attempts: **${low:.8f}–${high:.8f} USD**. "
        'Rates use reported cache hit/miss counts and [official off-peak/peak prices](https://api-docs.deepseek.com/quick_start/pricing/), checked 10 November 2026. '
        'The estimate excludes CPU/storage and counts each paid exchange once; replayed responses add zero API usage.','',
        '## Tests and next evaluation','',
        'The final local suite passed **182 tests, seven skips**. The seven skips are existing raw-data-dependent tests; the local saved-model integration tests ran. '
        'Compilation, dependency consistency, result/quote validation, immutable execution snapshots and global cost reconciliation passed. '
        'Tests cover adaptive later-window selection, exact source offsets, source caps, normalization/amount preservation, fixed workflow, missing provider, '
        'invalid/repeated/unknown tools, ungrounded decisions/evidence IDs, clarification provenance, timeout snapshots, final-answer reserves and free paths after a paid cap.','',
        'Next evaluation: compare the fixed model, fixed workflow and adaptive Agent on fresh independent TEST sources with the same source/tool access, classifier policy and budget. '
        'Report recall, false positives, full-denominator success, review/abstention, unreviewed misses, tool work, latency and cost. '
        'Keep the development VAL messages and previously examined long-mail failure separate from those TEST sources. '
        'The comparison runner requires explicit --allow-live and additionally --allow-live-test for paid TEST execution. '
        'The comparison runner supports this next TEST study.','',
        '## Local evidence and commands','',
        'Evidence: outputs/agent/document_triage_validation_v1/{manifest.json,run_config.yaml,run/,repair_v2/,repair_v3/,analysis.json}. '
        'Each execution retains its own source snapshot and protocol hashes. Later CLI paid-cap handling was added after the live snapshot; '
        'its no-call behavior is covered offline. Keys remain in macOS Keychain and local inputs/models/results stay ignored.','',
        '```sh','.venv/bin/python -m scripts.prepare_document_triage',
        '.venv/bin/python -m scripts.prepare_document_smoke',
        '.venv/bin/python -m scripts.compare_document_triage --manifest outputs/agent/document_triage_validation_v1/manifest.json --output outputs/agent/document_triage_validation_v1/run --config outputs/agent/document_triage_validation_v1/run_config.yaml --allow-live --token-budget 12000 --max-paid-messages 3',
        '.venv/bin/python -m scripts.revalidate_document_triage',
        '.venv/bin/python -m scripts.report_document_triage','```','',
        'Preparation and paid-run commands preserve existing output paths. New runs use separate paths and explicit budgets. '
        'For current offline use, run the frontend example or CLI with --mode document_classifier.']
    text='\n'.join(lines)+'\n';(ROOT/'docs/adaptive_agent_handoff.md').write_text(text);(BASE/'report.md').write_text(text)
    print(json.dumps({k:overview[k] for k in ['actual_requests_all_attempts','tokens_all_attempts','estimated_usd_low','estimated_usd_high']},indent=2))

if __name__=='__main__':main()
