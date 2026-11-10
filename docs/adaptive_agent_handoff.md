# Adaptive document investigation: implementation and validation

Date: 10 November 2026, Asia/Singapore. This is engineering validation on selected canonical VAL messages, not an independent TEST benchmark.

## Result

The backend now supports a free classifier path and an escalated single-agent investigation. The Agent chooses which source section to inspect, whether to cross-check, retrieve same-channel TRAIN cases, request explicit user context or finish. A deterministic fixed workflow exposes the same source and tool capability for comparison. Labels remain grounded in actual saved classifiers; evidence and next steps are rendered by the backend. **No accuracy improvement is claimed.** Earlier routing experiments found no gain, and this extension has not yet been evaluated on a fresh independent TEST sample.

## Behavior

- Default `configs/agent.yaml` mode: `adaptive_agent`. Historical modes remain explicit and reproducible.
- `document_classifier`: deterministic first-substantive-window selection, normalisation and one VAL-selected fixed model, no API.
- `document_fixed`: same gate and source index, predetermined section-reading/cross-check schedule, one compact LLM final decision only on escalation.
- `adaptive_agent`: same gate; genuine LLM function calls select the next section/action based on observed tool results. Finalization reserves are held before another exploratory turn.
- Gate reasons: long document, no substantive window, model-specific near-boundary distance or observed cross-check disagreement. Ordinary complete non-boundary messages make no API request.
- Default source limit 2,000,000 characters; oversized sources return source_too_large without silent truncation. Default window 2,000 Unicode code points. The full source stays request-local; JSON/LLM context contains bounded inspected views and an overview.
- Window ranking uses address/markup density and observed request phrases. This is a heuristic, not a guaranteed MIME/HTML parser or evidence of fraud. No URL fetching or external sender/domain verification is available.
- Near-boundary initial assessments automatically execute a distinct secondary before escalation. Later cross-checks are explicit allowlisted actions. Native margins/probabilities remain uncalibrated.
- Final JSON cites only backend-generated evidence IDs; the backend copies exact stored spans. Binary labels must match an executed assessment. Pending clarification forces abstention; no user answer is inferred.
- Negative partial-document findings become abstention and review. Conflicts and incomplete coverage require review. All-source coverage is a window-read measure, not a guarantee that the LLM semantically understood every character.
- Explicit boolean caller context is accepted only for transaction_initiated, sender_known, independent_verification and missing_body. Missing answers remain pending; no interactive UI or external verification service was added.
- Public CLI paid-message caps count actual paid analyses. After the cap, free fast-path messages can still complete; escalations stop before any provider call.

## VAL policy

The new policy evaluates the existing 12 canonical-TRAIN-only candidates on the deterministic initial normalised window of every canonical VAL message. Choose the primary by VAL macro-F1 and the distinct secondary by rescued primary VAL errors, then VAL F1/stable candidate order. The near-boundary distance threshold is the predeclared 10th percentile of primary VAL distances. All 2,330 VAL cases participate; no new classifier/vectorizer fitting or teammate artifact replacement occurred. Catalog/calibration metadata are ignored under models/triage_seed0/. Selection-set scores are not independent effectiveness.

## Actual live attempts and repair

Freeze three label-blind VAL engineering sources: a 2,362-character email with an unread request-bearing window, a short SMS and a short synthetic phone dialogue. The selected messages all have legitimate labels after selection; this is not a balanced fraud-recall test. Run three modes on each source (nine records). SMS and phone complete through free fast paths in all modes.

The initial email Agent actually requested the second section, observed the result and then requested a cross-check. Its final turn was blocked by the per-message token budget. The initial fixed workflow produced literal/cross-section evidence that failed strict validation. Both failures remain in original predictions.jsonl and are included in total API cost.

Repair changes: hold a closing reservation before permitting further exploration; send a deterministic compact summary for finalization; use backend evidence IDs instead of asking the LLM to copy newlines/offsets. The compact summary retains all executed assessment decisions and exact evidence options. It does not reproduce the full text context, so conservative abstention is permitted.

Revalidation reexecutes one saved authentic Agent section-selection action and verifies that its classifier outputs equal recorded original outputs. The repaired budget guard ends exploration before the original second action. A new real final generation completes that derived Agent result. The fixed schedule is reexecuted with unchanged actual classifier results and a new live final generation. These are saved-action revalidations, not fresh end-to-end effectiveness runs. A subsequent fresh end-to-end request tests the remaining-budget path: it finalizes early and abstains because some content is unread.

| Verification record | Actual outcome | Scope |
|---|---|---|
| adaptive_agent | legitimate; ok; coverage complete=True | Saved real action/schedule + new live finalization |
| document_fixed | abstain; needs_review; coverage complete=True | Saved real action/schedule + new live finalization |
| adaptive_agent | abstain; needs_review; coverage complete=False | Fresh end-to-end remaining-budget check |

An earlier repair preflight was stopped before any provider call (repair_v2 records zero requests). It is retained. No failed prediction, tool trace or original response was overwritten. No further paid call was made for report generation.

**All attempts combined: 6 actual requests, 9,744 reported tokens**, below the original shared 12,000-token ceiling. All ledger usage reconciles with filtered exchanges; zero unresolved reservations. Estimated total API list price, including failed attempts: **$0.00127411–$0.00254822 USD**. Rates use reported cache hit/miss counts and [official off-peak/peak prices](https://api-docs.deepseek.com/quick_start/pricing/), checked 10 November 2026. This excludes CPU/storage and is not a billing receipt. Replayed original responses are not charged a second time.

## Verification and remaining work

The final local suite passed **182 tests, seven skips**. The seven skips are existing raw-data-dependent tests; the local saved-model integration tests ran. Compilation, dependency consistency, result/quote validation, immutable execution snapshots and global cost reconciliation passed. Tests cover adaptive later-window selection, exact source offsets, source caps, normalization/amount preservation, fixed workflow, missing provider, invalid/repeated/unknown tools, ungrounded decisions/evidence IDs, clarification provenance, timeout snapshots, final-answer reserves and free paths after a paid cap.

Remaining: a frozen fresh independent TEST comparison with the same source/tool access, classifier policy and budget across fixed model, fixed workflow and Agent. Report recall, false positives, full-denominator success, review/abstention, unreviewed misses, tool work, latency and cost. Do not use the already observed long-mail failure or these VAL messages as new held-out evidence. The comparison runner requires explicit --allow-live and additionally --allow-live-test for paid TEST execution. No new paid TEST benchmark or performance superiority claim is included in this delivery.

## Local evidence and commands

Evidence: outputs/agent/document_triage_validation_v1/{manifest.json,run_config.yaml,run/,repair_v2/,repair_v3/,analysis.json}. Each execution retains its own source snapshot and protocol hashes. Later CLI paid-cap handling was added after the live snapshot; its no-call behavior is covered offline. Keys remain in macOS Keychain and local inputs/models/results stay ignored.

```sh
.venv/bin/python -m scripts.prepare_document_triage
.venv/bin/python -m scripts.prepare_document_smoke
.venv/bin/python -m scripts.compare_document_triage --manifest outputs/agent/document_triage_validation_v1/manifest.json --output outputs/agent/document_triage_validation_v1/run --config outputs/agent/document_triage_validation_v1/run_config.yaml --allow-live --token-budget 12000 --max-paid-messages 3
.venv/bin/python -m scripts.revalidate_document_triage
.venv/bin/python -m scripts.report_document_triage
```

Paid/preparation commands refuse existing output paths. Commands describe the recorded process, not permission to repeat spending. For current offline use, run the frontend example or CLI with --mode document_classifier.
