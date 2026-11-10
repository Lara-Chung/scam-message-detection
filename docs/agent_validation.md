# Agent validation record

**Current adaptive investigation update:** the default configuration now uses
`adaptive_agent` with a free classifier gate and bounded document tools. Final
suite: **182 passed / 7 raw-data skips**. Three selected VAL sources exercised
the free and escalated paths. Original budget/evidence failures were preserved;
repair reexecuted a saved real action with new live finalization, plus a fresh
remaining-budget request that safely abstained. All attempts combined: **6 real
requests / 9,744 tokens**, below the original 12,000 cap; accounting complete.
This is engineering verification, not a fresh held-out effectiveness result.
See [adaptive_agent_handoff.md](adaptive_agent_handoff.md) for exact scopes,
failures, correction and current setup.

**Normalised routing update (10 November 2026):** version-1.2 router results now
record common normalisation and mandatory VAL-calibrated boundary cross-checks.
The full suite passed **154 tests / 7 skips**. A fresh 12-input challenge excluded
114 prior TEST sources and produced 10/12 correct for fixed, rule and Agent modes,
with zero technical failures/abstentions. Actual Agent usage was 24 requests /
55,177 tokens; accounting is complete and no retries were made. Both misses are
paired prefixes of one long email. See the [full report](normalised_boundary_report.md)
for clipping limits, per-case traces, review behavior and cache-aware price estimates.
The sections below preserve historical acceptance snapshots.

**Current acceptance update:** the budget repair and a subsequent real three-channel
run recorded 6 calls / 9,164 tokens. All channels reached final responses;
email's unique literal quote required a visible coordinate correction and review.
See [agent_budget_fix.md](agent_budget_fix.md) for original versus offline
revalidated evidence, 121 passing tests, and remaining evaluation limits.


**Latest update:** local DeepSeek setup and the authorized three-message live run
are now complete as an execution record: 5 real API calls, 6,228 reported tokens,
one successful final answer and two local budget stops. See
[the live validation report](agent_live_validation.md). The sections below retain
the **initial implementation snapshot**, before this subsequent live run. The
latest full offline suite is 109 passed / 7 skipped.

Date: 10 November 2026, Asia/Singapore. Base commit:
`90cd4baa7d2bdc08b061b250de1e7f8909e46749`; local branch:
`feature/agent-backend`. Nothing was pushed or published.

## Executed checks

The active interpreter is
`/Users/bohanzhang/Projects/scam-message-detection/.venv/bin/python`, Python
3.12.15. This isolated environment uses `requirements-agent.txt`; the team's
original requirements/training environment was preserved.

| Check | Actual outcome |
|---|---|
| Full original + Agent test suite | **102 passed, 7 skipped** |
| Existing tests before Agent tests | 47 passed, 7 skipped |
| Dependency check | No broken requirements |
| Source compilation | Passed |
| Patch whitespace | Passed |
| Supplied export extraction/preflight | 8 archive members verified; existing identical files preserved |
| Data, label and split audit | Three canonical channels; all three seeds aligned with saved splits; no within-dataset split leakage |
| Legacy models | Six real LR/word-SVM artifacts loaded and smoke-predicted; unverified for held-out evaluation |
| New baseline training | 3,431 SMS / 5,091 email / 760 phone canonical TRAIN cases, metadata saved |
| Real Agent-model inference | All three channels; explicit class orientation; no inference fitting |
| Train-only retrieval | All three channels; same-channel TRAIN membership and disjoint held-out IDs verified |
| Frozen validation A | Nine cases, all answered; used only for development/smoke, not a benchmark claim |
| Frozen test pilot A | 90 cases, detailed below |
| B/C/D missing-provider path | Three cases per mode return `provider_not_configured`, zero provider calls |
| Genuine provider calls | **Not executed: all three required LLM variables absent** |
| Real Agent explanation quality / comparative gain | **Pending** |

The seven original-test skips require raw Mendeley, Nazario/Nigerian, BothBosu,
UCI or CEAS/SpamAssassin downloads. These raw files are not required by the
backend. The delivered processed-export tests and real saved-model tests did
run; there are no new local-artifact skips in the reported run.

Tests cover NA exclusions, empty canonical templates, saved split counts,
TRAIN retrieval membership, first-column positive class (`fraud, legitimate`),
LR probabilities and reversed binary SVM margins, artifact hash/mapping guards,
real inference/cache reuse without fitting, exact normaliser counts and unchanged
prices/numbers, span provenance, invalid and repeated calls, tool/output/turn/token
bounds, provider and tool deadlines, immutable timeout snapshots, prompt-injection
placement as data, absent artifacts/provider, malformed provider/final JSON,
provider usage/secret handling, global batch accounting, pending-case denominators,
explicit live-test gating, concurrency rejection, CLI JSONL separation, safe archive
extraction and strict result serialization. Scripted providers and HTTP mocks
validate mechanics; they do not establish live model behaviour.

Exact commands already executed from the repository root include:

```sh
.venv/bin/python -m scripts.agent_audit
.venv/bin/python -m scripts.prepare_agent_inputs
.venv/bin/python -m src.agent.baseline
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
.venv/bin/python -m compileall -q src/agent scripts tests/test_agent.py
git diff --check
.venv/bin/python -m src.agent.cli --text 'Please verify your account immediately.' --channel sms --mode classifier_only
.venv/bin/python -m src.agent.evaluate manifest --output outputs/agent/validation_manifest_v1.json --split val --per-channel 3
.venv/bin/python -m src.agent.evaluate run --manifest outputs/agent/validation_manifest_v1.json --output outputs/agent/validation_classifier_v1 --modes classifier_only
.venv/bin/python -m src.agent.evaluate run --manifest outputs/agent/validation_manifest_v1.json --output outputs/agent/provider_pending_v1 --modes direct_llm agent fixed_workflow --max-cases 3
.venv/bin/python -m src.agent.evaluate manifest --output outputs/agent/test_pilot_manifest_v1.json --split test --per-channel 30
.venv/bin/python -m src.agent.evaluate run --manifest outputs/agent/test_pilot_manifest_v1.json --output outputs/agent/classifier_test_pilot_v1 --modes classifier_only
```

Training and run directories refuse overwriting. Use a new output path when
repeating a command. The extraction command verifies identical existing files.
The unversioned early validation outputs are development records; `_v1` manifests
freeze system prompt, final schema, tool definitions and common clipping policy.

## Canonical classifier-only test pilot (A)

Selection was fixed before test predictions: canonical seed0 TEST, eligible fraud
labels only; SHA-256 ordering of `0:<case_id>`, first 30 per channel, no class
rebalancing or hard-case filtering. Train-only LR settings were fixed a priori.
All models' train IDs/export hashes were checked before inference. This is an
**in-domain small pilot**, not zero-shot transfer and not real-bank deployment.

| Channel | Legitimate / fraud support | Macro-F1 | Fraud precision | Fraud recall | FPR | Confusion matrix, true rows/predicted cols [0,1] |
|---|---|---:|---:|---:|---:|---|
| SMS | 24 / 6 | 1.00 | 1.00 | 1.00 | 0.00 | `[[24,0],[0,6]]` |
| Email | 11 / 19 | 1.00 | 1.00 | 1.00 | 0.00 | `[[11,0],[0,19]]` |
| Phone, **synthetic** | 9 / 21 | 1.00 | 1.00 | 1.00 | 0.00 | `[[9,0],[0,21]]` |
| Overall pooled | 44 / 46 | 1.00 | 1.00 | 1.00 | 0.00 | `[[44,0],[0,46]]` |

All 90 cases were answered; zero abstentions, review flags, technical failures or
pending cases. Conservative full-denominator success is 90/90. Actual correctly
oriented continuous LR scores were available; average precision (step-integral
PR summary) was approximately 1.00 per channel and overall. Tiny floating point
rounding in raw JSON is retained. No Agent PR score or LLM confidence was invented.

The saved initial run's warm-service mean measured latency was approximately
0.51 ms SMS / 0.87 ms email / 0.75 ms phone / 0.71 ms overall, before final result
schema-validation overhead in that initial instrumentation. The current service
also includes final validation in `elapsed_ms`. Evaluation preloads models for
provenance checks; these are not cold-start or frontend-network timings.
There were 90 actual classifier tool executions, zero LLM calls and zero LLM
tokens. Cost is null because pricing is not configured, not a claim that an LLM
benchmark was free or completed.

SMS recall is based on only **six positives**. Perfect results on this small,
frozen sample do not establish perfect generalisation. BothBosu dialogues are
synthetic and formulaic; paraphrase leakage beyond the supplied template rules
is not measured. No performance gain from an Agent is claimed.

Per-case predictions and bounded tool results were flushed before aggregation:

- `outputs/agent/classifier_test_pilot_v1/run.json`: exact manifest hash,
  provider/config, bounds and selected IDs.
- `outputs/agent/classifier_test_pilot_v1/predictions.jsonl`: 90 actual results.
- `outputs/agent/classifier_test_pilot_v1/summary.json`: per-channel and pooled metrics.
- `outputs/agent/test_pilot_manifest_v1.json`: frozen membership, label support,
  source and message hashes, protocol/prompt hash.

These files contain local data/traces and are ignored, not committed.

## Failure handling and evidence inspection

The nine-case validation subset has eight legitimate cases and one synthetic
phone fraud. Correct labels on this tiny set are a smoke check. SMS/email each
have no positives: binary macro-F1 is defined over both labels and is 0.5 for
these all-legitimate channel subsets even with every case correct; this is why
class support is always reported.

Three real validation inputs were also passed through normalisation, both
classifier views, indicator extraction and TRAIN retrieval, with actual outputs
saved as `outputs/agent/component_validation.json`. These are labelled
`offline_components_no_llm`; they are not an Agent conversation or LLM trace.
A manual reading of those messages against observed indicator quotes found:

| Validation case | Observed evidence check | Interpretation limit |
|---|---|---|
| SMS `mendeley_2681` | Travel/subletting conversation; rules returned no indicators; raw/normalised classifier decisions both legitimate | No indicators alone cannot establish safety |
| Email `nigerian_5_1019` | Announced an association website; the URL, email addresses and two phone spans are exact substrings | These contacts do not establish malice or verified ownership |
| Phone `bothbosu_train_1140` (canonical **val**, despite ID prefix) | Wrong-number dialogue; `555-1234` matches the original at [23,31) | Official source-ID prefix is not canonical Agent split membership |

This is a human-readable structural review of component evidence, not a live
LLM explanation assessment. The three pairs of raw/normalised classifier outputs
were unchanged, so no measured normalisation gain or harm is inferred. Fixture
tests demonstrate that a fabricated quote forces abstention, provider timeouts
surface as technical failures, and missing artifacts produce no score.
No false positive/negative occurred in this 90-case pilot; therefore there is
no observed error case to present as failure analysis. Larger held-out evaluation
and live comparisons must look for such failures rather than assuming improvement.

Live B/C/D comparative error analysis, cases where extra tools hurt, stochastic
variation and contextual support of Agent decisions are **pending**. Do not use
the same LLM as the sole judge. For the next real validation run, manually inspect
short quotes against the named view, whether they actually support the decision,
and whether the Agent overweights a URL, a mentioned organisation or a similar
TRAIN case. Record any unsupported interpretation even if offsets are valid.

## Comparison runner and denominators

`src.agent.evaluate` provides A classifier, B direct LLM, C bounded Agent and
D fixed workflow. Selected IDs/channel information, label definition, final
schema, provider/model and original clipping policy are shared. No current-case
ID, dataset name, subtype or truth label enters the LLM input. Training references
may include their own IDs and labels, as allowed by the retrieval contract.

The runner refuses unverified model provenance, altered manifest labels/texts,
duplicate case IDs, changed prompt/input policies and live TEST calls without
`--live-test`. Each case/mode is recorded before aggregation. Global budget or
live-message caps leave explicit `pending_budget` records; cases are not silently
dropped. All planned cases remain in coverage and conservative-success denominators.
Technical-failure rate counts result-level errors; recovered tool errors remain
visible in traces and route the result to review. Answered-case metrics include
binary answers with review flags, with review coverage reported separately.

The summary reports label support, binary macro-F1, fraud precision/recall, FPR,
confusion matrices, coverage, abstentions, failures, pending cases, review routing,
full-denominator correct/total, latency, tool/LLM calls, usage and accounting
completeness. Average precision is computed only for actual oriented continuous
classifier scores, not LLM decisions. Pooling uncalibrated channel-specific LR
scores is descriptive; channel-specific metrics are the primary comparison.

`provider_pending_v1` records nine actual missing-configuration errors (three
modes × the same three validation cases). All have zero LLM calls and no
answered cases. Its null accuracy metrics and zero coverage are **not** evidence
about LLM model quality. Full B/C/D pilot comparisons remain unexecuted.

## Smallest remaining action and next command

Privately configure `LLM_API_KEY`, `LLM_BASE_URL`, and the actual tool-capable
`LLM_MODEL` for the DeepSeek Chat Completions endpoint. No key is requested in
chat. Then run this one bounded validation smoke (new output directory):

```sh
cd /Users/bohanzhang/Projects/scam-message-detection
.venv/bin/python -m src.agent.evaluate run \
  --manifest outputs/agent/validation_manifest_v1.json \
  --output outputs/agent/live_agent_smoke_v1 \
  --modes agent --max-cases 3 --max-live-messages 3 --token-budget 12000
```

Maximum: **three validation messages**, up to **12 LLM calls** and **18 attempted
tool calls**, with a **12,000-token global budget** using conservative pre-call
reservations and reported usage. Per-message deadline is 60 seconds, output cap
700 tokens/call, and turn/tool caps are 4/6. It may stop before three messages if
budget or usage accounting requires it. Actual spend cannot be estimated in
currency until pricing is configured. This command was **not run live** here.

For an immediate provider-free frontend check, the next command is:

```sh
.venv/bin/python -m scripts.frontend_agent_example
```

Larger paid comparisons need an explicitly planned budget and explicit raised
`--max-live-messages`/`--token-budget` (plus `--live-test` for TEST). They are not
run automatically. Repeated stochastic validation can reuse the same frozen
validation manifest with distinct output directories and the same provider/model;
record variation rather than selecting a favourable run. New seeds, cross-channel
transfer and context-guidance ablations were not implemented.
