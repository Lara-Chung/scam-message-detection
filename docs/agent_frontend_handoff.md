# Agent backend frontend handoff

**Current default:** `configs/agent.yaml` selects `adaptive_agent` with result
schema 1.3. Start with the [adaptive investigation handoff](adaptive_agent_handoff.md)
and the integration section below. Earlier sections retain the historical modes
and their 1.0/1.1/1.2 contracts for reproducibility.

## Quick start

Run from the actual repository root. Agent setup is isolated from the team's
training environment; keep using the delivered split columns.

```sh
cd /Users/bohanzhang/Projects/scam-message-detection
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-agent.txt
.venv/bin/python -m scripts.prepare_agent_inputs
.venv/bin/python -m scripts.prepare_agent_tokenizer
# Only needed on a fresh checkout without the local Agent models:
.venv/bin/python -m src.agent.baseline
.venv/bin/python -m scripts.frontend_agent_example
```

The supplied ZIP must already be at `local_inputs/processed_data_export.zip`.
The baseline command refuses a nonempty output directory. Existing trained
artifacts are already available in this checkout and need not be rebuilt.
Do not commit local inputs, models or per-case evaluation output.

## Python API

```python
from src.agent.service import ScamAgentService

service = ScamAgentService.from_config("configs/agent.yaml")
result = service.analyze_message(text=message, channel="sms", mode="agent")
# result is a validated, JSON-serialisable dict; no Streamlit dependency.
```

Use `sms`, `email` or `phone`. The config path may be absolute; model/export paths
resolve against the config's `project_root`, independent of the caller's cwd.
Cache one service instance for repeated calls: models and optional TRAIN matrices
are loaded lazily once per instance. Use a separate instance per worker, or
serialize calls. Concurrent calls on one instance immediately return
`service_busy`; they are not queued. Construction raises configuration/file/YAML
errors; request-level failures return the versioned result object.

| Mode | Behaviour |
|---|---|
| `classifier_only` (A) | Saved classifier on the original clipped view, no LLM |
| `direct_llm` (B) | One provider call, same label/final schema and input policy, no tools |
| `agent` (C) | Model chooses zero or more allowlisted tools, receives results in the same conversation, then chooses later actions/final JSON |
| `fixed_workflow` (D) | Normalise, classify original, classify normalised, extract original, optional retrieve original, then one LLM call |

D logs `source=fixed_workflow`; it never inserts fake assistant function calls.
A logs `source=classifier_only`. Only genuine model-originated tool requests in
C use `source=model_tool_call`. A tool call fixture under test is explicitly
`provider.mock=true`; it is not a real LLM run. The CLI has no mock fallback.

## Provider setup and bounds

Only DeepSeek non-thinking Chat Completions is implemented. Configure privately:

- `LLM_API_KEY`: provider credential; never send it in chat or commit it.
- `LLM_BASE_URL`: HTTPS API base, e.g. `https://api.deepseek.com` (without
  `/chat/completions`). An explicitly configured compatible gateway may be used,
  but its protocol still requires live verification.
- `LLM_MODEL`: an actual tool-capable model available to your account. There is
  deliberately no default model ID. No automatic provider switching occurs.

The implementation follows the official [tool call guide](https://api-docs.deepseek.com/guides/tool_calls/):
function schemas, assistant `tool_calls`, matching tool-call IDs in `role=tool`
replies, and a later assistant response. `thinking` is explicitly disabled;
hidden reasoning is neither logged nor returned. There are no implicit retries.
The configured endpoint is operator-controlled; message text cannot change it.
`.env` files are ignored by Git but are not automatically loaded.

Default per-message bounds in `configs/agent.yaml`: 4 LLM turns, 6 attempted tool
calls (including invalid/repeated calls), 60 seconds, 700 output tokens per call,
12,000 total tokens, and 20,000 characters per tool result. Identical repeated
calls return a structured error; invalid tools/arguments are not executed.
An invalid call may be corrected in a later turn while budgets remain.

With the configured official tokenizer, each request reserves the complete
serialized token count plus 25% and 256 framing tokens, then the output cap.
The service and durable meter share this estimator. Profiles without an explicit
tokenizer retain the UTF-8-byte fallback. The estimate is not billed usage. Actual
provider usage is accumulated. A provider failure/timeout or absent usage marks
accounting incomplete and prevents subsequent batch spending. Provider-specific
hidden billing is not claimed to be known. No cost is estimated without explicit
input/output USD-per-million pricing. Estimates do not include unconfigured
provider discounts or surcharges.

The caller deadline is enforced by an isolated daemon operation. An in-flight
HTTP/native inference operation cannot be safely cancelled and may finish after
an error is returned, with possible provider billing. It schedules no later
actions and cannot mutate a returned snapshot. Unknown accounting is not treated
as zero usage. Avoid automatic retries after such errors.

## Result contract

Authoritative schema: [`src/agent/schema.json`](../src/agent/schema.json), version
`1.0`. The LLM submits the smaller [`final_schema.json`](../src/agent/final_schema.json);
the backend fills IDs, actual scores, timings, trace, usage and errors itself.

| Fields | Frontend meaning |
|---|---|
| `request_id`, `schema_version`, `mode`, `channel` | Request identity and contract |
| `status` | `ok`, `needs_review`, or `error`; show separately from the decision |
| `decision` | `fraud`, `legitimate`, or `abstain` |
| `needs_review` | Review routing; errors always abstain, never display as legitimate |
| `explanation`, `evidence`, `unsupported_claims` | Short quote-based assessment; invalid/missing support for a binary LLM answer forces abstention |
| `views` | Exact original **used** prefix; optional normalised view, null until used |
| `input_policy` | Full original character count, used count, clipping flag, prefix offset zero |
| `indicators` | Observed rule matches, named view, quote and start/end; no domain reputation claims |
| `classifier_results` | Saved model identity, provenance, original class order, original prediction, task decision, score kind, scores and preprocessing |
| `retrieved_cases` | TRAIN-only references/excerpts, labels and similarity; never current test metadata |
| `tool_trace` | Tool name, validated arguments, bounded result, elapsed time, error and source |
| `elapsed_ms`, `llm_calls`, `usage` | Actual local duration/calls and provider-reported tokens; inspect `accounting_complete` |
| `provider`, `cost_estimate` | Actual configured provider/model, explicit fixture flag, optional configured-pricing estimate |
| `stop_reason`, `errors` | Why work ended, including recoverable tool errors |

Quotes/spans use **Python Unicode code points**, zero-based, end-exclusive, into
the named view. For JavaScript highlighting, use `Array.from(viewText).slice(start,
end).join("")` or convert indices to UTF-16 before slicing. Normalised offsets
must never be used against original text. The deterministic clipping policy is
the first 8,000 characters for **all** A/B/C/D modes. Nothing is dropped for being
long: clipped results require review, and original-prefix offsets stay valid.
The caller retains the full original if needed; returned `views.original` is
only the analyzed prefix, not an invented reconstruction.

LR's `model_probability_uncalibrated` is a class-weighted model output, not a
real-world fraud risk percentage. SVM's `decision_margin` is not a probability.
`fraud_score` is oriented by class label, including legacy email's first-column
fraud class; `scores` keys and `predicted_label` use string representations of
original labels, while `original_classes` preserves their original JSON types.
There is no LLM confidence field.

Exact quote matching proves attribution, not that the quote establishes fraud.
The backend renders explanations from quotes instead of publishing free-form
LLM allegations about an organisation or domain. Contextual evidence quality
still needs human review. `ok` means contract validation passed; it is not
external verification. Indicator absence also does not prove legitimacy.

Examples (actual local classifier/error outputs on handwritten smoke inputs,
not evaluation cases):

- [`success_classifier.json`](examples/success_classifier.json)
- [`needs_review_clipped.json`](examples/needs_review_clipped.json)
- [`error_provider_missing.json`](examples/error_provider_missing.json)

The later authorized live run produced
[`success_agent_live.json`](examples/success_agent_live.json), with a real
model-selected tool loop and `provider.mock=false`. See the
[live validation report](agent_live_validation.md) for its scope and two other
budget-stopped cases. `tests/test_agent.py` contains explicitly marked offline
loop fixtures.

## CLI and JSONL

```sh
.venv/bin/python -m src.agent.cli --text 'Please verify your account immediately.' --channel sms --mode classifier_only
.venv/bin/python -m src.agent.cli --text 'Your parcel is ready.' --channel sms --mode agent
.venv/bin/python -m src.agent.cli --jsonl messages.jsonl --mode agent --max-live-messages 3 --token-budget 12000
```

Each input line is `{"text":"...","channel":"sms"}`. Mode is fixed by the
CLI flag, not overridden by untrusted row fields. Each processed line produces
one validated result on stdout. Malformed JSONL and global-stop diagnostics go
to stderr. Exit 0 means no technical errors; exit 2 means an error or a stopped
batch. `needs_review` alone does not cause exit 2. Use `--jsonl -` for stdin.
The token budget and default three-live-message cap apply to the whole invocation.
For a larger planned run, explicitly configure both limits. Missing provider
configuration returns `provider_not_configured`; missing models return
`artifact_missing`. Neither produces fake scores or a heuristic Agent fallback.

## Retrieval and experiments

Optional retrieval is same-channel only, canonical seed0 eligible TRAIN only,
using the saved TRAIN-fitted vectorizer. Validation and test never enter the
index. Indexes across channels or seeds and zero-shot transfer are unsupported.

See [`agent_validation.md`](agent_validation.md) for executed checks, a frozen
90-case classifier pilot and the exact bounded live validation command. The
comparison runner uses identical frozen case IDs for each mode, hides current
row IDs/labels/dataset/subtype from the provider, and persists predictions/traces
before aggregation. Three validation cases per channel are available for initial
debugging. No HTTP server or frontend changes are needed for this handoff.


## Local macOS Keychain profile

This checkout now has ignored `configs/agent.local.yaml`, containing only a
Keychain reference, explicit `deepseek-flash` model and official endpoint. The
API key is in macOS Keychain, not the YAML or an `.env` file. To initialize a
frontend that is intentionally allowed to make paid calls, use:

```python
service = ScamAgentService.from_config("configs/agent.local.yaml")
```

This is explicit opt-in; the default config retains its environment-only
behaviour. The completed three-message smoke consumed 6,228 tokens across five
real calls, with one valid final answer and two conservative budget stops. See
[agent_live_validation.md](agent_live_validation.md). The original frozen prompt
and validation manifest are unchanged. Do not automatically repeat paid calls.
`run_live_agent_smoke.py` adds a durable pre-call ledger and writes sanitized wire
exchanges, without keys, HTTP headers or hidden reasoning.


## Current three-channel acceptance

The budget repair is covered by 121 passing tests. A new real run completed all
three provider conversations in 9,164 tokens. SMS and phone returned `ok`.
Email's actual quote was unique and exact but had a wrong end offset; the shared
validator now records a deterministic `evidence_span_repaired` event, retains the
unchanged quote, and returns `needs_review` with the original legitimate decision.
Only a unique exact quote can be re-anchored. Absent or ambiguous support still
abstains. See [the acceptance record](agent_budget_fix.md) for original live
responses versus offline postprocessing; no extra paid call was used for that
coordinate repair. Human review routing is part of the frontend contract.


## Classifier router extension (result version 1.1)

The original four modes keep their version 1.0 contract. Three new modes expose
explicit classifier selection:

```python
service = ScamAgentService.from_config("configs/agent.local.yaml")
result = service.analyze_message(text=message, channel="sms", mode="agent_router")
# mode="fixed_model": best global VAL candidate for this channel, no LLM.
# mode="rule_router": frozen VAL bucket rule, no LLM.
```

Run `.venv/bin/python -m src.agent.candidates` once to create the ignored unified
candidate directory and catalog. It refuses to replace an existing directory.
`routing_catalog_path` in the config names the catalog. Missing candidates or
provenance/policy mismatches return structured errors, with no fallback score.

`routing.selected_model_id` is one of the aliases word_lr, word_nb, word_svm,
char_svm; `routing.candidates_called` and tool_trace record what actually ran.
`classifier_results.model_id` is the channel-qualified artifact identity.
`routing.selection_reason` and input_features describe the selection. Final
fraud/legitimate decisions come from the selected called classifier, not an LLM
label override. Conflicting outputs force review. SVM margins and native
uncalibrated probabilities remain distinct.

Same-channel allowlisted model IDs are the only routing tool argument. The
Agent can choose up to two classifier calls and three LLM turns; lower configured
limits still apply. Model caches are reused per service instance; its existing
single-request concurrency restriction remains. The local policies do not make
provider requests and work with the default environment-only offline config.
The Agent mode needs the explicit configured provider/Keychain profile.

See [the router report](classifier_router_report.md) and
[an actual live version-1.1 example](examples/agent_router_live.json). The current
initial 12-case pilot found no accuracy gain over the best fixed model; Agent routing
is an optional experimental mode, not the recommended default.

## Normalised routing and mandatory boundary checks (result version 1.2)

Select `configs/agent.normalised.yaml` for the new policy. The original config
retains the earlier router for reproducibility. On this configured local machine,
`configs/agent.normalised.local.yaml` adds the existing Keychain provider reference;
it is ignored and contains no literal API key. The public profile reads explicit
provider environment settings. The original four modes retain their version-1.0
behavior; this setting affects only the three classifier-routing modes.

```python
service = ScamAgentService.from_config("configs/agent.normalised.yaml")
result = service.analyze_message(message, "sms", mode="rule_router")
# fixed_model: normalise + one VAL-selected model; no LLM or second classifier.
# rule_router: normalise + VAL profile primary + mandatory boundary check; no LLM.
# agent_router: LLM selects one primary; backend enforces the same boundary guard.
```

The same `normalise_with_changes` text is supplied to every actual classifier;
the Agent sees that view and observable original-text change counts. Each primary
has a VAL-selected native-distance threshold and distinct secondary model. A
near-boundary primary must execute that secondary, even when the LLM would not
request it. There are at most two actual classifier calls and three LLM turns;
lower configured limits still apply. If a configured limit blocks the required
check, return a technical error and review rather than silently skipping it.

The LLM receives both actual results through its one tool reply and proposes an
executed model. The backend applies the frozen VAL conflict policy. Conflicting
models always require review; insufficient distinct VAL conflict support produces
`decision="abstain"`, `status="needs_review"`. A supported winner may remain fraud
or legitimate but still requests review. Agreement is not proof of correctness.
Margins and uncalibrated probabilities are never shown as calibrated fraud risk.

Version 1.2 adds these fields under `routing`:

| Field | Frontend interpretation |
|---|---|
| `preprocessing.classifier_view` | Always `normalised` for this profile |
| `preprocessing.normalisation_changes` | Actual existing normaliser change counts |
| `boundary_checks` | Primary ID, native score kind/distance, VAL threshold, near flag, policy-enabled flag, actual secondary execution and resolution |
| `proposed_model_id` | LLM/local primary proposal before enforced conflict resolution |
| `selected_model_id` | Actual final called model, or null on abstention/error |
| `candidates_called` | Actual classifiers executed, in order |

Automatic secondary trace entries have `source="boundary_policy"`; genuine LLM
tool requests have `source="model_tool_call"`. Do not describe the automatic call
as an LLM choice. Show original and normalised text with explicit view names. Use
`needs_review` to route review regardless of whether a binary decision is present.
See the [real 1.2 review example](examples/agent_router_boundary_live.json).

Build unified TRAIN candidates with `python -m src.agent.candidates`, then build
the normalised VAL catalog with `python -m scripts.prepare_boundary_router validation`
using the original `configs/agent.local.yaml` provider profile. That calibration
step sends no API requests; the current script loads local provider configuration.
It refuses an existing experiment/catalog. The runtime profile references
`models/router_normalised_seed0/catalog.json`; missing files return structured
errors. Do not add the ignored export, catalogs, models or local provider profiles
to Git. The new public profile is ready to use with the existing local artifacts.

The [independent pilot](normalised_boundary_report.md) excluded 114 prior TEST
source IDs and scored 12 views from nine new sources. All methods scored 10/12;
the Agent used 24 requests and 55,177 tokens, with no accuracy gain. VAL calibration
and the pilot use 4,000-character source prefixes; the runtime ceiling remains
8,000 to match candidate metadata. Longer runtime inputs have not been validated
under these thresholds. If a caller clips text before calling the service, it
must retain and disclose that upstream clipping; the service cannot detect it.
The pilot's two misses are one million-character email's paired prefixes, mostly
recipient addresses. All four local candidates missed both views. Do not infer
that full-message classification was evaluated. The seven raw-data-dependent
test skips remain; 154 offline tests passed and this new live experiment completed.

## Adaptive investigation integration (schema 1.3)

```python
from src.agent.service import ScamAgentService

service = ScamAgentService.from_config("configs/agent.yaml")
result = service.analyze_message(message, channel="email")  # configured adaptive_agent
# Explicit free baseline:
local = service.analyze_message(message, channel="email", mode="document_classifier")
# Explicit fixed-workflow comparison:
fixed = service.analyze_message(message, channel="email", mode="document_fixed")
```

Use the existing ignored `configs/agent.local.yaml` on this machine for its
explicit Keychain provider reference. Missing provider configuration only blocks
escalated paid investigations; the free path remains usable. Cache one service
instance and serialize requests; its immediate `service_busy` guard still applies.
For parallel use, create separate instances. No UI framework or HTTP server is
required. The original four modes and classifier-router experiments are retained.

New modes require the ignored candidate artifacts and
`models/triage_seed0/catalog.json`. From a fresh checkout, prepare the supplied
processed export, run `python -m src.agent.candidates`, then
`python -m scripts.prepare_document_triage`. Calibration uses canonical VAL only,
refuses existing outputs and makes no provider request. Historical modes may also
need the separate `python -m src.agent.baseline` artifacts. Models/catalogs are
machine-local and must be generated rather than copied into Git.

Default bounds: 2,000,000 source characters, 2,000-character source windows,
4 LLM turns, 6 total tool attempts including the initial classifier/automatic
boundary check, 60 seconds and 12,000 tokens per message. A source above the cap
returns `source_too_large`; it is not silently cropped to its prefix. Provide the
whole original message when possible so the backend can disclose coverage.
Character/markup heuristics suggest windows; this does not implement a MIME parser.

Only the following request-bound tools are exposed:

| Tool | Allowlisted arguments | Actual action |
|---|---|---|
| `inspect_section` | `section_id` | Read the indexed source window, normalise, classify with fixed primary, extract indicators/evidence IDs |
| `cross_check` | `section_id` | Run a distinct VAL-selected secondary on an already inspected window |
| `retrieve_similar_cases` | `section_id`, `k` (1–2) | Same-channel canonical TRAIN retrieval; disabled when retrieval=false |
| `request_clarification` | `question_id` | Return a caller-supplied boolean answer or a pending question |

The Agent sees the same source index available to the fixed schedule. It chooses
later actions after actual observations. It cannot supply replacement message
text, arbitrary file paths, code, model names or URLs to execute. Initial
observations use `tool_trace.source="triage_gate"`; mandatory initial cross-checks
use `boundary_policy`; authentic LLM choices use `model_tool_call`. Keep these
sources distinct in the UI. Fixed-schedule calls use `fixed_workflow`.

Final internal responses use [triage_final_schema.json](../src/agent/triage_final_schema.json).
The LLM selects an executed assessment and existing evidence IDs; the backend
copies exact stored quotes/spans and rejects invented IDs or binary label
overrides. Evidence may span inspected sections, but a binary decision needs
support from its selected assessment's section. No generated confidence, sender
verification or free-form factual explanation is accepted. Finalization budget
is reserved before further exploration; a compact factual state is used to close.

Public schema 1.3 adds `triage`:

| Field | UI meaning |
|---|---|
| `path`, `escalation_reasons` | Free versus escalated route and observed triggers |
| `overview` | Source length/hash, window count, suggested IDs and bounded previews |
| `read_sections` | Actual source start/end, original/normalised view IDs and normaliser changes |
| `assessments`, `selected_assessment_id` | Actual model results and grounded final selection |
| `coverage_complete` | Whether every source window was inspected; not proof of semantic understanding |
| `clarification`, `context_answers` | Actual pending question and explicitly supplied answers |
| `next_steps` | Backend-rendered, bounded action guidance |
| `finalization_compacted` | Whether the final request used a compact observation summary |

`views.original` is the initial source prefix for backward presentation;
`views.section_0000_original` and `views.section_0000_normalised` (and later IDs)
are the actual inspected section views. Evidence offsets are Unicode code points
within their named view. `read_sections.source_start/source_end` locate the raw
section in the full caller source. Normalised character offsets are not raw-source
offsets; use the retained original section and change counts. Only bounded views
are returned, not the full potentially million-character source.

`input_policy.name="document_windows_v1"`; `used_chars` is the sum of distinct
inspected raw-window lengths, potentially larger than a single `max_chars` window.
`clipped=true` means some source content remains unread. Negative partial-source
findings become abstention/review; positive partial findings still require review.
Conflicts require review. A near-boundary gate is not calibrated error probability.

To continue clarification, retain the same original message and pass explicit
answers from the user on a new request:

```python
result = service.analyze_message(
    message, "email", mode="adaptive_agent",
    context_answers={"transaction_initiated": False},
)
```

Allowed question IDs are transaction_initiated, sender_known,
independent_verification and missing_body. Values must be booleans. A missing
answer remains pending; never fill it from the message or an LLM guess. A false
answer is still a supplied answer. The frontend must collect it; no interactive
frontend or conversation database is part of this backend.

Pass `allow_paid_calls=False` to disable paid escalation while still permitting
free classifications. Escalations then return `global_budget_exhausted` with
zero provider calls and complete zero-token accounting. In the CLI, the default
batch cap is three paid analyses / 12,000 tokens; free fast paths do not consume
the paid-message count. Original modes retain their historical batch behavior.

```sh
.venv/bin/python -m scripts.frontend_agent_example
.venv/bin/python -m src.agent.cli --text 'Meeting confirmed for tomorrow.' --channel sms --mode document_classifier
.venv/bin/python -m src.agent.cli --config configs/agent.local.yaml --jsonl messages.jsonl --mode adaptive_agent --max-live-messages 3 --token-budget 12000
```

An independent paid comparison is intentionally not automatic. The prepared
`compare_document_triage` runner accepts a frozen canonical manifest and common
bounds, preserves pending/failure/review results, and requires `--allow-live` plus
`--allow-live-test` for paid TEST runs. See the [validation record](adaptive_agent_handoff.md)
for original failures, saved-action revalidation, one fresh constrained-budget
execution and exact cumulative cost. Performance superiority remains unverified.
