# Agent backend frontend handoff

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
