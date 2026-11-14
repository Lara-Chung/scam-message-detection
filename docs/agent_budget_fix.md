# Budget repair and three-channel acceptance

Date: 10 November 2026, Asia/Singapore. Branch: `feature/agent-backend`.
This record supersedes the initial budget-stopped live run described in
[agent_live_validation.md](agent_live_validation.md). Original run files remain
unchanged. No additional paid calls were made after the three-message run below.

## Implemented repair

The previous reservation used serialized UTF-8 bytes as tokens. That is a safe
fallback but rejected affordable continuations after tool results accumulated.
`agent/runtime/budget.py` now supplies one shared `PromptBudget` instance to the
service and durable meter:

1. Load the official DeepSeek V4 tokenizer data from the explicitly configured
   local file. Verify the reviewed SHA-256 before parsing. Only the reviewed
   `deepseek-flash` / `deepseek-v4-pro` IDs can use this profile.
2. Count the complete serialized conversation and tool definitions, without
   truncation or padding. Keep every tool result and the same input policy.
3. Reserve `ceil(token_count * 1.25) + 256` input tokens for protocol differences,
   plus the actual requested output cap. Clamp output to the remaining budget.
4. Persist the reservation before a request; reconcile actual reported input and
   output after it. Stop subsequent spending if either component exceeds its
   reservation, usage is unavailable, or a request fails.

This is a buffered estimate, not an assertion that the provider's private chat
framing can be reproduced exactly. Reported usage remains authoritative. The
counter covered all five historical provider input counts and every request in
the new live run. The byte-based fallback remains for explicit profiles without
a tokenizer. There is no silent tokenizer/model substitution.

Tokenizer source: [DeepSeek token usage documentation](https://api-docs.deepseek.com/quick_start/token_usage/).
Pinned data SHA-256:
`89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614`.
`prepare_agent_tokenizer.py` downloads JSON data only, verifies it, and executes
no downloaded Python or remote model code. Data stays in ignored `models/`.

## Actual bounded validation

The same frozen canonical validation manifest, prompt, final schema, tool
definitions, model artifacts and clipping policy were reused. The model selected
its own tools. No workflow was forced. All samples are legitimate; this run is
integration acceptance, not a fraud recall or performance-gain experiment.

| Channel | Actual calls | Actual tool executions | Prompt + output tokens | Final accepted decision |
|---|---:|---:|---:|---|
| SMS | 2 | 3 | 2,680 | legitimate, `ok` |
| Email | 2 | 3 | 3,888 | legitimate, `needs_review` after exact-quote coordinate correction |
| Phone, synthetic | 2 | 2 | 2,596 | legitimate, `ok` |
| Total | **6** | **8** | **9,164** | Three final responses, zero technical failures |

Provider totals: **8,480 prompt + 684 completion = 9,164 tokens**, below the
new run's 12,000-token cap. Ledger accounting is complete; outstanding
reservations are zero. No message hit the token, turn or tool budget.

The paid execution command was `.venv/bin/python -m scripts.run_live_agent_smoke`;
it wrote new `_v2` output and audit directories. It initially returned exit 2
because the email answer's first evidence quote had a wrong end offset. It did
not fail the budget or provider protocol. That original response/result is
preserved; it must not be presented as an unqualified initial success.

## Exact-quote repair and offline revalidation

The email LLM returned a legitimate decision and a real, unique literal quote,
but attached [0,100) instead of its actual [0,136) coordinates. Its other two
cited spans were exact. This is a coordinate error, not invented quote text.

The shared final-answer validator now accepts a bad coordinate only when the
unchanged, nonempty quote occurs exactly once in its explicitly named view.
It computes that unique span, records `evidence_span_repaired` with old/new
coordinates, and routes the result to **needs_review**. Missing or ambiguous
quotes still force abstention. No spelling, quote content, normalised view or
classification is guessed to make a response pass.

The three unmodified saved real final responses were then processed by that
same validator **offline**, without another API request. Derived files are
explicitly labelled `deterministic_revalidation_of_saved_live_final`; original
results and raw exchanges were not overwritten. The final validator returns
legitimate for all three, with the email review event retained. This is
postprocessing verification of actual live responses, not a claim of another
fresh three-message live run.

Acceptance requires each channel to reach `final_answer`, return schema-valid
JSON with exact attributable quotes, preserve complete usage and real tool
results, and avoid technical failure or unresolved unsupported claims. A
visible review event is allowed. All three meet these integration criteria;
the email case remains routed for human review. Human reading confirmed the
association-website announcement matches that literal quote; a URL or email
address alone is not evidence of fraud or verified ownership.

## Saved artifacts and frontend examples

Ignored local evidence:

- `outputs/agent/live_agent_smoke_v2/predictions.jsonl`: original paid-run results.
- `outputs/agent/live_agent_smoke_v2/revalidated_predictions.jsonl`: derived,
  source-linked postprocessing results with zero additional API calls.
- `outputs/agent/live_agent_smoke_v2/{summary,revalidated_summary}.json`: both
  original and derived aggregate views.
- `outputs/agent/live_agent_smoke_v2_api_audit/budget_ledger.json`: all six real
  reservations and usage reconciliation.
- `outputs/agent/live_agent_smoke_v2_api_audit/exchange_01.json` through
  `exchange_06.json`: actual model-generated calls and final responses, no key,
  Authorization headers or hidden reasoning.

Shareable schema-valid examples:

- [SMS](examples/success_agent_live.json): unmodified successful live result.
- [Phone](examples/success_phone_agent_live.json): unmodified successful live result.
- [Email](examples/needs_review_email_agent_live.json): actual saved response
  revalidated with a visible exact-quote coordinate correction, not a fresh API call.

The offline revalidation command was:

```sh
.venv/bin/python -m scripts.revalidate_live_agent_results
```

It refuses to overwrite existing derived files and cannot complete a response
that never reached a real provider final answer.

## Checks and reproducible setup

Full suite: **121 passed, 7 skipped**. Added coverage includes complete-request
counting, official data/model/hash guards, no input truncation, historical
provider count coverage, previously blocked continuations, shared three-channel
budget replay, per-component reservation overruns, and unique/ambiguous/absent
quote handling. Replay final answers used in unit tests are explicitly labelled
fixtures and are not included as live evidence.

Fresh local setup for an authorized future run:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-agent.txt
.venv/bin/python -m scripts.prepare_agent_inputs
.venv/bin/python -m src.agent.baseline
.venv/bin/python -m scripts.prepare_agent_tokenizer
.venv/bin/python -m pytest -q
```

Use existing artifacts instead of retraining when already present. Data inputs,
model/tokenizer files, local Keychain profile and per-case audit output stay
ignored by Git. There is no need to configure an API key to run offline tests.
The seven skips require unavailable raw datasets; the current local processed
export, real artifact and tokenizer/regression checks executed.

The comparison B/direct LLM versus C/Agent versus D/fixed workflow and a
fraud-positive live evaluation remain future work. This repair verifies
functionality and budget handling; it does not claim that the Agent improves
accuracy or is production-ready.
