# DeepSeek local setup and real Agent validation

**Current acceptance update:** the budget repair and a subsequent real three-channel
run recorded 6 calls / 9,164 tokens. All channels reached final responses;
email's unique literal quote required a visible coordinate correction and review.
See [agent_budget_fix.md](agent_budget_fix.md) for original versus offline
revalidated evidence, 121 passing tests, and remaining evaluation limits.


Executed on 10 November 2026 (Asia/Singapore), on local
`feature/agent-backend`. No push was made. This report supersedes the initial
implementation record's statement that live verification was pending.

## Outcome

Three frozen canonical validation messages were attempted, one per channel.
There were **5 actual DeepSeek generation calls**, **8 executed tool calls**, and
**6,228 provider-reported tokens** (5,834 prompt + 394 completion). Accounting is
complete; unresolved reservations are zero. No generation was retried or added
after the three-message run. The authorized maximum was three messages and
12,000 total tokens.

| Channel | Validation case | Actual provider calls | Tokens | Executed tools | Result |
|---|---|---:|---:|---|---|
| SMS | `mendeley_2681` | 2 | 2,347 | classify original, extract indicators | Valid final JSON; legitimate; exact evidence span |
| Email | `nigerian_5_1019` | 1 | 1,382 | classify original, extract indicators, retrieve TRAIN cases | `token_budget_exhausted`; technical abstention |
| Phone, synthetic | `bothbosu_train_1140` | 2 | 2,499 | classify original, extract indicators, retrieve TRAIN cases | `token_budget_exhausted`; technical abstention |

This proves one complete real model-chosen tool loop and actual model-chosen
retrieval in the other conversations. It does **not** prove three successful
end-to-end analyses. The two budget stops remain recorded as technical failures,
not legitimate predictions or successful abstentions. Normalisation was offered
but not selected by the model in this run.

All three selected cases are legitimate; there are no fraud positives. Binary
coverage is 1/3, technical-failure rate 2/3, and conservative full-denominator
success is 1/3. The one answered case is correct, but no meaningful fraud recall
or comparative Agent accuracy claim follows. B/direct LLM and D/fixed workflow
were not run live. No paid benchmark or stochastic rerun was performed.

## Why two conversations stopped below 12,000 actual tokens

The cap is checked before every request using a deliberately conservative
reservation: serialized UTF-8 bytes of the full conversation/tool definitions,
plus framing and the maximum permitted output. As tool and retrieval results
accumulate, that reserved upper bound can exceed the remaining budget even when
the actual provider token count would be much smaller. The email continuation
and the final phone continuation were rejected **locally before another API
request**. No provider-side quota exhaustion or missing account balance is
inferred from these stops.

There were 5,772 unspent reported-token units after the run, but a safe
reservation for the next continuation did not fit. The guard was not relaxed,
contexts were not silently dropped, prompts were not changed, and extra paid
messages were not sent. A later implementation can test a documented,
model-compatible token counter or reduce disclosed context/tool-result size on
validation; that requires a new frozen input/prompt policy where applicable.
The present strict guard and the original results are preserved.

## Credential configuration

The key was entered through a native hidden-input macOS dialog. It is stored in
macOS Keychain as the project-specific generic-password item
`ee6405.scam-message-detection.deepseek`, under the local OS account. It was never
printed, passed as a shell command argument, embedded in code, or written into
an environment file.

Ignored `configs/agent.local.yaml` contains only:

- official base URL `https://api.deepseek.com`;
- explicit model `deepseek-flash`;
- a Keychain service/account reference;
- the existing artifact paths, tool settings and limits.

That profile has mode 0600. `ScamAgentService.from_config()` accesses Keychain
only when a config explicitly contains the reference. The default
`configs/agent.yaml` still uses environment configuration and does not silently
search other apps or credential stores. Explicit LLM environment values override
the local profile. The setup script is `scripts/configure_deepseek.py`; it has no
embedded secret and refuses to overwrite an existing local profile.

The selected ID and non-thinking tool-call protocol were checked against
[DeepSeek's official model listing and pricing page](https://api-docs.deepseek.com/quick_start/pricing/)
and [tool-call documentation](https://api-docs.deepseek.com/guides/tool_calls/).
An authenticated, non-generative `GET /models` returned HTTP 200 and confirmed
that `deepseek-flash` was available. It was not an additional validation message
or generation. There was no model/provider fallback. `thinking` was disabled.
Pricing remains unconfigured, so monetary cost is null; 6,228 tokens is the
measured usage, not a claim about a charged currency amount.

## Durable budget and saved evidence

`src/agent/live_meter.py` wraps the actual provider. It writes a reservation to
disk before each network request; after a valid response it replaces the
reservation with provider-reported usage. A failed request or missing usage
freezes later spending. Existing ledger/output paths block accidental reruns.
The wrapper also caps actual generation attempts at 12. The evaluation runner
independently caps the run at three canonical validation messages and 12,000
tokens, with per-message limits of 4 turns, 6 tools and 60 seconds.

Actual local artifacts (all ignored by Git):

- `outputs/agent/live_agent_smoke_v1/run.json`: exact frozen manifest/config.
- `outputs/agent/live_agent_smoke_v1/predictions.jsonl`: all three validated results,
  including the two technical failures.
- `outputs/agent/live_agent_smoke_v1/summary.json`: full-denominator metrics.
- `outputs/agent/live_agent_smoke_v1_api_audit/budget_ledger.json`: all five
  reservations and reported usage, no unknown accounting.
- `outputs/agent/live_agent_smoke_v1_api_audit/exchange_01.json` through
  `exchange_05.json`: actual messages/tool definitions and adapter-filtered
  responses. No Authorization headers, credentials or hidden reasoning.
- `outputs/agent/live_agent_smoke_v1_api_audit/model_preflight.json`: model-ID
  availability check, no key.

The SMS response is also supplied as a frontend example:
[`examples/success_agent_live.json`](examples/success_agent_live.json).
It has `provider.mock=false` and contains real tool outputs, rather than a
scripted provider fixture.

The executed command was:

```sh
.venv/bin/python -m scripts.run_live_agent_smoke
```

Do not automatically repeat this command to obtain better results. It refuses
the existing output path. This authorization has been used for the recorded
three-message run; a later paid run needs its own explicit scope/budget.

## Verification after local setup

- Full offline suite: **109 passed, 7 skipped**; no tests send live requests.
- Source compilation and `git diff --check`: passed.
- Default environment-only and explicit Keychain profiles tested separately.
- Durable reservation, over-budget rejection before a call, fail-closed unknown
  accounting, call cap and refusal to reuse a ledger tested with no-network stubs.
- All three actual results validated against `schema.json`, including evidence
  quote/span checks.
- Actual assistant call IDs were matched to subsequent `role=tool` messages in
  both conversations that made a second provider request.
- Five audit exchanges reconcile exactly to 5,834 + 394 = 6,228 reported tokens.
- A key-value scan of 39 source/config/document/run files found no persisted key.
  The scan read the key in process memory and emitted only its pass/fail result.

Manual inspection of the successful SMS quote confirmed it exactly matches the
original [0,136) span and describes travel/subletting, supporting a legitimate
assessment in context. The model first chose classifier and indicator tools,
received those real results, then returned the final answer. The email and phone
retrieval results are actual canonical TRAIN matches, but the LLM did not reach
a final interpretation of them; no explanation-quality success is claimed.
