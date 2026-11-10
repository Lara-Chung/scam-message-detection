# Unified classifier routing: implementation and pilot

Date: 10 November 2026, Asia/Singapore. All original artifacts, teammate training scripts and previous experiments are preserved.

## Outcome

**Classifier routing is implemented and genuine model-generated selection was observed. This pilot does not demonstrate an accuracy benefit from LLM routing.** All three policies return 12/12 correct decisions. The Agent calls two SVM candidates on every case and selects the same final family within each channel; it has not demonstrated fine-grained per-message adaptation in this sample.

## Unified candidate provenance

Four newly trained candidates per channel: word TF-IDF + LogisticRegression, MultinomialNB, LinearSVC, and character TF-IDF + LinearSVC. All use only the same eligible canonical seed0 TRAIN IDs, nullable fraud labels (NA excluded), and the original 8,000-character prefix. Every learned vectorizer and classifier is TRAIN-fitted; all labels are numeric [0,1] with fraud=1. No legacy seed42 artifact is accepted for routing. No candidate was fitted or refitted on VAL or TEST.

| Channel | TRAIN / VAL | Word LR VAL F1 | Word NB VAL F1 | Word SVM VAL F1 | Char SVM VAL F1 | Selected fixed model |
|---|---:|---:|---:|---:|---:|---|
| sms | 3431 / 858 | 0.9557 | 0.9554 | 0.9770 | 0.9934 | `char_svm` |
| email | 5091 / 1273 | 0.9926 | 0.9910 | 0.9934 | 0.9943 | `char_svm` |
| phone (synthetic) | 760 / 199 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | `word_lr` |

Fixed hyperparameters: word 1–2 grams/max_features=50,000/min_df=1; char_wb 3–5 grams/max_features=60,000/min_df=2; sublinear TF and float32 sparse vectors. LR/SVM use C=1, balanced class weights, seed0; NB alpha=1 with uniform class priors. Candidate settings and tie order (LR, NB, word SVM, char SVM) were declared before fitting. The best fixed model is selected by VAL macro-F1.

Artifacts live in ignored `models/router_candidates_seed0/`. Each metadata file saves identical TRAIN IDs/membership hashes, original and fitted-prefix text hashes, export/artifact hashes, dependency versions, labels and hyperparameters. The runtime checks these before loading and caches each channel/model once. Inference does not fit or download models.

## Complementary errors, inspected on VAL before TEST

| Channel | Cases | Fixed correct | Oracle-any-candidate correct | Fixed errors another candidate can rescue | All candidates wrong | Candidate disagreements |
|---|---:|---:|---:|---:|---:|---:|
| sms | 858 | 856 | 858 | 2 | 0 | 25 |
| email | 1273 | 1266 | 1270 | 4 | 3 | 12 |
| phone | 199 | 199 | 199 | 0 | 0 | 0 |

Across 2,330 validation cases, fixed models are correct on 2,321; the hindsight oracle reaches 2,327. The maximum observed rescue space is only six cases (about 0.26 percentage points). The oracle uses true VAL labels and is a non-deployable diagnostic upper bound, not a router or measured Agent result. Complementary errors do not establish that text features let a router identify the right model.

Examples of actual rescuable VAL cases (0=legitimate, 1=fraud):

| VAL case | Truth | LR | NB | Word SVM | Char SVM / fixed |
|---|---:|---:|---:|---:|---:|
| `mendeley_4310` | 1 | 0 | 1 | 0 | 0 |
| `mendeley_4366` | 0 | 1 | 0 | 0 | 1 |
| `nigerian_5_158` | 0 | 0 | 1 | 1 | 1 |
| `nigerian_5_2085` | 1 | 0 | 0 | 1 | 0 |
| `nigerian_5_2868` | 1 | 0 | 1 | 0 | 0 |
| `nigerian_5_412` | 0 | 0 | 1 | 1 | 1 |

For example, one SMS fraud case is detected only by NB, although NB has lower global VAL F1. That is genuine complementarity. It is not evidence that a routing rule or LLM can recognize that case without its label.

## Rule and Agent policies

The deterministic rules inspect only original-text features: first obfuscated (normaliser changes or a digit inside a word), then long (>1,200 characters), short (<=160), otherwise standard. The best VAL model is selected within each bucket only if n>=30 and both classes have at least three cases; otherwise use the best global model. Thresholds were declared before evaluation. VAL rule performance is selection-set/apparent performance, not independent generalisation.

| Channel | Obfuscated | Long | Short | Standard |
|---|---|---|---|---|
| sms | char_svm | char_svm | char_svm | word_lr |
| email | char_svm | word_lr | word_svm | word_lr |
| phone | word_lr | word_lr | word_lr | word_lr |

The Agent receives the original message, channel, observable features and aggregate candidate VAL cards. It receives no current row ID, dataset/subtype, label, per-case oracle, TEST candidate predictions or rule lookup answer. It can call only `classify_message(model_id=...)`, with a strict same-channel allowlist and at most two candidate calls. Later choices can depend on actual results. Its final selected ID must have been called; the backend copies that classifier's decision. The LLM cannot invent or override a fraud label. Conflicting called-model predictions force review. SVM margins and uncalibrated probabilities retain their native types and are not treated as comparable confidence.

## Frozen TEST pilot

Twelve canonical seed0 TEST rows: two legitimate and two fraud cases per channel, SHA-256 sample seed 6405. Balanced sampling is diagnostic, not real prevalence. All production prompts, catalog/policies, code hashes and the input policy were frozen before predictions. There are no prompt or rule fixes, retries or resampling after results. Phone is synthetic.

| Policy | Correct / 12 | Macro-F1 | Fraud precision / recall | False-positive rate | Failure / abstention / review cases | Mean latency |
|---|---:|---:|---:|---:|---|---:|
| Best fixed per channel | 12/12 | 1.00 | 1.00 / 1.00 | 0.00 | 0 / 0 / 0 | 0.0051s |
| VAL bucket rules | 12/12 | 1.00 | 1.00 / 1.00 | 0.00 | 0 / 0 / 0 | 0.0030s |
| Agent router | 12/12 | 1.00 | 1.00 / 1.00 | 0.00 | 0 / 0 / 0 | 2.1453s |

Coverage is 12/12 and full-denominator success 12/12 for all policies. Every channel is 4/4. Confusion matrices are [[6,0],[0,6]] overall and [[2,0],[0,2]] per channel. No case was dropped or pending. Timing includes feature calculation, routing/model inference and API round trips, with all artifacts preloaded. The small local timing difference between fixed and rules is not a speedup claim. Neither local policy uses an LLM.

| Policy | Classifier calls | Actual API requests | Reported tokens | Estimated API price for all 12 cases |
|---|---:|---:|---:|---:|
| Best fixed per channel | 12 | 0 | 0 | $0 incremental API |
| VAL bucket rules | 12 | 0 | 0 | $0 incremental API |
| Agent router | 24 | 24 | 45,839 | $0.003594–$0.007187 |

The actual ledger reconciles **24 requests, 44,603 input + 1,236 output = 45,839 tokens**. Input cache counts: 26,112 hit / 18,491 miss. All reservations held and unresolved usage is zero. The estimate applies current official off-peak/peak list rates and reported cache counts; it is not measured account-balance deduction. Local CPU, training and storage costs are excluded. [Official pricing checked 10 November 2026](https://api-docs.deepseek.com/quick_start/pricing/).

## What the Agent actually selected

- SMS and email: called char SVM and word SVM together for every case, selected char SVM.
- Phone: called word SVM and char SVM together for every case, selected word SVM.
- All twelve finals used cross_check_agreement. The recorded calls really agree; none was a fabricated classifier result.
- The choice changed between channels but not between messages within a channel. Conditional second-call capability is unit-tested, but this particular live run used parallel pairs rather than adapting a later request after the first result.
- The post-run TEST oracle is 12/12, equal to the fixed baseline. One fraud SMS is missed by NB alone; all other candidate decisions in this sample agree. Oracle predictions were computed only after routing and never fed to the LLM.

No observed label rescue or harm occurred versus either local policy. The sample reached a ceiling; it does not measure whether an Agent can find the six rare VAL rescue opportunities. No hand-picked TEST disagreement challenge was added after seeing these results.

## Interface and reproducibility

The original four modes still work. Three additive modes use the new version **1.1** routing metadata; historical 1.0 results remain valid:

```python
from src.agent.service import ScamAgentService
service = ScamAgentService.from_config("configs/agent.local.yaml")
result = service.analyze_message(text=message, channel="sms", mode="agent_router")
# Alternatives: mode="fixed_model" or mode="rule_router" (no provider calls).
selected_alias = result["routing"]["selected_model_id"]
```

Candidate aliases are word_lr, word_nb, word_svm, char_svm. Artifact model IDs in classifier_results are channel-qualified; tool_trace.inputs.model_id and routing metadata use the allowlisted aliases. Routing output records the actual calls, selection reason, features, native scores, latency, usage, errors and review status. It has no invented LLM probability.

```sh
.venv/bin/python -m src.agent.candidates
.venv/bin/python -m scripts.run_router_comparison prepare
.venv/bin/python -m scripts.run_router_comparison run
.venv/bin/python -m scripts.report_router_comparison
```

Existing model/experiment paths refuse overwrite. Keys remain in Keychain; candidate artifacts, catalog, raw predictions and wire traces are Git-ignored. Default offline profiles need no API key; they need the candidate artifacts for local routing.

Records under ignored outputs/agent/router_comparison_v1 include protocol.json, manifest.json, results/predictions.jsonl (36 actual case/policy results), results/summary.json, results/candidate_test_predictions.jsonl (post-run diagnostic), api_audit/budget_ledger.json and 24 filtered exchanges. analysis.json stores the reconciled costs and paired outcomes. Candidate VAL predictions are saved beside the catalog.

## Supported conclusion

Unified candidate validation found real but small complementary error space in SMS/email and none in synthetic phone VAL. The router extension works, but this 12-case run supplies no evidence that LLM routing is worth its additional latency/API cost. For this task, the VAL-selected fixed model is the supported economical default; the rule and Agent policies remain alternatives for further independently reviewed evaluation. Model selection on a validation set is useful even when an LLM adds no gain.

Independent human review, a larger untouched test sample, stochastic repeats and obfuscation/shift stress tests remain pending. Do not interpret equal 12/12 scores as perfect real-world detection, statistically established equivalence or proof that no future routing strategy can help. No additional paid run was launched after this pilot.

Verification after implementation: **135 tests passed, 7 skipped**. Compilation, dependency consistency, patch whitespace, all saved result schemas, TRAIN membership, selected-classifier grounding, frozen source hashes and usage reconciliation passed. No report-verification command sent an API request.
