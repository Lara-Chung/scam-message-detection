# Independent character-obfuscation and disagreement challenge

Date: 10 November 2026, Asia/Singapore. This is a deliberately conditioned challenge, not a random population-performance estimate.

## Finding

**No additional label accuracy was observed from Agent routing.** All three policies return 11/12 correct inputs, including 4/5 among preselected disagreement inputs. One fraud SMS is correctly classified clean but missed after four invisible-character insertions. Other candidates can classify it correctly; the Agent does not select or cross-check them on that input. This exposes a real unused routing opportunity, not merely a clean-sample accuracy ceiling.

## VAL-only design and independent TEST boundary

- Reuse the same 12 candidate artifacts fitted only on canonical seed0 eligible TRAIN. No vectorizer/classifier fitting on augmented VAL or TEST.
- Evaluate the full 2,330 VAL bases under clean text plus four letter-only stress families at severities 0.25 and 0.50: zero-width characters, Cyrillic lookalikes, fullwidth letters and spaced letters. This produces 20,970 correlated validation observations, not 20,970 independent messages.
- Use the first 4,000 source characters for all conditions to bound cost, with transformed text kept below 8,000 characters. Keep digits, punctuation and protected URLs, emails and amount spans intact. Labels are inherited under an intended-meaning assumption; this is a synthetic perturbation test, not observed real-world attacker traffic.
- Select the robust fixed classifier using mean per-condition VAL macro-F1 over all nine conditions. Select profile rules using only VAL results and observable normaliser changes; require 30 distinct base IDs and at least three per class, otherwise fall back to the robust fixed model. Rules use no current-message true label.
- Select the stress family/severity for TEST by highest VAL candidate-disagreement rate among variants changing at least half the messages. Ties prefer lower severity then declared family order. Freeze the catalog and validation_design.json before examining TEST candidates.
- Exclude 105 base IDs from all previous recorded TEST manifests. The new selector receives only id/text and candidate predictions; its interface rejects labels. Only after selecting IDs are source labels joined for scoring.
- Freeze one natural disagreement (or declared ordinary fallback), one clean/obfuscated pair and one random clean control per channel. Select by SHA-256 seed 6406; do not select cases where a particular method is known to be right or wrong.
- This yields 12 inputs from nine previously unused TEST base messages. Paired views are correlated. Disagreement conditioning changes the evaluated distribution: report strata and pairs separately, not as general TEST accuracy.
- Lock prompt/guidance, current-profile VAL cards, policies, manifest, input construction and code hashes before live requests. Copies of all frozen source files are saved under frozen_source/. No post-result changes, reruns or sample replacements.

## Selected VAL policies

| Channel | Word LR mean F1 | Word NB mean F1 | Word SVM mean F1 | Char SVM mean F1 | Robust fixed | TEST stress from VAL |
|---|---:|---:|---:|---:|---|---|
| sms | 0.949922 | 0.930247 | 0.971683 | 0.989219 | char_svm | invisible:0.5 |
| email | 0.986046 | 0.987094 | 0.992449 | 0.992450 | char_svm | invisible:0.5 |
| phone | 1.000000 | 1.000000 | 1.000000 | 1.000000 | word_lr | invisible:0.25 |

The email word/character SVM means differ by less than one millionth; the deterministic ranking is not evidence of statistically meaningful superiority. VAL rule results are selection-set/apparent results, not independent performance. Agent cards contain only aggregate VAL results for the current observable profile, not per-case outcomes, rule answers or TEST labels.

| Channel | none | digit-word | invisible | compatibility | lookalike | spaced | mixed |
|---|---|---|---|---|---|---|---|
| sms | char_svm | char_svm | char_svm | char_svm | char_svm | word_svm | char_svm |
| email | word_svm | word_nb | char_svm | char_svm | char_svm | word_svm | word_svm |
| phone | word_lr | word_lr | word_lr | word_lr | word_lr | word_lr | word_lr |

## Independent challenge availability

| Channel | Unused eligible TEST bases | Natural disagreements | Stress disagreements | Natural fallback |
|---|---:|---:|---:|---|
| sms | 1038 | 15 | 39 | False |
| email | 1555 | 16 | 41 | False |
| phone | 205 | 0 | 1 | True |

Stress counts are on the remaining pool after removing the selected natural case. Phone had zero VAL disagreement under all conditions; its TEST natural stratum therefore uses the predeclared ordinary fallback. One new phone base disagreed after the VAL-chosen stress transformation. No stronger perturbation was chosen after learning that fact.

Actual selected input support: 4 legitimate / 8 fraud. Source-prefix clipping affected 0 selected observations. No transformed input was subsequently clipped by the service. The 4,000-character base policy was fixed on VAL even though it did not truncate these selected sources.

A binary candidate-disagreement stratum necessarily has at least one candidate predicting each possible label. Its oracle upper bound is therefore 100% by construction, not a discovery or deployable algorithm. Selection is label-blind but the challenge is still conditional, not an unbiased sample.

## Actual outcomes, including all failures and review routing

| Policy | Correct / 12 | Macro-F1 | Fraud precision | Fraud recall | FPR | Review / failures / abstentions | Mean latency |
|---|---:|---:|---:|---:|---:|---|---:|
| Robust VAL-selected fixed model | 11/12 | 0.9111 | 1.000 | 0.875 | 0.000 | 0 / 0 / 0 | 0.0048s |
| VAL profile rules | 11/12 | 0.9111 | 1.000 | 0.875 | 0.000 | 0 / 0 / 0 | 0.0037s |
| Agent router | 11/12 | 0.9111 | 1.000 | 0.875 | 0.000 | 0 / 0 / 0 | 2.0434s |

All 36 case/policy records executed. Answered coverage is 12/12 for each method; there are no pending cases. Overall confusion matrix is [[4,0],[1,7]] for each policy. This pooled result describes only this challenge mixture.

| Stratum | n | Fixed correct | Rule correct | Agent correct |
|---|---:|---:|---:|---:|
| natural_control_fallback | 1 | 1/1 | 1/1 | 1/1 |
| natural_disagreement | 2 | 2/2 | 2/2 | 2/2 |
| paired_clean | 3 | 3/3 | 3/3 | 3/3 |
| paired_obfuscated | 3 | 2/3 | 2/3 | 2/3 |
| random_clean_control | 3 | 3/3 | 3/3 | 3/3 |

The five explicitly selected disagreement inputs score 4/5 for all three policies. The other seven observations score 7/7. The three clean paired views score 3/3 and their perturbed partners 2/3. No average over these correlated views is claimed to be a population estimate.

| Paired source | Channel | Clean decision, all policies | Perturbed decision, all policies | Correct-to-wrong flip |
|---|---|---|---|---|
| nazario_5_1888 | email | fraud | fraud | False |
| bothbosu_test_199 | phone | legitimate | legitimate | False |
| mendeley_1043 | sms | fraud | legitimate | True |

## Observed failure and unexploited complementary model

For `mendeley_1043`, the supplied fraud label is inherited unchanged. The clean SMS is correctly labelled fraud by all four candidates. The controlled variant inserts U+200B only into `Your`, `Card`, `Code` and `Call`; the amount, card digits, date, reference and callback number remain unchanged.

- On the perturbed view, word LR and char SVM predict legitimate; word NB and word SVM predict fraud.
- Fixed and profile-rule policies choose char SVM and miss it. The Agent also calls only char SVM, chooses its legitimate result and does not request review.
- The actually observed char-SVM fraud margin is approximately -0.0193, near its zero decision boundary. This is a native decision margin, not a probability or calibrated confidence.
- The Agent cross-checks both SVMs on the clean version but does not cross-check the version that causes the only miss. Thus the additional LLM budget was not allocated to the observed rescue opportunity. Calling another classifier could expose disagreement; it does not by itself guarantee the Agent would select the correct one.
- No post-result threshold, prompt instruction or paid rerun was introduced to rescue this example.

Post-hoc, the existing deterministic normaliser removes the four zero-width characters and exactly reproduces the already scored clean input. That clean input was classified as fraud. This is recorded as an input-equivalence diagnostic, not a measured fourth policy or additional API result. The scored policies classify the same unnormalised views; they use normaliser change counts only as observable routing features. A normalise-then-strong-classifier baseline is a plausible inexpensive next experiment, but this run does not claim its general performance.

## Work, latency and API price

| Policy | Classifier calls | API requests | Reported tokens | Estimated API price, all 12 |
|---|---:|---:|---:|---:|
| Robust VAL-selected fixed model | 12 | 0 | 0 | $0 incremental API |
| VAL profile rules | 12 | 0 | 0 | $0 incremental API |
| Agent router | 21 | 24 | 43,865 | $0.003440–$0.006879 USD |

Actual usage: **24 generation requests; 42,731 prompt + 1,134 completion = 43,865 tokens**. Provider cache counts: 24,832 hit / 17,899 miss. Accounting is complete, all request reservations held, and outstanding reservations are zero. The ceiling was 75,000 total tokens and 36 requests, with at most 350 output tokens/request and no retries.

Costs apply the [official off-peak and peak list rates](https://api-docs.deepseek.com/quick_start/pricing/) checked 10 November 2026 to reported cache usage. They are price estimates, not account-balance deductions; local hardware, training and storage costs are excluded. Timing is warm-service, including routing, model inference and provider round trips. Small local fixed/rule timing differences are not reliable speedup evidence.

The Agent made one classifier call on three SMS inputs and two calls on the other nine inputs. It therefore varies some call decisions, unlike the preceding uniform two-call pilot. This observed variation did not improve labels or identify the only error.

## Limits and conclusion

This is stronger evidence than a clean-sample ceiling test because real candidate disagreements and a paired perturbation-induced error are present. It still supplies **no observed accuracy gain** for Agent routing over the two local policies. The result supports keeping the strong fixed classifier as the economical default rather than paying for an LLM on every message.

There are only nine independent source messages, one stochastic run and a deliberately conditioned class mix. Four phone views are synthetic. Character edits assume intended meaning/label preservation and have not been independently adjudicated by a human reviewer. The normaliser and profiles are English-oriented. Validation conditions repeat the same bases and are not independent observations. No p-value, general fraud recall, real-traffic prevalence estimate or universal claim about Agent usefulness follows from this result.

## Reproduction and evidence

```sh
.venv/bin/python -m scripts.prepare_router_stress validation
.venv/bin/python -m scripts.prepare_router_stress test
.venv/bin/python -m scripts.run_router_stress
.venv/bin/python -m scripts.report_router_stress
```

Existing design/manifest/results paths reject overwrite and automatic paid reruns. Original candidate artifacts, previous experiments and teammate training code are preserved. Only the independent stress experiment uses the new catalog, profile-specific cards and locked cost guidance; ordinary configurations retain the previous bucket behavior.

Ignored local evidence under outputs/agent/router_stress_v1/: validation_design.json; catalog.json; validation_predictions.jsonl (20,970 correlated VAL rows); manifest.json including source hashes and exact edit spans; protocol.json and frozen_source/; results/predictions.jsonl (36 actual results); results/summary.json; results/candidate_diagnostics.jsonl; api_audit/budget_ledger.json and 24 filtered wire exchanges; analysis.json.

Keys remain in macOS Keychain; no credential, Authorization header or hidden reasoning is saved in those wire exchanges. No further paid calls were made during analysis.

Verification after execution: **145 tests passed, 7 skipped**. Compilation, dependency consistency, patch whitespace, independent-source exclusions, all recorded result schemas, selected-model grounding, immutable source snapshots and exact token reconciliation passed. Verification sent no additional API request.
