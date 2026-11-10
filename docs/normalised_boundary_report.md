# Normalised routing with mandatory boundary cross-checks

Date: 10 November 2026, Asia/Singapore. This is a small, deliberately boundary-conditioned independent TEST challenge.

**No incremental classification gain was observed. All three policies scored 10/12 (83.3%), answered macro-F1 0.8125, fraud recall 0.60, and zero false positives.** All 36 analyses completed, without technical failures or abstentions. The rule and Agent guards made eight and six secondary calls respectively. Each routed one disagreement for review, retaining the already correct primary result. Neither exposed the observed false negatives. The Agent used 24 paid requests and 55,177 tokens; the local rule router reached the same labels without API calls.

## What was implemented

All three policies run the same existing English normaliser before every classifier. The fixed baseline runs one VAL-selected model. The rule and Agent routers must run a distinct, VAL-selected second classifier whenever their primary is within its own native-score boundary threshold. The Agent makes a genuine function call to choose exactly one primary and receives every actual classifier result in one tool reply. The backend, rather than an optional LLM decision, enforces the second call and conflict policy. At most two classifiers and three LLM turns are permitted. This hybrid Agent delegates primary selection; it is not an unconstrained model ensemble.

Schema 1.2 records the original/normalised views, preprocessing changes, primary distance/threshold, actual secondary execution, proposed versus enforced selection and trace source `boundary_policy`. A mandatory local call is never presented as a model-generated call. Unsupported conflicts abstain and request review; supported winners still require review. Classifier outputs remain the source of final labels.

## VAL-only calibration and TEST isolation

The same 12 eligible saved candidates use identical canonical seed0 TRAIN memberships within each channel. No new model/vectorizer fitting occurred. Calibration used all 2,330 VAL bases, each clean plus four character perturbation families at two severities (20,970 correlated views). Original source prefixes are capped at 4,000 characters for every method, then perturbed where specified and normalised. Stress edits preserve digits and protected URL/email/amount spans. Inherited labels assume intended-meaning preservation.

Fixed models maximize mean per-condition VAL macro-F1. Profile rules require at least 30 distinct VAL bases and three of each class. For each primary, choose a threshold quantile from 5%, 10%, 20%, 30% and a secondary using `(correct - 2*wrong - .25*abstain - .02*second_calls)/N`. Distances are abs(SVM margin) or abs(uncalibrated fraud probability minus .5); their numeric scales are model-specific, not calibrated risk and not compared between models. A conflict direction requires 10 distinct base IDs and at least 60% winner share; otherwise abstain. Correlated variants from one source count as one conflict vote. These constants and candidate quantiles were locked before TEST. Apparent VAL results are selection statistics, not an independent gain estimate.

| Channel | Normalised fixed | VAL mean macro-F1 | Primary boundary threshold | Secondary | VAL quantile |
|---|---|---:|---:|---|---:|
| sms | char_svm | 0.993402 | 0.626964 | word_nb | 5% |
| email | char_svm | 0.994262 | 0.534756 | word_svm | 5% |
| phone | word_lr | 1.000000 | 0.343926 | word_nb | 5% |

Exclude all 114 previously used TEST base IDs. The selector receives only IDs/text and normalised native distances, never true labels. Select one clean near-boundary input, one clean/perturbed near-boundary pair and one random control per channel, by deterministic hash seed 6407. Choose the perturbation condition on VAL candidate-disagreement rate after normalisation; no harder condition was chosen on TEST. Only after source IDs were fixed were labels joined for scoring. All channels had eligible near-boundary candidates, with no fallback. Freeze design, catalog, manifest, config and source hashes before paid calls.

The resulting 12 inputs have nine independent source messages, 7 legitimate and 5 fraud views. Four phone views are synthetic. The base-prefix policy clipped 3 selected observations. Paired views are correlated and boundary selection deliberately changes the distribution. This is not general population accuracy.

## Measured outcomes

| Policy | Correct / total | Coverage | Answered macro-F1 | Fraud recall, all fraud inputs | FPR, answered | Review | Abstain | Technical failure |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Normalisation + fixed model | 10/12 | 100.0% | 0.8125 | 0.6000 | 0.0000 | 0 | 0 | 0 |
| Rule router + boundary guard | 10/12 | 100.0% | 0.8125 | 0.6000 | 0.0000 | 1 | 0 | 0 |
| Agent router + boundary guard | 10/12 | 100.0% | 0.8125 | 0.6000 | 0.0000 | 1 | 0 | 0 |

Correct/total keeps abstentions, technical failures and pending inputs in the denominator. Answered-case F1/FPR must be read with coverage; abstention is not counted as correct. Confusion matrices, precision/recall, per-channel and per-stratum metrics are retained in analysis.json.

| Stratum | n | Fixed correct | Rule correct | Agent correct |
|---|---:|---:|---:|---:|
| clean_near_boundary | 3 | 3 | 3 | 3 |
| paired_clean | 3 | 2 | 2 | 2 |
| paired_obfuscated | 3 | 2 | 2 | 2 |
| random_control | 3 | 3 | 3 | 3 |

| Policy | Near primaries | Second classifiers | Conflicts | Primary errors exposed | Unreviewed wrong | Mean latency | API requests / tokens |
|---|---:|---:|---:|---:|---:|---:|---|
| Normalisation + fixed model | 8 | 0 | 0 | 0 | 2 | 0.0067s | 0 / 0 |
| Rule router + boundary guard | 8 | 8 | 1 | 0 | 2 | 0.0055s | 0 / 0 |
| Agent router + boundary guard | 6 | 6 | 1 | 0 | 2 | 2.0320s | 24 / 55,177 |

Near-boundary status alone is not an error label. Agreement between two classifiers does not establish correctness. The fixed baseline records distance for audit but deliberately does not cross-check. Every completed rule/Agent analysis with a near-boundary primary was verified to execute its mandatory second classifier.

| Input | Truth | Fixed | Rule | Agent | Agent actual calls |
|---|---|---|---|---|---|
| mendeley_4596::clean_near_boundary | fraud | fraud | fraud (review) | fraud (review) | char_svm, word_nb |
| mendeley_562::paired_clean | legitimate | legitimate | legitimate | legitimate | char_svm, word_nb |
| mendeley_562::paired_obfuscated | legitimate | legitimate | legitimate | legitimate | char_svm, word_nb |
| mendeley_2754::random_control | legitimate | legitimate | legitimate | legitimate | char_svm |
| nazario_5_2332::clean_near_boundary | fraud | fraud | fraud | fraud | char_svm |
| nazario_5_2033::paired_clean | fraud | legitimate | legitimate | legitimate | char_svm, word_svm |
| nazario_5_2033::paired_obfuscated | fraud | legitimate | legitimate | legitimate | char_svm, word_svm |
| nigerian_5_5435::random_control | legitimate | legitimate | legitimate | legitimate | char_svm |
| bothbosu_test_186::clean_near_boundary | legitimate | legitimate | legitimate | legitimate | word_svm, word_lr |
| bothbosu_test_180::paired_clean | legitimate | legitimate | legitimate | legitimate | word_svm |
| bothbosu_test_180::paired_obfuscated | legitimate | legitimate | legitimate | legitimate | word_svm |
| bothbosu_test_59::random_control | fraud | fraud | fraud | fraud | word_svm |

## Failure and complementary-error analysis

The two incorrect observations are the clean and perturbed views of the same fraud-labelled email, `nazario_5_2033`; they are one independent source failure. Its source has 1,025,016 characters. Under the common, predeclared 4,000-character prefix policy, the observed input is dominated by a long recipient address list. Later content is unavailable to every policy. This clipping limitation is a plausible cause of the shared failure; the experiment does not establish the full-message outcome. A post-run local diagnostic ran all four normalised candidates on the same bounded inputs: every candidate predicted legitimate on both views. Thus this candidate set and input policy offer no routing rescue for that source. The Agent actually cross-checked char SVM with word SVM on both views, obtained agreement, and did not request review. The rule router used word NB alone on the clean view and both SVMs on the perturbed view, with the same missed-fraud label.

The only observed disagreement is `mendeley_4596`: char SVM correctly predicts fraud while word NB predicts legitimate. The locked VAL conflict policy retains char SVM and requests review. Post-run local diagnostics show word LR and word SVM also predict legitimate; a majority vote would be wrong. The guard adds review work on a correct fixed-model decision, not a classification improvement. These post-run diagnostics did not change policies, prompts, samples or paid results. They are saved separately in candidate_diagnostics.jsonl.

Per-channel correct counts are SMS 4/4, email 2/4 and synthetic phone 4/4 for all three methods. All three clean/perturbed pairs have identical outcomes within each policy. This new set differs from earlier experiments; 10/12 here versus 11/12 previously is not a measured regression caused by normalisation.

## Cost and evidence

Actual API usage: 24 requests, 54,341 prompt + 836 completion = 55,177 tokens. Cache accounting: 32,384 hit / 21,957 miss. Estimated total API price: **$0.00389230–$0.00778460 USD**, using [official off-peak/peak list rates](https://api-docs.deepseek.com/quick_start/pricing/) checked 10 November 2026. This is not a billing receipt. Local CPU, training and storage costs are excluded. Fixed/rule policies make zero API requests. Warm-service timings include routing, local inference and provider round trips.

The frozen ceiling was 75,000 tokens, 36 requests, 12 paid analyses, 350 output tokens/request, no retries or post-result policy changes. All filtered wire usage reconciles with per-result usage and the global ledger; outstanding reservations are zero and accounting is complete. Keys remain in macOS Keychain and are excluded from outputs. Analysis/verification sends no additional API requests.

Local ignored evidence: outputs/agent/normalised_boundary_v1/{validation_design.json,validation_predictions.jsonl,manifest.json,protocol.json,frozen_source/,run_config.yaml,results/,api_audit/,analysis.json}. The runtime catalog is models/router_normalised_seed0/catalog.json; original candidate models and teammate artifacts are preserved.

```sh
.venv/bin/python -m scripts.prepare_boundary_router validation
.venv/bin/python -m scripts.prepare_boundary_router test
.venv/bin/python -m scripts.run_boundary_router
.venv/bin/python -m scripts.report_boundary_router
```

Preparation and paid execution refuse existing experiment/results paths. The report command only reads saved results and refreshes the derived report. Reproduction needs the ignored canonical export, candidate artifacts and explicit provider credentials for the paid step.

## Interpretation

Compare the observed counts above without treating review/abstention as a correct fraud decision. All three methods share normalisation, so this experiment measures incremental cross-check/routing behavior; it cannot isolate normalisation gain against a raw-input baseline. Nine independent source messages, one provider run, synthetic phone data and deliberate boundary conditioning cannot establish statistical superiority or real-traffic recall. The Agent has less freedom than earlier experiments: its primary choice is LLM-generated, but secondary execution and conflict handling are deterministic. Any gain shared by the rule router is not evidence that an LLM is needed. The present evidence supports the normalised fixed model as the economical baseline and the local rule guard as an optional review mechanism; it does not support paying for an LLM on every message.

## Verification

The full offline suite passed: 154 tests, seven original raw-data-dependent skips. Dependency consistency and compilation passed. Every recorded result validates against schema.json. Source snapshots, config/catalog/input hashes, prior-source exclusions, actual normalised views, mandatory secondary execution, selected-model grounding and exact usage reconciliation were checked. Offline fixtures exercise unsupported-conflict abstention and prevention of an LLM overriding the locked winner; no unsupported-conflict abstention occurred in this live sample. No extra paid verification or retries were sent.
