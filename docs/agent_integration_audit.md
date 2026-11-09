# Agent integration audit

**Current acceptance update:** the budget repair and a subsequent real three-channel
run recorded 6 calls / 9,164 tokens. All channels reached final responses;
email's unique literal quote required a visible coordinate correction and review.
See [agent_budget_fix.md](agent_budget_fix.md) for original versus offline
revalidated evidence, 121 passing tests, and remaining evaluation limits.


Inspected on 10 November 2026 (Asia/Singapore). Repository root:
`/Users/bohanzhang/Projects/scam-message-detection`.
Base commit: `90cd4baa7d2bdc08b061b250de1e7f8909e46749`.
The checkout already had the local `feature/agent-backend` branch. Initial untracked
files were the supplied prompt and `local_inputs/`. No applicable `AGENTS.md` was
found in the repository or ancestor directories. No push, publication or PR was made.

The complete `EE6405_Agent_Codex_Prompt.txt`, repository README, shared data/split/
normalisation/export modules, tests, four training scripts, ZIP README and pipeline
HTML notes were read. Original datasets, training scripts, results and frontend
were preserved. Only `.gitignore`, a README integration section and new Agent
files are needed by this change.

## Local inputs and export

- Archive: `local_inputs/processed_data_export.zip` (eight members: seven gzip CSVs
  and README under `processed_data_export/`).
- Notes: `local_inputs/EE6405_Data_Pipeline_Notes.html`.
- Extracted export: `data/processed/export/`, ignored by Git. The preparation
  script rejects traversal, links, duplicate destinations and oversized members;
  existing differing files are never overwritten.
- Read labels as nullable `Int64`; only empty **label** cells become NA. Empty
  canonical templates are real values: two SMS emoticons and three punctuation-only
  emails have empty templates. They are retained. Templates are not reconstructed.

| Channel / dataset | All rows | Fraud | Legitimate | NA excluded | Eligible train / val / test, seed0 |
|---|---:|---:|---:|---:|---|
| SMS / Mendeley | 5,797 | 528 | 4,834 | 435 | 3,431 / 858 / 1,073 |
| Email / email_fraud | 7,955 | 4,796 | 3,159 | 0 | 5,091 / 1,273 / 1,591 |
| Phone / BothBosu, synthetic | 1,598 | 799 | 399 | 400 | 760 / 199 / 239 |

Mendeley spam and BothBosu insurance/telemarketing are inapplicable, never
legitimate negatives. UCI (5,159), UCI replication (5,574), CEAS_08 (37,949), and
SpamAssassin (5,773) each have zero eligible fraud labels and are not evaluated.
For all three primary datasets and all three seeds, export assignments match the
tracked `data/processed/splits.csv` exactly. Shared group/template/text counts
across partitions are all zero. BothBosu's official test identities stay fixed
across seeds; its seeds are not independent test samples.

Verified archive-file SHA-256 values (hashes refer to compressed bytes):

| Export | SHA-256 |
|---|---|
| mendeley.csv.gz | `b9faca5808e97ce9ce4b8f0c0b2073da501ba50930a8e152b08f54bf0b84fdc9` |
| email_fraud.csv.gz | `45b05974af8348ce1393e35f88770e434f33f77002062c6bb7b170d0140485f6` |
| bothbosu.csv.gz | `3fa5effc94999a220045ee0e029452ca252ff7818fc14d18f5ff37d7660cab6c` |

Reproduce the machine-readable audit with `.venv/bin/python -m scripts.agent_audit`;
full paths, artifact hashes, class orders, warnings and smoke labels are recorded
in ignored `outputs/agent/audit.json`. The audit loads only user-supplied trusted
joblib files. Pickle/joblib is executable serialization, not a format for arbitrary
message-supplied files.

## Legacy model reconciliation

The nested directory is
`models/teammate_legacy/drive-download-20261109T173956Z-1-001/`;
there is also the original downloaded ZIP. It contains 84 joblib files. The four
nested training scripts have byte-identical SHA-256 hashes to the root scripts.
There was no saved training-ID manifest or independently verified invocation
record establishing canonical training membership. Result JSONs provide seeds
42/43/44, rather than canonical 0/1/2.

Loaded the split0 weighted LR **and** word-SVM Pipeline in each of
`outputs_mendeley_*`, `outputs_email_*`, `outputs_bothbosu_*` (six artifacts).
All expose saved TF-IDF preprocessing; all six loaded without version warnings
in the isolated scikit-learn 1.7.2 runtime. Their classifier random_state is 42.
One handwritten smoke input per model was predicted; these are integration
checks, not accuracy evidence.

| Channel | Actual original class order | Legacy reported split0 test size | Canonical eligible test | Reconstructed legacy TRAIN overlap with canonical eligible TEST |
|---|---|---:|---:|---:|
| SMS | ham, smishing, spam | 1,173 | 1,073 | 878 / 1,073 |
| Email | fraud, legitimate | 1,590 | 1,591 | 1,293 / 1,591 |
| Phone | non-scam, scam | 320 | 239 | 203 / 239 |

The reconstruction calls the supplied LR script's actual `template_key()` and
`grouped_split()` on matching processed text and original labels, seed 42. These
are **reconstructed memberships**, not recovered saved train IDs. Nevertheless,
they disqualify these artifacts from the canonical held-out experiment. Phone
canonical test has 319 total rows, 239 after exclusions; the old result's 320 rows
also show that the protocols differ. No legacy benchmark numbers were reused.

Explicit demo mappings are ham -> legitimate, smishing -> fraud, spam -> abstain;
email fraud -> fraud, legitimate -> legitimate; phone scam -> fraud, non-scam ->
legitimate (legacy phone training eligibility remains unverified). Legacy loading
is integration-only and results require review. Retrieval and held-out evaluation
reject unverified artifacts.

## Dependency reconciliation and repaired evaluation artifacts

The repository pins scikit-learn 1.9.1; the notes describe Python 3.11 / pandas
3.1.0 / sklearn 1.9.1. The supplied legacy serialization was reported as 1.7.2;
loading these six artifacts under 1.7.2 produced no mismatch warnings. A matching
sklearn version does not recover unrecorded original numpy/scipy versions or
prove numerical equivalence. Cross-version equivalence is not assumed (see
[scikit-learn persistence documentation](https://scikit-learn.org/stable/model_persistence.html)).

An isolated `.venv` uses Python 3.12.15, scikit-learn 1.7.2, pandas 2.3.3,
numpy 2.5.4, scipy 1.18.1 and joblib 1.6.0. See `requirements-agent.txt`.
The team's requirements were not edited. Delivered split columns are reused;
canonical splits are never regenerated under this different runtime.

Because no verified corrected teammate artifacts were available, the authorized
minimal repair created `models/agent_seed0/{sms,email,phone}.joblib`. Each is a
TF-IDF (word 1–2 grams, max_features 50,000, sublinear_tf) + balanced LR
(C=1, max_iter=1000, random_state=0), fitted **only** on canonical seed0 eligible
TRAIN. Settings were fixed before test evaluation; there was no test tuning,
validation fitting or train+val refit. No neural model was trained.

Each `.metadata.json` records train IDs, sorted-membership hash, per-row text
hashes, export hash, artifact hash, class order, task definition, versions,
hyperparameters and selection policy. Runtime loading checks these against the
local canonical export and exact numerical dependency versions. This is local
reproducibility/provenance evidence produced by our training code, not an
independent certification of third-party artifacts.

## Integration boundaries

The adapter reads actual class order. In particular, email fraud is column zero
in the legacy artifacts. LR probabilities are named
`model_probability_uncalibrated`; SVM scores are `decision_margin`, with binary
margin negated when fraud is the first class. Neither is an LLM confidence.
Models are cached; inference never fits a model/vectorizer.

Normalisation wraps `normalise_with_changes()` unchanged. It is English-oriented,
preserves prices/numbers, and does not undo `acc0unt` or `v.e.r.i.f.y`. Indicator
rules return observed spans, not verified malicious domains or impersonation.
Retrieval uses only the same channel's canonical seed0 TRAIN and the saved
TRAIN-fitted TF-IDF vectorizer. No cross-channel union, other-seed union, validation
or test indexing is enabled. Cross-channel transfer is not claimed; adding it
requires `remove_overlap()` against the target held-out texts/templates before
any model/index fitting.

The only provider implementation is DeepSeek non-thinking Chat Completions,
following its [tool calling protocol](https://api-docs.deepseek.com/guides/tool_calls/)
and [API schema](https://api-docs.deepseek.com/api/create-chat-completion/).
Assistant-generated function calls are followed by matching `role=tool` messages.
The chosen model must be supplied through `LLM_MODEL`; no model ID is invented or
selected automatically. All three LLM environment variables were absent during
this work. Real provider/tool-calling compatibility and B/C/D accuracy remain
pending. Fixtures and HTTP mock transport prove mechanics only.

## Subsequent local API verification

The initial missing-configuration finding above is historical. The project now
has an ignored local Keychain profile and a real three-message validation run:
5 actual DeepSeek calls, 6,228 reported tokens, one successful final answer and
two locally enforced budget stops. See
[the full live report](agent_live_validation.md). No legacy artifact policy,
canonical split, model, frozen prompt or test result was changed for this run.
