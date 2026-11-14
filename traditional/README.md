# Traditional classifiers

Training entry points for the team's word/character TF-IDF models:

| File | Model |
|---|---|
| `lr_scam.py` | Logistic regression |
| `nb_scam.py` | Multinomial naive Bayes |
| `svm_word_scam.py` | Word TF-IDF + linear SVM |
| `svm_char_scam.py` | Character TF-IDF + linear SVM |

Run from the repository root:

```sh
python traditional/lr_scam.py --help
python traditional/nb_scam.py --help
python traditional/svm_word_scam.py --help
python traditional/svm_char_scam.py --help
```

These are the original training implementations moved from `baselines/`. Their
model settings and split procedures are unchanged. The older `baselines/` entry
points and root `lr_scam.py` forward to these files for compatibility.

The frontend's saved conventional-model reports remain under [`artifacts/`](../artifacts).
Canonical models used by Agent routing are fitted by
`python -m src.agent.candidates`, with configuration and provenance checks shared
with the Agent component. Prepared model weights remain ignored under `models/`.
