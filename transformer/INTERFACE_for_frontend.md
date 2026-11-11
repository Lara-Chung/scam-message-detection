# Transformer prediction interface (for the frontend / agent)

## 1. Files needed

```
transformer/
├── models/distilbert_mendeley_3class_v1/   # model (about 270 MB, shared separately, not in git)
└── inference/predict.py                    # prediction interface
```

Environment: Python ≥ 3.10, `pip install torch "transformers==5.16.1" pandas` (plus streamlit for the frontend).
A CPU is enough, no GPU needed; on an M-series MacBook one message takes about 20 ms (the first load takes about 1–2 s).
The model folder defaults to `inference/../models`; set the environment variable `MODELS_DIR` if it is elsewhere.

## 2. How to call it

```python
import sys; sys.path.insert(0, "inference")
from predict import predict, predict_batch

r = predict("URGENT: your bank account has been suspended...", "distilbert_mendeley_3class_v1")
rs = predict_batch(["msg1", "msg2"], "distilbert_mendeley_3class_v1")
r2 = predict(text, "distilbert_mendeley_3class_v1", threshold=0.3)  # manual threshold for the demo
```

The model is loaded on the first call and then kept in memory.

## 3. Returned fields

| Field | Meaning |
|---|---|
| status | "ok" or "error" |
| predicted_label | the highest-scoring of ham / spam / smishing |
| class_scores | scores of the three classes, summing to 1 |
| positive_labels | ["smishing"]: only smishing counts as fraud |
| risk_score | the smishing score |
| threshold | threshold used for this call (default 0.5, chosen on the validation set) |
| flagged | risk_score ≥ threshold |
| truncated | the input was longer than 256 tokens and was cut |
| inference_ms | time for tokenization + forward pass (ms), without model loading; the frontend measures end-to-end time itself |
| error_code / message | only on error: empty_input, model_not_found |

## 4. Display rules

- **spam = marketing spam, not fraud.** Do not show spam as "scam", and do not show every non-smishing message as "safe". Suggested wording:
  - flagged = true → "Suspected scam message (smishing)"
  - predicted_label = spam and not flagged → "Marketing / spam message"
  - predicted_label = ham and not flagged → "Nothing suspicious found"
- risk_score is the model's score, not a guaranteed probability that the message is a scam.
- This model was trained only on English SMS (Mendeley); email and phone texts are outside its training data.
- When status = "error", show the message, not scores.
- A threshold changed by hand only affects the demo; the report uses 0.5 (and the cost-based thresholds of section 5.5).

## 5. Minimal Streamlit example

```python
import sys; sys.path.insert(0, "inference")
import streamlit as st
from predict import predict

MODEL = "distilbert_mendeley_3class_v1"
text = st.text_area("Paste an English SMS")
thr = st.slider("Scam threshold", 0.05, 0.95, 0.5, 0.05)
if st.button("Check"):
    r = predict(text, MODEL, threshold=thr)
    if r["status"] != "ok":
        st.error(r["message"])
    else:
        if r["flagged"]:
            st.error(f"Suspected scam message (smishing), risk score {r['risk_score']:.2f}")
        elif r["predicted_label"] == "spam":
            st.warning("Marketing / spam message (not a scam)")
        else:
            st.success("Nothing suspicious found")
        st.bar_chart(r["class_scores"])
        if r["truncated"]:
            st.info("Text too long: only the first 256 tokens were analysed")
        st.caption(f"Model inference {r['inference_ms']} ms")
```

## 6. Test results (Mendeley test set, mean of 3 split seeds)

macro-F1 0.887 ± 0.011, accuracy 0.969; smishing recall about 0.89, false alarm rate about 1.4%.
Ham is never predicted as smishing; the errors are between spam and smishing.
