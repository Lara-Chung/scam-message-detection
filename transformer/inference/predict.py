"""Prediction interface for the Transformer models (prediction protocol v0.1, Task Brief section 7).

Loads a fine-tuned DistilBERT model by model_id and returns one result dict per text
with the predicted label, class scores, fraud risk score and flag, used by the
Streamlit frontend (proposal 4.6).

Model folder: MODELS_DIR environment variable, default ../models next to this file.
Each model has its own sub-folder named after its model_id, containing config.json,
model.safetensors, the tokenizer files and task_config.json.

inference_ms covers tokenization + forward pass + softmax per message (model loading
is not counted; on GPU the device is synchronised before timing). End-to-end latency
is measured separately by the frontend.

Usage:
    from predict import predict, predict_batch
    predict("Your account is locked, verify at http://...", "distilbert_mendeley_3class_v1")
    python predict.py [model_id]      # small self-test

Outputs: dicts with status, model_id, task_id, predicted_label, class_scores,
positive_labels, risk_score, threshold, flagged, truncated, inference_ms; or, on
error, status="error" with error_code (empty_input / model_not_found) and message.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_loaded: dict[str, tuple[Any, Any, dict[str, Any], list[str]]] = {}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _error(code: str, message: str, model_id: Optional[str] = None) -> dict[str, Any]:
    """Error result in the protocol format."""
    return {"status": "error", "error_code": code, "message": message, "model_id": model_id}


def _load(model_id: str) -> Optional[tuple[Any, Any, dict[str, Any], list[str]]]:
    """Load (tokenizer, model, task_config, labels) once and cache it; None if the model is missing."""
    if model_id in _loaded:
        return _loaded[model_id]
    d = MODELS_DIR / model_id
    if not (d / "task_config.json").exists():
        return None
    cfg = json.loads((d / "task_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(d)
    model = AutoModelForSequenceClassification.from_pretrained(d).to(DEVICE).eval()
    # Label order comes from the model's own saved id2label, not from assumed indices
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    _loaded[model_id] = (tok, model, cfg, labels)
    return _loaded[model_id]


def _sync() -> None:
    """Wait for queued GPU work so timings are accurate (no-op on CPU)."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()


# --------------------------------------------------------------------------
# Public interface
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_batch(texts: list[str], model_id: str, threshold: Optional[float] = None,
                  batch_size: int = 32) -> list[dict[str, Any]]:
    """Predict a list of texts; returns one result dict per text, in the same order.

    threshold overrides the model's default_threshold from task_config.json.
    """
    m = _load(model_id)
    if m is None:
        return [_error("model_not_found", f"model {model_id} not found (folder {MODELS_DIR})", model_id)] * len(texts)
    tok, model, cfg, labels = m
    max_len = cfg.get("max_length", 256)
    pos = cfg.get("positive_labels") or []
    thr = cfg.get("default_threshold") if threshold is None else threshold

    results: list[Optional[dict[str, Any]]] = [None] * len(texts)
    valid = []
    for i, t in enumerate(texts):
        if not isinstance(t, str) or not t.strip():
            results[i] = _error("empty_input", "input text is empty", model_id)
        else:
            valid.append(i)

    for s in range(0, len(valid), batch_size):
        idx = valid[s:s + batch_size]
        batch = [texts[i] for i in idx]
        full_lens = [len(ids) for ids in tok(batch, truncation=False)["input_ids"]]
        _sync()
        t0 = time.perf_counter()
        enc = tok(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(DEVICE)
        probs = torch.softmax(model(**enc).logits.float(), dim=-1).cpu().tolist()
        _sync()
        ms = (time.perf_counter() - t0) * 1000 / len(idx)

        for i, p, n in zip(idx, probs, full_lens):
            scores = {l: round(v, 6) for l, v in zip(labels, p)}
            risk = round(sum(scores[l] for l in pos), 6) if pos else None
            results[i] = {
                "status": "ok",
                "model_id": model_id,
                "task_id": cfg.get("task_id"),
                "predicted_label": max(scores, key=scores.get),
                "class_scores": scores,
                "positive_labels": pos,
                "risk_score": risk,
                "threshold": thr if risk is not None else None,
                "flagged": (risk >= thr) if (risk is not None and thr is not None) else None,
                "truncated": n > max_len,
                "inference_ms": round(ms, 2),
            }
    return results


def predict(text: str, model_id: str, threshold: Optional[float] = None) -> dict[str, Any]:
    """Predict one text; returns one result dict (fields in Task Brief section 7)."""
    return predict_batch([text], model_id, threshold)[0]


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    mid = sys.argv[1] if len(sys.argv) > 1 else "distilbert_mendeley_3class_v1"
    tests = [
        "Hey, are we still meeting for lunch tomorrow?",
        "Congratulations! You've won a free cruise. Reply YES to claim, std rates apply.",
        "URGENT: your bank account has been suspended. Verify now at http://secure-bank-login.xyz",
        "",
        "please " * 400,
    ]
    predict(tests[0], mid)  # warm-up so the first timing does not include initialisation
    for t in tests:
        print(repr(t[:60]), "->", json.dumps(predict(t, mid), ensure_ascii=False))
    print(json.dumps(predict("hi", "no_such_model"), ensure_ascii=False))
