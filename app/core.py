"""Inference, artifact access and demonstration logic; no model fitting occurs here."""
from __future__ import annotations

import html
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(relative: str) -> Path:
    """Resolve a project-relative path without allowing directory escapes."""
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError('Artifact paths must stay inside the project directory.')
    return path


def registry() -> list[dict]:
    """Read the team's editable model manifest."""
    return json.loads((ROOT / 'app/models.json').read_text())['models']


def results_for(spec: dict) -> dict:
    """Read original exported metrics, preserving their split and class order."""
    return json.loads((resolve_path(spec['output_dir']) / spec['results_file']).read_text())


def model_path(spec: dict, split: int) -> Path:
    """Resolve the fitted pipeline for an explicitly selected run."""
    return resolve_path(spec['output_dir']) / spec['model_pattern'].format(split=split)


def load_pipeline(path: str) -> Any:
    """Load a team-supplied joblib pipeline from disk, never from a web upload."""
    return joblib.load(path)


def input_text(text: str, spec: dict) -> str:
    """Match the HTML unescaping used by src.data.clean on incoming user text."""
    return html.unescape(text) if spec.get('preprocess') == 'html_unescape' else text


def predict_batch(model: Any, texts: list[str], positive_label: str) -> tuple[np.ndarray, float]:
    """Return actual probabilities and batch wall time; never turn SVM margins into probabilities."""
    if not hasattr(model, 'predict_proba'):
        raise ValueError('This artifact exposes decision scores only. Provide a probability-calibrated model to use thresholds.')
    classes = list(model.classes_)
    if positive_label not in classes:
        raise ValueError(f'Configured positive label {positive_label!r} is absent from {classes}.')
    start = time.perf_counter()
    probabilities = np.asarray(model.predict_proba(texts), dtype=float)
    elapsed_ms = (time.perf_counter() - start) * 1000
    if probabilities.shape != (len(texts), len(classes)) or not np.isfinite(probabilities).all():
        raise ValueError('Model returned invalid probability values.')
    return probabilities[:, classes.index(positive_label)], elapsed_ms


def grouped_test_rows(data: pd.DataFrame, seed: int, spec: dict) -> pd.DataFrame:
    """Reproduce lr_scam.py's grouped_split, NOT the independent split_seed columns."""
    if spec.get('protocol') != 'lr_grouped_v1':
        raise ValueError('Test-set reconstruction is not configured for this protocol.')
    text_col, label_col = spec['text_col'], spec['label_col']
    frame = data.dropna(subset=[text_col, label_col]).copy().reset_index(drop=True)
    text = frame[text_col].astype(str)
    labels = frame[label_col].astype(str)
    group = text.str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    group = group.str.replace(r'https?://\S+|www\.\S+', ' URL ', regex=True)
    group = group.str.replace(r'\d+', '0', regex=True)
    for offset in range(50):
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed + offset * 1000)
        train, test = next(splitter.split(frame, labels, groups=group))
        if set(labels.iloc[train]) == set(labels.iloc[test]) == set(labels):
            return frame.iloc[test].copy()
    raise ValueError('No valid template-grouped test split found.')


def verified_test_predictions(model: Any, data: pd.DataFrame, spec: dict, run: dict) -> pd.DataFrame:
    """Expose held-out errors only if reconstructed predictions match the saved matrix."""
    frame = grouped_test_rows(data, run['split_seed'], spec)
    frame['actual'] = frame[spec['label_col']].astype(str)
    # Training files already contain processed text; do not reprocess these rows.
    frame['predicted'] = model.predict(frame[spec['text_col']].astype(str))
    matrix = confusion_matrix(frame['actual'], frame['predicted'], labels=run['classes'])
    if matrix.tolist() != run['confusion_matrix']:
        raise ValueError('Reconstructed confusion matrix differs from the export. Error samples are withheld; confirm the training CSV and protocol.')
    frame['positive_score'], _ = predict_batch(model, frame[spec['text_col']].astype(str).tolist(), spec['positive_label'])
    return frame


SUBSTITUTIONS = str.maketrans({'a': '4', 'e': '3', 'i': '1', 'o': '0', 's': '5', 'A': '4', 'E': '3', 'I': '1', 'O': '0', 'S': '5'})
HOMOGLYPHS = str.maketrans({'a': 'а', 'e': 'е', 'o': 'о', 'c': 'с', 'p': 'р', 'x': 'х', 'i': 'і'})


def perturb(text: str, kinds: list[str], seed: int = 42, fraction: float = .3) -> tuple[str, list[dict]]:
    """Perturb a reproducible fraction of eligible words."""
    rng = random.Random(seed)
    changes: list[dict] = []
    for kind in kinds:
        matches = [m for m in re.finditer(r'\b[A-Za-z]{4,}\b', text)
                   if m.group().lower() not in ENGLISH_STOP_WORDS]
        if kind == 'P1':
            matches = [m for m in matches if m.group().translate(SUBSTITUTIONS) != m.group()]
        if kind == 'P3 look-alikes':
            matches = [m for m in matches if m.group().translate(HOMOGLYPHS) != m.group()]
        selected = rng.sample(matches, min(len(matches), math.ceil(len(matches) * fraction)))
        for match in sorted(selected, key=lambda m: m.start(), reverse=True):
            word = match.group()
            if kind == 'P1':
                replacement = word.translate(SUBSTITUTIONS)
            elif kind == 'P2 spaces':
                replacement = ' '.join(word)
            elif kind == 'P2 dots':
                replacement = '.'.join(word)
            elif kind == 'P3 look-alikes':
                replacement = word.translate(HOMOGLYPHS)
            elif kind == 'P3 zero-width':
                replacement = word[:1] + '\u200b' + word[1:]
            else:
                raise ValueError(f'Unknown perturbation {kind}.')
            text = text[:match.start()] + replacement + text[match.end():]
            changes.append({'type': kind, 'original': word, 'changed': replacement})
    return text, changes


FIELD_PATTERNS = {
    'URL': r'\b(?:https?://|www\.)[^\s<>]+|\b[\w-]+(?:\.[\w-]+)*\.(?:com|net|org|info|biz|io|ly|uk)\b(?:/[^\s<>]*)?',
    'Email address': r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b',
    'Phone number': r'(?<!\w)\+?\d[\d ()-]{5,}\d(?!\w)',
    'Amount': r'(?:[£$€]\s?\d[\d,.]*|\b(?:SGD|USD|GBP)\s?\d[\d,.]*)',
    'Deadline / urgency': r'\b(?:within\s+\d+\s+(?:hours?|days?|minutes?)|immediately|urgent|today|now|as soon as possible)\b',
    'Information mentioned': r'\b(?:one[- ]time password|social security number|card number|bank account details|gift cards?|password|PIN|OTP)\b',
}


def extract_indicators(text: str, nlp: Any = None) -> list[dict]:
    """Extract candidate mentions, not verified scam facts or inferred requests."""
    spans: list[dict] = []
    for field, pattern in FIELD_PATTERNS.items():
        for match in re.finditer(pattern, text, flags=re.I):
            if field == 'Phone number' and not 7 <= len(re.sub(r'\D', '', match.group())) <= 15:
                continue
            if field == 'URL' and match.start() > 0 and text[match.start() - 1] == '@':
                continue
            spans.append({'field': field, 'value': match.group(), 'start': match.start(), 'end': match.end(), 'source': 'Regex'})
    if nlp is not None:
        fields = {'ORG': 'Organisation mentioned', 'PERSON': 'Person mentioned', 'MONEY': 'Amount', 'DATE': 'Date mentioned', 'TIME': 'Time mentioned'}
        for ent in nlp(text).ents:
            if ent.label_ in fields:
                spans.append({'field': fields[ent.label_], 'value': ent.text, 'start': ent.start_char, 'end': ent.end_char, 'source': 'spaCy NER'})
    return spans


def highlighted_html(text: str, spans: list[dict]) -> str:
    """Escape user text and highlight non-overlapping candidate spans safely."""
    chunks, end = [], 0
    for span in sorted(spans, key=lambda x: (x['start'], -x['end'])):
        if span['start'] < end:
            continue
        chunks.append(html.escape(text[end:span['start']]))
        chunks.append('<mark>' + html.escape(text[span['start']:span['end']]) + '</mark>')
        end = span['end']
    chunks.append(html.escape(text[end:]))
    return '<div class="message-box">' + ''.join(chunks).replace('\n', '<br>') + '</div>'


def lime_explanation(model: Any, text: str, positive_label: str) -> dict:
    """Run 500-sample LIME only on explicit request; report local surrogate fidelity."""
    from lime.lime_text import LimeTextExplainer
    index = list(model.classes_).index(positive_label)
    start = time.perf_counter()
    explainer = LimeTextExplainer(class_names=list(map(str, model.classes_)), random_state=42)
    explanation = explainer.explain_instance(text, model.predict_proba, labels=(index,), num_features=10, num_samples=500)
    weights = explanation.as_list(label=index)
    return {'weights': weights, 'html': weighted_html(text, weights),
            'r2': float(explanation.score), 'seconds': time.perf_counter() - start}


def weighted_html(text: str, weights: list) -> str:
    """Render signed token weights and escape the source text."""
    token_weights = {token.casefold(): weight for token, weight in weights}
    pieces = []
    for part in re.split(r'(\W+)', text):
        safe = html.escape(part)
        weight = token_weights.get(part.casefold(), 0)
        if weight:
            colour = '#fce0dd' if weight > 0 else '#d9efeb'
            safe = f'<span style="background:{colour};padding:2px" title="Score contribution {weight:+.4f}">{safe}</span>'
        pieces.append(safe)
    return '<div class="message-box">' + ''.join(pieces).replace('\n', '<br>') + '</div>'
