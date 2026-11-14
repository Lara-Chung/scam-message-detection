"""Cost-based review thresholds for the neural models (proposal section 5.5).

Reads the thresholds that analysis/neural/scripts/lime_threshold_report.py selected on
split_seed0 validation predictions and lets the user pick one in the sidebar of the
RNN and Transformer pages. Each threshold minimises

    Cost = C_FN * p * FNR + (1 - p) * FPR

where p is the assumed share of scam messages and C_FN is the cost of one missed
scam relative to one false alarm.

Usage:
    from app.thresholds import threshold_controls, apply_threshold
    threshold = threshold_controls(model_id, channel, key)    # inside the page layout
    rows = apply_threshold(rows, threshold)                   # re-flag saved results

Outputs: a float threshold (or None for the model's saved default) and, for the
all-channel models, the expected missed scams / false alarms per 10,000 messages on
the selected channel's test split, shown as a caption.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Optional

import pandas as pd
import streamlit as st

from src.project_paths import NEURAL_RESULTS
from app.team_models import CHANNELS

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
REPORT_DIR = NEURAL_RESULTS / 'results_app_lime' / 'results_report'
THRESHOLDS_JSON = REPORT_DIR / 'app_thresholds.json'
THRESHOLDS_CSV = REPORT_DIR / 'app_thresholds.csv'
MODES = ('Model default', 'Cost-based', 'Custom')


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_thresholds() -> dict[str, dict[str, float]]:
    """Read {model_id: {'p=..|C_FN=..': threshold}}; empty if the report is missing."""
    if not THRESHOLDS_JSON.is_file():
        return {}
    return json.loads(THRESHOLDS_JSON.read_text(encoding='utf-8'))['thresholds']


@lru_cache(maxsize=1)
def load_threshold_table() -> pd.DataFrame:
    """Read the per-channel error rates that go with each cost-based threshold."""
    if not THRESHOLDS_CSV.is_file():
        return pd.DataFrame()
    return pd.read_csv(THRESHOLDS_CSV)


def cost_options(model_id: str) -> tuple[list[float], list[int]]:
    """List the scam rates p and miss costs C_FN that have a saved threshold."""
    keys = load_thresholds().get(model_id, {})
    pairs = [dict(part.split('=') for part in key.split('|')) for key in keys]
    rates = sorted({float(pair['p']) for pair in pairs})
    costs = sorted({int(float(pair['C_FN'])) for pair in pairs})
    return rates, costs


def cost_threshold(model_id: str, p: float, c_fn: int) -> Optional[float]:
    """Look up the saved cost-optimal threshold for one (p, C_FN) setting."""
    table = load_thresholds().get(model_id, {})
    for key, value in table.items():
        pair = dict(part.split('=') for part in key.split('|'))
        if float(pair['p']) == p and int(float(pair['C_FN'])) == c_fn:
            return float(value)
    return None


def expected_errors(model_id: str, p: float, c_fn: int, channel: str) -> Optional[tuple[float, float]]:
    """Return expected (missed scams, false alarms) per 10,000 messages on the channel's test split."""
    table = load_threshold_table()
    missed, alarms = f'{channel}_missed_per10k', f'{channel}_false_alarms_per10k'
    if table.empty or missed not in table:
        return None
    row = table[(table['model'] == model_id) & (table['p'] == p) & (table['C_FN'] == c_fn)]
    if row.empty or row[[missed, alarms]].isna().any(axis=None):
        return None
    return float(row[missed].iloc[0]), float(row[alarms].iloc[0])


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------
def threshold_controls(model_id: str, channel: str, key: str) -> Optional[float]:
    """Render the threshold choice and return the threshold to use (None = model default)."""
    has_cost = model_id in load_thresholds()
    modes = MODES if has_cost else (MODES[0], MODES[2])
    mode = st.radio('Review threshold', modes, index=1 if has_cost else 0, horizontal=True,
                    key=f'{key}:threshold-mode')
    if mode == 'Custom':
        return st.slider('Flag threshold', .01, .99, .50, .01, key=f'{key}:threshold')
    if mode == 'Model default':
        st.caption('Uses the threshold saved with the model (0.50).')
        return None

    rates, costs = cost_options(model_id)
    left, right = st.columns(2)
    p = left.selectbox('Assumed scam rate p', rates, index=len(rates) - 1,
                       format_func=lambda value: f'{value:.0%}', key=f'{key}:p')
    c_fn = right.select_slider('Cost of a missed scam (C_FN)', costs,
                               value=10 if 10 in costs else costs[0], key=f'{key}:cfn')
    threshold = cost_threshold(model_id, p, c_fn)
    errors = expected_errors(model_id, p, c_fn, channel)
    note = f'Threshold {threshold:.4f}, chosen on seed 0 validation data to minimise C_FN·p·FNR + (1−p)·FPR.'
    if errors:
        note += (f' Expected on the {CHANNELS.get(channel, channel)} test split at this scam rate: {errors[0]:.1f} missed scams and '
                 f'{errors[1]:.1f} false alarms per 10,000 messages.')
    st.caption(note)
    return threshold


def apply_threshold(rows: list[dict], threshold: Optional[float]) -> list[dict]:
    """Re-flag saved predictions for a new threshold without rerunning the model."""
    if threshold is None:
        return rows
    return [row | {'flagged': row['score'] >= threshold, 'threshold': threshold} for row in rows]
