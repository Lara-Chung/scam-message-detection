"""Readable execution details for an Agent response (frontend only, no backend changes).

The Agent backend (src/agent) returns a JSON response with classifier outputs, a tool
trace, routing and triage records. The demo used to print these as raw collapsed JSON.
This module turns them into short tables a viewer can read, and keeps the raw response
in a collapsed debug section for troubleshooting.

Usage:
    from app.agent_details import show_execution_details
    show_execution_details(result)          # inside the 'Execution details' expander

Outputs: Streamlit tables for run summary, classifier outputs, tool steps, boundary
checks and similar training cases, plus the raw JSON under 'Raw response (debug)'.
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_NAMES = {
    'char_svm': 'Character TF-IDF + linear SVM',
    'word_svm': 'Word TF-IDF + linear SVM',
    'word_lr': 'Word TF-IDF + logistic regression',
    'word_nb': 'Word TF-IDF + naive Bayes',
    'agent_lr': 'Logistic regression (Agent baseline)',
}
SCORE_KINDS = {
    'decision_margin': 'SVM margin (>0 = fraud)',
    'model_probability_uncalibrated': 'Probability (uncalibrated)',
}
TOOL_NAMES = {
    'classify_message': 'Classify message',
    'extract_indicators': 'Extract indicators',
    'retrieve_similar_cases': 'Retrieve similar training cases',
    'inspect_section': 'Read document section',
    'normalise_message': 'Normalise text (D1)',
    'request_clarification': 'Ask a clarification question',
}


# --------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------
def model_name(model_id: str | None) -> str:
    """Turn a saved model id such as router_sms_char_svm_seed0_v1 into a readable name."""
    if not model_id:
        return '—'
    for key, name in MODEL_NAMES.items():
        if key in model_id:
            return name
    return model_id


def summarise_tool_result(tool: str, result: Any) -> str:
    """Describe one tool result in a single line."""
    if result is None:
        return '—'
    if isinstance(result, dict) and 'decision' in result:
        score = result.get('fraud_score')
        score_text = f', score {score:.3f}' if isinstance(score, (int, float)) else ''
        return f"{model_name(result.get('model_id'))} → {result['decision']}{score_text}"
    if isinstance(result, dict) and 'assessment' in result:
        assessment = result['assessment'] or {}
        return (f"Section {result.get('section_id')} (characters {result.get('source_start')}–"
                f"{result.get('source_end')}) → {assessment.get('decision', 'assessed')}")
    if tool == 'extract_indicators' and isinstance(result, list):
        values = [f"{item.get('field', item.get('type', '?'))}: {item.get('value', item.get('text', ''))}"
                  for item in result[:4] if isinstance(item, dict)]
        more = f' (+{len(result) - 4} more)' if len(result) > 4 else ''
        return f'{len(result)} found' + (': ' + '; '.join(values) + more if values else '')
    if isinstance(result, list):
        return f'{len(result)} item(s)'
    if isinstance(result, dict):
        return ', '.join(f'{key}={value}' for key, value in list(result.items())[:3]
                         if not isinstance(value, (dict, list)))
    return str(result)[:120]


def classifier_table(results: list[dict]) -> pd.DataFrame:
    """One row per classifier call: model, text view, decision and score."""
    return pd.DataFrame([{
        'Model': model_name(item.get('model_id')),
        'Text view': str(item.get('view', '')).replace('_', ' '),
        'Decision': item.get('decision'),
        'Fraud score': round(item['fraud_score'], 4) if isinstance(item.get('fraud_score'), (int, float)) else None,
        'Score type': SCORE_KINDS.get(item.get('score_kind'), item.get('score_kind')),
    } for item in results])


def tool_table(trace: list[dict]) -> pd.DataFrame:
    """One row per tool call, in the order the Agent made them."""
    return pd.DataFrame([{
        'Step': index,
        'Action': TOOL_NAMES.get(step.get('tool'), step.get('tool')),
        'Result': step['error'] if step.get('error') else summarise_tool_result(step.get('tool'), step.get('result')),
        'Time (ms)': round(step['elapsed_ms'], 1) if isinstance(step.get('elapsed_ms'), (int, float)) else None,
    } for index, step in enumerate(trace, start=1)])


def show_table(frame: pd.DataFrame, *, hide_index: bool = True) -> None:
    """Static table that wraps long text, so it fits the narrow decision column."""
    frame = frame.fillna('—').astype(str)
    st.table(frame.style.hide(axis='index') if hide_index else frame)


# --------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------
def show_execution_details(result: dict) -> None:
    """Render the Agent's run record as readable tables, with the raw JSON for debugging."""
    routing = result.get('routing') or {}
    triage = result.get('triage') or {}
    usage = result.get('usage') or {}
    summary = {
        'Status': str(result.get('status', '')).replace('_', ' '),
        'LLM calls': result.get('llm_calls', 0),
        'Elapsed time': f"{result['elapsed_ms']:.1f} ms" if result.get('elapsed_ms') is not None else '—',
    }
    if result.get('llm_calls'):
        summary['Tokens used'] = usage.get('total_tokens', '—')
    if cost := result.get('cost_estimate'):
        summary['Estimated cost'] = f"{cost.get('amount', 0):.4f} {cost.get('currency', '')}"
    if triage.get('path'):
        summary['Triage path'] = triage['path']
    st.markdown('**Run summary**')
    show_table(pd.DataFrame(summary.items(), columns=['Item', 'Value']), hide_index=True)

    if results := result.get('classifier_results'):
        st.markdown('**Classifier outputs**')
        show_table(classifier_table(results), hide_index=True)
    if trace := result.get('tool_trace'):
        st.markdown('**Steps taken**')
        show_table(tool_table(trace), hide_index=True)
    if checks := routing.get('boundary_checks'):
        st.markdown('**Boundary cross-checks**')
        st.caption('When the first classifier is close to its decision boundary, a second classifier is run.')
        show_table(pd.DataFrame([{
            'First model': model_name(check.get('primary_model')),
            'Near boundary': 'Yes' if check.get('near_boundary') else 'No',
            'Second model': model_name(check.get('secondary_model')) if check.get('secondary_executed') else '—',
            'Final choice': model_name(check.get('resolution')),
        } for check in checks]), hide_index=True)
    if cases := result.get('retrieved_cases'):
        st.markdown('**Similar training cases**')
        show_table(pd.DataFrame([{
            'Case': case.get('case_id'), 'Label': case.get('training_label'),
            'Similarity': round(case['similarity'], 3) if isinstance(case.get('similarity'), (int, float)) else None,
            'Excerpt': case.get('excerpt'),
        } for case in cases]), hide_index=True)
    if st.toggle('Show raw response (debug)', key=f"agent:raw:{result.get('request_id')}"):
        st.code(json.dumps(result, indent=2, ensure_ascii=False, default=str), language='json')
