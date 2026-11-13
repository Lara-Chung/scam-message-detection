"""Run the traditional models in a shared prediction and evaluation workspace."""

import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.core import (
    extract_indicators, highlighted_html, input_text, lime_explanation, model_path, perturb, predict_batch, registry,
    weighted_html,
)
from app.evaluation import show_chart, show_table, show_traditional, show_uci
from app.team_models import router, traditional_batch
from app.team_pages import message_editor, optional_ner, result_for, show_prediction, uci_model
from src.normalise import normalise

D1_HELP = 'D1 normalizes look-alike letters, zero-width characters and spaced-out words.'
PERTURBATION_HELP = {
    'P1': 'replaces letters with similar digits, such as “offer” becoming “0ff3r”.',
    'P2 spaces': 'inserts spaces between letters, as in “o f f e r”.',
    'P2 dots': 'inserts dots between letters, as in “o.f.f.e.r”.',
    'P3 look-alikes': 'uses similar-looking letters from another alphabet.',
    'P3 zero-width': 'inserts an invisible character. The preview marks it as [ZWSP].',
}
ROBUST_LABELS = ['Original', 'Perturbed', 'Original + D1', 'Perturbed + D1']


@st.cache_data
def token_contributions(model_id: str, channel: str, text: str, artifact_revision: tuple[str, int] | None) -> dict:
    """Explain probabilities with LIME and SVM margins with token removal."""
    if model_id == 'uci_lr':
        model = uci_model(*artifact_revision)
        return lime_explanation(model, text, 'spam') | {'method': 'LIME', 'target': 'P(spam)'}
    model = router().adapter(channel, model_id).model
    if hasattr(model, 'predict_proba'):
        return lime_explanation(model, text, 1) | {'method': 'LIME', 'target': 'P(fraud)'}

    parts = re.split(r'(\W+)', text)
    folded = [part.casefold() for part in parts]
    tokens = list(dict.fromkeys(part for part in folded if part and re.search(r'\w', part)))[:80]
    variants = [''.join(part for part, token in zip(parts, folded) if token != word) for word in tokens]
    margins = np.asarray(model.decision_function([text, *variants]), dtype=float)
    orientation = 1 if model.classes_[1] == 1 else -1
    weights = sorted(zip(tokens, orientation * (margins[0] - margins[1:])),
                     key=lambda pair: abs(pair[1]), reverse=True)[:10]
    weights = [(word, float(weight)) for word, weight in weights if weight]
    return {'weights': weights, 'html': weighted_html(text, weights),
            'method': 'Token removal', 'target': 'fraud decision margin'}


def show_message_indicators(text: str) -> None:
    """Highlight rule-based warning signs in the message."""
    spans = extract_indicators(text, optional_ner())
    st.markdown(highlighted_html(text, spans), unsafe_allow_html=True)
    if spans:
        st.markdown('**Candidate indicators**')
        frame = pd.DataFrame(spans)[['field', 'value', 'source']].drop_duplicates()
        show_table(frame, hide_index=True)
    else:
        st.caption('No indicator patterns matched.')


def show_token_contributions(model_id, channel, text, revision) -> None:
    """Show the cached explanation as soon as a prediction is available."""
    with st.expander('Why this prediction?'):
        if not re.search(r'\w', text):
            st.caption('There are no words to explain in this message.')
            return
        try:
            result = token_contributions(model_id, channel, text, revision)
        except (ValueError, ImportError) as error:
            st.error(f'Explanation unavailable: {error}')
            return
        st.markdown(result['html'], unsafe_allow_html=True)
        target = result['target']
        st.caption(f'Red raises {target}. Green lowers it. Bar length shows the size of the contribution.')
        if result['weights']:
            frame = pd.DataFrame(result['weights'], columns=['Token', 'Contribution'])
            frame['Effect'] = np.where(frame['Contribution'] > 0, 'Raises score', 'Lowers score')
            chart = px.bar(frame.sort_values('Contribution'), x='Contribution', y='Token', orientation='h',
                           color='Effect', color_discrete_map={'Raises score': '#d58a85', 'Lowers score': '#76b2a2'})
            chart.update_layout(height=310, showlegend=False, bargap=.45, margin=dict(t=8, b=25))
            show_chart(chart)
        if result['method'] == 'LIME':
            st.caption(f"LIME approximates this message's score using 500 word-removal samples. Fit R²: {result['r2']:.3f}.")
        else:
            st.caption('Token removal measures the change in the SVM margin after removing every occurrence of one word. It checks the first 80 distinct words.')


def apply_review_threshold(rows: list[dict], threshold: float | None) -> list[dict]:
    """Update review flags without rerunning inference."""
    if threshold is None:
        return rows
    return [row | {'flagged': row['score'] >= threshold, 'threshold': threshold} for row in rows]


def show_robustness(text, identity, threshold, key, disabled, score_views) -> None:
    """Compare original and perturbed text and score four D1 conditions."""
    with st.expander('Robustness test'):
        if not text.strip():
            st.caption('Enter a message above to compare perturbations.')
            return
        kinds = st.multiselect('Perturbations', list(PERTURBATION_HELP), default=['P3 zero-width'], key=f'{key}:kinds')
        for kind in kinds:
            st.caption(f'**{kind}** {PERTURBATION_HELP[kind]}')
        changed, edits = perturb(text, kinds, 42)
        original_col, changed_col = st.columns(2, gap='large')
        original_col.markdown('**Original text**')
        original_col.markdown(highlighted_html(text, extract_indicators(text)), unsafe_allow_html=True)
        changed_col.markdown('**Perturbed text**')
        visible = changed.replace('\u200b', '[ZWSP]')
        changed_col.markdown(highlighted_html(visible, extract_indicators(visible)), unsafe_allow_html=True)
        st.caption(f'{len(edits)} edits using seed 42.')
        signature = (identity, text, tuple(kinds))
        if st.button('Compare four conditions', key=f'{key}:compare', disabled=disabled, type='primary'):
            st.session_state.pop(f'{key}:robust', None)
            try:
                scored = score_views([text, changed, text, changed], [False, False, True, True])
                st.session_state[f'{key}:robust'] = (signature, scored)
            except Exception as error:
                st.error(f'Comparison failed: {error}')
        saved = result_for(f'{key}:robust', signature)
        if saved:
            scored = apply_review_threshold(saved, threshold)
            label = scored[0]['score_name']
            frame = pd.DataFrame({
                'Condition': ROBUST_LABELS, label: [item['score'] for item in scored],
                'Flagged': ['Yes' if item['flagged'] else 'No' for item in scored],
            })
            fig = px.bar(frame, x='Condition', y=label, color_discrete_sequence=['#087f8c'])
            fig.update_traces(width=.30)
            fig.update_layout(height=285, showlegend=False, margin=dict(t=15, b=30))
            if threshold is not None:
                fig.update_yaxes(range=[0, 1])
                fig.add_hline(y=threshold, line_dash='dot', line_color='#435468')
                threshold_note = 'Dotted line means the review threshold.'
            else:
                fig.add_hline(y=0, line_dash='dot', line_color='#435468')
                threshold_note = 'The dotted line marks the SVM decision threshold at 0.'
            show_chart(fig)
            show_table(frame, hide_index=True)
            st.caption(f'{threshold_note} This comparison uses one message.')


def run_traditional(model_id: str, channel: str) -> None:
    """Show prediction, robustness and evaluation together for the selected model."""
    spec = None
    model = None
    revision = None
    if model_id == 'uci_lr':
        spec = next(item for item in registry() if item['id'] == model_id)
        path = model_path(spec, 0)
        revision = (str(path), path.stat().st_mtime_ns) if path.is_file() else None
        if revision:
            model = uci_model(*revision)
        else:
            st.warning('The saved UCI model is missing.')
    target = 'P(spam)' if spec else 'P(fraud)'
    with st.sidebar:
        threshold = None
        if model_id in ('uci_lr', 'word_lr', 'word_nb'):
            threshold = st.slider(f'Review threshold for {target}', .01, .99, .50, .01,
                                  key=f'review:{model_id}')
            st.caption(f'Messages with {target} at or above this threshold are flagged for review. Evaluation uses the saved model boundary.')
        else:
            st.markdown('**Review threshold for the SVM margin**')
            st.caption('Linear SVM outputs a decision margin with a fixed threshold of 0. '
                       'Positive margins are flagged for review.')
    if spec:
        st.caption('UCI SMS predicts spam or ham. The spam label includes marketing.')
    key = f'traditional:{channel}'
    left, right = st.columns([1.12, 1], gap='large')
    unavailable = bool(spec and model is None)

    def score_views(messages, defences):
        """Score shared input views at the model's default boundary."""
        if spec:
            prepared = [input_text(message, spec) for message in messages]
            inputs = [normalise(message) if defence else message
                      for message, defence in zip(prepared, defences, strict=True)]
            scores, elapsed = predict_batch(model, inputs, 'spam')
            return [{
                'label': 'spam' if score >= .5 else 'ham', 'score': float(score), 'score_name': 'P(spam)',
                'flagged': bool(score >= .5), 'truncated': False, 'inference_ms': elapsed / len(inputs),
                'scores': {'spam': float(score), 'ham': 1 - float(score)},
            } for score in scores]
        rows = [traditional_batch(model_id, channel, [message], defence)[0]
                for message, defence in zip(messages, defences, strict=True)]
        if threshold is not None:
            for row in rows:
                row['score_name'] = 'P(fraud)'
        return rows

    with left:
        st.subheader('Message input')
        text = message_editor(key, channel, show_channel=False)
        d1 = st.toggle('Apply D1 normalization', key=f'{key}:d1')
        st.caption(D1_HELP)
        signature = (model_id, text, d1, revision)
        if st.button('Analyse message', type='primary', width='stretch', disabled=unavailable or not text.strip()):
            st.session_state.pop(f'{key}:prediction', None)
            try:
                row = score_views([text], [d1])[0]
                limit = router().limits['max_input_chars'] if not spec else len(text)
                prepared = input_text(text, spec) if spec else text[:limit]
                analysed = normalise(prepared) if d1 else prepared
                st.session_state[f'{key}:prediction'] = (signature, (row, analysed))
            except Exception as error:
                st.error(f'Inference failed: {error}')
    with right:
        st.subheader('Prediction')
        result = result_for(f'{key}:prediction', signature)
        if result:
            row, analysed = result
            shown = apply_review_threshold([row], threshold)[0]
            show_prediction(shown, compact=True)
            show_message_indicators(analysed)
        else:
            st.caption('Load or paste a message, then select Analyse message.')
    if result:
        show_token_contributions(model_id, channel, analysed, revision)
    show_robustness(text, (model_id, revision), threshold, key, unavailable, score_views)
    with st.expander('Evaluation'):
        if spec:
            show_uci(spec)
        else:
            show_traditional(model_id, channel)
