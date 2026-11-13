"""Render the original two-column demonstration layout for each method."""

import json
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.core import ROOT, load_pipeline, perturb, registry
from app.evaluation import show_agent_evaluation, show_chart, show_neural, show_table
from app.agent_details import show_execution_details
from app.examples import EXAMPLES, next_example
from app.thresholds import apply_threshold, threshold_controls
from app.team_models import (
    CHANNELS, TRADITIONAL, model_root, neural_batch, neural_models,
    neural_requirements,
)
from src.normalise import normalise

MODE_GROUPS = {
    'Classifier routing': {
        'rule_router': 'Rule-based model router',
        'agent_router': 'LLM model router',
        'fixed_model': 'Fixed validation-selected classifier',
    },
    'Adaptive investigation': {
        'adaptive_agent': 'Adaptive investigation Agent',
        'document_classifier': 'Local document classifier',
        'document_fixed': 'Fixed document workflow',
    },
    'Message analysis': {
        'agent': 'Tool-calling Agent',
        'classifier_only': 'Local logistic regression',
        'direct_llm': 'Direct LLM',
        'fixed_workflow': 'Fixed tool workflow',
    },
}
MODE_DESCRIPTIONS = {
    'agent_router': 'The LLM selects up to two same-channel TF-IDF classifiers.',
    'rule_router': 'A validation-selected rule chooses a same-channel TF-IDF classifier.',
    'fixed_model': 'Runs the fixed classifier selected for this channel on validation data.',
    'adaptive_agent': 'Local screening escalates ambiguous documents for section inspection, cross-checks or clarification.',
    'document_classifier': 'Runs the local document triage gate and reports coverage and review status.',
    'document_fixed': 'Runs the predefined document inspection schedule with LLM finalization when required.',
    'agent': 'The LLM can choose normalization, classification, indicator extraction and similar-case retrieval.',
    'classifier_only': 'Runs the channel-specific logistic regression classifier.',
    'direct_llm': 'Runs one LLM call with the same evidence contract as message analysis.',
    'fixed_workflow': 'Runs a fixed sequence of local tools before LLM finalization.',
}
ALWAYS_PAID = {'agent_router', 'agent', 'direct_llm', 'fixed_workflow'}
OPTIONAL_PAID = {'adaptive_agent', 'document_fixed'}
DEEPSEEK_URL = 'https://api.deepseek.com'
DEEPSEEK_MODEL = 'deepseek-flash'
RECORDED = {
    ('agent_router', False): 'agent_router_live.json',
    ('agent_router', True): 'agent_router_boundary_live.json',
    ('agent', False): 'success_agent_live.json',
}


METHODS = {
    'Traditional modelling': 'Classical classifiers trained on word or character TF-IDF features.',
    'RNN': 'BiLSTM with a trainable embedding layer.',
    'Transformer': 'DistilBERT fine-tuned with Hugging Face Trainer.',
    'AI Agent': 'Routing and investigation workflows for fraud review.',
}
AGENT_GROUPS = dict(zip(MODE_GROUPS, ('routing', 'adaptive', 'analysis'), strict=True))
PERTURBATIONS = ['P1', 'P2 spaces', 'P2 dots', 'P3 look-alikes', 'P3 zero-width']


def model_options(family: str) -> dict:
    """List model names with the architecture and documented training task."""
    if family == 'Traditional modelling':
        return TRADITIONAL | {'uci_lr': next(item['name'] for item in registry() if item['id'] == 'uci_lr')}
    architecture = 'BiLSTM' if family == 'RNN' else 'DistilBERT'
    models = neural_models(family)
    labels = ['fraud across SMS, email and phone', 'three-class Mendeley SMS']
    featured = {key: f'{architecture} on {label}' for key, label in zip(models, labels)}
    return {key: featured.get(key, key) for key in models}


def message_editor(key: str, channel: str, *, show_channel: bool = True,
                   initial_example: tuple[str, str] | None = None) -> str:
    """Load a fictional example or accept user-provided message text."""
    prefix = CHANNELS[channel].split()[0]
    options = [name for name in EXAMPLES if name.startswith(f'{prefix}: ')]
    choice = st.selectbox(
        'Fictional examples', options, key=f'{key}:type',
        index=options.index(initial_example[0]) if initial_example else 0,
        format_func=lambda name: name if show_channel else name.split(': ', 1)[1],
    )
    refresh = st.button('Load example', key=f'{key}:load')
    if refresh:
        st.session_state[key] = next_example(choice, st.session_state, f'{key}:examples')
    elif key not in st.session_state:
        st.session_state[key] = (initial_example[1] if initial_example else
                                 next_example(choice, st.session_state, f'{key}:examples'))
    return st.text_area('Message text', key=key, height=185, max_chars=12000)


def result_for(key: str, signature: tuple):
    """Return the result only when its inputs and settings still match."""
    entry = st.session_state.get(key)
    return entry[1] if entry and entry[0] == signature else None


@st.cache_resource
def uci_model(path: str, modified: int):
    """Cache a UCI model until the saved artifact changes."""
    return load_pipeline(path)


@st.cache_resource
def optional_ner():
    """Load the optional spaCy English entity recognizer."""
    try:
        import spacy
        return spacy.load('en_core_web_sm')
    except (ImportError, OSError):
        return None


def show_prediction(result: dict, *, compact: bool = False) -> None:
    """Show the decision, native model score and per-class outputs."""
    flagged = result['flagged']
    if flagged is True:
        if compact:
            st.markdown(
                '<div role="status" style="background:#fce4eb;color:#8a2144;'
                'border-radius:.5rem;padding:1rem;margin-bottom:1rem">Flag for review</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning('Flag for review')
    elif flagged is False and compact:
        st.markdown(
            '<div role="status" style="background:#d9f1e5;color:#125c47;'
            'border-radius:.5rem;padding:1rem;margin-bottom:1rem">Below review threshold</div>',
            unsafe_allow_html=True,
        )
    elif flagged is False:
        st.success('Below review threshold')
    if not compact:
        st.caption(f"Predicted class: {result['label']}.")
    score, latency = st.columns(2)
    if result['score'] is not None:
        score.metric(result['score_name'], f"{result['score']:.4f}")
    latency.metric('Inference time', f"{result['inference_ms']:.2f} ms")
    if not compact:
        if result['score'] is not None and result['score_name'] != 'Fraud decision margin':
            st.progress(float(result['score']))
        if result.get('threshold') is not None:
            st.caption(f"Flag threshold: {result['threshold']:.3f}.")
    if result.get('truncated') or result.get('possible_truncation'):
        st.warning('The message exceeds the model input limit.')
    if not compact:
        with st.expander('Class scores'):
            show_table(pd.DataFrame(result['scores'].items(), columns=['Class', 'Score']), hide_index=True)


def run_models(family: str, model_id: str, channel: str, page: str) -> None:
    """Run prediction, robustness or saved evaluation for the selected model."""
    if page == 'Evaluation':
        show_neural(model_id, family, channel)
        return

    key = f'{family}:{page}:{channel}'
    robust = page == 'Robustness test'
    missing = neural_requirements(family, model_id)
    if missing:
        st.warning('Live inference requires the original team checkpoint. Open Evaluation to see saved results.')
    left, right = st.columns([1.12, 1], gap='large')
    with left:
        st.subheader('Message input')
        text = message_editor(key, channel)
        if robust:
            kinds = st.multiselect('Perturbations', PERTURBATIONS, default=['P3 zero-width'], key=f'{key}:kinds')
            changed, changes = perturb(text, kinds, 42)
            with st.expander('Preview perturbed message'):
                st.text(changed.replace('\u200b', r'\u200b'))
                st.caption(f'{len(changes)} edits using seed 42.')
            messages = [text, changed, text, changed]
            defences = [False, False, True, True]
        else:
            kinds = []
            messages = [text]
            defences = [st.toggle('Apply D1 normalization', key=f'{key}:d1')]
        threshold = None
        if not missing:
            with st.expander('Prediction settings', expanded=True):
                threshold = threshold_controls(model_id, channel, f'{key}:{model_id}')
        # The threshold is not part of the signature: changing it re-flags the saved scores.
        signature = (model_id, text, tuple(kinds), tuple(defences))
        blocked = bool(missing)
        button = 'Compare four conditions' if robust else 'Analyse message'
        if st.button(button, type='primary', width='stretch', disabled=blocked or not text.strip()):
            st.session_state.pop(f'{key}:result', None)
            try:
                inputs = [normalise(message) if defence else message
                          for message, defence in zip(messages, defences, strict=True)]
                with st.spinner('Running the model… The first run loads it from disk, which takes a few seconds.'):
                    rows = neural_batch(family, model_id, inputs, None)
                st.session_state[f'{key}:result'] = (signature, rows)
            except ImportError:
                st.error('Install the model dependencies listed in START_HERE.md.')
            except Exception as error:
                st.error(f'Inference failed: {error}')
    with right:
        st.subheader('Robustness comparison' if robust else 'Prediction')
        rows = result_for(f'{key}:result', signature)
        if rows:
            rows = apply_threshold(rows, threshold)
        if missing:
            with st.expander('Checkpoint setup'):
                st.code(str(model_root(family) / model_id), language=None, wrap_lines=True)
                st.write('Missing files: ' + ', '.join(missing))
                st.caption('Installation instructions are in START_HERE.md.')
        elif not rows:
            st.caption('Load or paste a message to see its result here.')
        elif robust:
            conditions = ['Original', 'Perturbed', 'Original + D1', 'Perturbed + D1']
            frame = pd.DataFrame({
                'Condition': conditions, rows[0]['score_name']: [row['score'] for row in rows],
                'Model label': [row['label'] for row in rows], 'Flagged': [row['flagged'] for row in rows],
            })
            show_table(frame, hide_index=True)
            show_chart(px.bar(frame, x='Condition', y=rows[0]['score_name'], color_discrete_sequence=['#087f8c']))
            st.caption('Interactive comparison on a single message.')
        else:
            result = rows[0]
            show_prediction(result)
            analysed = normalise(text) if defences[0] else text
            from app.traditional import show_message_indicators  # imported here: traditional imports this module
            show_message_indicators(analysed)
    if rows and not robust and not missing:
        show_neural_explanation(family, model_id, normalise(text) if defences[0] else text, key)


@st.cache_data(show_spinner=False, max_entries=50)
def neural_lime(family: str, model_id: str, text: str) -> dict:
    """Run 500-sample LIME on one message for a neural checkpoint (proposal 5.7)."""
    from app.core import lime_explanation
    from app.team_models import NeuralProbabilities

    model = NeuralProbabilities(family, model_id)
    return lime_explanation(model, text, model.positive_label) | {'target': f'P({model.positive_label})'}


def show_neural_explanation(family: str, model_id: str, text: str, key: str) -> None:
    """Explain the current prediction with LIME on request, because it needs 500 extra model calls."""
    with st.expander('Why this prediction?', expanded=True):
        if not re.search(r'\w', text):
            st.caption('There are no words to explain in this message.')
            return
        signature = (family, model_id, text)
        cached = result_for(f'{key}:lime', signature)
        if cached is None:
            st.caption('LIME removes words at random 500 times and fits a local linear model to the scores. '
                       'This takes a few seconds for BiLSTM and longer for DistilBERT on CPU.')
            if not st.button('Explain with LIME', key=f'{key}:lime-run'):
                return
            try:
                with st.spinner('Running LIME (500 model calls)…'):
                    cached = neural_lime(family, model_id, text)
            except Exception as error:
                st.error(f'Explanation unavailable: {error}')
                return
            st.session_state[f'{key}:lime'] = (signature, cached)
            st.rerun()  # redraw without the button now that the explanation is stored
        st.markdown(cached['html'], unsafe_allow_html=True)
        st.caption(f"Red raises {cached['target']}. Green lowers it.")
        if cached['weights']:
            frame = pd.DataFrame(cached['weights'], columns=['Token', 'Contribution'])
            frame['Effect'] = np.where(frame['Contribution'] > 0, 'Raises score', 'Lowers score')
            chart = px.bar(frame.sort_values('Contribution'), x='Contribution', y='Token', orientation='h',
                           color='Effect', color_discrete_map={'Raises score': '#d58a85', 'Lowers score': '#76b2a2'})
            chart.update_layout(height=310, showlegend=False, bargap=.45, margin=dict(t=8, b=25))
            show_chart(chart)
        st.caption(f"Fit R² of the local surrogate: {cached['r2']:.3f}. Took {cached['seconds']:.1f} s.")


@st.cache_resource
def local_agent_service(profile: str):
    """Cache the local service without credentials."""
    from src.agent.service import ScamAgentService

    return ScamAgentService.from_config(ROOT / 'configs' / profile, enable_provider=False)


def live_agent_service(profile: str, connection: tuple[str, str, str]):
    """Create a request-local provider with the selected DeepSeek connection."""
    from src.agent.provider import DeepSeekProvider
    from src.agent.service import ScamAgentService

    service = ScamAgentService.from_config(ROOT / 'configs' / profile, enable_provider=False)
    service.provider = DeepSeekProvider(*connection, tokenizer_path=service.config['provider'].get('tokenizer_path'))
    return service


def deepseek_connection() -> tuple[str, str, str] | None:
    """Read an optional DeepSeek connection from the launch environment."""
    env_key = os.getenv('LLM_API_KEY', '').strip()
    if env_key:
        return env_key, os.getenv('LLM_BASE_URL') or DEEPSEEK_URL, os.getenv('LLM_MODEL') or DEEPSEEK_MODEL
    return None


def render_agent(result: dict, *, provider_ready: bool = False) -> None:
    """Show the fields actually returned by the Agent."""
    decision, review = st.columns(2)
    label = result['decision'].replace('_', ' ').title()
    if result['decision'] == 'fraud':
        with decision.container(key=f"fraud-decision-{result['request_id']}"):
            st.metric('Decision', label)
    else:
        decision.metric('Decision', label)
    review.metric('Review needed', 'Yes' if result['needs_review'] else 'No')
    routing = result.get('routing') or {}
    selected = routing.get('selected_model_id')
    explanation = result.get('explanation')
    blocked_escalation = result.get('stop_reason') == 'global_budget_exhausted' and bool(result.get('triage'))
    if blocked_escalation:
        action = ('Enable DeepSeek for this message and select Analyse message again.' if provider_ready else
                  'The message was left for human review in this offline run.')
        st.warning(f'Local screening finished. This message needs further investigation. {action}')
    elif explanation and (not selected or routing.get('boundary_checks')):
        st.text(explanation)
    for error in result.get('errors', []):
        if blocked_escalation and error['code'] == 'global_budget_exhausted':
            continue
        show_error = st.error if result['status'] == 'error' else st.warning
        show_error(error['message'])
    if (result.get('input_policy') or {}).get('clipped'):
        st.caption('This run inspected only part of the supplied input.')
    if selected:
        st.write(f"**Selected classifier:** {TRADITIONAL.get(selected, selected)}")
        if reason := routing.get('selection_reason'):
            reason_labels = {
                'validation_bucket_rule': 'The router matched this message to a rule learned from validation data.',
                'best_global_validation': 'The classifier was chosen in advance using this channel’s validation results.',
            }
            st.caption(f"Selection reason: {reason_labels.get(reason, reason.replace('_', ' ').capitalize())}")
    triage = result.get('triage') or {}
    if triage.get('read_sections'):
        st.write(f"**Document coverage** {'Complete' if triage['coverage_complete'] else 'Partial'}")
    if evidence := result.get('evidence'):
        st.markdown('**Evidence**')
        for item in evidence:
            st.text(item['quote'])
    if actions := triage.get('next_steps'):
        st.markdown('**Next steps**')
        for action in actions:
            st.text(action)
    with st.expander('Execution details'):
        show_execution_details(result)


def run_agent_page(workflow: str, mode: str, channel: str) -> None:
    """Show the selected workflow and its saved evaluation together."""
    group = AGENT_GROUPS[workflow]
    connection = deepseek_connection()
    if connection is None and mode in ALWAYS_PAID:
        show_offline_agent(mode, channel)
    else:
        run_agent_demo(group, mode, channel, connection)
    with st.expander('Evaluation'):
        show_agent_evaluation(group)


def recorded_agent_example(mode: str, channel: str, boundary: bool = False) -> dict | None:
    """Load a saved response for the selected mode when available."""
    name = RECORDED.get((mode, boundary))
    if mode == 'agent':
        name = {'sms': 'success_agent_live.json', 'email': 'needs_review_email_agent_live.json',
                'phone': 'success_phone_agent_live.json'}[channel]
    return json.loads((ROOT / 'docs' / 'examples' / name).read_text()) if name else None


def show_offline_agent(mode: str, channel: str) -> None:
    """Display an authentic saved run or point to the saved evaluation."""
    boundary = mode == 'agent_router' and st.checkbox('D1 + boundary cross-check', key='agent:recorded:boundary')
    example = recorded_agent_example(mode, channel, boundary)
    if example is None:
        st.info('This mode calls the DeepSeek LLM, so it needs an API key (set LLM_API_KEY before starting Streamlit). '
                'No saved run is available for it; open Evaluation below for its recorded results.')
        return
    st.info('This mode calls the DeepSeek LLM, so it needs an API key (set LLM_API_KEY before starting Streamlit). '
            f"Without a key, the page shows a saved run on {CHANNELS[example['channel']]}: its input and the Agent's decision.")
    left, right = st.columns([1.12, 1], gap='large')
    with left:
        st.subheader('Recorded message')
        if source := (example.get('views') or {}).get('original'):
            st.text(source)
    with right:
        st.subheader('Recorded decision')
        render_agent(example)


def run_agent_demo(group: str, mode: str, channel: str, connection: tuple[str, str, str] | None) -> None:
    """Run local screening or an explicitly enabled live request."""
    left, right = st.columns([1.12, 1], gap='large')
    boundary = False
    key = f'agent:{mode}:{channel}'
    with left:
        st.subheader('Message input')
        initial_example = ('SMS: everyday message', 'Hi, are we still meeting for lunch?') if group == 'adaptive' and channel == 'sms' else None
        text = message_editor(f'agent:{group}:{channel}', channel, show_channel=False, initial_example=initial_example)
        if group == 'routing':
            boundary = st.checkbox('D1 + boundary cross-check', key=f'{key}:boundary')
        paid = False
        if mode in ALWAYS_PAID | OPTIONAL_PAID:
            if connection:
                paid = st.checkbox('Allow DeepSeek to process this input', key=f'{key}:paid')
            else:
                st.caption('Offline screening is available. Messages that need further investigation are marked for review.')
        profile = 'agent.normalised.yaml' if boundary else 'agent.yaml'
        signature = (mode, channel, profile, text, paid)
        blocked = not text.strip() or mode in ALWAYS_PAID and not paid
        if st.button('Analyse message', type='primary', width='stretch', disabled=blocked):
            st.session_state.pop(f'{key}:result', None)
            execute_agent(profile, mode, channel, text, connection if paid else None, key, signature)
    with right:
        st.subheader('Decision')
        entry = result_for(f'{key}:result', signature)
        if entry:
            result, answers = entry
            render_agent(result, provider_ready=connection is not None)
            clarification = (result.get('triage') or {}).get('clarification')
            if clarification:
                choice = st.radio(clarification['question'], ['Yes', 'No'], index=None,
                                  key=f"agent:answer:{result['request_id']}")
                if st.button('Continue with answer', disabled=choice is None):
                    updated = answers | {clarification['question_id']: choice == 'Yes'}
                    if execute_agent(profile, mode, channel, text, connection if paid else None, key, signature, updated):
                        st.rerun()
        else:
            st.caption('Load or paste a message to see its decision here.')
        if example := recorded_agent_example(mode, channel, boundary):
            with st.expander('Recorded repository example'):
                st.caption(f"Saved example on {CHANNELS[example['channel']]}.")
                source = (example.get('views') or {}).get('original')
                if source:
                    st.text(source)
                render_agent(example)


def execute_agent(profile, mode, channel, text, connection, key, signature, answers=None) -> bool:
    """Store the Agent response for the current input and settings."""
    try:
        with st.spinner('Running Agent workflow… The first run loads and verifies the saved classifiers, '
                        'which takes a few seconds.'):
            service = live_agent_service(profile, connection) if connection else local_agent_service(profile)
            result = service.analyze_message(text, channel, mode=mode, allow_paid_calls=connection is not None,
                                             context_answers=answers)
        st.session_state[f'{key}:result'] = (signature, (result, answers or {}))
        return True
    except Exception as error:
        st.error(f'Agent could not run: {error}')
        return False
