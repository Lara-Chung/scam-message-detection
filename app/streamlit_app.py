"""Start the Streamlit message triage demo from the project root."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from app.traditional import run_traditional
from app.team_models import CHANNELS, neural_domain
from app.team_pages import METHODS, MODE_DESCRIPTIONS, MODE_GROUPS, model_options, run_agent_page, run_models

st.set_page_config(page_title='Signal | Message triage lab', page_icon='◈', layout='wide')
st.markdown('''<style>
/* Hide Streamlit's deployment/menu controls without hiding sidebar navigation. */
[data-testid="stToolbar"], [data-testid="stAppDeployButton"],
[data-testid="stMainMenu"], .stAppDeployButton, #MainMenu {display:none!important}
.block-container {max-width:1240px;padding-top:3.4rem;padding-bottom:3rem}
[data-testid="stSidebar"] {border-right:1px solid #e1e7ef}
h1 {letter-spacing:-.045em;font-size:2.9rem!important;color:#142d41}
h2 {letter-spacing:-.025em} h3 {font-size:1.15rem!important}
.message-box {background:#fff;border:1px solid #dbe4ec;border-radius:12px;padding:20px;line-height:1.9;overflow-wrap:anywhere;white-space:normal}
mark {background:#fff0bc;color:#24374b;border-radius:3px;padding:2px}
[data-testid="stMetric"] {background:white;border:1px solid #e0e7ee;border-radius:12px;padding:16px}
[class*="st-key-fraud-decision-"] [data-testid="stMetric"] {background:#fce4eb;color:#8a2144;border-color:#fce4eb}
[class*="st-key-fraud-decision-"] [data-testid="stMetric"] p,
[class*="st-key-fraud-decision-"] [data-testid="stMetricValue"] {color:#8a2144}
.stButton>button[kind="primary"] {border-radius:9px}
</style>''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('## Signal')
    family = st.radio('Method', list(METHODS), key='family')
    if family == 'AI Agent':
        workflow = st.selectbox('Workflow', list(MODE_GROUPS), key='agent:workflow')
        options = MODE_GROUPS[workflow]
        model_id = st.selectbox('Mode', list(options), format_func=options.get, key=f'agent:mode:{workflow}')
    else:
        options = model_options(family)
        model_id = st.selectbox('Model / workflow', list(options), format_func=options.get, key=f'model:{family}')
    domain = neural_domain(model_id) if family in ('RNN', 'Transformer') else 'all'
    if model_id.startswith('uci_'):
        domain = 'sms'
    channels = list(CHANNELS) if domain == 'all' else [domain]
    channel = st.selectbox('Channel', channels, format_func=CHANNELS.get, key=f'channel:{family}',
                           disabled=len(channels) == 1)

st.title(options[model_id])
st.caption(MODE_DESCRIPTIONS.get(model_id, METHODS[family]))
if family == 'Traditional modelling':
    run_traditional(model_id, channel)
elif family == 'AI Agent':
    run_agent_page(workflow, model_id, channel)
else:
    views = ['Check a message', 'Robustness test', 'Evaluation']
    page = st.radio('Workspace', views, horizontal=True, label_visibility='collapsed', key='workspace')
    run_models(family, model_id, channel, page)
