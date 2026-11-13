"""Exercise the new model and Agent navigation against the packaged artifacts."""
import pytest
from streamlit.testing.v1 import AppTest

from app.core import ROOT
from app.team_models import CHANNELS, TRADITIONAL
from app.team_pages import MODE_GROUPS


def start():
    return AppTest.from_file(str(ROOT / 'app/streamlit_app.py'), default_timeout=40).run()


def click(app, label):
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception
    assert not app.error, [item.value for item in app.error]


@pytest.mark.skipif(not (ROOT / 'models/router_candidates_seed0/catalog.json').is_file(),
                    reason='Requires canonical router models')
def test_canonical_models_and_robustness():
    app = start()
    for channel in CHANNELS:
        app.selectbox(key='channel:Traditional modelling').set_value(channel).run()
        for model in TRADITIONAL:
            app.selectbox(key='model:Traditional modelling').set_value(model).run()
            click(app, 'Analyse message')
            score_label = 'Fraud decision margin' if 'svm' in model else 'P(fraud)'
            assert any(metric.label == score_label for metric in app.metric)
    click(app, 'Compare four conditions')
    assert any('Condition' in item.value for item in app.dataframe)
    app.text_area[0].set_value('Edited input').run()
    assert not any('Condition' in item.value for item in app.dataframe)


@pytest.mark.skipif(not (ROOT / 'models/triage_seed0/catalog.json').is_file(),
                    reason='Requires local Agent artifacts')
def test_local_agent_modes():
    app = start()
    app.sidebar.radio(key='family').set_value('AI Agent').run()
    assert [item.label for item in app.sidebar.selectbox] == ['Workflow', 'Mode', 'Channel']
    app.text_area[0].set_value('Hi, are we still meeting for lunch?').run()
    local_modes = {'rule_router', 'fixed_model', 'document_classifier', 'adaptive_agent',
                   'document_fixed', 'classifier_only'}
    for workflow, modes in MODE_GROUPS.items():
        app.selectbox(key='agent:workflow').set_value(workflow).run()
        assert app.selectbox(key=f'agent:mode:{workflow}').options == list(modes.values())
        assert any(item.label == 'Evaluation' for item in app.expander)
        for mode in modes:
            if mode in local_modes:
                app.selectbox(key=f'agent:mode:{workflow}').set_value(mode).run()
                click(app, 'Analyse message')
                assert any(metric.label == 'Decision' for metric in app.metric)
                # Execution details shows a run summary table; local modes make no LLM calls.
                summaries = [item.value for item in app.table if 'LLM calls' in item.value.to_string()]
                assert summaries
                rows = dict(zip(summaries[0]['Item'], summaries[0]['Value']))
                assert rows['LLM calls'] == '0'
    app.selectbox(key='agent:workflow').set_value('Classifier routing').run()
    app.selectbox(key='agent:mode:Classifier routing').set_value('rule_router').run()
    app.checkbox(key='agent:rule_router:sms:boundary').set_value(True).run()
    click(app, 'Analyse message')
    assert any('Boundary cross-check' in item.value for item in app.subheader)
    assert [item.label for item in app.sidebar.radio] == ['Method']
    assert not any(item.key == 'workspace' for item in app.radio)


@pytest.mark.parametrize('family', ['RNN', 'Transformer'])
def test_neural_pages_report_missing_checkpoints(monkeypatch, tmp_path, family):
    monkeypatch.setenv('BILSTM_MODELS_DIR' if family == 'RNN' else 'MODELS_DIR', str(tmp_path))
    app = start()
    app.sidebar.radio(key='family').set_value(family).run()
    assert not app.exception
    assert any('Live inference requires' in entry.value for entry in app.warning)
    assert not app.metric
    assert app.selectbox(key=f'model:{family}').options[0].startswith('BiLSTM' if family == 'RNN' else 'DistilBERT')
    app.radio(key='workspace').set_value('Evaluation').run()
    assert not app.exception
    assert any(item.label == 'Macro-F1' for item in app.metric)
    assert any(item.label == 'E5 robustness on held-out data' for item in app.expander)
