"""Meaningful checks against supplied artifacts and the interactive demo logic."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.core import (extract_indicators, highlighted_html, lime_explanation,
                      load_pipeline, model_path, perturb, predict_batch, registry,
                      resolve_path, results_for, verified_test_predictions)
from src.normalise import normalise


@pytest.fixture(scope='module')
def spec():
    return registry()[0]


@pytest.mark.parametrize('split', [0, 1, 2])
def test_reproduce_exported_test_predictions(spec, split):
    model = load_pipeline(str(model_path(spec, split)))
    run = results_for(spec)['runs']['weighted'][split]
    data = pd.read_csv(resolve_path(spec['data_path']))
    predictions = verified_test_predictions(model, data, spec, run)
    assert len(predictions) == sum(map(sum, run['confusion_matrix']))
    assert (predictions.actual != predictions.predicted).sum() == sum(map(sum, run['confusion_matrix'])) - np.trace(run['confusion_matrix'])


def test_probability_label_order_is_explicit():
    class ReversedClasses:
        classes_ = np.array(['spam', 'ham'])
        def predict_proba(self, texts):
            return np.array([[.8, .2]] * len(texts))
    score, _ = predict_batch(ReversedClasses(), ['hello'], 'spam')
    assert score[0] == .8
    with pytest.raises(ValueError, match='absent'):
        predict_batch(ReversedClasses(), ['hello'], 'scam')


def test_score_only_model_is_not_labelled_probability():
    with pytest.raises(ValueError, match='decision scores'):
        predict_batch(object(), ['hello'], 'spam')


def test_defence_recovers_zero_width_and_leaves_digits():
    original = 'Verify account password immediately. Pay £200 to 12345678.'
    changed, log = perturb(original, ['P3 zero-width'], seed=42)
    assert changed != original and log
    assert normalise(changed) == original
    assert '£200' in changed and '12345678' in changed
    p1, _ = perturb(original, ['P1'], seed=42)
    assert normalise(p1) != original


def test_untrusted_message_html_is_escaped():
    text = '<script>alert(1)</script> Pay £200 now.'
    output = highlighted_html(text, extract_indicators(text))
    assert '<script>' not in output
    assert '&lt;script&gt;' in output
    assert '<mark>£200</mark>' in output


def test_lime_works_with_actual_pipeline(spec):
    model = load_pipeline(str(model_path(spec, 0)))
    result = lime_explanation(model, 'Congratulations! Claim your cash prize by calling now.', 'spam')
    assert result['weights'] and np.isfinite(result['r2'])


def click_button(app, label):
    next(item for item in app.button if item.label == label).click().run()
    assert not app.exception
    assert not app.error


def uci_app():
    app = AppTest.from_file(str(resolve_path('app/streamlit_app.py')), default_timeout=40).run()
    app.selectbox(key='model:Traditional modelling').set_value('uci_lr').run()
    return app


def test_streamlit_end_to_end():
    app = uci_app()
    assert not app.exception
    click_button(app, 'Analyse message')
    assert any('P(spam)' in item.label for item in app.metric)
    assert any('Red raises P(spam)' in item.value for item in app.caption)
    assert not any(item.label.startswith(('Explain', 'Verify')) for item in app.button)
    assert any('D1 normalizes' in item.value for item in app.caption)
    slider = app.sidebar.slider[0]
    slider.set_value(.8).run()
    assert any(item.label == 'P(spam)' for item in app.metric)
    assert not any('Flag threshold:' in item.value for item in app.caption)
    assert not any(item.label in ('Class scores', 'Candidate indicators') for item in app.expander)
    assert not app.get('progress')
    # Empty input must not leave a stale prediction visible.
    app.text_area[0].set_value('').run()
    assert not any('P(spam)' in item.label for item in app.metric)
    app.text_area[0].set_value('Claim your prize today.').run()
    click_button(app, 'Compare four conditions')
    frame = next(item.value for item in app.dataframe if 'Condition' in item.value)
    assert set(frame['Flagged']) <= {'Yes', 'No'}
    app.text_area[0].set_value('Different message without prior analysis').run()
    assert not any('P(spam)' in item.value.columns for item in app.dataframe)


def test_shuffled_examples_cycle_without_repeats():
    from app.examples import EXAMPLES, next_example
    for category, pool in EXAMPLES.items():
        state = {}
        values = [next_example(category, state, 'test') for _ in pool]
        assert len(set(values)) == len(pool)
        assert set(values) == set(pool)
        assert next_example(category, state, 'test') != values[-1]


@pytest.mark.parametrize('model_index', [1, 2, 3])
def test_supplementary_models_reproduce_saved_metrics(model_index):
    spec = registry()[model_index]
    model = load_pipeline(str(model_path(spec, 0)))
    run = results_for(spec)['runs']['weighted'][0]
    frame = verified_test_predictions(model, pd.read_csv(resolve_path(spec['data_path'])), spec, run)
    assert len(frame) == 1038
    if spec.get('defence') == 'D2':
        assert hasattr(model, 'calibrated_classifiers_')


def test_random_refresh_and_model_switching():
    app = uci_app()
    original = app.text_area[0].value
    click_button(app, 'Load example')
    assert app.text_area[0].value != original
    chosen = next(x for x in app.selectbox if x.key == 'traditional:sms:type')
    chosen.set_value('SMS: everyday message').run()
    click_button(app, 'Load example')
    from app.examples import EXAMPLES
    assert app.text_area[0].value in EXAMPLES['SMS: everyday message']
    before = app.text_area[0].value
    click_button(app, 'Load example')
    assert app.text_area[0].value != before
    models = app.selectbox(key='model:Traditional modelling')
    assert not any('demo' in label for label in models.options)
    assert all('SMS:' not in label for label in chosen.options)
    models.set_value('char_svm').run()
    app.toggle[0].set_value(True).run()
    click_button(app, 'Analyse message')
    assert any(x.label == 'Fraud decision margin' for x in app.metric)
    assert any('Red raises fraud decision margin' in item.value for item in app.caption)
    assert any('validation split, seed 0' in item.value for item in app.caption)
