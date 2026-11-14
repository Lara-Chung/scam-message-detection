"""Render evaluation outputs from the selected method's saved files."""

import json
import re

from sklearn.metrics import classification_report

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.core import ROOT, results_for
from src.project_paths import NEURAL_RESULTS
from app.team_models import METRICS, router

RESULTS = NEURAL_RESULTS
APP_RESULTS = RESULTS / 'results_app_lime'
AGENT_REPORTS = {
    'routing': ('classifier_router_report.md', 'Frozen TEST pilot', 'Frozen test pilot on 12 canonical cases'),
    'boundary': ('normalised_boundary_report.md', 'Measured outcomes', 'Boundary cross-check on 12 test views'),
    'adaptive': ('adaptive_agent_handoff.md', 'Actual live attempts and repair', 'Development verification on validation sources'),
    'analysis': ('agent_comparison_pilot.md', 'Objective outcomes', 'Paired comparison on six cases'),
    'cost': ('agent_comparison_pilot.md', 'Latency, work and cost', 'Provider usage and latency on six cases'),
}


def show_table(frame: pd.DataFrame, *, hide_index: bool = False) -> None:
    """Display table headers and values with consistent left alignment."""
    left = st.column_config.Column(alignment='left')
    columns = {name: left for name in frame.columns}
    if not hide_index:
        columns['_index'] = left
    st.dataframe(frame, width='stretch', hide_index=hide_index, column_config=columns)


def show_chart(figure) -> None:
    """Display a chart without zoom or selection controls."""
    figure.update_layout(dragmode=False)
    figure.update_xaxes(fixedrange=True)
    figure.update_yaxes(fixedrange=True)
    st.plotly_chart(figure, width='stretch', config={'displayModeBar': False, 'scrollZoom': False})


def square_confusion(matrix, labels, colors) -> go.Figure:
    """Draw the matrix and its color scale on the same vertical axis."""
    size = len(labels)
    peak = max(1, max(max(row) for row in matrix))
    steps = 128
    ticks = sorted({round(peak * index / 4) for index in range(5)})
    positions = list(range(size))
    figure = go.Figure(go.Heatmap(z=matrix, texttemplate='%{z}', coloraxis='coloraxis',
                                 customdata=[[f'True {actual}<br>Predicted {predicted}' for predicted in labels]
                                             for actual in labels],
                                 hovertemplate='%{customdata}<br>Messages %{z}<extra></extra>'))
    figure.add_trace(go.Heatmap(z=[[peak * (steps - index - .5) / steps] for index in range(steps)],
                                y0=-.5 + size / (2 * steps), dy=size / steps, xaxis='x2',
                                coloraxis='coloraxis', hoverinfo='skip'))
    figure.add_trace(go.Scatter(x=[.67] * len(ticks), y=[size - .5 - size * count / peak for count in ticks],
                                text=ticks, mode='text', textposition='middle right', cliponaxis=False,
                                xaxis='x2', hoverinfo='skip', showlegend=False))
    figure.update_layout(height=255, margin=dict(t=9, b=57, l=10, r=14),
                         coloraxis=dict(colorscale=colors, cmin=0, cmax=peak, showscale=False),
                         xaxis=dict(domain=[0, .80], range=[-.5, size - .5], tickvals=positions, ticktext=labels,
                                    type='linear', constrain='domain'),
                         xaxis2=dict(domain=[.85, 1], range=[-.5, 2.7], visible=False))
    figure.update_yaxes(title='True label', range=[size - .5, -.5], tickvals=positions, ticktext=labels,
                        scaleanchor='x', scaleratio=1, constrain='domain', constraintoward='top')
    for axis, title in [('x domain', 'Predicted label'), ('x2 domain', 'Messages')]:
        figure.add_annotation(x=.5, xref=axis, y=size - .5, yref='y', yshift=-42, text=title, showarrow=False)
    return figure


def show_confusion(matrix, labels, *, square: bool = False) -> None:
    """Draw a saved confusion matrix in the recorded label order."""
    colors = ['#eff7f8', '#087f8c']
    if square:
        show_chart(square_confusion(matrix, labels, colors))
        return
    frame = pd.DataFrame(matrix, index=labels, columns=labels)
    figure = px.imshow(frame, text_auto=True, aspect='auto', color_continuous_scale=colors,
                       labels={'x': 'Predicted label', 'y': 'True label', 'color': 'Messages'})
    figure.update_layout(height=325, margin=dict(t=10, b=25, l=10, r=10))
    show_chart(figure)


def show_classification(metrics: dict, *, square: bool = False) -> None:
    """Show available classification metrics and the associated class details."""
    columns = {
        'macro_f1': 'Macro-F1', 'accuracy': 'Accuracy', 'positive_recall': 'Positive recall',
        'smishing_recall': 'Smishing recall', 'fraud_recall': 'Fraud recall',
        'fraud_precision': 'Fraud precision', 'positive_false_positive_rate': 'False-positive rate',
        'smishing_false_positive_rate': 'Smishing false-positive rate', 'false_positive_rate': 'False-positive rate',
        'positive_pr_auc': 'PR-AUC', 'smishing_pr_auc': 'Smishing PR-AUC',
        'truncated_rate': 'Truncation rate', 'predict_ms_per_sample': 'Prediction ms per message',
    }
    items = [(label, metrics[key]) for key, label in columns.items() if metrics.get(key) is not None]
    descriptions = {'Macro-F1': 'The mean F1 across classes, giving each class equal weight.',
                    'Accuracy': 'The share of messages classified correctly.',
                    'Fraud recall': 'The share of actual fraud messages detected.'}
    cards = st.columns(min(3, len(items)))
    for card, (label, value) in zip(cards, items):
        card.metric(label, f'{value:.4f}')
        if square and label in descriptions:
            card.caption(descriptions[label])
    extra_metrics = items[3:]
    if extra_metrics:
        if square and all(value in (0, 1) for _, value in extra_metrics):
            summary = ', '.join(f'{label} {value:.0%}' for label, value in extra_metrics)
            st.caption(summary + '.')
        else:
            show_table(pd.DataFrame(extra_metrics, columns=['Metric', 'Value']), hide_index=True)
    per_class = metrics.get('per_class')
    matrix = metrics.get('confusion_matrix', metrics.get('confusion_matrix_0_1'))
    if matrix:
        left, right = st.columns(2, gap='medium' if square else 'large')
        with left:
            st.subheader('Confusion matrix')
            show_confusion(matrix, list(per_class) if per_class else ['legitimate', 'fraud'], square=square)
        with right:
            if per_class:
                st.subheader('Per-class metrics')
                show_table(pd.DataFrame(per_class).T)
            elif 'fraud_support' in metrics:
                st.subheader('Class support')
                st.write(f"{metrics['legitimate_support']} legitimate and {metrics['fraud_support']} fraud messages.")


def show_traditional(model_id: str, channel: str) -> None:
    """Show saved validation metrics with the dataset and split above them."""
    catalog = router().catalog['channels'][channel]
    candidate = catalog['models'][model_id]
    metrics = candidate['validation']
    matrix = metrics['confusion_matrix_0_1']
    labels = ['legitimate', 'fraud']
    true = [label for label, row in zip(labels, matrix) for count in row for _ in range(count)]
    predicted = [label for row in matrix for label, count in zip(labels, row) for _ in range(count)]
    report = classification_report(true, predicted, labels=labels, output_dict=True, zero_division=0)
    st.subheader('Validation results')
    st.caption(f"{catalog['dataset']} validation split, seed 0. "
               f"{catalog['train_n']:,} training messages and {metrics['n']:,} validation messages.")
    per_class = {label: report[label] for label in labels}
    show_classification(metrics | {'per_class': per_class}, square=True)


def show_uci(spec: dict) -> None:
    """Show the original UCI holdout results for the selected run."""
    report = results_for(spec)
    runs = report['runs'][spec['variant']]
    summary = report['summary'][spec['variant']]
    st.caption(f"{spec['provenance']}. Grouped 80/20 holdout with saved argmax predictions.")
    cards = [('Mean Macro-F1', 'macro_f1_mean', 'The mean F1 across classes and runs.'),
             ('Mean PR-AUC', 'pr_auc_mean', 'Summarizes the precision-recall tradeoff.'),
             ('Mean inference ms per message', 'infer_ms_per_msg_mean', 'Batch timing from the original runs.')]
    for card, (label, key, help_text) in zip(st.columns(3), cards):
        card.metric(label, f'{summary[key]:.4f}')
        card.caption(help_text)
    fields = {'split_seed': 'Seed', 'macro_f1': 'Macro-F1', 'pr_auc': 'PR-AUC', 'train_seconds': 'Train (s)',
              'tune_seconds': 'Tune (s)', 'infer_ms_per_msg': 'Inference (ms/message)', 'model_size_mb': 'Model (MB)'}
    show_table(pd.DataFrame(runs)[list(fields)].rename(columns=fields), hide_index=True)
    index = st.selectbox('Evaluation run', range(len(runs)), format_func=lambda i: f"Run {i + 1}, seed {runs[i]['split_seed']}")
    run = runs[index]
    left, right = st.columns(2, gap='medium')
    with left:
        st.subheader(f'Run {index + 1} confusion matrix')
        show_confusion(run['confusion_matrix'], run['classes'], square=True)
    with right:
        st.subheader(f'Run {index + 1} per-class metrics')
        show_table(pd.DataFrame(run['per_class']).T)


def show_neural(model_id: str, family: str, channel: str) -> None:
    """Show the neural team's saved metrics, thresholds and robustness results."""
    report = METRICS.get(model_id)
    if report:
        data = json.loads((RESULTS / report).read_text())
        partitions = {name: values for name, values in data.get('eval', data).items()
                      if isinstance(values, dict) and 'macro_f1' in values}
        default = next((i for i, name in enumerate(partitions) if name == f'test_{channel}' or name == 'test'), 0)
        split = st.selectbox('Evaluation split', list(partitions), index=default, key=f'eval:{model_id}:{channel}',
                             format_func=lambda name: name.replace('_', ' ').title())
        show_classification(partitions[split])
        if 'config' in data:
            with st.expander('Training configuration'):
                st.json({key: data[key] for key in ('config', 'seq_len', 'best_epoch', 'train_seconds') if key in data})
    else:
        st.info('This checkpoint has no matching per-run evaluation file in the package.')
    with st.expander('E1 benchmark by dataset'):
        if family == 'RNN':
            frame = pd.read_csv(RESULTS / 'results_bilstm' / 'bilstm_summary.csv')
            frame = frame[(frame['experiment'] == 'E1') & frame['eval_set'].str.startswith('test', na=False)]
            fields = {'dataset': 'Dataset', 'run': 'Run', 'eval_set': 'Split', 'macro_f1': 'Macro-F1',
                      'accuracy': 'Accuracy', 'positive': 'Positive class', 'pos_recall': 'Recall', 'pos_pr_auc': 'PR-AUC'}
        else:
            frame = pd.read_csv(RESULTS / 'results_e1' / 'e1_summary.csv')
            fields = {'dataset': 'Dataset', 'run': 'Run', 'test_macro_f1': 'Macro-F1', 'test_accuracy': 'Accuracy',
                      'positive': 'Positive class', 'test_pos_recall': 'Recall', 'test_pos_pr_auc': 'PR-AUC'}
        show_table(frame[list(fields)].rename(columns=fields), hide_index=True)
    with st.expander('E3 and E4 transfer results'):
        path = (RESULTS / 'results_bilstm' / 'bilstm_summary.csv' if family == 'RNN'
                else RESULTS / 'results_e3_e4' / 'e3_e4_summary.csv')
        frame = pd.read_csv(path)
        selected = frame[frame['experiment'].isin(['E3', 'E4']) & frame['eval_set'].str.startswith('test', na=False)]
        fields = {'experiment': 'Experiment', 'run': 'Run', 'source': 'Training source',
                  'eval_set': 'Test split', 'text': 'Input variant', 'macro_f1': 'Macro-F1',
                  'pos_recall': 'Positive recall', 'pos_fpr': 'False-positive rate'}
        show_table(selected[list(fields)].rename(columns=fields), hide_index=True)
    with st.expander('E5 robustness on held-out data'):
        frame = pd.read_csv(RESULTS / 'results_e5' / 'e5_summary.csv')
        name = 'bilstm' if family == 'RNN' else 'distilbert'
        selected = frame[(frame['model'] == name) & (frame['channel'] == channel)]
        fields = {'perturbation': 'Perturbation', 'defence': 'Defence', 'fraud_recall_mean': 'Mean fraud recall',
                  'false_alarm_rate_mean': 'Mean false-alarm rate', 'flip_rate_mean': 'Mean flip rate',
                  'macro_f1_mean': 'Mean Macro-F1', 'n_seeds': 'Seeds'}
        st.caption('Separate E5 runs by model family and channel.')
        show_table(selected[list(fields)].rename(columns=fields), hide_index=True)
    thresholds = pd.read_csv(APP_RESULTS / 'results_report' / 'app_thresholds.csv')
    thresholds = thresholds[thresholds['model'] == model_id]
    if not thresholds.empty:
        with st.expander('Cost-based thresholds'):
            fields = {'p': 'Assumed fraud rate', 'C_FN': 'Relative cost of a missed fraud',
                      'threshold': 'Threshold', f'{channel}_missed_per10k': 'Missed fraud per 10,000',
                      f'{channel}_false_alarms_per10k': 'False alarms per 10,000'}
            st.caption('Thresholds selected on seed 0 validation predictions.')
            show_table(thresholds[list(fields)].rename(columns=fields), hide_index=True)
    if family == 'Transformer':
        with st.expander('Saved DistilBERT LIME failure analysis'):
            frame = pd.read_csv(APP_RESULTS / 'results_report' / 'lime_category_summary.csv')
            st.caption('Offline LIME categories for the saved E3 transfer and E5 robustness failure cases.')
            show_table(frame, hide_index=True)


def show_agent_evaluation(group: str) -> None:
    """Read the Agent owner's evaluation tables for the selected workflow."""
    groups = {'routing': ('routing', 'boundary'), 'adaptive': ('adaptive',), 'analysis': ('analysis', 'cost')}
    for report in groups[group]:
        name, heading, title = AGENT_REPORTS[report]
        text = (ROOT / 'docs' / name).read_text()
        section = text.split(f'\n## {heading}\n', 1)[1].split('\n## ', 1)[0]
        st.subheader(title)
        table = []
        for line in section.splitlines() + ['']:
            if line.startswith('|'):
                table.append(line)
            elif table:
                table[1] = re.sub(r':?-+:?', ':---', table[1])
                st.markdown('\n'.join(table))
                table.clear()
        if report == 'adaptive':
            st.caption('The handoff records 6 provider requests and 9,744 tokens across all development attempts.')
        st.caption(f'Source: docs/{name}')
