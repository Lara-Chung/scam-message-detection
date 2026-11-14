"""Shared locations for component code, saved comparisons and local checkpoints."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEURAL_RESULTS = ROOT / 'analysis' / 'neural' / 'results'


def neural_model_roots(family: str) -> tuple[Path, ...]:
    """List canonical/compatible roots; explicit environment overrides take precedence."""
    if family == 'Transformer':
        return (Path(os.getenv('MODELS_DIR', ROOT / 'transformer' / 'models')).resolve(),)
    if family != 'RNN':
        raise ValueError('Expected RNN or Transformer model family.')
    if os.getenv('BILSTM_MODELS_DIR'):
        return (Path(os.environ['BILSTM_MODELS_DIR']).resolve(),)
    return ((ROOT / 'recurrent' / 'models').resolve(), (ROOT / 'transformer' / 'models_bilstm').resolve())


def neural_models_root(family: str) -> Path:
    """Pick the first populated root, so an empty new folder does not hide old weights."""
    roots = neural_model_roots(family)
    return next((p for p in roots if next(p.glob('*/task_config.json'), None)), roots[0])


def neural_checkpoint_dir(family: str, model_id: str) -> Path:
    """Support models migrated individually while keeping explicit overrides strict."""
    roots = neural_model_roots(family)
    return next((p / model_id for p in roots if (p / model_id / 'task_config.json').is_file()), roots[0] / model_id)
