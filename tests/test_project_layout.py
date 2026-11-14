"""Integration checks for canonical component paths and retained public interfaces."""
import importlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.project_paths import ROOT, NEURAL_RESULTS


def test_agent_imports_and_preparation_commands_share_canonical_files():
    service=importlib.import_module('src.agent.service')
    prepare=importlib.import_module('scripts.prepare_document_triage')
    assert Path(service.__file__).parent==ROOT/'agent/runtime'
    assert Path(prepare.__file__).parent==ROOT/'agent/scripts'
    assert prepare.ROOT==ROOT
    from scripts.prepare_agent_tokenizer import TARGET
    assert TARGET==ROOT/'models/agent_tokenizer/tokenizer.json'


def test_legacy_traditional_imports_are_aliases_not_copies():
    canonical=importlib.import_module('traditional.lr_scam')
    assert importlib.import_module('baselines.lr_scam') is canonical
    assert importlib.import_module('lr_scam') is canonical


@pytest.mark.parametrize('entry', ['traditional/lr_scam.py','baselines/lr_scam.py','lr_scam.py'])
def test_traditional_entry_points_remain_runnable(entry):
    completed=subprocess.run([sys.executable,str(ROOT/entry),'--help'],capture_output=True,text=True,timeout=30)
    assert completed.returncode==0,completed.stderr
    assert '--label-col' in completed.stdout and '--out-dir' in completed.stdout


def test_app_reads_relocated_metrics_and_thresholds():
    from app import evaluation,thresholds
    from app.team_models import METRICS
    assert evaluation.RESULTS==NEURAL_RESULTS
    assert thresholds.REPORT_DIR==NEURAL_RESULTS/'results_app_lime/results_report'
    assert thresholds.load_thresholds()
    assert all((NEURAL_RESULTS/path).is_file() for path in METRICS.values())


def test_new_and_existing_checkpoint_locations_can_coexist(monkeypatch,tmp_path):
    import src.project_paths as paths
    monkeypatch.setattr(paths,'ROOT',tmp_path)
    monkeypatch.delenv('BILSTM_MODELS_DIR',raising=False)
    canonical=tmp_path/'recurrent/models';legacy=tmp_path/'transformer/models_bilstm'
    canonical.mkdir(parents=True)
    (legacy/'old_model').mkdir(parents=True);(legacy/'old_model/task_config.json').write_text('{}')
    assert paths.neural_models_root('RNN')==legacy
    assert paths.neural_checkpoint_dir('RNN','old_model')==legacy/'old_model'
    (canonical/'new_model').mkdir();(canonical/'new_model/task_config.json').write_text('{}')
    assert paths.neural_checkpoint_dir('RNN','new_model')==canonical/'new_model'
    assert paths.neural_checkpoint_dir('RNN','old_model')==legacy/'old_model'
    override=tmp_path/'override';monkeypatch.setenv('BILSTM_MODELS_DIR',str(override))
    assert paths.neural_checkpoint_dir('RNN','old_model')==override/'old_model'


def test_recurrent_adapter_preserves_class_order_prefix_batching_and_threshold(monkeypatch):
    from recurrent.inference import predict as adapter
    class TF:
        string='string'
        @staticmethod
        def constant(values,dtype):
            assert dtype=='string'
            assert all(v.startswith('xxbos ') for v in values)
            return values
    batches=[]
    def model(values):
        batches.append(len(values));return np.array([[.8,.2]]*len(values))
    metadata={'labels':['fraud','legitimate'],'positive_labels':['fraud'],'input_prefix':'xxbos ',
              'seq_len':3,'default_threshold':.9}
    monkeypatch.setattr(adapter,'load_model',lambda directory:(TF,model,metadata))
    rows=adapter.predict_batch(['three input words']*65,'bilstm_fixture',threshold=.75)
    assert batches==[64,1]
    assert all(r['label']=='fraud' and r['score']==.8 and r['flagged'] for r in rows)
    assert all(r['possible_truncation'] for r in rows)
    assert adapter.predict_batch([],'bilstm_fixture')==[]
    assert not adapter.predict('hello','bilstm_fixture')['flagged']


def test_shared_analyses_discover_both_families_from_canonical_results():
    from analysis.neural.scripts.cost_thresholds import find_runs,RESULTS_DIR
    from analysis.neural.scripts.significance import find_pairs
    from analysis.neural.scripts.e5_analysis import load_runs
    assert RESULTS_DIR==NEURAL_RESULTS
    assert {run.model for run in find_runs(RESULTS_DIR)}=={'distilbert','bilstm'}
    assert find_pairs(RESULTS_DIR)
    assert set(load_runs(RESULTS_DIR)['model'])=={'distilbert','bilstm'}


def test_transformer_reload_help_uses_shared_paths_without_loading_weights(tmp_path):
    completed=subprocess.run([sys.executable,str(ROOT/'transformer/inference/check_reload.py'),'--help'],
                             cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert completed.returncode==0,completed.stderr
    assert '--predictions' in completed.stdout and '--data' in completed.stdout
