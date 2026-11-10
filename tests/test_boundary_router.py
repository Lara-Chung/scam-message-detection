"""Native-scale boundary calibration and guaranteed backend checks. No live API."""
import json
from pathlib import Path

import numpy as np
import pytest

from src.agent.boundary import calibrate,distance,resolution
from src.agent.boundary_routing import BoundaryRouter
from test_agent import FixtureProvider,call
from test_classifier_routing import engine,final


def configured(monkeypatch,threshold=.4,winner='abstain',limits=None):
    e,calls=engine(monkeypatch,limits)
    e.config={'normalised_boundary_routing':True}
    e._catalog['boundary_policy_version']=1;e._catalog['inference_normalisation']=True
    for ch in e._catalog['channels'].values():
        ch['boundary_policies']={name:{'threshold':threshold,'secondary_model':'char_svm' if name!='char_svm' else 'word_lr',
            'distance_kind':'fixture_native','conflict_by_primary_prediction':{
                '0':{'selected_model':winner},'1':{'selected_model':winner}}} for name in ch['models']}
    return e,calls


def test_native_score_scales_remain_separate():
    assert distance({'fraud_score':-.02,'score_kind':'decision_margin'})==.02
    assert distance({'fraud_score':.52,'score_kind':'model_probability_uncalibrated'})==pytest.approx(.02)
    with pytest.raises(ValueError):distance({'fraud_score':None,'score_kind':'label_only'})


def test_duplicate_variants_do_not_fake_conflict_support():
    y=np.ones(90,dtype=int);p=np.zeros(90,dtype=int);s=np.ones(90,dtype=int)
    policies,_,abstain=resolution(y,p,s,['same-base']*90,np.ones(90,dtype=bool),'word_lr','char_svm')
    assert policies['0']['unique_conflict_bases']==1 and policies['0']['selected_model']=='abstain'
    assert abstain.all()


def test_calibration_uses_declared_quantile_candidates_and_unique_votes():
    y=np.array([0,1]*30);ids=[f'val-{i}' for i in range(60)]
    preds={'word_lr':np.zeros(60,dtype=int),'word_nb':y.copy(),'word_svm':y.copy(),'char_svm':y.copy()}
    distances={n:np.linspace(0,1,60) for n in preds}
    result=calibrate(y,preds,distances,ids,'word_lr',list(preds))
    assert result['quantile'] in [.05,.10,.20,.30]
    assert result['secondary_model']!='word_lr'
    assert result['threshold']==pytest.approx(np.quantile(distances['word_lr'],result['quantile']))


def test_normalisation_is_actual_classifier_input_and_fixed_has_no_crosscheck(monkeypatch):
    e,calls=configured(monkeypatch)
    original=e.adapter
    seen=[]
    def adapter(ch,name):
        a=original(ch,name);old=a.classify
        def classify(text,view):seen.append((text,view));return old(text,view)
        a.classify=classify;return a
    monkeypatch.setattr(e,'adapter',adapter)
    result=BoundaryRouter(e).analyze_message('he\u200bllo £200','sms','fixed_model',None)
    assert calls==['word_lr'] and seen==[('hello £200','normalised')]
    assert result['views']=={'original':'he\u200bllo £200','normalised':'hello £200'}
    assert result['schema_version']=='1.2' and result['classifier_results'][0]['view']=='normalised'
    assert not result['routing']['boundary_checks'][0]['policy_enabled']


def test_rule_near_boundary_runs_secondary_and_abstains_on_insufficient_conflict_evidence(monkeypatch):
    e,calls=configured(monkeypatch)
    for ch in e._catalog['channels'].values():ch['rules']={b:'word_lr' for b in ['short','standard','long','obfuscated']}
    result=BoundaryRouter(e).analyze_message('hello','sms','rule_router',None)
    assert calls==['word_lr','char_svm']
    assert result['decision']=='abstain' and result['status']=='needs_review'
    assert result['routing']['boundary_checks'][0]['secondary_executed']
    assert result['tool_trace'][1]['source']=='boundary_policy'


def test_agent_cannot_skip_mandatory_check_or_override_locked_conflict_winner(monkeypatch):
    e,calls=configured(monkeypatch,winner='char_svm')
    provider=FixtureProvider([call('classify_message',{'model_id':'word_lr'}),final('word_lr')])
    result=BoundaryRouter(e).analyze_message('he\u200bllo','sms','agent_router',provider)
    assert calls==['word_lr','char_svm']
    assert result['decision']=='fraud' and result['needs_review']
    assert result['routing']['proposed_model_id']=='word_lr'
    assert result['routing']['selected_model_id']=='char_svm'
    outcome=json.loads(provider.messages[1][-1]['content'])
    assert outcome['secondary_result']['decision']=='fraud'
    assert outcome['actual_candidates_called']==calls
    payload=json.loads(provider.messages[0][1]['content'])
    assert payload['untrusted_message']=='hello' and payload['message_view']=='normalised'


def test_non_near_case_uses_one_classifier(monkeypatch):
    e,calls=configured(monkeypatch,threshold=.01)
    result=BoundaryRouter(e).analyze_message('hello','sms','agent_router',FixtureProvider([
        call('classify_message',{'model_id':'word_lr'}),final()]))
    assert calls==['word_lr'] and result['decision']=='legitimate' and result['status']=='ok'


def test_insufficient_call_budget_is_not_silently_ignored(monkeypatch):
    e,calls=configured(monkeypatch,limits={'tool_calls':1})
    result=BoundaryRouter(e).analyze_message('hello','sms','agent_router',FixtureProvider([
        call('classify_message',{'model_id':'word_lr'}),final()]))
    assert result['status']=='error' and result['decision']=='abstain'
    assert result['stop_reason']=='cross_check_missing'


def test_previous_challenge_sources_are_excluded_for_a_new_experiment(tmp_path,monkeypatch):
    import scripts.prepare_router_stress as module
    root=tmp_path/'outputs/agent';old=root/'old';current=root/'current'
    old.mkdir(parents=True);current.mkdir()
    for folder,ident in [(old,'old-id'),(current,'current-id')]:
        (folder/'manifest.json').write_text(json.dumps({'partition':'independent_canonical_seed0_test_challenge',
            'cases':[{'channel':'sms','case_id':ident+'::paired_clean','source_id':ident}]}))
    monkeypatch.setattr(module,'ROOT',tmp_path)
    excluded,_=module.exclude_previous(current)
    assert ('sms','old-id') in excluded and ('sms','current-id') not in excluded
