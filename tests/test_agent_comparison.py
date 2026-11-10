"""Experimental design and accounting tests; no real provider requests."""
import json
from collections import Counter
from pathlib import Path

import pytest

from src.agent.data import EXPORT,load_channel
from src.agent.evaluate import create_manifest,ordered_modes

MODES=['classifier_only','direct_llm','fixed_workflow','agent']


def test_rotation_balances_paid_mode_positions():
    positions={mode:Counter() for mode in MODES[1:]}
    for i in range(6):
        order=ordered_modes(MODES,i,'rotate_llm')
        assert order[0]=='classifier_only' and set(order)==set(MODES)
        for j,mode in enumerate(order[1:]):positions[mode][j]+=1
    assert all(counts=={0:2,1:2,2:2} for counts in positions.values())
    assert ordered_modes(MODES,2)==MODES


@pytest.mark.skipif(not (EXPORT/'mendeley.csv.gz').exists(),reason='Local export not supplied')
def test_balanced_manifest_deterministic_and_test_only(tmp_path):
    first=create_manifest(tmp_path/'first.json',split='test',per_channel=2,sample_seed=20261110,balanced=True)
    second=create_manifest(tmp_path/'second.json',split='test',per_channel=2,sample_seed=20261110,balanced=True)
    assert first==second and first['sampling_policy']=='equal_per_class_hash_order'
    assert len(first['cases'])==6
    for channel in ['sms','email','phone']:
        cases=[c for c in first['cases'] if c['channel']==channel]
        assert {c['label'] for c in cases}=={0,1}
        df=load_channel(channel).set_index('id')
        assert all(df.loc[c['case_id'],'split_seed0']=='test' for c in cases)
    with pytest.raises(ValueError,match='even'):
        create_manifest(tmp_path/'odd.json',per_channel=3,balanced=True)


def test_reported_cache_tokens_affect_cost_and_unknown_cache_is_not_guessed():
    from src.agent.comparison import cost_range
    rates={'off_peak':{'cache_hit':.003,'cache_miss':.15,'output':.6},
           'peak':{'cache_hit':.006,'cache_miss':.3,'output':1.2}}
    usage={'prompt_tokens':300,'completion_tokens':30,'prompt_cache_hit_tokens':100,'prompt_cache_miss_tokens':200}
    cost=cost_range(usage,rates)
    assert cost['cache_counts_reported']
    assert cost['usd_low']==pytest.approx((.3+30+18)/1e6)
    assert cost['usd_high']==pytest.approx(cost['usd_low']*2)
    unknown=cost_range({'prompt_tokens':300,'completion_tokens':30},rates)
    assert not unknown['cache_counts_reported'] and unknown['cache_hit_tokens'] is None
    assert unknown['usd_low']<cost['usd_low'] and unknown['usd_high']>cost['usd_high']
