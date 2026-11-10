"""Pre-registered stress transforms and label-blind sampling; no live API calls."""
import pytest

from src.agent.candidates import features
from src.agent.stress import FAMILIES,SEVERITIES,perturb,profile,select_label_blind


@pytest.mark.parametrize('family',FAMILIES)
@pytest.mark.parametrize('severity',SEVERITIES)
def test_transform_is_reproducible_preserves_digits_contacts_and_price(family,severity):
    text=('Please verify your account urgently using official customer support. '*20+
          ' Pay $15 Million USD, call +65 8123 4567 or email help@example.com at https://example.com/login?id=42.')
    changed,edits=perturb(text,family,severity,'fixture')
    assert (changed,edits)==perturb(text,family,severity,'fixture')
    assert edits and changed!=text
    for literal in ['$15 Million USD','+65 8123 4567','help@example.com','https://example.com/login?id=42']:
        assert literal in changed
    assert ''.join(c for c in text if c.isdigit())==''.join(c for c in changed if c.isdigit())
    for edit in edits:assert text[edit['start']:edit['end']]==edit['original']


def test_disagreement_selector_never_accepts_truth_labels():
    pool=[{'id':'a','text':'hello'},{'id':'b','text':'world'}]
    pred=[{'a':0,'b':0},{'a':0,'b':1}]
    assert select_label_blind(pool,pred,1,17,True)==[pool[1]]
    assert select_label_blind(pool,pred,1,17,False)==select_label_blind(pool,pred,1,17,False)
    with pytest.raises(ValueError,match='labels'):
        select_label_blind([dict(pool[0],label=1)],pred[:1],1,17,True)


def test_observable_profile_does_not_need_scenario_or_true_label():
    assert profile(features('ver\u200bify'))=='invisible'
    assert profile(features('v e r i f y'))=='spaced'
    assert profile(features('аccount'))=='lookalike'
    assert profile(features('ｖｅｒｉｆｙ'))=='compat'
    assert profile(features('ｖｅｒ\u200bｉｆｙ'))=='mixed'
