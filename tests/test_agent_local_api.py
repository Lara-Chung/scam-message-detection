"""Local credential/profile and durable-budget mechanics. No live HTTP calls."""
import json

import pytest

from src.agent.live_meter import LiveMeter
from src.agent.models import AgentError
from src.agent.provider import DeepSeekProvider


class NoNetworkProviderStub:
    # Advertise the production interface to test the live-only meter. This stub
    # never leaves this test module and never sends HTTP or reports real results.
    is_mock=False
    kind='unit_test_no_network'
    model='unit-test-model'

    def __init__(self,usage=20,fail=False):
        self.calls=0
        self.usage=usage
        self.fail=fail

    def complete(self,messages,tools,max_tokens,timeout):
        self.calls+=1
        if self.fail:
            raise AgentError('provider_failure','Simulated transport failure.')
        return {'message':{'role':'assistant','content':'{}'},'usage':None if self.usage is None else
                {'prompt_tokens':self.usage-5,'completion_tokens':5,'total_tokens':self.usage}}


def test_explicit_keychain_profile_and_environment_override(monkeypatch):
    for name in ['LLM_API_KEY','LLM_BASE_URL','LLM_MODEL']:
        monkeypatch.delenv(name,raising=False)
    lookups=[]
    def lookup(service,account):
        lookups.append((service,account))
        return 'unit-test-placeholder'
    monkeypatch.setattr('src.agent.local_credentials.read_keychain_secret',lookup)
    profile={'keychain':{'service':'project-test','account':'tester'},
             'base_url':'https://api.deepseek.com','model':'deepseek-flash'}
    provider=DeepSeekProvider.from_config(profile)
    assert lookups==[('project-test','tester')]
    assert provider.model=='deepseek-flash'
    monkeypatch.setenv('LLM_API_KEY','unit-test-env-placeholder')
    assert DeepSeekProvider.from_config(profile)._key=='unit-test-env-placeholder'
    assert len(lookups)==1


def test_no_implicit_keychain_lookup_without_profile(monkeypatch):
    for name in ['LLM_API_KEY','LLM_BASE_URL','LLM_MODEL']:
        monkeypatch.delenv(name,raising=False)
    def forbidden(*args):
        raise AssertionError('Unrequested credential lookup')
    monkeypatch.setattr('src.agent.local_credentials.read_keychain_secret',forbidden)
    assert DeepSeekProvider.from_config({'kind':'deepseek_chat'}) is None


def test_meter_reserves_before_call_and_persists_exact_usage(tmp_path):
    stub=NoNetworkProviderStub()
    meter=LiveMeter(stub,tmp_path/'audit',token_limit=1200)
    def complete(messages,tools,max_tokens,timeout):
        state=json.loads(meter.ledger_path.read_text())
        assert state['actual_calls']==1 and state['unresolved_reserved_tokens']>0
        assert state['calls'][0]['status']=='in_flight'
        return NoNetworkProviderStub.complete(stub,messages,tools,max_tokens,timeout)
    stub.complete=complete
    meter.complete([],[],50,1)
    state=json.loads(meter.ledger_path.read_text())
    assert state['reported_tokens']==20 and state['unresolved_reserved_tokens']==0
    assert state['calls'][0]['status']=='completed'
    assert (meter.directory/'exchange_01.json').exists()
    with pytest.raises(FileExistsError):
        LiveMeter(stub,meter.directory)


def test_meter_rejects_over_budget_before_network(tmp_path):
    stub=NoNetworkProviderStub()
    meter=LiveMeter(stub,tmp_path/'audit',token_limit=1000)
    with pytest.raises(AgentError) as err:
        meter.complete([],[],50,1)
    assert err.value.code=='global_token_budget_exhausted' and stub.calls==0


@pytest.mark.parametrize('usage,fail',[(None,False),(20,True)])
def test_meter_freezes_after_unknown_accounting(tmp_path,usage,fail):
    stub=NoNetworkProviderStub(usage,fail)
    meter=LiveMeter(stub,tmp_path/'audit',token_limit=12000)
    if fail:
        with pytest.raises(AgentError):
            meter.complete([],[],50,1)
    else:
        meter.complete([],[],50,1)
    assert not meter.state['accounting_complete'] and meter.state['unresolved_reserved_tokens']>0
    with pytest.raises(AgentError) as err:
        meter.complete([],[],50,1)
    assert err.value.code=='accounting_incomplete' and stub.calls==1


def test_meter_enforces_call_limit(tmp_path):
    stub=NoNetworkProviderStub()
    meter=LiveMeter(stub,tmp_path/'audit',call_limit=1)
    meter.complete([],[],50,1)
    with pytest.raises(AgentError) as err:
        meter.complete([],[],50,1)
    assert err.value.code=='global_call_budget_exhausted' and stub.calls==1
