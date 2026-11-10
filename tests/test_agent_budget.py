"""Prompt-budget regression tests. Recorded wire data is replayed only offline."""
import copy
import json
from pathlib import Path

import pytest

from src.agent.budget import PromptBudget, TOKENIZER_SHA256
from src.agent.live_meter import LiveMeter
from src.agent.models import AgentError
from src.agent.service import ScamAgentService

ROOT=Path(__file__).resolve().parents[1]
TOKENIZER=ROOT/'models/agent_tokenizer/tokenizer.json'
OLD_AUDIT=ROOT/'outputs/agent/live_agent_smoke_v1_api_audit'
OLD_PREDICTIONS=ROOT/'outputs/agent/live_agent_smoke_v1/predictions.jsonl'


class TinyEncoder:
    """Deterministic no-network tokenizer interface for framing arithmetic only."""
    def encode(self,text,add_special_tokens):
        assert not add_special_tokens
        self.seen=text
        return type('Encoded',(),{'ids':list(range(len(text)//4))})()


def test_count_includes_full_history_and_tool_schema():
    encoder=TinyEncoder()
    counter=PromptBudget(encoder)
    a=counter.estimate([{'role':'user','content':'hello'}],[])
    b=counter.estimate([{'role':'user','content':'hello'},{'role':'tool','content':'x'*2000}],
                       [{'type':'function','function':{'name':'classify_message'}}])
    assert b>a
    assert 'classify_message' in encoder.seen and 'x'*2000 in encoder.seen
    assert counter.describe()['tokenizer_sha256']==TOKENIZER_SHA256


def test_byte_fallback_remains_available():
    body={'messages':[{'role':'user','content':'你好'}],'tools':[]}
    assert PromptBudget().estimate(body['messages'],[])==len(json.dumps(body,ensure_ascii=False).encode())+1024


def test_tokenizer_file_is_explicit_pinned_and_model_specific(tmp_path):
    with pytest.raises(AgentError) as err:
        PromptBudget.from_file(tmp_path/'absent.json','deepseek-flash')
    assert err.value.code=='tokenizer_missing'
    path=tmp_path/'bad.json';path.write_text('{}')
    with pytest.raises(AgentError) as err:
        PromptBudget.from_file(path,'deepseek-flash')
    assert err.value.code=='tokenizer_hash_mismatch'
    with pytest.raises(AgentError) as err:
        PromptBudget.from_file(path,'other-provider-model')
    assert err.value.code=='tokenizer_model_mismatch'


@pytest.mark.skipif(not TOKENIZER.exists() or not OLD_AUDIT.exists(),reason='Requires local official tokenizer and historical audit')
def test_official_counter_covers_all_previously_billed_inputs():
    counter=PromptBudget.from_file(TOKENIZER,'deepseek-flash')
    assert counter.tokenizer.truncation is None
    for path in sorted(OLD_AUDIT.glob('exchange_*.json')):
        exchange=json.loads(path.read_text())
        estimate=counter.estimate(exchange['messages'],exchange['tools'])
        assert estimate>=exchange['response']['usage']['prompt_tokens']
        old=PromptBudget().estimate(exchange['messages'],exchange['tools'])
        assert estimate<old


@pytest.mark.skipif(not TOKENIZER.exists() or not OLD_PREDICTIONS.exists(),reason='Requires local historical audit')
@pytest.mark.parametrize('wire_index,channel,remaining',[(3,'email',8271),(5,'phone',5772)])
def test_previously_blocked_continuation_fits_without_dropping_context(wire_index,channel,remaining):
    exchange=json.loads((OLD_AUDIT/f'exchange_{wire_index:02d}.json').read_text())
    case=next(json.loads(line) for line in OLD_PREDICTIONS.read_text().splitlines() if json.loads(line)['channel']==channel)
    messages=exchange['messages']+[exchange['response']['message']]
    for call in exchange['response']['message']['tool_calls']:
        trace=next(t for t in case['result']['tool_trace'] if t['tool']==call['function']['name'])
        messages.append({'role':'tool','tool_call_id':call['id'],
                         'content':json.dumps({'result':trace['result'],'error':trace['error']},ensure_ascii=False)})
    assert PromptBudget().estimate(messages,exchange['tools'])>=remaining
    counter=PromptBudget.from_file(TOKENIZER,'deepseek-flash')
    assert counter.estimate(messages,exchange['tools'])+700<remaining


def test_service_stops_when_provider_exceeds_prompt_reservation():
    class ExactSmallBudget:
        def estimate(self,*args):return 10
    class ViolatingProvider:
        prompt_budget=ExactSmallBudget()
        kind,model,is_mock='offline_fixture','unit-test',True
        def complete(self,*args):
            return {'message':{'role':'assistant','content':'{}'},
                    'usage':{'prompt_tokens':11,'completion_tokens':1,'total_tokens':12}}
    result=ScamAgentService({},ViolatingProvider()).analyze_message('hello')
    assert result['stop_reason']=='token_reservation_exceeded'
    assert result['usage']['total_tokens']==12 and not result['usage']['accounting_complete']


def test_meter_uses_provider_counter_and_checks_input_independently(tmp_path):
    class SmallBudget:
        def estimate(self,*args):return 30
        def describe(self):return {'name':'offline_fixture'}
    class NoNetworkStub:
        prompt_budget=SmallBudget()
        kind,model,is_mock='unit_test_no_network','fixture',False
        def complete(self,messages,tools,max_tokens,timeout):
            # Total is under the combined reserve, but prompt exceeds its part.
            return {'message':{'role':'assistant','content':'{}'},
                    'usage':{'prompt_tokens':31,'completion_tokens':1,'total_tokens':32}}
    meter=LiveMeter(NoNetworkStub(),tmp_path/'audit',token_limit=200)
    meter.complete([],[],50,1)
    entry=meter.state['calls'][0]
    assert entry['reserved_tokens']==80 and entry['prompt_reservation']==30
    assert entry['status']=='reservation_exceeded'
    assert not meter.state['accounting_complete']


@pytest.mark.skipif(not TOKENIZER.exists() or not OLD_AUDIT.exists(),reason='Requires local historical audit')
def test_full_three_channel_offline_replay_fits_shared_budget():
    # Captured tool requests are real historical responses. The last two final
    # answers below are EXPLICIT TEST FIXTURES, not new live verification.
    rows=[json.loads(line) for line in OLD_PREDICTIONS.read_text().splitlines()]
    config=ScamAgentService.from_config(ROOT/'configs/agent.yaml').config
    exchanges=[json.loads(p.read_text()) for p in sorted(OLD_AUDIT.glob('exchange_*.json'))]
    class Replay:
        kind,model,is_mock='offline_replay','scripted-historical-mechanics',True
        prompt_budget=PromptBudget.from_file(TOKENIZER,'deepseek-flash')
        def __init__(self,responses):self.responses=list(responses)
        def complete(self,messages,tools,max_tokens,timeout):
            if self.responses:
                return copy.deepcopy(self.responses.pop(0))
            original=json.loads(messages[1]['content'])['untrusted_message']
            quote=original[:30]
            return {'message':{'role':'assistant','content':json.dumps({'decision':'legitimate','needs_review':False,
                'evidence':[{'view':'original','start':0,'end':len(quote),'quote':quote}]})},
                'usage':{'prompt_tokens':2500,'completion_tokens':100,'total_tokens':2600}}
    remaining=12000
    for row,indices in zip(rows,[[0,1],[2],[3,4]]):
        replay=Replay([exchanges[i]['response'] for i in indices])
        original=row['result']['views']['original']
        result=ScamAgentService(config,replay).analyze_message(original,row['channel'],mode='agent',token_budget=remaining)
        assert result['status']=='ok',(row['channel'],remaining,result['errors'])
        assert result['provider']['mock']
        remaining-=result['usage']['total_tokens']
        assert remaining>=0


def test_unique_exact_quote_offset_repair_stays_visible_and_requires_review():
    from src.agent.contract import resolve_span
    repaired,previous=resolve_span({'view':'original','quote':'hello','start':0,'end':2}, {'original':'say hello'})
    assert previous==(0,2) and repaired['start']==4 and repaired['end']==9
    assert repaired['quote']=='hello'
    # The public service uses the same helper and retains a review event.
    from test_agent import FixtureProvider, final, service
    provider=FixtureProvider([final('hello')])
    result=service(provider).analyze_message('say hello')
    assert result['status']=='needs_review' and result['decision']=='legitimate'
    assert result['errors'][0]['code']=='evidence_span_repaired'
    assert result['evidence'][0]['start']==4


@pytest.mark.parametrize('text,quote',[('hello hello','hello'),('hello','invented')])
def test_ambiguous_or_absent_quote_does_not_get_reanchored(text,quote):
    from src.agent.contract import resolve_span
    with pytest.raises(ValueError):
        resolve_span({'view':'original','quote':quote,'start':1,'end':2},{'original':text})
