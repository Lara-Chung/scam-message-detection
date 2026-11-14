"""Persistent fail-closed accounting around a single explicitly authorized live run."""
from __future__ import annotations

import json
import threading
from pathlib import Path

from src.agent.models import AgentError
from src.agent.budget import PromptBudget


class LiveMeter:
    def __init__(self, provider, directory: Path, token_limit=12000, call_limit=12):
        self.kind, self.model, self.is_mock = provider.kind, provider.model, provider.is_mock
        if self.is_mock:
            raise ValueError('This audit meter is for an actual provider only.')
        self._provider, self._lock = provider, threading.Lock()
        self.prompt_budget = getattr(provider, 'prompt_budget', PromptBudget())
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.ledger_path = directory/'budget_ledger.json'
        if self.ledger_path.exists():
            raise FileExistsError('Live ledger exists; refusing an accidental repeat of paid calls.')
        self.state = {'global_token_limit':token_limit, 'call_limit':call_limit, 'actual_calls':0,
                      'reported_tokens':0, 'unresolved_reserved_tokens':0, 'accounting_complete':True,
                      'budget_policy':self.prompt_budget.describe(), 'calls':[]}
        self._save()

    def _save(self):
        temporary = self.ledger_path.with_suffix('.pending.json')
        with temporary.open('w') as stream:
            json.dump(self.state, stream, indent=2)
            stream.flush()
            __import__('os').fsync(stream.fileno())
        temporary.chmod(0o600)
        temporary.replace(self.ledger_path)

    def complete(self, messages, tools, max_tokens, timeout):
        with self._lock:
            prompt_reservation = self.prompt_budget.estimate(messages, tools)
            reservation = prompt_reservation + max_tokens
            if not self.state['accounting_complete']:
                raise AgentError('accounting_incomplete', 'Earlier provider accounting is unresolved; live calls stopped.')
            if self.state['actual_calls'] >= self.state['call_limit']:
                raise AgentError('global_call_budget_exhausted', 'Authorized live call limit reached.')
            if self.state['reported_tokens']+self.state['unresolved_reserved_tokens']+reservation > self.state['global_token_limit']:
                raise AgentError('global_token_budget_exhausted', 'Conservative reservation exceeds the remaining authorized token budget.')
            index = self.state['actual_calls']+1
            entry = {'index':index, 'reserved_tokens':reservation, 'prompt_reservation':prompt_reservation, 'max_output_tokens':max_tokens, 'status':'in_flight', 'usage':None}
            self.state['calls'].append(entry)
            self.state['actual_calls'] += 1
            self.state['unresolved_reserved_tokens'] += reservation
            self._save()  # Reserve durably BEFORE any network request; never auto retry.
        try:
            response = self._provider.complete(messages,tools,max_tokens,timeout)
        except Exception:
            with self._lock:
                entry['status']='failed_usage_unknown'
                self.state['accounting_complete']=False
                self._save()
            raise
        # Only structured input and the provider adapter's already-filtered output
        # are saved. No key, HTTP headers, HTTP error body or hidden reasoning.
        exchange={'index':index,'messages':messages,'tools':tools,'max_output_tokens':max_tokens,
                  'response':response,'mock':False,'model':self.model}
        with (self.directory/f'exchange_{index:02d}.json').open('x') as stream:
            json.dump(exchange,stream,indent=2,ensure_ascii=False)
        usage=response.get('usage')
        with self._lock:
            if usage is None:
                entry['status']='response_usage_unknown'
                self.state['accounting_complete']=False
            else:
                entry['usage']=usage
                entry['status']='completed'
                self.state['reported_tokens'] += usage['total_tokens']
                self.state['unresolved_reserved_tokens'] -= reservation
                if (usage['total_tokens'] > reservation or usage['prompt_tokens'] > prompt_reservation
                    or usage['completion_tokens'] > max_tokens):
                    entry['status']='reservation_exceeded'
                    self.state['accounting_complete']=False
            self._save()
        return response
