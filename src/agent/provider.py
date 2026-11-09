"""DeepSeek non-thinking Chat Completions protocol, without SDK or implicit retries.

Protocol: https://api-docs.deepseek.com/guides/tool_calls/
No provider/model fallback. The configured endpoint is trusted operator input.
"""
from __future__ import annotations

import json
import os
import queue
import threading
from urllib.parse import urlparse

import httpx

from src.agent.models import AgentError


def bounded_call(function, seconds):
    """Bound caller wall time. A timed-out native/HTTP operation may finish in its daemon.

    It cannot schedule later actions. Request state is isolated and is discarded
    after timeout. There is no unsafe thread cancellation or automatic retry.
    """
    if seconds <= 0:
        raise AgentError('timeout', 'Message wall-clock deadline exhausted.')
    result = queue.Queue(maxsize=1)

    def run():
        try:
            result.put((True, function()))
        except Exception as exc:
            result.put((False, exc))

    threading.Thread(target=run, daemon=True).start()
    try:
        success, value = result.get(timeout=seconds)
    except queue.Empty as exc:
        raise AgentError('timeout', 'Message wall-clock deadline exhausted.') from exc
    if success:
        return value
    raise value


class DeepSeekProvider:
    is_mock = False
    kind = 'deepseek_chat'

    def __init__(self, api_key: str, base_url: str, model: str, transport=None, tokenizer_path=None):
        parsed = urlparse(base_url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('LLM_BASE_URL must be an HTTPS API base URL without credentials/query.')
        self._key, self.base_url, self.model = api_key, base_url.rstrip('/'), model
        self._transport = transport
        from src.agent.budget import PromptBudget
        self.prompt_budget = PromptBudget.from_file(tokenizer_path, model) if tokenizer_path else PromptBudget()

    @classmethod
    def from_env(cls):
        key, base, model = (os.getenv(k) for k in ['LLM_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL'])
        if not all([key, base, model]):
            return None
        return cls(key, base, model)

    @classmethod
    def from_config(cls, config):
        """Read explicit local profile values and an opt-in Keychain reference."""
        key = os.getenv('LLM_API_KEY')
        reference = config.get('keychain')
        if not key and reference:
            from src.agent.local_credentials import read_keychain_secret
            key = read_keychain_secret(reference['service'], reference['account'])
        base = os.getenv('LLM_BASE_URL') or config.get('base_url')
        model = os.getenv('LLM_MODEL') or config.get('model')
        if not all([key, base, model]):
            return None
        return cls(key, base, model, tokenizer_path=config.get('tokenizer_path'))

    def complete(self, messages, tools, max_tokens, timeout):
        body = {'model': self.model, 'messages': messages, 'max_tokens': max_tokens,
                'stream': False, 'thinking': {'type': 'disabled'}, 'temperature': 0}
        if tools:
            body.update(tools=tools, tool_choice='auto')
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout), transport=self._transport, follow_redirects=False) as client:
                with client.stream('POST', self.base_url+'/chat/completions',
                                   headers={'Authorization': 'Bearer '+self._key}, json=body) as response:
                    response.raise_for_status()
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        if size > 1_000_000:
                            raise AgentError('provider_response_too_large', 'Provider response exceeded the byte limit.')
                        chunks.append(chunk)
                    data = json.loads(b''.join(chunks))
        except AgentError:
            raise
        except httpx.TimeoutException as exc:
            raise AgentError('timeout', 'Provider request timed out.') from exc
        except Exception as exc:
            # Do not echo HTTP bodies, headers, credentials or provider diagnostics.
            raise AgentError('provider_failure', 'Provider request failed; inspect endpoint/configuration privately.') from exc
        try:
            message = data['choices'][0]['message']
            finish = data['choices'][0].get('finish_reason')
            if finish not in ['stop', 'tool_calls']:
                raise AgentError('provider_incomplete', 'Provider did not finish a complete answer or tool request.')
            if message.get('role') != 'assistant':
                raise ValueError('invalid role')
            # Non-thinking mode: never persist or expose reasoning_content.
            clean = {'role': 'assistant', 'content': message.get('content')}
            calls = message.get('tool_calls')
            if calls:
                if not isinstance(calls, list) or len(calls) > 32:
                    raise ValueError('invalid tool calls')
                clean['tool_calls'] = []
                ids = set()
                for call in calls:
                    name, args, ident = call['function']['name'], call['function']['arguments'], call['id']
                    if (call.get('type') != 'function' or not all(isinstance(v, str) for v in [name, args, ident])
                        or not ident or ident in ids or len(args) > 4000 or len(name) > 100 or len(ident) > 200):
                        raise ValueError('malformed call')
                    ids.add(ident)
                    clean['tool_calls'].append({'id': ident, 'type': 'function', 'function': {'name': name, 'arguments': args}})
            if not calls and not isinstance(clean['content'], str):
                raise ValueError('missing final content')
            usage = data.get('usage')
            if usage is not None:
                for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
                    if type(usage[key]) is not int or usage[key] < 0:
                        raise ValueError('invalid usage')
                if usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']:
                    raise ValueError('inconsistent usage')
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError('provider_protocol_error', 'Malformed Chat Completions response.') from exc
        return {'message': clean, 'usage': usage}
