"""Shared prompt reservations, with a pinned official tokenizer and framing margin."""
from __future__ import annotations

import math
from pathlib import Path

from src.agent.data import file_hash
from src.agent.models import AgentError

TOKENIZER_SHA256 = '89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614'
TOKENIZER_MODELS = {'deepseek-flash', 'deepseek-v4-pro'}


class PromptBudget:
    """The local tokenizer does not encode the server's private chat/tool framing.

    Count the complete serialized request, then reserve 25% plus 256 tokens for
    framing/tokenizer differences. Never treat this estimate as billed usage.
    With no explicit tokenizer, keep the conservative UTF-8-byte fallback.
    """
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    @classmethod
    def from_file(cls, path, model):
        if model not in TOKENIZER_MODELS:
            raise AgentError('tokenizer_model_mismatch', 'Pinned DeepSeek V4 tokenizer is not configured for this model ID.')
        path = Path(path)
        if not path.is_file():
            raise AgentError('tokenizer_missing', 'Run python -m scripts.prepare_agent_tokenizer before live inference.')
        if file_hash(path) != TOKENIZER_SHA256:
            raise AgentError('tokenizer_hash_mismatch', 'Tokenizer differs from the reviewed official data.')
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(path))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        return cls(tokenizer)

    def estimate(self, messages, tools):
        # Count the same complete JSON envelope that the provider serializes.
        import json
        serialized = json.dumps({'messages':messages, 'tools':tools}, ensure_ascii=False)
        if self.tokenizer is None:
            return len(serialized.encode()) + 1024
        count = len(self.tokenizer.encode(serialized, add_special_tokens=False).ids)
        return math.ceil(count * 1.25) + 256

    def describe(self):
        if self.tokenizer is None:
            return {'name':'utf8_bytes_v1','framing_tokens':1024}
        return {'name':'deepseek_v4_tokenizer_v1','tokenizer_sha256':TOKENIZER_SHA256,
                'multiplier':1.25,'framing_tokens':256,
                'source':'https://api-docs.deepseek.com/quick_start/token_usage/'}
