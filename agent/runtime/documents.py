"""Bounded, request-local document windows with exact source coordinates.

No network, arbitrary files, HTML execution, fitted preprocessing or hidden text
replacement. Window ranking is a deterministic observation available to all modes.
"""
from __future__ import annotations

import hashlib
import re

from src.agent.models import AgentError
from src.agent.tools import extract_indicators
from src.normalise import normalise_with_changes

EMAIL = re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}')
TAG = re.compile(r'<[^>]{0,300}>')


class Document:
    def __init__(self, text, window_chars=2000, max_source_chars=2_000_000):
        if len(text) > max_source_chars:
            raise AgentError('source_too_large', 'Source exceeds the explicit document limit; no silent truncation.')
        self.text = text
        self.window_chars = window_chars
        self.sections = []
        for start in range(0, len(text), window_chars):
            chunk = text[start:start + window_chars]
            email_ratio = sum(m.end() - m.start() for m in EMAIL.finditer(chunk)) / len(chunk)
            markup_ratio = sum(m.end() - m.start() for m in TAG.finditer(chunk)) / len(chunk)
            indicators = extract_indicators(chunk)
            requests = sum(i['kind'] in ['information_request_phrase', 'urgency_phrase'] for i in indicators)
            letter_ratio = sum(c.isalpha() for c in chunk) / len(chunk)
            self.sections.append({
                'section_id': f's{len(self.sections):04d}', 'source_start': start,
                'source_end': start + len(chunk), 'email_ratio': round(email_ratio, 4),
                'markup_ratio': round(markup_ratio, 4), 'request_phrase_count': requests,
                'substantive': email_ratio < .35 and markup_ratio < .50 and letter_ratio > .12,
                'preview': chunk[:100],
            })
        self.by_id = {s['section_id']: s for s in self.sections}
        substantive = [s for s in self.sections if s['substantive']]
        self.initial_id = (substantive or self.sections)[0]['section_id']
        # Same declared candidates for fixed workflow and adaptive tool access.
        ranked = sorted(self.sections, key=lambda s: (-s['request_phrase_count'], not s['substantive'], s['source_start']))
        picks = [self.by_id[self.initial_id], self.sections[0], *ranked[:3], self.sections[-1]]
        self.recommended_ids = list(dict.fromkeys(s['section_id'] for s in picks))

    def overview(self):
        return {'source_chars': len(self.text), 'source_sha256': hashlib.sha256(self.text.encode()).hexdigest(),
                'window_chars': self.window_chars, 'total_sections': len(self.sections),
                'initial_section_id': self.initial_id,
                'recommended_sections': [self.by_id[k] for k in self.recommended_ids],
                'limitations': 'Fixed non-overlapping source windows; rankings are observations, not fraud labels. Unread content remains unknown.'}

    def read(self, section_id):
        if section_id not in self.by_id:
            raise AgentError('unknown_section', 'Section ID is not in this request-bound document.')
        meta = self.by_id[section_id]
        raw = self.text[meta['source_start']:meta['source_end']]
        normal, changes = normalise_with_changes(raw)
        suffix = section_id[1:]
        return dict(meta, original=raw, normalised=normal, changes=changes,
                    original_view=f'section_{suffix}_original', normalised_view=f'section_{suffix}_normalised')
