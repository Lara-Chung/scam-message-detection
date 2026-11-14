"""Allowlisted request-bound tools. No network, arbitrary files, or executable input."""
from __future__ import annotations

import re
from typing import Literal, TypedDict

from jsonschema import Draft202012Validator, ValidationError

from src.agent.models import AgentError
from src.normalise import normalise_with_changes


class ViewArgs(TypedDict):
    view: Literal['original', 'normalised']


PATTERNS = {
    'url': r'(?i)\b(?:https?://|www\.)[^\s<>"\x27]+|(?<![@\w])(?:[a-z0-9-]+\.)+(?:com|org|net|io|co\.uk|sg)(?:/[^\s<>"\x27]*)?',
    'email': r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b',
    'phone': r'(?<!\w)\+?\d[\d ()-]{6,}\d(?!\w)',
    'amount': r'(?i)(?:[$£€]\s*\d[\d,]*(?:\.\d{1,2})?|\b(?:USD|SGD|GBP|EUR)\s*\d[\d,]*(?:\.\d{1,2})?)',
    'urgency_phrase': r'(?i)\b(?:urgent|immediately|act now|within \d+ hours?|account (?:is |will be )?(?:locked|suspended)|final warning)\b',
    'information_request_phrase': r'(?i)\b(?:verify your (?:account|identity)|send (?:your )?(?:password|otp|code)|provide (?:your )?(?:password|bank details|ssn)|click (?:here|the link))\b',
}


def extract_indicators(text: str, view: str = 'original') -> list[dict]:
    found = []
    for kind, pattern in PATTERNS.items():
        for match in re.finditer(pattern, text):
            start, end = match.span()
            if kind == 'url':
                end = start + len(text[start:end].rstrip('.,;!?)'))
            if end > start:
                found.append({'kind': kind, 'view': view, 'start': start, 'end': end, 'quote': text[start:end][:500]})
                # Keep spans consistent when an unusually long URL is clipped.
                found[-1]['end'] = start + len(found[-1]['quote'])
            if len(found) >= 40:
                return found
    return found


def _schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


VIEW = {'type': 'string', 'enum': ['original', 'normalised']}
TOOL_SPECS = {
    'normalise_message': ('Normalise the current message; return exact change counts.', _schema({})),
    'classify_message': ('Classify a current text view using the configured saved model.', _schema({'view': VIEW})),
    'extract_indicators': ('Extract observed entities and request phrases with text spans.', _schema({'view': VIEW})),
    'retrieve_similar_cases': ('Find up to three similar canonical TRAIN cases in this channel.',
                               _schema({'view': VIEW, 'k': {'type': 'integer', 'minimum': 1, 'maximum': 3}})),
}


def definitions(retrieval=True):
    return [{'type': 'function', 'function': {'name': name, 'description': desc, 'parameters': schema}}
            for name, (desc, schema) in TOOL_SPECS.items() if retrieval or name != 'retrieve_similar_cases']


class RequestTools:
    def __init__(self, original, get_adapter, retrieval=True):
        self.views = {'original': original, 'normalised': None}
        self.get_adapter, self.retrieval = get_adapter, retrieval
        self.indicators, self.classifiers, self.cases = [], [], []

    def execute(self, name: str, args: dict):
        if name not in TOOL_SPECS or (name == 'retrieve_similar_cases' and not self.retrieval):
            raise AgentError('unknown_tool', 'Tool name is not allowlisted.')
        try:
            Draft202012Validator(TOOL_SPECS[name][1]).validate(args)
        except ValidationError as exc:
            raise AgentError('invalid_tool_arguments', 'Tool arguments fail the allowlisted schema.') from exc
        if name == 'normalise_message':
            text, changes = normalise_with_changes(self.views['original'])
            self.views['normalised'] = text
            return {'normalised_text': text, 'changes': changes,
                    'limitations': 'English only; no digit swaps or dotted-letter reconstruction.'}
        view = args['view']
        text = self.views[view]
        if text is None:
            raise AgentError('view_unavailable', 'Call normalise_message before selecting the normalised view.')
        if name == 'classify_message':
            result = self.get_adapter().classify(text, view)
            self.classifiers.append(result)
            return result
        if name == 'extract_indicators':
            result = extract_indicators(text, view)
            self.indicators.extend(x for x in result if x not in self.indicators)
            return result
        result = self.get_adapter().retrieve(text, args['k'])
        self.cases.extend(x for x in result if x not in self.cases)
        return result
