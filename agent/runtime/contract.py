"""Versioned public JSON contract and exact evidence validation."""
import json
from pathlib import Path

from jsonschema import Draft202012Validator

HERE = Path(__file__).parent
RESULT_SCHEMA = json.loads((HERE/'schema.json').read_text())
FINAL_SCHEMA = json.loads((HERE/'final_schema.json').read_text())
RESULT_VALIDATOR = Draft202012Validator(RESULT_SCHEMA)
FINAL_VALIDATOR = Draft202012Validator(FINAL_SCHEMA)


def validate_result(result):
    RESULT_VALIDATOR.validate(result)
    for item in result['evidence'] + result['indicators']:
        validate_span(item, result['views'])
    json.dumps(result, allow_nan=False)
    return result


def validate_span(item, views):
    text = views.get(item['view'])
    a, b = item['start'], item['end']
    if text is None or not 0 <= a < b <= len(text) or text[a:b] != item['quote']:
        raise ValueError('Evidence quote/span does not match its named text view.')


def resolve_span(item, views):
    """Accept exact offsets, or anchor an unchanged unique literal quote.

    Absent/ambiguous text never gets repaired. Original coordinates are returned
    for the caller to record as a visible review event, rather than hidden edits.
    """
    try:
        validate_span(item, views)
        return dict(item), None
    except ValueError:
        text = views.get(item['view'])
        quote = item['quote']
        if text is None or not quote:
            raise ValueError('Evidence quote is unavailable in its named view.')
        start = text.find(quote)
        if start < 0 or text.find(quote, start+1) >= 0:
            raise ValueError('Evidence quote is absent or ambiguous in its named view.')
        fixed = dict(item, start=start, end=start+len(quote))
        validate_span(fixed, views)
        return fixed, (item['start'], item['end'])


def apply_final_answer(result, final):
    """Keep exact quote text; unresolved evidence support yields abstention."""
    FINAL_VALIDATOR.validate(final)
    valid = []
    for quote in final['evidence']:
        try:
            fixed, previous = resolve_span(quote, result['views'])
            valid.append(fixed)
            if previous:
                result['errors'].append({'code':'evidence_span_repaired','source':'validation',
                    'message':f"Unique exact quote in {fixed['view']} re-anchored from [{previous[0]},{previous[1]}) to [{fixed['start']},{fixed['end']}); review required."})
        except ValueError:
            result['unsupported_claims'].append('A proposed evidence quote did not match its named text view.')
    result['evidence'] = valid
    result['decision'] = final['decision']
    result['needs_review'] = final['needs_review'] or final['decision'] == 'abstain' or bool(result['errors'])
    if result['unsupported_claims'] or (not valid and final['decision'] != 'abstain'):
        result['needs_review'] = True
        result['decision'] = 'abstain'
        if not valid:
            result['unsupported_claims'].append('Binary assessment lacked a validated supporting quote.')
    if any(c['provenance'] != 'canonical_train_verified' for c in result['classifier_results']):
        result['needs_review'] = True
    quotes = '; '.join(json.dumps(q['quote'], ensure_ascii=False) for q in valid)
    result['explanation'] = (f"Assessment: {result['decision']}. Cited message text: {quotes or 'none'}. "
                             'External verification unavailable; quotes require contextual human review.')[:2000]
    result['stop_reason'] = 'final_answer'
    return result


def new_result(raw, channel, mode, max_chars, views=None, indicators=None, classifiers=None, cases=None):
    """Shared version 1.0 envelope; route modes add their explicit 1.1 metadata."""
    import uuid
    used = raw[:max_chars]
    views = views if views is not None else {'original':used,'normalised':None}
    indicators = indicators if indicators is not None else []
    classifiers = classifiers if classifiers is not None else []
    cases = cases if cases is not None else []
    result = {'schema_version': '1.0', 'request_id': str(uuid.uuid4()), 'mode': mode, 'channel': channel,
            'status': 'error', 'decision': 'abstain', 'needs_review': True, 'explanation': '',
            'views': views, 'input_policy': {'name': 'prefix_chars_v1', 'max_chars': max_chars,
                'original_chars': len(raw), 'used_chars': len(used), 'clipped': len(raw)>len(used), 'original_offset': 0},
            'indicators': indicators, 'evidence': [], 'classifier_results': classifiers,
            'retrieved_cases': cases, 'tool_trace': [], 'elapsed_ms': 0, 'llm_calls': 0,
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0, 'accounting_complete': True},
            'cost_estimate': None, 'provider': None, 'stop_reason': '', 'errors': [], 'unsupported_claims': []}
    return result
