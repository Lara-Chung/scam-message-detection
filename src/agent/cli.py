"""Single-message and JSONL CLI. stdout contains JSON only; exit 2 means errors."""
import argparse
import json
import sys

from src.agent.service import ScamAgentService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/agent.yaml')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--text')
    source.add_argument('--jsonl', help='UTF-8 JSONL objects with text and channel fields; use - for stdin.')
    parser.add_argument('--channel', default='sms')
    parser.add_argument('--mode', default=None, choices=['adaptive_agent', 'document_classifier', 'document_fixed', 'agent', 'classifier_only', 'direct_llm', 'fixed_workflow','fixed_model','rule_router','agent_router'])
    parser.add_argument('--token-budget', type=int, default=12000, help='Global token cap for this invocation, not per row.')
    parser.add_argument('--max-live-messages', type=int, default=3, help='Default validation smoke cap; raise explicitly for a planned run.')
    args = parser.parse_args()
    try:
        service = ScamAgentService.from_config(args.config)
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({'error': 'configuration_error', 'message': str(exc)}), file=sys.stderr)
        return 2
    if args.token_budget <= 0 or args.max_live_messages <= 0:
        parser.error('Budgets must be positive.')
    args.mode = args.mode or service.config.get('default_mode', 'agent')
    remaining, live, failed = args.token_budget, 0, False

    def run(row):
        nonlocal remaining, live, failed
        limited = False
        if args.mode in ['agent','direct_llm','fixed_workflow','agent_router','adaptive_agent','document_fixed'] and service.provider is not None:
            if live >= args.max_live_messages or remaining <= 0:
                limited = True
                if args.mode not in ['adaptive_agent', 'document_fixed']:
                    print(json.dumps({'error': 'global_budget_exhausted', 'message': 'Batch stopped before additional calls.'}), file=sys.stderr)
                    return False
        result = service.analyze_message(row.get('text'), row.get('channel', args.channel), args.mode,
                                         token_budget=max(1, remaining), context_answers=row.get('context_answers'),
                                         allow_paid_calls=not limited)
        if result['llm_calls']:
            live += 1
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        failed |= result['status'] == 'error'
        remaining -= result['usage']['total_tokens']
        if not result['usage']['accounting_complete']:
            remaining = 0
        return True

    if args.text is not None:
        run({'text': args.text, 'channel': args.channel})
    else:
        stream = sys.stdin if args.jsonl == '-' else open(args.jsonl, encoding='utf-8')
        try:
            for line_no, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError('JSON object required')
                except ValueError:
                    print(json.dumps({'error': 'invalid_jsonl', 'line': line_no}), file=sys.stderr)
                    failed = True
                    continue
                if not run(row):
                    failed = True
                    break
        finally:
            if stream is not sys.stdin:
                stream.close()
    return 2 if failed else 0

if __name__ == '__main__':
    sys.exit(main())
