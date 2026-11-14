"""Minimal Python frontend integration: no UI imports or session required."""
import json
from pathlib import Path

from src.agent.service import ScamAgentService

CONFIG = Path(__file__).resolve().parents[2]/'configs/agent.yaml'


def main():
    service = ScamAgentService.from_config(CONFIG)  # Cache this instance in the frontend.
    # Explicit free mode; omit mode to use the configured adaptive workflow.
    result = service.analyze_message(
        text='Please verify your account immediately.', channel='sms', mode='document_classifier')
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main()
