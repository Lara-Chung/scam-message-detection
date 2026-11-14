# Agent backend

This component contains the adaptive message-investigation backend and its commands.

| Directory | Content |
|---|---|
| `runtime/` | Model adapters, document tools, provider, bounded Agent loop and JSON schemas |
| `scripts/` | Data preparation, calibration, live-run accounting, comparisons and reports |
| `__main__.py` | CLI entry point for `python -m agent` |

Run from the repository root:

```sh
python -m agent --help
python -m agent --text 'Meeting confirmed for tomorrow.' --channel sms --mode document_classifier
python -m agent.scripts.prepare_agent_inputs --help
```

The existing API and command aliases remain available:

```python
from src.agent.service import ScamAgentService

service = ScamAgentService.from_config('configs/agent.yaml')
result = service.analyze_message(message, channel='sms')
```

`python -m src.agent` and `python -m scripts.<command>` load the same canonical
files. `src/agent/` and `scripts/__init__.py` retain these public import paths;
there are no copied backend implementations. Runtime configuration remains in
[`configs/`](../configs), shared by the CLI and frontend.

Use the isolated [`requirements-agent.txt`](../requirements-agent.txt) environment.
See the [frontend handoff](../docs/agent_frontend_handoff.md),
[design and validation](../docs/adaptive_agent_handoff.md) and
[project directory guide](../docs/PROJECT_STRUCTURE.md).

Local inputs, prepared models, provider profiles and execution audits remain
ignored in their existing locations. Reorganising code does not rerun paid experiments.
