"""Compatibility namespace for the backend now grouped under agent/runtime.

The public ``src.agent`` imports and ``python -m src.agent`` commands are stable.
Submodules load directly from the canonical files; implementation is not copied.
"""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[2] / 'agent' / 'runtime')]
