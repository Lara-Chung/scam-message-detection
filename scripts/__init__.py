"""Shared scripts plus compatibility imports for agent/scripts commands."""
from pathlib import Path

_DIRECTORY = Path(__file__).resolve().parent
__path__ = [str(_DIRECTORY), str(_DIRECTORY.parent / 'agent' / 'scripts')]
