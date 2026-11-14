"""Compatibility entry point; implementation lives in traditional/lr_scam.py."""
import importlib
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == '__main__':
    runpy.run_module('traditional.lr_scam', run_name='__main__')
else:
    sys.modules[__name__] = importlib.import_module('traditional.lr_scam')
