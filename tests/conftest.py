"""Pytest configuration.

Adds the in-repo ``src/`` to ``sys.path`` so tests run against the
working tree without needing ``pip install -e .`` first. CI does an
editable install anyway, but this makes the local loop friction-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
