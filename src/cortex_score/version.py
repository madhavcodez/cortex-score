"""Single source of truth for the runtime version.

``hatch-vcs`` (``[tool.hatch.version] source = "vcs"`` in ``pyproject.toml``)
computes the version from git tags and bakes it into the installed
distribution metadata at build/install time. We read it back at runtime
via ``importlib.metadata`` so there is exactly one authority and no
generated ``_version.py`` artifact shipped in the wheel.

The fallback only triggers for a raw source tree that was never installed
(``importlib.metadata`` has no distribution to read). It is a
self-identifying sentinel — never a plausible-looking real version — so a
``ScoreResult`` produced from an uninstalled checkout cannot silently
claim a wrong provenance version.
"""

from __future__ import annotations

from importlib import metadata

try:
    __version__ = metadata.version("cortex-score")
except metadata.PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    __version__ = "0.0.0+unknown"
