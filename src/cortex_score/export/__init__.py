"""Optional export formats.

The MZ3 scalar overlay writer is a thin port from Cortexia
(``apps/worker/src/clipcortex_worker/pipeline/postprocess.py``). It has
no runtime dep beyond stdlib + NumPy, so it lives in the base install,
but it's only useful for callers visualizing cortical scalars in
Niivue.
"""

from __future__ import annotations

from cortex_score.export.niivue import write_mz3_scalar_overlay

__all__ = ["write_mz3_scalar_overlay"]
