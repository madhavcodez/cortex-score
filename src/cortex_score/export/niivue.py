"""Niivue-compatible MZ3 scalar overlay writer.

Direct port of Cortexia's
``apps/worker/src/clipcortex_worker/pipeline/postprocess.py::write_mz3_scalar_overlay``.
The MZ3 format is a small zlib-compressed header + float32 scalars per
vertex; no external dependency is required to write it.

Why this is in ``export/`` rather than the main path: the MZ3 file is
useful for callers who want to drop their score into Niivue / NiftyWeb
visualization, but is NOT part of the JSON contract. It is opt-in.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import numpy.typing as npt

_MZ3_MAGIC = 0x4D5A_0003
_MZ3_ATTR_SCALAR = 1


def write_mz3_scalar_overlay(
    vertex_data: npt.NDArray[np.floating],
    output_path: str | Path,
) -> Path:
    """Write a Niivue-compatible MZ3 scalar overlay.

    Args:
        vertex_data: shape ``(V,)``, float32-compatible. One scalar per
            fsaverage5 cortical vertex.
        output_path: destination ``.mz3`` path.

    Returns:
        Resolved output path.
    """
    if vertex_data.ndim != 1:
        msg = f"vertex_data must be 1D (V,), got shape {vertex_data.shape}"
        raise ValueError(msg)

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    scalars = np.ascontiguousarray(vertex_data, dtype=np.float32)
    header = struct.pack(
        "<I H I I I",
        _MZ3_MAGIC,
        _MZ3_ATTR_SCALAR,
        0,
        int(scalars.shape[0]),
        0,
    )
    dest.write_bytes(zlib.compress(header + scalars.tobytes(), level=6))
    return dest
