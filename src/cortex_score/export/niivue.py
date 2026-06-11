"""Niivue-compatible MZ3 scalar overlay writer.

Writes a per-vertex scalar overlay in the MZ3 format that NiiVue
(and surf-ice) read. The MZ3 layout is a fixed 16-byte little-endian
header followed by the float32 scalar payload, gzip-compressed as a
whole; no external dependency is required to write it.

Header (16 bytes, ``<HHIII``):

    offset 0  uint16  magic = 0x5A4D (23117, ASCII "MZ")
    offset 2  uint16  attr  = 8 (isSCALAR; per-vertex float32 scalars)
    offset 4  uint32  nface = 0 (no faces in an overlay)
    offset 8  uint32  nvert = number of scalars (one per vertex)
    offset 12 uint32  nskip = 0 (no extra bytes before the payload)

Why this is in ``export/`` rather than the main path: the MZ3 file is
useful for callers who want to drop their score into NiiVue / surf-ice
visualization, but is NOT part of the JSON contract. It is opt-in.

NOTE: an earlier port shipped a 0x4D5A0003 uint32 magic, ``attr=1``
(isFACE), an 18-byte header, and a raw zlib (not gzip) stream — NiiVue
rejects all of those at the magic check. The constants below are the
verified, NiiVue-readable values.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import numpy.typing as npt

# uint16 little-endian "MZ"; NiiVue requires getUint16(0) == 23117.
_MZ3_MAGIC = 0x5A4D
# attr bitfield: 1=isFACE, 2=isVERT, 4=isRGBA, 8=isSCALAR.
_MZ3_ATTR_ISSCALAR = 8
_MZ3_HEADER = "<HHIII"


def write_mz3_scalar_overlay(
    vertex_data: npt.NDArray[np.floating],
    output_path: str | Path,
) -> Path:
    """Write a NiiVue-compatible MZ3 scalar overlay.

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
        _MZ3_HEADER,
        _MZ3_MAGIC,
        _MZ3_ATTR_ISSCALAR,
        0,  # nface
        int(scalars.shape[0]),  # nvert
        0,  # nskip
    )
    # gzip (not raw zlib) so NiiVue's gzip-magic sniff recognizes it;
    # mtime=0 keeps the bytes deterministic for the same input.
    payload = gzip.compress(header + scalars.tobytes(), compresslevel=6, mtime=0)
    dest.write_bytes(payload)
    return dest
