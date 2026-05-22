"""Unit tests for the MZ3 scalar overlay writer."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from cortex_score.export.niivue import write_mz3_scalar_overlay

_MZ3_MAGIC = 0x4D5A_0003


def test_writes_file(tmp_path: Path) -> None:
    data = np.arange(20484, dtype=np.float32)
    out = write_mz3_scalar_overlay(data, tmp_path / "overlay.mz3")
    assert out.exists()
    assert out.stat().st_size > 0


def test_payload_has_magic_after_decompression(tmp_path: Path) -> None:
    data = np.zeros(100, dtype=np.float32)
    out = write_mz3_scalar_overlay(data, tmp_path / "x.mz3")
    raw = zlib.decompress(out.read_bytes())
    magic = struct.unpack("<I", raw[:4])[0]
    assert magic == _MZ3_MAGIC


def test_scalar_count_matches_input(tmp_path: Path) -> None:
    n = 250
    data = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    out = write_mz3_scalar_overlay(data, tmp_path / "x.mz3")
    raw = zlib.decompress(out.read_bytes())
    # Header layout: <I H I I I  (4+2+4+4+4 = 18 bytes); scalar count is field 4
    count = struct.unpack("<I", raw[10:14])[0]
    assert count == n
    # Scalars round-trip exactly.
    scalars = np.frombuffer(raw[18:], dtype=np.float32)
    np.testing.assert_array_equal(scalars, data)


def test_rejects_2d_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be 1D"):
        write_mz3_scalar_overlay(np.zeros((10, 10), dtype=np.float32), tmp_path / "x.mz3")
