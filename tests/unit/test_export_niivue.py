"""Unit tests for the MZ3 scalar overlay writer.

These validate the bytes against the *real* MZ3 format NiiVue reads
(uint16 magic 23117 at offset 0, attr bitfield at offset 2 with
isSCALAR=8, 16-byte header, gzip-compressed), not a self-referential
constant — so a regression to the old broken layout fails here.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import pytest

from cortex_score.export.niivue import write_mz3_scalar_overlay

_MZ3_MAGIC = 0x5A4D  # 23117, ASCII "MZ"
_MZ3_ATTR_ISSCALAR = 8


def _decode_header(path: Path) -> tuple[int, int, int, int, int, np.ndarray]:
    raw = gzip.decompress(path.read_bytes())
    magic, attr, nface, nvert, nskip = struct.unpack("<HHIII", raw[:16])
    scalars = np.frombuffer(raw[16:], dtype=np.float32)
    return magic, attr, nface, nvert, nskip, scalars


def test_writes_file(tmp_path: Path) -> None:
    data = np.arange(20484, dtype=np.float32)
    out = write_mz3_scalar_overlay(data, tmp_path / "overlay.mz3")
    assert out.exists()
    assert out.stat().st_size > 0


def test_payload_is_gzip_with_niivue_magic(tmp_path: Path) -> None:
    data = np.zeros(100, dtype=np.float32)
    out = write_mz3_scalar_overlay(data, tmp_path / "x.mz3")
    # gzip stream (RFC 1952) so NiiVue's gzip-magic sniff recognizes it.
    assert out.read_bytes()[:2] == b"\x1f\x8b"
    magic, attr, nface, _nvert, nskip, _scalars = _decode_header(out)
    assert magic == _MZ3_MAGIC  # NiiVue rejects anything != 23117
    assert attr == _MZ3_ATTR_ISSCALAR
    assert nface == 0
    assert nskip == 0


def test_scalar_count_and_values_round_trip(tmp_path: Path) -> None:
    n = 250
    data = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    out = write_mz3_scalar_overlay(data, tmp_path / "x.mz3")
    _magic, _attr, _nface, nvert, _nskip, scalars = _decode_header(out)
    assert nvert == n
    np.testing.assert_array_equal(scalars, data)


def test_output_is_deterministic(tmp_path: Path) -> None:
    """Same input -> identical bytes (mtime pinned), so overlays are
    content-addressable and reproducible."""
    data = np.linspace(0.0, 5.0, 64, dtype=np.float32)
    a = write_mz3_scalar_overlay(data, tmp_path / "a.mz3").read_bytes()
    b = write_mz3_scalar_overlay(data, tmp_path / "b.mz3").read_bytes()
    assert a == b


def test_casts_non_float32_input(tmp_path: Path) -> None:
    data = np.arange(10, dtype=np.float64)
    out = write_mz3_scalar_overlay(data, tmp_path / "x.mz3")
    _magic, _attr, _nface, nvert, _nskip, scalars = _decode_header(out)
    assert nvert == 10
    assert scalars.dtype == np.float32
    np.testing.assert_array_equal(scalars, data.astype(np.float32))


def test_rejects_2d_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be 1D"):
        write_mz3_scalar_overlay(np.zeros((10, 10), dtype=np.float32), tmp_path / "x.mz3")
