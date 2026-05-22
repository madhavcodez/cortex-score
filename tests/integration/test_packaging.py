"""Reviewer-required: wheel + sdist + twine check.

This test builds the actual artifacts that would ship to PyPI:

* ``python -m build`` (hatchling) -> sdist + wheel under ``dist/``
* ``twine check`` -> verifies long-description / metadata are valid
* Wheel contents must include bundled atlas data + license notices

Marked ``slow`` because ``python -m build`` is ~10 seconds. CI runs it
on tagged releases; the local loop should skip it via ``-m "not slow"``.
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


def _have_module(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


@pytest.mark.slow
@pytest.mark.skipif(not _have_module("build"), reason="build module not installed")
def test_python_m_build_produces_wheel_and_sdist(tmp_path: Path) -> None:
    out_dir = tmp_path / "dist"
    out_dir.mkdir()
    proc = subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(out_dir), str(_REPO)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"build failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    wheels = list(out_dir.glob("*.whl"))
    sdists = list(out_dir.glob("*.tar.gz"))
    assert len(wheels) == 1, f"expected 1 wheel, found {wheels}"
    assert len(sdists) == 1, f"expected 1 sdist, found {sdists}"

    # Wheel must include the bundled atlas data and license notices.
    expected_data = {
        "cortex_score/data/schaefer400_vertex.npy",
        "cortex_score/data/yeo17_vertex.npy",
        "cortex_score/data/schaefer400_to_yeo17.npy",
        "cortex_score/data/labels_schaefer400.json",
        "cortex_score/data/labels_yeo17.json",
        "cortex_score/data/network_groups.json",
        "cortex_score/data/manifest.json",
    }
    with zipfile.ZipFile(wheels[0]) as zf:
        names = set(zf.namelist())
    missing = expected_data - names
    assert not missing, f"wheel missing atlas data: {missing}"
    assert any("LICENSE" in n for n in names), "wheel must include LICENSE"


@pytest.mark.slow
@pytest.mark.skipif(not _have_module("twine"), reason="twine not installed")
def test_twine_check_passes(tmp_path: Path) -> None:
    out_dir = tmp_path / "dist"
    out_dir.mkdir()
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(out_dir), str(_REPO)],
        check=True,
        capture_output=True,
    )
    proc = subprocess.run(
        [sys.executable, "-m", "twine", "check", *map(str, out_dir.iterdir())],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"twine check failed:\n{proc.stdout}\n{proc.stderr}"
