"""Unit tests for the bundled-atlas loader.

These tests prove the data fingerprints actually match what
``data/manifest.json`` says — protecting against a corrupted wheel,
a bad release script, or an accidental swap of the .npy contents.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files

import pytest

from cortex_score.atlas import (
    load_manifest,
    load_schaefer400,
    load_schaefer400_to_yeo17,
    load_yeo17,
)
from cortex_score.exceptions import AtlasMismatchError


def _file_sha(filename: str) -> str:
    return hashlib.sha256(files("cortex_score.data").joinpath(filename).read_bytes()).hexdigest()


def test_manifest_parses() -> None:
    m = load_manifest()
    assert m.schema_version == "1.0"
    assert m.mesh == "fsaverage5"
    assert m.n_vertices == 20484
    assert m.atlas_version == "schaefer2018-400-yeo17-fsaverage5"
    assert m.network_group_source == "cortexia-network-groups-v1"


def test_manifest_shas_match_bundled_files() -> None:
    m = load_manifest()
    for name, expected in m.file_shas.items():
        actual = _file_sha(name)
        assert actual == expected, f"{name}: expected {expected[:12]}, got {actual[:12]}"


def test_schaefer400_loads_400_parcels() -> None:
    a = load_schaefer400()
    assert a.n_parcels == 400
    assert a.vertex_to_parcel.shape == (20484,)
    assert a.vertex_to_parcel.dtype.name == "int64"
    assert len(a.parcel_labels) == 400


def test_yeo17_loads_17_networks() -> None:
    a = load_yeo17()
    assert a.n_parcels == 17
    assert a.vertex_to_parcel.shape == (20484,)
    assert len(a.parcel_labels) == 17


def test_schaefer400_to_yeo17_map_in_range() -> None:
    m = load_schaefer400_to_yeo17()
    assert m.shape == (400,)
    assert m.min() >= 0
    assert m.max() <= 16  # 17 Yeo networks, ids 0..16


def test_load_manifest_is_cached() -> None:
    assert load_manifest() is load_manifest()


def test_atlas_mismatch_error_is_raised_on_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Manually invalidate the SHA cache and confirm a tampered manifest
    surfaces ``AtlasMismatchError``."""
    from typing import NoReturn

    from cortex_score import atlas as atlas_mod

    atlas_mod.load_schaefer400.cache_clear()  # type: ignore[attr-defined]
    atlas_mod.load_manifest.cache_clear()  # type: ignore[attr-defined]

    original = atlas_mod._assert_sha_matches

    def fake_assert(
        manifest: atlas_mod.AtlasManifest,
        filename: str,
        observed: str,
    ) -> NoReturn:
        raise AtlasMismatchError("simulated corruption")

    monkeypatch.setattr(atlas_mod, "_assert_sha_matches", fake_assert)
    with pytest.raises(AtlasMismatchError, match="simulated corruption"):
        atlas_mod.load_schaefer400()

    monkeypatch.setattr(atlas_mod, "_assert_sha_matches", original)
    atlas_mod.load_schaefer400.cache_clear()  # type: ignore[attr-defined]
    atlas_mod.load_manifest.cache_clear()  # type: ignore[attr-defined]
