"""Bundled atlas loader.

All atlas resources live inside the wheel under ``cortex_score/data/``
and are loaded via ``importlib.resources``. The total payload is < 400
KB (six small files), so the package stays offline-first and the atlas
version is pinned to the cortex-score version — no surprise drift if
upstream Schaefer/Yeo releases ever change.

We deliberately do NOT support runtime atlas swapping in v0.1: every
``ScoreResult`` is computed against the bundled atlas, fingerprinted by
the SHA-256 in ``data/manifest.json``. If a future version needs custom
atlases, the entry point will be a separate ``AtlasProvider`` Protocol
rather than ad-hoc kwargs.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from cortex_score.exceptions import AtlasMismatchError

_DATA_PKG = "cortex_score.data"


@dataclass(frozen=True)
class AtlasAssets:
    """Pre-loaded atlas tensors for one parcellation.

    Attributes:
        name: Stable atlas identifier (e.g. ``"schaefer400"`` or ``"yeo17"``).
        n_parcels: Number of parcels (excluding the medial wall).
        vertex_to_parcel: shape ``(V,)`` int64. Parcel id per fsaverage5
            vertex; ``-1`` for medial wall.
        parcel_labels: Length-``n_parcels`` list of human-readable labels.
        sha256: SHA-256 of the underlying ``.npy`` file (for provenance).
    """

    name: str
    n_parcels: int
    vertex_to_parcel: npt.NDArray[np.int64]
    parcel_labels: list[str]
    sha256: str


@dataclass(frozen=True)
class AtlasManifest:
    """Parsed ``data/manifest.json`` (mirror of the on-disk JSON).

    Used by ScoreResult.atlas provenance and by the cache to invalidate
    on atlas changes.
    """

    schema_version: str
    atlas_version: str
    mesh: str
    n_vertices: int
    network_group_source: str
    file_shas: dict[str, str]


def _read_bytes(filename: str) -> bytes:
    return cast(bytes, files(_DATA_PKG).joinpath(filename).read_bytes())


def _read_text(filename: str) -> str:
    return cast(str, files(_DATA_PKG).joinpath(filename).read_text(encoding="utf-8"))


def _load_npy(filename: str) -> tuple[np.ndarray, str]:
    """Load a bundled ``.npy`` file and return ``(array, sha256_hex)``."""
    raw = _read_bytes(filename)
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    return arr, hashlib.sha256(raw).hexdigest()


def _load_labels(filename: str) -> list[str]:
    raw = json.loads(_read_text(filename))
    if not isinstance(raw, list):
        msg = f"labels file {filename} must contain a JSON list of strings"
        raise AtlasMismatchError(msg)
    return [str(x) for x in raw]


def _validate_atlas(
    *,
    name: str,
    vertex_to_parcel: np.ndarray,
    labels: list[str],
) -> int:
    """Run integrity checks on an atlas. Returns ``n_parcels``."""
    if vertex_to_parcel.ndim != 1:
        msg = f"{name}: vertex_to_parcel must be 1D, got shape {vertex_to_parcel.shape}"
        raise AtlasMismatchError(msg)

    valid_ids = vertex_to_parcel[vertex_to_parcel >= 0]
    if valid_ids.size == 0:
        msg = f"{name}: no valid parcel assignments (entire atlas is medial wall)"
        raise AtlasMismatchError(msg)

    n_parcels = int(valid_ids.max()) + 1
    if n_parcels != len(labels):
        msg = (
            f"{name}: vertex assignment max id ({n_parcels - 1}) "
            f"does not match labels length ({len(labels)})"
        )
        raise AtlasMismatchError(msg)
    return n_parcels


@lru_cache(maxsize=1)
def load_manifest() -> AtlasManifest:
    """Load and parse ``data/manifest.json``."""
    raw: dict[str, Any] = json.loads(_read_text("manifest.json"))
    files_list = raw.get("files", [])
    file_shas = {entry["name"]: entry["sha256"] for entry in files_list}
    return AtlasManifest(
        schema_version=str(raw["schema_version"]),
        atlas_version=str(raw["atlas_version"]),
        mesh=str(raw["mesh"]),
        n_vertices=int(raw["n_vertices"]),
        network_group_source=str(raw.get("network_group_source", "unknown")),
        file_shas=file_shas,
    )


def _assert_sha_matches(manifest: AtlasManifest, filename: str, observed: str) -> None:
    expected = manifest.file_shas.get(filename)
    if expected and expected != observed:
        msg = (
            f"SHA-256 of bundled '{filename}' ({observed}) does not match "
            f"manifest.json ({expected}). The wheel may be corrupted."
        )
        raise AtlasMismatchError(msg)


@lru_cache(maxsize=1)
def load_schaefer400() -> AtlasAssets:
    """Load the Schaefer-400 (17-network projection) atlas."""
    manifest = load_manifest()
    arr, sha = _load_npy("schaefer400_vertex.npy")
    _assert_sha_matches(manifest, "schaefer400_vertex.npy", sha)
    labels = _load_labels("labels_schaefer400.json")
    n_parcels = _validate_atlas(
        name="schaefer400",
        vertex_to_parcel=arr,
        labels=labels,
    )
    return AtlasAssets(
        name="schaefer400",
        n_parcels=n_parcels,
        vertex_to_parcel=arr.astype(np.int64, copy=False),
        parcel_labels=labels,
        sha256=sha,
    )


@lru_cache(maxsize=1)
def load_yeo17() -> AtlasAssets:
    """Load the Yeo-17 functional network atlas."""
    manifest = load_manifest()
    arr, sha = _load_npy("yeo17_vertex.npy")
    _assert_sha_matches(manifest, "yeo17_vertex.npy", sha)
    labels = _load_labels("labels_yeo17.json")
    n_parcels = _validate_atlas(
        name="yeo17",
        vertex_to_parcel=arr,
        labels=labels,
    )
    return AtlasAssets(
        name="yeo17",
        n_parcels=n_parcels,
        vertex_to_parcel=arr.astype(np.int64, copy=False),
        parcel_labels=labels,
        sha256=sha,
    )


@lru_cache(maxsize=1)
def load_schaefer400_to_yeo17() -> npt.NDArray[np.int64]:
    """Load the Schaefer-400 -> Yeo-17 mapping (shape ``(400,)``)."""
    manifest = load_manifest()
    arr, sha = _load_npy("schaefer400_to_yeo17.npy")
    _assert_sha_matches(manifest, "schaefer400_to_yeo17.npy", sha)
    if arr.ndim != 1 or arr.shape[0] != load_schaefer400().n_parcels:
        msg = (
            f"schaefer400_to_yeo17 has shape {arr.shape}; "
            f"expected (n_schaefer={load_schaefer400().n_parcels},)"
        )
        raise AtlasMismatchError(msg)
    return arr.astype(np.int64, copy=False)
