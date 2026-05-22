"""5-network rollup (visual / language / faces / attention / motion).

The grouping itself is a Cortexia product-design decision (the
``cortexia-network-groups-v1`` source), NOT a canonical Yeo-17
decomposition. The mapping is loaded from
``cortex_score/data/network_groups.json`` and exposed here as a typed,
immutable tuple. Each group carries its own ``yeo_indices`` so it stays
fully auditable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt

NetworkId = Literal["visual", "language", "faces", "attention", "motion"]
"""Canonical id type for the 5-network rollup.

This is the typed contract downstream code can match on. Changes here
are breaking (require ``schema_version`` bump).
"""

_DATA_PKG = "cortex_score.data"


@dataclass(frozen=True)
class NetworkGroup:
    """One dashboard network group.

    Attributes:
        id: Stable lowercase id used in JSON and Python.
        label: Human-facing label.
        description: Short human-facing description.
        color: Hex color used by visualization tooling.
        color_rgb: RGB triple in 0..1 float space (matches ``color``).
        yeo_indices: Zero-indexed Yeo-17 parcel ids included in this group.
        yeo_labels: Yeo-17 parcel labels included in this group (parallel to ``yeo_indices``).
    """

    id: NetworkId
    label: str
    description: str
    color: str
    color_rgb: tuple[float, float, float]
    yeo_indices: tuple[int, ...]
    yeo_labels: tuple[str, ...]


def _load_raw() -> dict[str, Any]:
    text = files(_DATA_PKG).joinpath("network_groups.json").read_text(encoding="utf-8")
    return cast(dict[str, Any], json.loads(text))


def _parse_groups(raw: dict[str, Any]) -> tuple[NetworkGroup, ...]:
    groups: list[NetworkGroup] = []
    for item in raw["groups"]:
        groups.append(
            NetworkGroup(
                id=cast(NetworkId, item["id"]),
                label=str(item["label"]),
                description=str(item["description"]),
                color=str(item["color"]),
                color_rgb=cast(
                    tuple[float, float, float],
                    tuple(float(c) for c in item["colorRgb"]),
                ),
                yeo_indices=tuple(int(i) for i in item["yeoIndices"]),
                yeo_labels=tuple(str(label) for label in item["yeoLabels"]),
            )
        )
    return tuple(groups)


_RAW = _load_raw()

YEO_LABELS: tuple[str, ...] = tuple(str(label) for label in _RAW["yeoLabels"])
"""All 17 Yeo network labels in canonical order."""

NETWORK_GROUPS: tuple[NetworkGroup, ...] = _parse_groups(_RAW)
"""The 5-network rollup (visual, language, faces, attention, motion)."""

NETWORK_IDS: tuple[NetworkId, ...] = tuple(g.id for g in NETWORK_GROUPS)


@lru_cache(maxsize=1)
def yeo_to_network_indices() -> dict[int, NetworkId]:
    """Reverse map: yeo parcel id -> network id.

    Yeo parcels that aren't included in any of the 5 dashboard groups
    (e.g. LimbicA, LimbicB, SomMotA when only motor cortex is grouped,
    etc.) simply don't appear in the dict.
    """
    out: dict[int, NetworkId] = {}
    for group in NETWORK_GROUPS:
        for yeo_id in group.yeo_indices:
            out[yeo_id] = group.id
    return out


def build_network_summary(
    z_yeo_preds: npt.NDArray[np.float32],
) -> list[dict[str, Any]]:
    """Build per-network summary dicts from z-scored Yeo-17 predictions.

    Args:
        z_yeo_preds: shape ``(T, n_yeo)``, float32. Each Yeo parcel
            already z-scored within the clip (see
            ``processing.normalize.zscore_within_atlas``).

    Returns:
        A list of dicts (one per 5-network group) ready to be passed
        into ``ScoreResult.networks``. Each dict carries enough fields
        to construct a ``NetworkScore`` Pydantic model.

    Notes:
        ``mean_energy`` and ``peak_energy`` are computed from the mean
        absolute z-score across the group's member Yeo parcels at each
        timestep. ``mean_z_timeseries`` is the (signed) mean across
        member parcels.

        Definitions (canonical):

            energy_t          = mean_i( |z_yeo_preds[t, i]| )  for i in group
            mean_z_t          =  mean_i(  z_yeo_preds[t, i]  )  for i in group
            mean_energy       =  mean_t( energy_t )
            peak_energy       =  max_t(  energy_t )

        These metrics make sense ONLY relative to the same clip
        because the input is z-scored within-clip (see ``schemas.NormalizationMeta``).
    """
    if z_yeo_preds.ndim != 2:
        msg = f"z_yeo_preds must be 2D (T, R), got shape {z_yeo_preds.shape}"
        raise ValueError(msg)

    n_yeo = z_yeo_preds.shape[1]
    summaries: list[dict[str, Any]] = []

    for group in NETWORK_GROUPS:
        indices = [idx for idx in group.yeo_indices if idx < n_yeo]
        if not indices:
            energy_ts = np.zeros(z_yeo_preds.shape[0], dtype=np.float32)
            mean_z_ts = np.zeros(z_yeo_preds.shape[0], dtype=np.float32)
        else:
            group_ts = z_yeo_preds[:, indices]
            energy_ts = np.abs(group_ts).mean(axis=1).astype(np.float32)
            mean_z_ts = group_ts.mean(axis=1).astype(np.float32)

        summaries.append(
            {
                "id": group.id,
                "label": group.label,
                "description": group.description,
                "color": group.color,
                "yeo_indices": list(group.yeo_indices),
                "yeo_labels": list(group.yeo_labels),
                "mean_energy": float(energy_ts.mean()),
                "peak_energy": float(energy_ts.max()) if energy_ts.size else 0.0,
                "energy_timeseries": [float(v) for v in energy_ts],
                "mean_z_timeseries": [float(v) for v in mean_z_ts],
            }
        )

    return summaries
