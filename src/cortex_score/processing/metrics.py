"""Per-ROI scalar metrics computed from a z-scored time series.

Three scalar numbers per ROI:

    mean_energy         = mean_t( |z_roi[t]| )
    peak_energy         =  max_t( |z_roi[t]| )
    temporal_volatility = std_t( diff(z_roi)[t] )    # 0 if T < 2

All fields are unit-less because the input is z-scored. Definitions are
intentionally simple and stable; anything more complex should live in
downstream consumers where it can be revised without re-running the
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

METRICS_VERSION: str = "1.0"
"""Bumped on any change to ROIMetrics or ``compute_roi_metrics``.

Recorded into the score cache key so old cached results are invalidated.
"""


@dataclass(frozen=True)
class ROIMetrics:
    """Scalar metrics for a single ROI's time course."""

    mean_energy: float
    peak_energy: float
    temporal_volatility: float


def compute_roi_metrics(
    z_roi_preds: npt.NDArray[np.float32],
) -> list[ROIMetrics]:
    """Compute per-ROI summary metrics over a z-scored ``(T, R)`` array.

    Args:
        z_roi_preds: shape ``(T, R)``, float32. Z-scored ROI time
            series, as produced by ``zscore_within_atlas``.

    Returns:
        Length-R list of ``ROIMetrics``, ordered by ROI id 0..R-1.
    """
    if z_roi_preds.ndim != 2:
        msg = f"z_roi_preds must be 2D (T, R); got shape {z_roi_preds.shape}"
        raise ValueError(msg)

    abs_z = np.abs(z_roi_preds)
    mean_energy = abs_z.mean(axis=0)  # shape (R,)
    peak_energy = abs_z.max(axis=0)  # shape (R,)

    if z_roi_preds.shape[0] >= 2:
        diffs = np.diff(z_roi_preds, axis=0)
        temporal_volatility = diffs.std(axis=0)
    else:
        temporal_volatility = np.zeros(z_roi_preds.shape[1], dtype=np.float32)

    return [
        ROIMetrics(
            mean_energy=float(mean_energy[i]),
            peak_energy=float(peak_energy[i]),
            temporal_volatility=float(temporal_volatility[i]),
        )
        for i in range(z_roi_preds.shape[1])
    ]
