"""Per-ROI normalization.

We z-score *within* each atlas (each column is independent) so that ROI
energy is comparable across ROIs within a clip. Cross-atlas z-scoring
is intentionally avoided.

Important scientific caveat: the z-score is computed within a single
clip, so the resulting numbers are only meaningful relative to that
clip's own timeline. To compare two clips on the same axis, callers
need a shared reference distribution (a future ``NormalizationMeta.scope
== "reference_distribution"`` mode — see ``schemas.NormalizationMeta``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

DEFAULT_EPS: float = 1e-6
"""Default ridge added to the std denominator.

A small constant prevents division by zero on flat ROIs. Recorded into
``ScoreResult.normalization.epsilon`` for provenance.
"""


def zscore_within_atlas(
    roi_preds: npt.NDArray[np.float32],
    *,
    eps: float = DEFAULT_EPS,
) -> npt.NDArray[np.float32]:
    """Z-score each ROI's time series independently.

    Args:
        roi_preds: shape ``(T, R)``, float32. ROI-level predictions.
        eps: small constant added to the std denominator to avoid
            division by zero on flat ROIs.

    Returns:
        shape ``(T, R)``, float32. Each column has approximately zero
        mean and unit std (constant columns return zeros).
    """
    if roi_preds.ndim != 2:
        msg = f"roi_preds must be 2D (T, R); got shape {roi_preds.shape}"
        raise ValueError(msg)

    # Accumulate in float64 so a column of constant float32 values
    # produces an exact zero std rather than a tiny numerical residue
    # that gets amplified to ~0.3 once divided by `eps`. (Bug history:
    # carried over from Cortexia normalize.py:36-42.)
    in_64 = roi_preds.astype(np.float64, copy=False)
    mean = in_64.mean(axis=0, keepdims=True)
    std = in_64.std(axis=0, keepdims=True)
    out = (in_64 - mean) / (std + eps)
    result: npt.NDArray[np.float32] = out.astype(np.float32)
    return result
