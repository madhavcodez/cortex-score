"""Pre-flight validators for the postprocessing path.

These run before any aggregation/normalization and surface
shape/mesh/atlas mismatches with actionable messages. The high-level
``score_from_predictions`` and ``score_from_prediction_bundle`` APIs
call into these so that a bad input fails fast at the boundary, not
mid-pipeline.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from cortex_score.exceptions import IncompatiblePredictionShapeError


def validate_predictions_against_mesh(
    preds: npt.NDArray[np.floating],
    *,
    mesh: str,
    expected_n_vertices: int,
) -> None:
    """Raise if ``preds`` shape disagrees with the declared mesh.

    Args:
        preds: candidate prediction tensor, expected shape ``(T, V)``.
        mesh: declared mesh identifier (currently only ``"fsaverage5"`` is
            supported; future runners may add others).
        expected_n_vertices: vertex count of the bundled atlas for this mesh.

    Raises:
        IncompatiblePredictionShapeError: if ``preds`` is not 2D, has
            zero timesteps, or vertex count does not match.
        ValueError: on totally malformed input (zero dimensions).
    """
    if preds.ndim != 2:
        msg = (
            f"predictions must be 2D (T, V); got shape {preds.shape}. "
            f"If you have a 1D time-mean array, reshape to (1, V)."
        )
        raise ValueError(msg)

    t, v = preds.shape
    if t < 1:
        msg = f"predictions must have at least 1 timestep; got T={t}"
        raise ValueError(msg)

    if v != expected_n_vertices:
        raise IncompatiblePredictionShapeError(
            expected_n_vertices=expected_n_vertices,
            actual_n_vertices=v,
            mesh=mesh,
        )


def coerce_float32(preds: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """Return ``preds`` as float32, copying only when needed.

    TRIBE v2 returns float32; downstream code assumes float32. If a
    caller hands us float64 (e.g. they loaded ``.npy`` without dtype
    awareness), we cast once at the boundary.
    """
    return np.asarray(preds, dtype=np.float32)
