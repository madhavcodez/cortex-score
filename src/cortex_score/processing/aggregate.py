"""Vertex-level -> ROI-level aggregation.

Both functions take the mean across vertices assigned to each ROI.
Vertices with id ``-1`` are dropped (used for the fsaverage5 medial
wall mask). This is a direct port of Cortexia's
``apps/worker/src/clipcortex_worker/pipeline/aggregate.py`` (which is
already a pure function with no R2 or cloud coupling), so the
behaviour is identical and the existing Cortexia test suite proves it
out.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def aggregate_to_rois(
    preds: npt.NDArray[np.float32],
    vertex_to_roi: npt.NDArray[np.int64],
    n_rois: int,
) -> npt.NDArray[np.float32]:
    """Aggregate vertex-level predictions to ROI level by taking the mean.

    Args:
        preds: shape ``(T, V)``, float32. Vertex-level predictions where
            ``T`` is the number of time segments and ``V`` is the number
            of fsaverage5 cortical vertices (typically 20484 for both
            hemispheres combined).
        vertex_to_roi: shape ``(V,)``, int64. ROI assignment per vertex.
            ROI ids must be in the inclusive range ``[0, n_rois - 1]``.
            Vertices with id ``-1`` are dropped (medial wall mask).
        n_rois: total number of ROIs in the target atlas.

    Returns:
        shape ``(T, R)``, float32. ROI-level predictions, where each
        entry is the mean of all vertex predictions assigned to that
        ROI at that time step. ROIs with zero vertex assignments
        contain ``0.0``.

    Raises:
        ValueError: if ``preds.shape[1] != vertex_to_roi.shape[0]`` or
            if any non-medial-wall ROI id is outside ``[0, n_rois)``.
    """
    if preds.ndim != 2:
        msg = f"preds must be 2D (T, V); got shape {preds.shape}"
        raise ValueError(msg)
    if vertex_to_roi.ndim != 1:
        msg = f"vertex_to_roi must be 1D; got shape {vertex_to_roi.shape}"
        raise ValueError(msg)
    if preds.shape[1] != vertex_to_roi.shape[0]:
        msg = (
            f"preds.shape[1]={preds.shape[1]} does not match "
            f"vertex_to_roi.shape[0]={vertex_to_roi.shape[0]}"
        )
        raise ValueError(msg)

    valid = vertex_to_roi >= 0
    valid_ids = vertex_to_roi[valid]
    if valid_ids.size > 0 and (valid_ids.min() < 0 or valid_ids.max() >= n_rois):
        msg = (
            f"vertex_to_roi contains id outside [0, {n_rois}); "
            f"min={valid_ids.min()}, max={valid_ids.max()}"
        )
        raise ValueError(msg)

    t = preds.shape[0]
    out = np.zeros((t, n_rois), dtype=np.float32)
    counts = np.zeros(n_rois, dtype=np.int64)

    for roi_id in range(n_rois):
        mask = vertex_to_roi == roi_id
        n_verts = int(mask.sum())
        if n_verts == 0:
            continue
        counts[roi_id] = n_verts
        out[:, roi_id] = preds[:, mask].mean(axis=1, dtype=np.float32)

    return out


def remap_atlas(
    roi_preds: npt.NDArray[np.float32],
    roi_to_network: npt.NDArray[np.int64],
    n_networks: int,
) -> npt.NDArray[np.float32]:
    """Aggregate fine-grained ROI predictions to a coarser network atlas.

    Used to project Schaefer-400 -> Yeo-17. Same averaging semantics as
    ``aggregate_to_rois``: each network is the mean of its member ROIs.

    Args:
        roi_preds: shape ``(T, R_fine)``, float32.
        roi_to_network: shape ``(R_fine,)``, int64. Network id per ROI.
        n_networks: number of target networks.

    Returns:
        shape ``(T, n_networks)``, float32.
    """
    return aggregate_to_rois(roi_preds, roi_to_network, n_networks)
