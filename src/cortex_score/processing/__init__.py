"""Pure-NumPy processing core.

Every module here is safe to import in a CPU-only environment. No torch,
no tribev2, no whisperx. Heavy ML deps only enter via runners/.
"""

from __future__ import annotations

from cortex_score.processing.aggregate import aggregate_to_rois, remap_atlas
from cortex_score.processing.metrics import ROIMetrics, compute_roi_metrics
from cortex_score.processing.networks import (
    NETWORK_GROUPS,
    NETWORK_IDS,
    NetworkGroup,
    NetworkId,
    build_network_summary,
    yeo_to_network_indices,
)
from cortex_score.processing.normalize import zscore_within_atlas
from cortex_score.processing.validate import (
    validate_predictions_against_mesh,
)

__all__ = [
    "NETWORK_GROUPS",
    "NETWORK_IDS",
    "NetworkGroup",
    "NetworkId",
    "ROIMetrics",
    "aggregate_to_rois",
    "build_network_summary",
    "compute_roi_metrics",
    "remap_atlas",
    "validate_predictions_against_mesh",
    "yeo_to_network_indices",
    "zscore_within_atlas",
]
