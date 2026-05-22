"""cortex-score

Score any video for predicted cortical engagement across 5 brain networks
(visual, language, faces, attention, motion). Built on Meta FAIR's TRIBE v2.

What this actually does: summarizes TRIBE v2 predicted cortical responses
for any video across five Cortexia-defined network groups.

What this does NOT do: measure real viewer engagement. The numbers are
predictions for an average subject on the fsaverage5 cortical mesh,
not measurements from a real viewer.

Public API:

    from cortex_score import (
        score,                          # full pipeline (requires [gpu-deps] + TRIBE)
        score_from_predictions,         # CPU-only: friendly form, requires explicit metadata
        score_from_prediction_bundle,   # CPU-only: type-safe form
        PredictionBundle,
        ScoreResult,
        NetworkScore,
        CortexScoreError,
    )

The full ``score()`` path lazily imports torch and TRIBE v2 inside the
runner; importing this module by itself is fast and does not require a
GPU. See README for the install matrix.
"""

from __future__ import annotations

from cortex_score._version import __version__
from cortex_score.exceptions import (
    AtlasMismatchError,
    CortexScoreError,
    IncompatiblePredictionShapeError,
    MissingExternalToolError,
    MissingOptionalDependencyError,
    ModelLicenseError,
    PreprocessingWarning,
)

# Schemas and processing modules are pure NumPy / Pydantic — safe to
# import eagerly.
from cortex_score.schemas import (
    AtlasMeta,
    InputMeta,
    NetworkScore,
    NormalizationMeta,
    PredictionBundle,
    ProvenanceMeta,
    ScoreResult,
    SegmentMeta,
    TimingMeta,
)

# High-level API. score_from_predictions / score_from_prediction_bundle
# stay CPU-only; score() lazily resolves a runner.
from cortex_score.api import (
    CortexScorer,
    score,
    score_from_prediction_bundle,
    score_from_predictions,
)

__all__ = [
    # Version
    "__version__",
    # Top-level API
    "score",
    "score_from_predictions",
    "score_from_prediction_bundle",
    "CortexScorer",
    # Schemas
    "PredictionBundle",
    "ScoreResult",
    "NetworkScore",
    "InputMeta",
    "TimingMeta",
    "AtlasMeta",
    "NormalizationMeta",
    "ProvenanceMeta",
    "SegmentMeta",
    # Exceptions
    "CortexScoreError",
    "MissingOptionalDependencyError",
    "MissingExternalToolError",
    "IncompatiblePredictionShapeError",
    "AtlasMismatchError",
    "ModelLicenseError",
    "PreprocessingWarning",
]
