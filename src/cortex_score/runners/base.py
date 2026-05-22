"""Runner Protocol.

A ``PredictionRunner`` consumes a video file and produces a validated
``PredictionBundle``. The Protocol gives the core library a stable seam
to plug in future encoders (TRIBE v2-mini, replay-from-npy, remote
Modal sidecars) without changing the public API or the score path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from cortex_score.schemas import PredictionBundle


@runtime_checkable
class PredictionRunner(Protocol):
    """A producer of ``PredictionBundle``s from video files.

    Implementations must:

    1. Set ``model_id`` and ``model_revision`` attributes so the result
       carries faithful provenance.
    2. Validate that the returned bundle's ``mesh`` matches the runner's
       output mesh.
    3. Raise ``MissingOptionalDependencyError`` (not ``ImportError``) when
       a required runtime dep is absent — runners are the gatekeepers
       for optional ML deps.

    Attributes are read by the orchestrator in ``api.score`` to populate
    ``ProvenanceMeta`` before any inference call.
    """

    model_id: str
    model_revision: str

    def predict_video(self, path: Path) -> PredictionBundle:
        """Run the encoder on ``path`` and return a validated bundle."""
        ...
