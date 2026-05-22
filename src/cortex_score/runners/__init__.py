"""Prediction runners.

A runner produces a ``PredictionBundle`` from a video file. The base
``PredictionRunner`` Protocol makes the architecture pluggable so the
core library can score with TRIBE v2 today and with future encoders
tomorrow without rewriting the public API.
"""

from __future__ import annotations

from cortex_score.runners.base import PredictionRunner

__all__ = ["PredictionRunner"]
