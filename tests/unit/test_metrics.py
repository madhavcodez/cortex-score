"""Unit tests for per-ROI scalar metrics."""

from __future__ import annotations

import numpy as np
import pytest

from cortex_score.processing.metrics import (
    METRICS_VERSION,
    ROIMetrics,
    compute_roi_metrics,
)


def test_metrics_basic_values() -> None:
    z = np.array(
        [
            [1.0, -2.0],
            [-1.0, 2.0],
            [1.0, -2.0],
        ],
        dtype=np.float32,
    )
    metrics = compute_roi_metrics(z)
    assert len(metrics) == 2
    # |z| mean: col 0 -> 1.0, col 1 -> 2.0
    assert metrics[0].mean_energy == pytest.approx(1.0)
    assert metrics[1].mean_energy == pytest.approx(2.0)
    # peak |z|: col 0 -> 1.0, col 1 -> 2.0
    assert metrics[0].peak_energy == pytest.approx(1.0)
    assert metrics[1].peak_energy == pytest.approx(2.0)
    # temporal_volatility: diffs are [-2, 2], [4, -4], std = 2.0 and 4.0
    assert metrics[0].temporal_volatility == pytest.approx(2.0)
    assert metrics[1].temporal_volatility == pytest.approx(4.0)


def test_metrics_single_timestep_zero_volatility() -> None:
    z = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    metrics = compute_roi_metrics(z)
    assert all(m.temporal_volatility == 0.0 for m in metrics)


def test_metrics_shape_validation() -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        compute_roi_metrics(np.zeros(10, dtype=np.float32))


def test_metrics_returns_frozen_roimetrics() -> None:
    z = np.zeros((5, 1), dtype=np.float32)
    metrics = compute_roi_metrics(z)
    with pytest.raises(Exception):
        metrics[0].mean_energy = 99.0  # type: ignore[misc]


def test_metrics_version_is_stable() -> None:
    """Bumping ``METRICS_VERSION`` is intentional; this test exists so a
    silent constant change shows up in the diff."""
    assert METRICS_VERSION == "1.0"


def test_roimetrics_dataclass_fields() -> None:
    m = ROIMetrics(mean_energy=1.0, peak_energy=2.0, temporal_volatility=0.5)
    assert m.mean_energy == 1.0
    assert m.peak_energy == 2.0
    assert m.temporal_volatility == 0.5
