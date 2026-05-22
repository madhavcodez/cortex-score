"""Unit tests for per-ROI z-score normalization."""

from __future__ import annotations

import numpy as np
import pytest

from cortex_score.processing.normalize import DEFAULT_EPS, zscore_within_atlas


def test_zscore_zero_mean_unit_std() -> None:
    rng = np.random.default_rng(0)
    preds = rng.standard_normal((1000, 5)).astype(np.float32)
    z = zscore_within_atlas(preds)
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-3)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=5e-3)


def test_zscore_constant_column_returns_zeros() -> None:
    """Avoid the eps-amplification bug Cortexia documented in
    ``normalize.py``: a column of identical float32 values must
    produce exact zero rather than a tiny residue scaled up by 1/eps.
    """
    preds = np.full((100, 3), 7.0, dtype=np.float32)
    z = zscore_within_atlas(preds, eps=DEFAULT_EPS)
    np.testing.assert_array_equal(z, np.zeros_like(z))


def test_zscore_eps_is_recorded_default() -> None:
    assert DEFAULT_EPS > 0
    assert DEFAULT_EPS < 1e-3


def test_zscore_ndim_validation() -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        zscore_within_atlas(np.zeros(10, dtype=np.float32))


def test_zscore_returns_float32() -> None:
    preds = np.random.randn(10, 4).astype(np.float64)
    z = zscore_within_atlas(preds.astype(np.float32))
    assert z.dtype == np.float32
