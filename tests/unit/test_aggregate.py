"""Unit + property tests for vertex/ROI aggregation."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cortex_score.processing.aggregate import aggregate_to_rois, remap_atlas


def test_aggregate_basic_mean_invariant() -> None:
    # 4 vertices: roi 0 -> verts 0,1; roi 1 -> verts 2,3.
    preds = np.array(
        [
            [1.0, 3.0, 5.0, 7.0],
            [2.0, 4.0, 6.0, 8.0],
        ],
        dtype=np.float32,
    )
    v2r = np.array([0, 0, 1, 1], dtype=np.int64)
    out = aggregate_to_rois(preds, v2r, n_rois=2)
    np.testing.assert_array_almost_equal(out, np.array([[2.0, 6.0], [3.0, 7.0]]))


def test_aggregate_medial_wall_dropped() -> None:
    preds = np.array([[1.0, 99.0, 3.0]], dtype=np.float32)
    v2r = np.array([0, -1, 1], dtype=np.int64)  # vertex 1 is medial wall
    out = aggregate_to_rois(preds, v2r, n_rois=2)
    np.testing.assert_array_equal(out, np.array([[1.0, 3.0]]))


def test_aggregate_empty_roi_returns_zero() -> None:
    preds = np.array([[1.0, 2.0]], dtype=np.float32)
    v2r = np.array([0, 0], dtype=np.int64)  # both vertices map to roi 0; roi 1 empty
    out = aggregate_to_rois(preds, v2r, n_rois=2)
    np.testing.assert_array_equal(out, np.array([[1.5, 0.0]]))


def test_aggregate_shape_mismatch_raises() -> None:
    preds = np.zeros((1, 10), dtype=np.float32)
    v2r = np.zeros(5, dtype=np.int64)
    with pytest.raises(ValueError, match="does not match"):
        aggregate_to_rois(preds, v2r, n_rois=1)


def test_aggregate_ndim_validation() -> None:
    preds = np.zeros(10, dtype=np.float32)  # 1D
    v2r = np.zeros(10, dtype=np.int64)
    with pytest.raises(ValueError, match="must be 2D"):
        aggregate_to_rois(preds, v2r, n_rois=1)


def test_aggregate_id_out_of_range_raises() -> None:
    preds = np.zeros((1, 4), dtype=np.float32)
    v2r = np.array([0, 1, 5, 0], dtype=np.int64)  # 5 >= n_rois
    with pytest.raises(ValueError, match="outside"):
        aggregate_to_rois(preds, v2r, n_rois=2)


def test_remap_atlas_is_alias_of_aggregate() -> None:
    roi_preds = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    r2n = np.array([0, 0, 1], dtype=np.int64)
    out = remap_atlas(roi_preds, r2n, n_networks=2)
    np.testing.assert_array_almost_equal(out, np.array([[1.5, 3.0]]))


# ---- Property tests ----------------------------------------------------


@given(
    t=st.integers(min_value=1, max_value=20),
    fill=st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False,
        width=32,
    ),
)
@settings(max_examples=50, deadline=2000)
def test_aggregate_constant_input_returns_constant_per_roi(t: int, fill: float) -> None:
    """If every vertex carries the same value, every non-empty ROI mean
    equals that value (exactly)."""
    v = 12
    preds = np.full((t, v), fill, dtype=np.float32)
    v2r = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
    out = aggregate_to_rois(preds, v2r, n_rois=3)
    np.testing.assert_array_equal(out, np.full((t, 3), np.float32(fill)))


@given(perm_seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=25, deadline=2000)
def test_aggregate_invariant_to_vertex_order_within_roi(perm_seed: int) -> None:
    """Permuting vertices that share an ROI must not change the ROI mean."""
    rng = np.random.default_rng(perm_seed)
    v = 16
    preds = rng.standard_normal((3, v)).astype(np.float32)
    # All 16 vertices go to ROI 0 -> mean is invariant to any permutation.
    v2r = np.zeros(v, dtype=np.int64)
    perm = rng.permutation(v)
    out_a = aggregate_to_rois(preds, v2r, n_rois=1)
    out_b = aggregate_to_rois(preds[:, perm], v2r, n_rois=1)
    np.testing.assert_array_almost_equal(out_a, out_b, decimal=5)
