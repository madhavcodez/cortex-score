"""Unit tests for the boundary validators."""

from __future__ import annotations

import numpy as np
import pytest

from cortex_score.exceptions import IncompatiblePredictionShapeError
from cortex_score.processing.validate import (
    coerce_float32,
    validate_predictions_against_mesh,
)


def test_validate_accepts_correct_shape() -> None:
    preds = np.zeros((10, 20484), dtype=np.float32)
    validate_predictions_against_mesh(
        preds, mesh="fsaverage5", expected_n_vertices=20484
    )


def test_validate_raises_on_wrong_vertex_count() -> None:
    preds = np.zeros((10, 99), dtype=np.float32)
    with pytest.raises(IncompatiblePredictionShapeError) as info:
        validate_predictions_against_mesh(
            preds, mesh="fsaverage5", expected_n_vertices=20484
        )
    assert info.value.expected_n_vertices == 20484
    assert info.value.actual_n_vertices == 99
    assert info.value.mesh == "fsaverage5"


def test_validate_raises_on_1d_input() -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        validate_predictions_against_mesh(
            np.zeros(20484, dtype=np.float32),
            mesh="fsaverage5",
            expected_n_vertices=20484,
        )


def test_validate_raises_on_zero_timesteps() -> None:
    preds = np.zeros((0, 20484), dtype=np.float32)
    with pytest.raises(ValueError, match="at least 1 timestep"):
        validate_predictions_against_mesh(
            preds, mesh="fsaverage5", expected_n_vertices=20484
        )


def test_coerce_float32_passes_through_float32() -> None:
    a = np.zeros((2, 4), dtype=np.float32)
    out = coerce_float32(a)
    assert out.dtype == np.float32
    # Note: when a.dtype already matches, np.asarray returns same object
    assert out is a or np.shares_memory(out, a)


def test_coerce_float32_casts_float64() -> None:
    a = np.zeros((2, 4), dtype=np.float64)
    out = coerce_float32(a)
    assert out.dtype == np.float32
