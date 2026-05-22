"""Unit tests for the exception hierarchy."""

from __future__ import annotations

import warnings

import pytest

from cortex_score.exceptions import (
    AtlasMismatchError,
    CortexScoreError,
    IncompatiblePredictionShapeError,
    MissingExternalToolError,
    MissingOptionalDependencyError,
    ModelLicenseError,
    PreprocessingWarning,
)


def test_missing_optional_dep_carries_package_and_hint() -> None:
    e = MissingOptionalDependencyError(
        package="tribev2",
        install_hint="pip install -r requirements/tribev2-gpu.txt",
    )
    assert e.package == "tribev2"
    assert "pip install" in str(e)
    # Inherits ImportError so `except ImportError` still works.
    assert isinstance(e, ImportError)
    assert isinstance(e, CortexScoreError)


def test_missing_external_tool_carries_tool_name() -> None:
    e = MissingExternalToolError(tool="ffmpeg", install_hint="https://ffmpeg.org")
    assert e.tool == "ffmpeg"
    assert "ffmpeg.org" in str(e)


def test_incompatible_prediction_shape_message_actionable() -> None:
    e = IncompatiblePredictionShapeError(
        expected_n_vertices=20484,
        actual_n_vertices=10,
        mesh="fsaverage5",
    )
    assert e.expected_n_vertices == 20484
    assert e.actual_n_vertices == 10
    assert e.mesh == "fsaverage5"
    assert "20484" in str(e)
    assert "fsaverage5" in str(e)


def test_atlas_mismatch_carries_detail() -> None:
    e = AtlasMismatchError("bad sha")
    assert "bad sha" in str(e)
    assert e.detail == "bad sha"


def test_model_license_error_carries_license_name() -> None:
    e = ModelLicenseError(
        model_id="facebook/tribev2",
        license_name="CC-BY-NC-4.0",
        note="non-commercial",
    )
    assert e.model_id == "facebook/tribev2"
    assert e.license_name == "CC-BY-NC-4.0"


def test_preprocessing_warning_is_userwarning() -> None:
    with pytest.warns(PreprocessingWarning):
        warnings.warn("letterboxed input", PreprocessingWarning, stacklevel=2)


def test_all_inherit_cortex_score_error() -> None:
    for cls in (
        MissingOptionalDependencyError,
        MissingExternalToolError,
        IncompatiblePredictionShapeError,
        AtlasMismatchError,
        ModelLicenseError,
    ):
        assert issubclass(cls, CortexScoreError)
