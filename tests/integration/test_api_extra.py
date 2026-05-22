"""Coverage for the api.py helpers not exercised by the smoke path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cortex_score import CortexScorer, PredictionBundle, ScoreConfig
from cortex_score.api import (
    _build_input_meta,
    _runner_class_path,
    _sha256_file,
    _validate_runner,
    score,
)
from cortex_score.exceptions import MissingOptionalDependencyError


class _FakeRunner:
    """Trivial PredictionRunner stub for testing the orchestration.

    Returns a bundle whose ``model_revision`` matches the runner's own
    attribute, mirroring how a real runner records provenance.
    """

    model_id = "facebook/tribev2"
    model_revision = "fake-rev"

    def predict_video(self, path: Path) -> PredictionBundle:
        return PredictionBundle(
            vertex_predictions=np.random.default_rng(0)
            .standard_normal((4, 20484))
            .astype(np.float32),
            mesh="fsaverage5",
            n_vertices=20484,
            tr_seconds=1.0,
            hrf_lag_seconds=5.0,
            model_id=self.model_id,
            model_revision=self.model_revision,
            source="tribev2",
        )


def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello world")
    sha = _sha256_file(p)
    assert sha == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_build_input_meta_records_basename_only_by_default(tmp_path: Path) -> None:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"mp4-bytes")
    meta = _build_input_meta(p)
    # PII fix: default does NOT serialize the absolute filesystem path.
    assert meta.filename == "clip.mp4"
    assert meta.absolute_path is None
    assert meta.content_sha256 is not None
    assert len(meta.content_sha256) == 64


def test_build_input_meta_opt_in_absolute_path(tmp_path: Path) -> None:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"mp4-bytes")
    meta = _build_input_meta(p, include_absolute_path=True)
    assert meta.filename == "clip.mp4"
    assert meta.absolute_path == str(p.resolve())


def test_runner_class_path() -> None:
    assert _runner_class_path("tribev2") == "cortex_score.runners.tribev2.TribeV2Runner"
    assert _runner_class_path("npy") == "external"
    assert _runner_class_path("unknown") == "external"


def test_score_with_explicit_runner(tmp_path: Path) -> None:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"mp4-bytes")
    result = score(p, runner=_FakeRunner())
    assert result.provenance.model_revision == "fake-rev"
    assert result.input.filename == "clip.mp4"
    assert result.input.absolute_path is None  # opt-in default off


def test_score_with_explicit_runner_opt_in_absolute_path(tmp_path: Path) -> None:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"mp4-bytes")
    config = ScoreConfig(include_absolute_path=True)
    result = score(p, runner=_FakeRunner(), config=config)
    assert result.input.absolute_path == str(p.resolve())


def test_score_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        score(tmp_path / "does-not-exist.mp4", runner=_FakeRunner())


def test_score_without_runner_raises_missing_optional_dep(tmp_path: Path) -> None:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"mp4-bytes")
    # No tribev2 installed in the test env -> friendly error.
    with pytest.raises(MissingOptionalDependencyError):
        score(p)


def test_cortex_scorer_reuses_runner(tmp_path: Path) -> None:
    p1 = tmp_path / "a.mp4"
    p2 = tmp_path / "b.mp4"
    p1.write_bytes(b"a")
    p2.write_bytes(b"b")
    scorer = CortexScorer(runner=_FakeRunner())
    r1 = scorer.score(p1)
    r2 = scorer.score(p2)
    assert r1.provenance.model_revision == "fake-rev"
    assert r2.provenance.model_revision == "fake-rev"
    # Same runner instance is reused
    assert scorer.runner is scorer.runner


# ---- Runner structural guard -----------------------------------------


class _RunnerMissingModelId:
    """Custom runner that 'forgets' to declare model_id."""

    model_revision = "x"

    def predict_video(self, path: Path) -> PredictionBundle:
        raise NotImplementedError


class _RunnerMissingModelRevision:
    """Custom runner that 'forgets' to declare model_revision."""

    model_id = "x"

    def predict_video(self, path: Path) -> PredictionBundle:
        raise NotImplementedError


class _RunnerEmptyModelId:
    """Custom runner with empty model_id."""

    model_id = ""
    model_revision = "x"

    def predict_video(self, path: Path) -> PredictionBundle:
        raise NotImplementedError


def test_validate_runner_accepts_well_formed_runner() -> None:
    # No exception means the well-formed _FakeRunner passes.
    _validate_runner(_FakeRunner())  # type: ignore[arg-type]


def test_validate_runner_rejects_missing_model_id() -> None:
    with pytest.raises(ValueError, match="model_id"):
        _validate_runner(_RunnerMissingModelId())  # type: ignore[arg-type]


def test_validate_runner_rejects_empty_model_id() -> None:
    with pytest.raises(ValueError, match="model_id"):
        _validate_runner(_RunnerEmptyModelId())  # type: ignore[arg-type]


def test_validate_runner_rejects_missing_model_revision() -> None:
    with pytest.raises(ValueError, match="model_revision"):
        _validate_runner(_RunnerMissingModelRevision())  # type: ignore[arg-type]


def test_score_rejects_runner_missing_metadata(tmp_path: Path) -> None:
    """End-to-end: score() rejects a malformed runner before predict_video()."""
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"mp4-bytes")
    with pytest.raises(ValueError, match="model_id"):
        score(p, runner=_RunnerMissingModelId())  # type: ignore[arg-type]
