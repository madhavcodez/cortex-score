"""End-to-end test of the CPU-only scoring path."""

from __future__ import annotations

import json

import numpy as np

from cortex_score import (
    PredictionBundle,
    ScoreResult,
    score_from_prediction_bundle,
    score_from_predictions,
)
from cortex_score.api import ScoreConfig
from cortex_score.schemas import (
    FRAMING_DISCLAIMER,
    FRAMING_PRIMARY,
    InputMeta,
)


def _synthetic_preds(t: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((t, 20484)).astype(np.float32)


def test_score_from_predictions_smoke() -> None:
    result = score_from_predictions(
        _synthetic_preds(),
        mesh="fsaverage5",
        tr_seconds=1.0,
        hrf_lag_seconds=5.0,
        model_id="facebook/tribev2",
        model_revision="test",
    )
    assert isinstance(result, ScoreResult)
    assert len(result.networks) == 5
    assert result.timing.n_segments == 8


def test_score_from_predictions_includes_framing() -> None:
    result = score_from_predictions(_synthetic_preds(), model_revision="test")
    assert result.framing == FRAMING_PRIMARY
    assert result.framing_disclaimer == FRAMING_DISCLAIMER


def test_score_from_predictions_records_provenance() -> None:
    result = score_from_predictions(_synthetic_preds(), model_revision="test-rev")
    assert result.provenance.model_revision == "test-rev"
    assert result.provenance.cortex_score_version  # non-empty
    assert result.provenance.python_version  # non-empty
    assert result.provenance.schema_version == "1.0"


def test_score_from_predictions_records_atlas_shas() -> None:
    result = score_from_predictions(_synthetic_preds(), model_revision="test")
    assert len(result.atlas.atlas_sha256) == 64
    assert len(result.atlas.yeo_atlas_sha256) == 64
    assert len(result.atlas.network_groups_sha256) == 64
    assert result.atlas.network_group_source == "cortexia-network-groups-v1"


def test_score_from_predictions_records_license_restriction() -> None:
    result = score_from_predictions(_synthetic_preds(), model_revision="test")
    assert len(result.license_restrictions) >= 1
    tribe = next(r for r in result.license_restrictions if r.component == "TRIBE v2")
    assert "NC-4.0" in tribe.license or "NonCommercial" in tribe.license


def test_score_from_predictions_records_normalization_scope() -> None:
    result = score_from_predictions(_synthetic_preds(), model_revision="test")
    assert result.normalization.scope == "within_video"
    assert result.normalization.method == "zscore"


def test_score_result_json_roundtrip() -> None:
    result = score_from_predictions(_synthetic_preds(), model_revision="test")
    j = result.to_json()
    rebuilt = ScoreResult.model_validate_json(j)
    assert rebuilt.result_id == result.result_id
    assert rebuilt.atlas.atlas_sha256 == result.atlas.atlas_sha256


def test_score_result_is_deterministic_for_same_input() -> None:
    """Same predictions + same config → same network scores (created_at
    differs but result_id is constructed before created_at is inserted)."""
    a = score_from_predictions(_synthetic_preds(seed=1), model_revision="test")
    b = score_from_predictions(_synthetic_preds(seed=1), model_revision="test")
    for na, nb in zip(a.networks, b.networks, strict=True):
        assert na.mean_energy == nb.mean_energy
        assert na.peak_energy == nb.peak_energy


def test_score_from_prediction_bundle_takes_validated_bundle() -> None:
    bundle = PredictionBundle(
        vertex_predictions=_synthetic_preds(),
        mesh="fsaverage5",
        n_vertices=20484,
        tr_seconds=1.0,
        hrf_lag_seconds=5.0,
        model_id="facebook/tribev2",
        model_revision="bundled",
        source="npy",
    )
    result = score_from_prediction_bundle(bundle)
    assert result.provenance.model_revision == "bundled"


def test_score_from_prediction_bundle_with_input_meta() -> None:
    bundle = PredictionBundle(
        vertex_predictions=_synthetic_preds(),
        mesh="fsaverage5",
        n_vertices=20484,
        tr_seconds=1.0,
        hrf_lag_seconds=5.0,
        model_id="facebook/tribev2",
        model_revision="bundled",
    )
    meta = InputMeta(
        path="/fake/clip.mp4",
        content_sha256="d" * 64,
        duration_s=12.4,
        fps=30.0,
        resolution="720x1280",
    )
    result = score_from_prediction_bundle(bundle, input_meta=meta)
    assert result.input.path == "/fake/clip.mp4"
    assert result.input.duration_s == 12.4


def test_score_config_normalization_scope_propagates() -> None:
    config = ScoreConfig(
        normalization_scope="reference_distribution",
        reference_id="cortexia-v1-68clip",
    )
    result = score_from_predictions(
        _synthetic_preds(),
        model_revision="test",
        config=config,
    )
    assert result.normalization.scope == "reference_distribution"
    assert result.normalization.reference_id == "cortexia-v1-68clip"


def test_emitted_json_contains_top_level_provenance_fields() -> None:
    """Reviewer-required: every result must carry full provenance."""
    result = score_from_predictions(_synthetic_preds(), model_revision="test")
    payload = json.loads(result.to_json(indent=None))
    required = {
        "schema_version",
        "result_id",
        "created_at",
        "framing",
        "framing_scientific",
        "framing_disclaimer",
        "input",
        "timing",
        "normalization",
        "atlas",
        "provenance",
        "license_restrictions",
        "warnings",
        "networks",
    }
    assert required.issubset(payload.keys())
