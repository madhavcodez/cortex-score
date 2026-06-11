"""Unit tests for the schema layer."""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pytest
from pydantic import ValidationError

from cortex_score.schemas import (
    FRAMING_DISCLAIMER,
    FRAMING_PRIMARY,
    FRAMING_SCIENTIFIC,
    SCHEMA_VERSION,
    AtlasMeta,
    InputMeta,
    LicenseRestriction,
    NetworkScore,
    NormalizationMeta,
    PredictionBundle,
    ProvenanceMeta,
    ScoreResult,
    SegmentMeta,
    TimingMeta,
    build_provenance,
    compute_result_id,
)

_DUMMY_SHA = "a" * 64
_DUMMY_SHA_B = "b" * 64
_DUMMY_SHA_C = "c" * 64


def _atlas() -> AtlasMeta:
    return AtlasMeta(
        mesh="fsaverage5",
        n_vertices=20484,
        atlas_version="schaefer2018-400-yeo17-fsaverage5",
        atlas_sha256=_DUMMY_SHA,
        yeo_atlas_sha256=_DUMMY_SHA_B,
        network_groups_sha256=_DUMMY_SHA_C,
        network_group_source="cortexia-network-groups-v1",
    )


def _net(net_id: str, n_segments: int) -> NetworkScore:
    return NetworkScore(
        id=net_id,  # type: ignore[arg-type]
        label=net_id.capitalize(),
        description="test",
        color="#1234AB",
        yeo_indices=(0, 1),
        yeo_labels=("VisCent", "VisPeri"),
        mean_energy=1.0,
        peak_energy=2.0,
        energy_timeseries=tuple(0.0 for _ in range(n_segments)),
        mean_z_timeseries=tuple(0.0 for _ in range(n_segments)),
        group_definition_sha256=_DUMMY_SHA,
    )


def test_prediction_bundle_post_init_validates_shape() -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        PredictionBundle(
            vertex_predictions=np.zeros(10, dtype=np.float32),  # type: ignore[arg-type]
            mesh="fsaverage5",
            n_vertices=10,
            tr_seconds=1.0,
            hrf_lag_seconds=5.0,
            model_id="facebook/tribev2",
            model_revision="x",
        )


def test_prediction_bundle_n_vertices_must_match() -> None:
    with pytest.raises(ValueError, match="does not match"):
        PredictionBundle(
            vertex_predictions=np.zeros((2, 10), dtype=np.float32),
            mesh="fsaverage5",
            n_vertices=99,  # wrong
            tr_seconds=1.0,
            hrf_lag_seconds=5.0,
            model_id="facebook/tribev2",
            model_revision="x",
        )


def test_prediction_bundle_requires_model_revision() -> None:
    with pytest.raises(ValueError, match="model_revision"):
        PredictionBundle(
            vertex_predictions=np.zeros((2, 10), dtype=np.float32),
            mesh="fsaverage5",
            n_vertices=10,
            tr_seconds=1.0,
            hrf_lag_seconds=5.0,
            model_id="facebook/tribev2",
            model_revision="",
        )


def test_normalization_reference_id_required_for_distribution_scope() -> None:
    with pytest.raises(ValidationError, match="reference_id is required"):
        NormalizationMeta(
            scope="reference_distribution",
            epsilon=1e-6,
        )


def test_normalization_within_video_default() -> None:
    n = NormalizationMeta(epsilon=1e-6)
    assert n.scope == "within_video"
    assert n.reference_id is None
    assert n.method == "zscore"


def test_score_result_requires_exactly_five_networks() -> None:
    timing = TimingMeta(tr_seconds=1.0, hrf_lag_seconds=5.0, n_segments=2)
    payload = dict(
        result_id="z" * 64,
        created_at=_dt.datetime(2026, 1, 1, tzinfo=_dt.UTC),
        input=InputMeta(),
        timing=timing,
        normalization=NormalizationMeta(epsilon=1e-6),
        atlas=_atlas(),
        provenance=build_provenance(
            model_id="x",
            model_revision="y",
            runner="cortex_score.runners.tribev2.TribeV2Runner",
        ),
        networks=tuple(_net(i, 2) for i in ("visual", "language", "faces")),  # only 3
    )
    with pytest.raises(ValidationError, match="exactly 5"):
        ScoreResult(**payload)


def test_score_result_networks_must_be_in_order() -> None:
    timing = TimingMeta(tr_seconds=1.0, hrf_lag_seconds=5.0, n_segments=2)
    networks = tuple(_net(i, 2) for i in ("motion", "visual", "language", "faces", "attention"))
    payload = dict(
        result_id="z" * 64,
        created_at=_dt.datetime(2026, 1, 1, tzinfo=_dt.UTC),
        input=InputMeta(),
        timing=timing,
        normalization=NormalizationMeta(epsilon=1e-6),
        atlas=_atlas(),
        provenance=build_provenance(
            model_id="x",
            model_revision="y",
            runner="cortex_score.runners.tribev2.TribeV2Runner",
        ),
        networks=networks,
    )
    with pytest.raises(ValidationError, match="order"):
        ScoreResult(**payload)


def test_score_result_timeseries_length_must_match_n_segments() -> None:
    timing = TimingMeta(tr_seconds=1.0, hrf_lag_seconds=5.0, n_segments=4)
    networks = tuple(
        _net(i, 2)  # wrong: networks have 2 points but timing.n_segments = 4
        for i in ("visual", "language", "faces", "attention", "motion")
    )
    payload = dict(
        result_id="z" * 64,
        created_at=_dt.datetime(2026, 1, 1, tzinfo=_dt.UTC),
        input=InputMeta(),
        timing=timing,
        normalization=NormalizationMeta(epsilon=1e-6),
        atlas=_atlas(),
        provenance=build_provenance(
            model_id="x",
            model_revision="y",
            runner="external",
        ),
        networks=networks,
    )
    with pytest.raises(ValidationError, match="n_segments"):
        ScoreResult(**payload)


def test_score_result_framing_defaults_are_baked_in() -> None:
    timing = TimingMeta(tr_seconds=1.0, hrf_lag_seconds=5.0, n_segments=2)
    networks = tuple(_net(i, 2) for i in ("visual", "language", "faces", "attention", "motion"))
    r = ScoreResult(
        result_id="z" * 64,
        created_at=_dt.datetime(2026, 1, 1, tzinfo=_dt.UTC),
        input=InputMeta(),
        timing=timing,
        normalization=NormalizationMeta(epsilon=1e-6),
        atlas=_atlas(),
        provenance=build_provenance(model_id="x", model_revision="y", runner="external"),
        networks=networks,
    )
    assert r.framing == FRAMING_PRIMARY
    assert r.framing_scientific == FRAMING_SCIENTIFIC
    assert r.framing_disclaimer == FRAMING_DISCLAIMER


def _full_result(result_id: str = "z" * 64) -> ScoreResult:
    timing = TimingMeta(tr_seconds=1.0, hrf_lag_seconds=5.0, n_segments=2)
    networks = tuple(_net(i, 2) for i in ("visual", "language", "faces", "attention", "motion"))
    return ScoreResult(
        result_id=result_id,
        created_at=_dt.datetime(2026, 1, 1, tzinfo=_dt.UTC),
        input=InputMeta(),
        timing=timing,
        normalization=NormalizationMeta(epsilon=1e-6),
        atlas=_atlas(),
        provenance=build_provenance(model_id="x", model_revision="y", runner="external"),
        networks=networks,
    )


def test_compute_result_id_is_stable_and_blank_invariant() -> None:
    result = _full_result()
    a = compute_result_id(result)
    b = compute_result_id(result)
    assert a == b
    assert len(a) == 64
    # result_id is blanked before hashing, so the stored value can't
    # influence the hash.
    assert compute_result_id(result.model_copy(update={"result_id": "x" * 64})) == a


def test_result_id_is_verifiable_from_serialized_json() -> None:
    """The documented contract: re-hashing a serialized result (with
    result_id blanked) reproduces result_id. This is the round-trip the
    old hand-built-dict implementation silently broke (Z vs +00:00)."""
    result = _full_result().model_copy(update={"result_id": ""})
    stamped = result.model_copy(update={"result_id": compute_result_id(result)})

    rebuilt = ScoreResult.model_validate_json(stamped.to_json())
    assert rebuilt.result_id == stamped.result_id
    # Recompute from the deserialized object: must match.
    assert compute_result_id(rebuilt) == stamped.result_id


def test_segment_meta_is_frozen() -> None:
    import dataclasses

    s = SegmentMeta(index=0, start_s=0.0, end_s=1.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.start_s = 99.0  # type: ignore[misc]


def test_license_restriction_roundtrip() -> None:
    r = LicenseRestriction(component="TRIBE v2", license="CC-BY-NC-4.0", note="non-commercial")
    j = r.model_dump_json()
    rebuilt = LicenseRestriction.model_validate_json(j)
    assert rebuilt == r


def test_schema_version_is_stable() -> None:
    """If you bump SCHEMA_VERSION the diff lights up here so it never
    slips in silently."""
    assert SCHEMA_VERSION == "1.0"


def test_provenance_records_python_version() -> None:
    p = ProvenanceMeta(
        cortex_score_version="0.0.0+local",
        model_id="x",
        model_revision="y",
        runner="external",
        python_version="3.11.15",
    )
    assert p.python_version == "3.11.15"
    assert p.schema_version == SCHEMA_VERSION
