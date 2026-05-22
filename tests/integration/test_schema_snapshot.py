"""Reviewer-required: canonical ScoreResult JSON snapshot.

A frozen example fixture under tests/fixtures/score_result_v1.json is
the reference. We:

1. Run the CPU pipeline on a deterministic synthetic input.
2. Compare the resulting JSON Schema (Pydantic-generated) to the
   committed schema snapshot.
3. Compare the resulting payload (sans created_at/result_id) to the
   committed result snapshot.

A failed comparison is a contract change — the test author must either
roll back the change or bump SCHEMA_VERSION and rewrite the snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cortex_score import score_from_predictions
from cortex_score.schemas import SCHEMA_VERSION, ScoreResult

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
_SCHEMA_FIXTURE = _FIXTURES / "schema_v1.json"
_RESULT_FIXTURE = _FIXTURES / "score_result_v1.json"


def _generate_result() -> ScoreResult:
    np.random.seed(2026)
    preds = np.random.randn(6, 20484).astype(np.float32)
    return score_from_predictions(
        preds,
        mesh="fsaverage5",
        tr_seconds=1.0,
        hrf_lag_seconds=5.0,
        model_id="facebook/tribev2",
        model_revision="snapshot-test",
        source="npy",
    )


def _strip_volatile(payload: dict) -> dict:
    """Remove fields that legitimately change between runs."""
    out = dict(payload)
    out.pop("result_id", None)
    out.pop("created_at", None)
    if "provenance" in out:
        prov = dict(out["provenance"])
        prov.pop("python_version", None)
        prov.pop("torch_version", None)
        prov.pop("cuda_available", None)
        prov.pop("device", None)
        prov.pop("cortex_score_version", None)
        out["provenance"] = prov
    return out


def test_schema_snapshot_exists_and_matches() -> None:
    expected = json.loads(_SCHEMA_FIXTURE.read_text(encoding="utf-8"))
    actual = ScoreResult.model_json_schema()
    assert (
        actual == expected
    ), (
        f"ScoreResult JSON Schema diverged from {_SCHEMA_FIXTURE}. "
        "Either revert the schema change OR bump SCHEMA_VERSION and "
        f"re-snapshot. (current SCHEMA_VERSION={SCHEMA_VERSION!r})"
    )


def test_result_snapshot_matches() -> None:
    expected = _strip_volatile(json.loads(_RESULT_FIXTURE.read_text(encoding="utf-8")))
    actual = _strip_volatile(json.loads(_generate_result().to_json(indent=None)))
    assert actual == expected, (
        f"ScoreResult output diverged from {_RESULT_FIXTURE}. "
        "Investigate before merging; if intentional, re-snapshot."
    )
