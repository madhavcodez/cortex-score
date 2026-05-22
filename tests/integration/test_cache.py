"""Reviewer-required: cache invalidation matrix.

Mutating any of the inputs that go into a cache key must change the
resulting key. These tests would have caught the cortexia FU-1
"preprocessing_version" follow-up — a missed cache invalidation that
silently served stale data.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from cortex_score.cache.keys import (
    PredictionCacheInputs,
    ScoreCacheInputs,
    hash_canonical_config,
    prediction_cache_key,
    score_cache_key,
)
from cortex_score.cache.store import CacheStore


def _base_prediction_inputs() -> PredictionCacheInputs:
    return PredictionCacheInputs(
        input_content_sha256="a" * 64,
        preprocessing_config_sha256="b" * 64,
        runner_name="cortex_score.runners.tribev2.TribeV2Runner",
        model_id="facebook/tribev2",
        model_revision="34f52344",
        tribev2_code_revision="34f52344",
        device_settings_sha256="c" * 64,
    )


def _base_score_inputs(pred_key: str) -> ScoreCacheInputs:
    return ScoreCacheInputs(
        prediction_key=pred_key,
        cortex_score_version="0.1.0",
        schema_version="1.0",
        atlas_sha256="d" * 64,
        yeo_atlas_sha256="e" * 64,
        network_groups_sha256="f" * 64,
        normalization_config_sha256="9" * 64,
        metrics_version="1.0",
        serialization_version="1.0",
    )


@pytest.mark.parametrize(
    "field",
    [
        "input_content_sha256",
        "preprocessing_config_sha256",
        "runner_name",
        "model_id",
        "model_revision",
        "tribev2_code_revision",
        "device_settings_sha256",
    ],
)
def test_prediction_key_changes_on_each_field(field: str) -> None:
    base = _base_prediction_inputs()
    mutated = dataclasses.replace(base, **{field: "z" * 64 if "sha256" in field else "mutated"})
    assert prediction_cache_key(base) != prediction_cache_key(mutated)


@pytest.mark.parametrize(
    "field",
    [
        "prediction_key",
        "cortex_score_version",
        "schema_version",
        "atlas_sha256",
        "yeo_atlas_sha256",
        "network_groups_sha256",
        "normalization_config_sha256",
        "metrics_version",
        "serialization_version",
    ],
)
def test_score_key_changes_on_each_field(field: str) -> None:
    pred = prediction_cache_key(_base_prediction_inputs())
    base = _base_score_inputs(pred)
    # Sentinel for mutation. Use a value guaranteed not to match ANY
    # base field (none of the base shas use "0").
    mutation_value: str | int = "0" * 64 if "sha256" in field else "mutated"
    if field == "prediction_key":
        mutated = dataclasses.replace(base, prediction_key="x" * 64)
    else:
        mutated = dataclasses.replace(base, **{field: mutation_value})
    assert score_cache_key(base) != score_cache_key(mutated)


def test_same_inputs_same_key() -> None:
    a = _base_prediction_inputs()
    b = _base_prediction_inputs()
    assert prediction_cache_key(a) == prediction_cache_key(b)


def test_hash_canonical_config_is_order_independent() -> None:
    a = hash_canonical_config({"a": 1, "b": 2})
    b = hash_canonical_config({"b": 2, "a": 1})
    assert a == b


# ---- Store -------------------------------------------------------------


def test_cache_store_atomic_text_write(tmp_path: Path) -> None:
    store = CacheStore(root=tmp_path)
    store.init()
    payload = '{"hello": "world"}'
    written = store.put_score_json(
        "abc123",
        payload,
        inputs={"reason": "test"},
    )
    assert written.exists()
    assert written.read_text(encoding="utf-8") == payload
    # No leftover .tmp.* file
    tmps = list(tmp_path.rglob("*.tmp.*"))
    assert tmps == []


def test_cache_store_score_get(tmp_path: Path) -> None:
    store = CacheStore(root=tmp_path)
    store.put_score_json("key1", '{"x":1}', inputs={})
    assert store.has_score("key1")
    assert store.get_score_json("key1") == '{"x":1}'
    assert store.get_score_json("missing") is None


def test_cache_store_clear_removes_entries(tmp_path: Path) -> None:
    store = CacheStore(root=tmp_path)
    store.put_score_json("key1", '{"x":1}', inputs={})
    store.put_prediction_bytes("predk", b"\x01\x02", {"m": 1}, inputs={})
    store.clear()
    assert not store.has_score("key1")
    assert not store.has_prediction("predk")


def test_cache_store_manifest_is_written(tmp_path: Path) -> None:
    import json

    store = CacheStore(root=tmp_path)
    store.put_score_json("key1", '{"x":1}', inputs={"runner": "test"})
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    assert "key1" in manifest["scores"]
    assert manifest["scores"]["key1"]["inputs"]["runner"] == "test"
