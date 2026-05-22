"""Cache-key derivation.

Why two separate keys instead of one:

* A re-score after the user upgraded ``cortex-score`` or the network
  grouping changed is cheap (milliseconds of NumPy work) — but the
  prediction tensor is expensive (~30 s of TRIBE forward on an A100).
  Keeping the prediction key independent of postprocessing knobs means
  upgrades reuse the expensive part.
* Conversely, mutating a normalization knob must invalidate the
  resulting score immediately, even though the predictions did not
  change.

Stability rules:

1. Every input to either function must be JSON-serializable so we can
   pretty-print the key components in ``cache_manifest.json`` next to
   the cached file (helps debugging "why did this cache miss?").
2. Keys are SHA-256 hex digests. Truncation is acceptable for filenames
   (16 chars) but the full digest is what goes into the manifest.
3. Adding a new field is breaking. Bump ``METRICS_VERSION`` or
   ``SERIALIZATION_VERSION`` and document the migration in the
   changelog.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PredictionCacheInputs:
    """All inputs that must be identical for two prediction runs to be
    interchangeable.

    Equivalence of every field below is the contract: if any one
    differs, the cached prediction is invalid.
    """

    input_content_sha256: str
    preprocessing_config_sha256: str
    runner_name: str
    model_id: str
    model_revision: str
    tribev2_code_revision: str
    device_settings_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_content_sha256": self.input_content_sha256,
            "preprocessing_config_sha256": self.preprocessing_config_sha256,
            "runner_name": self.runner_name,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tribev2_code_revision": self.tribev2_code_revision,
            "device_settings_sha256": self.device_settings_sha256,
        }


@dataclass(frozen=True)
class ScoreCacheInputs:
    """Score-cache inputs — strict superset of the prediction key.

    A score key change implies recomputing the score; the underlying
    prediction may still be cached.
    """

    prediction_key: str
    cortex_score_version: str
    schema_version: str
    atlas_sha256: str
    yeo_atlas_sha256: str
    network_groups_sha256: str
    normalization_config_sha256: str
    metrics_version: str
    serialization_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_key": self.prediction_key,
            "cortex_score_version": self.cortex_score_version,
            "schema_version": self.schema_version,
            "atlas_sha256": self.atlas_sha256,
            "yeo_atlas_sha256": self.yeo_atlas_sha256,
            "network_groups_sha256": self.network_groups_sha256,
            "normalization_config_sha256": self.normalization_config_sha256,
            "metrics_version": self.metrics_version,
            "serialization_version": self.serialization_version,
        }


def _canonical_sha(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def prediction_cache_key(inputs: PredictionCacheInputs) -> str:
    """Stable SHA-256 of a ``PredictionCacheInputs``."""
    return _canonical_sha(inputs.to_dict())


def score_cache_key(inputs: ScoreCacheInputs) -> str:
    """Stable SHA-256 of a ``ScoreCacheInputs``."""
    return _canonical_sha(inputs.to_dict())


def hash_canonical_config(config: dict[str, Any]) -> str:
    """Hash any arbitrary config dict canonically (used for sub-component shas).

    Helpers like ``preprocessing_config_sha256`` and
    ``device_settings_sha256`` should be computed via this so two callers
    producing the same logical config also produce the same hash.
    """
    return _canonical_sha(config)
