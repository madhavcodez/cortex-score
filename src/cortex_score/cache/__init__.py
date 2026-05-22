"""Two-tier on-disk cache.

Prediction cache: keyed on input bytes + preprocessing + runner + model
revision. Reusable across normalization/atlas changes.

Score cache: keyed on prediction cache key PLUS cortex-score version +
schema version + atlas SHA + network-groups SHA + normalization config +
metrics version + serialization encoding. Reusable only within an exact
software pin set.

Splitting the two lets a re-score after a library upgrade reuse the
expensive TRIBE forward pass.
"""

from __future__ import annotations

from cortex_score.cache.keys import (
    prediction_cache_key,
    score_cache_key,
)
from cortex_score.cache.store import (
    CacheStore,
    default_cache_dir,
)

__all__ = [
    "CacheStore",
    "default_cache_dir",
    "prediction_cache_key",
    "score_cache_key",
]
