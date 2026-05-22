"""On-disk cache store.

Two top-level directories under the cache root:

    {cache_root}/
        predictions/
            <key>.npz                 (compressed vertex predictions + segments)
            <key>.json                (lightweight metadata: bundle fields minus the array)
        scores/
            <key>.json                (ScoreResult JSON)
        cache_manifest.json           (running index of all keys + their inputs)

Writes are atomic: contents go to ``{path}.tmp.{pid}.{uuid_hex}``, then
``os.replace`` to the final name. ``os.replace`` is atomic on all
modern filesystems (POSIX and NTFS) per Python's stdlib documentation.
The tmp suffix uses ``uuid.uuid4().hex`` (128 random bits) so two
writers in the same millisecond - or with the same PID after a fork -
never collide on the temp filename.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platformdirs import user_cache_path


def default_cache_dir() -> Path:
    """Return the default cross-platform cache root.

    - Linux/macOS: ``$XDG_CACHE_HOME/cortex-score`` or ``~/.cache/cortex-score``
    - Windows: ``%LOCALAPPDATA%\\cortex-score\\Cache``

    Override with the ``CORTEX_SCORE_CACHE_DIR`` env var.
    """
    override = os.environ.get("CORTEX_SCORE_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return user_cache_path(appname="cortex-score", appauthor=False)


@dataclass(frozen=True)
class CacheEntry:
    """One record returned by lookup."""

    key: str
    path: Path
    inputs: dict[str, Any]


class CacheStore:
    """Lightweight file-system cache with atomic writes.

    The store does not try to be a database. It exists to:

    1. Skip TRIBE forward on identical inputs (prediction cache).
    2. Skip postprocessing on identical inputs (score cache).
    3. Tell the user *why* a cache entry was found or missed via
       ``cache_manifest.json`` — readable JSON, no migrations needed.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or default_cache_dir()).resolve()
        self.predictions_dir = self.root / "predictions"
        self.scores_dir = self.root / "scores"
        self.manifest_path = self.root / "cache_manifest.json"

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def init(self) -> None:
        """Create the cache directories (idempotent)."""
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self._write_manifest({"version": "1.0", "predictions": {}, "scores": {}})

    def info(self) -> dict[str, Any]:
        """Return a small summary about the cache state."""
        manifest = self._read_manifest()
        n_pred = len(manifest.get("predictions", {}))
        n_score = len(manifest.get("scores", {}))
        return {
            "root": str(self.root),
            "predictions_count": n_pred,
            "scores_count": n_score,
            "exists": self.root.exists(),
        }

    def clear(self, *, predictions: bool = True, scores: bool = True) -> None:
        """Remove cache contents (does not delete the root directory)."""

        def _purge(dirpath: Path) -> None:
            if not dirpath.exists():
                return
            for p in dirpath.iterdir():
                if p.is_file():
                    p.unlink()

        if predictions:
            _purge(self.predictions_dir)
        if scores:
            _purge(self.scores_dir)
        # Sweep tmp orphans left by interrupted atomic writes (interrupted
        # process or SIGKILL between tmp write and os.replace).
        if self.root.exists():
            for p in self.root.glob("**/*.tmp.*"):
                with contextlib.suppress(OSError):
                    p.unlink()
        manifest = self._read_manifest()
        if predictions:
            manifest["predictions"] = {}
        if scores:
            manifest["scores"] = {}
        self._write_manifest(manifest)

    # ------------------------------------------------------------------
    # Score cache (the common path: JSON in / JSON out)
    # ------------------------------------------------------------------

    def score_path(self, key: str) -> Path:
        return self.scores_dir / f"{key}.json"

    def has_score(self, key: str) -> bool:
        return self.score_path(key).exists()

    def get_score_json(self, key: str) -> str | None:
        path = self.score_path(key)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def put_score_json(
        self,
        key: str,
        payload: str,
        *,
        inputs: dict[str, Any],
    ) -> Path:
        self.init()
        path = self.score_path(key)
        self._atomic_write_text(path, payload)
        self._record_in_manifest("scores", key, inputs, path)
        return path

    # ------------------------------------------------------------------
    # Prediction cache (NumPy + sidecar JSON)
    # ------------------------------------------------------------------

    def prediction_path(self, key: str) -> Path:
        return self.predictions_dir / f"{key}.npz"

    def prediction_meta_path(self, key: str) -> Path:
        return self.predictions_dir / f"{key}.json"

    def has_prediction(self, key: str) -> bool:
        return self.prediction_path(key).exists()

    def put_prediction_bytes(
        self,
        key: str,
        npz_bytes: bytes,
        meta: dict[str, Any],
        *,
        inputs: dict[str, Any],
    ) -> Path:
        self.init()
        npz_path = self.prediction_path(key)
        meta_path = self.prediction_meta_path(key)
        self._atomic_write_bytes(npz_path, npz_bytes)
        self._atomic_write_text(meta_path, json.dumps(meta, separators=(",", ":")))
        self._record_in_manifest("predictions", key, inputs, npz_path)
        return npz_path

    def get_prediction_bytes(self, key: str) -> tuple[bytes, dict[str, Any]] | None:
        npz_path = self.prediction_path(key)
        meta_path = self.prediction_meta_path(key)
        if not npz_path.exists() or not meta_path.exists():
            return None
        return npz_path.read_bytes(), json.loads(meta_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _record_in_manifest(
        self,
        bucket: str,
        key: str,
        inputs: dict[str, Any],
        path: Path,
    ) -> None:
        manifest = self._read_manifest()
        bucket_dict = manifest.setdefault(bucket, {})
        bucket_dict[key] = {
            "inputs": inputs,
            "path": str(path),
            "created_at": time.time(),
        }
        self._write_manifest(manifest)

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"version": "1.0", "predictions": {}, "scores": {}}
        try:
            parsed: dict[str, Any] = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            return parsed
        except json.JSONDecodeError:
            # Corrupted manifest — start fresh; cached files are still
            # on disk but become orphaned. Better than a fatal crash
            # mid-batch.
            return {"version": "1.0", "predictions": {}, "scores": {}}

    def _write_manifest(self, payload: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._atomic_write_text(
            self.manifest_path,
            json.dumps(payload, indent=2, sort_keys=True),
        )

    # ------------------------------------------------------------------
    # Atomic write primitives
    # ------------------------------------------------------------------

    def _atomic_write_text(self, path: Path, payload: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)

    def _atomic_write_bytes(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
        tmp.write_bytes(payload)
        os.replace(tmp, path)
