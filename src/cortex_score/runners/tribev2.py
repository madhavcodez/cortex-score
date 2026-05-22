"""TRIBE v2 ``PredictionRunner`` adapter.

Thin wrapper around the public TRIBE v2 API:

    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=...)
    events = model.get_events_dataframe(video_path=...)
    preds, segments = model.predict(events=events)

Why a wrapper rather than re-implementation: TRIBE v2 is upstream
research code and we explicitly do not want to fork it. Tracking the
public ``demo_utils.TribeModel`` surface keeps us forward-compatible
with whatever bug fixes Meta lands without us having to maintain
internals.

Why TRIBE is not in ``pyproject.toml``: PyPI rejects published package
metadata that contains direct-URL dependencies. The
``[project.optional-dependencies]`` ``gpu-deps`` group declares the
*environmental* matrix (torch / transformers / moviepy versions) that
TRIBE v2 itself requires, but TRIBE itself is installed from
``requirements/tribev2-gpu.txt`` after the extra is installed.

License: TRIBE v2 is CC-BY-NC-4.0. We emit a one-time runtime warning
on first construction so a caller using ``cortex-score`` in a
commercial context sees the restriction at runtime, not just in a
buried LICENSE file.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np

from cortex_score.exceptions import (
    MissingExternalToolError,
    MissingOptionalDependencyError,
)
from cortex_score.schemas import PredictionBundle, SegmentMeta

# The TRIBE v2 commit this runner was developed and tested against. Must
# match the SHA pin in ``requirements/tribev2-gpu.txt``. Bumping one
# requires bumping the other and re-running the GPU smoke test.
TRIBEV2_PINNED_REVISION: str = "34f52344e5ba96660fac877393e1954e399d3ef3"

# TRIBE's effective TR (seconds per segment) and HRF lag. Mirrors
# Cortexia's settings; both are baked into upstream TRIBE's preprocessing
# pipeline and are not user-tunable without retraining.
DEFAULT_TR_SECONDS: float = 1.0
DEFAULT_HRF_LAG_SECONDS: float = 5.0

_LICENSE_WARNING_EMITTED = False


def _emit_license_warning_once() -> None:
    global _LICENSE_WARNING_EMITTED
    if _LICENSE_WARNING_EMITTED:
        return
    _LICENSE_WARNING_EMITTED = True
    warnings.warn(
        (
            "TRIBE v2 is licensed under CC-BY-NC-4.0 (non-commercial). "
            "Scores produced via the full inference path inherit this restriction. "
            "Review https://huggingface.co/facebook/tribev2 before commercial use."
        ),
        stacklevel=3,
    )


def _default_cache_folder() -> Path:
    """Where to put the ~12 GB of model weights. Mirrors HF cache conventions."""
    env = os.environ.get("CORTEX_SCORE_MODEL_CACHE")
    if env:
        return Path(env).expanduser().resolve()
    # platformdirs would give us LocalAppData/Caches; for big model
    # weights that's correct on macOS/Linux but on Windows the user
    # may not want them on the C: drive. Default mirrors HF_HOME.
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser().resolve()
    return Path.home() / ".cache" / "huggingface"


class TribeV2Runner:
    """Public ``PredictionRunner`` for the TRIBE v2 model.

    Construction is cheap — model weights load lazily on the first
    ``predict_video`` call so ``CortexScorer(TribeV2Runner())`` doesn't
    pay the ~30 s cold start until it actually needs to.
    """

    model_id: str = "facebook/tribev2"
    model_revision: str = TRIBEV2_PINNED_REVISION

    def __init__(
        self,
        *,
        cache_folder: Path | None = None,
        device: str = "auto",
        tr_seconds: float = DEFAULT_TR_SECONDS,
        hrf_lag_seconds: float = DEFAULT_HRF_LAG_SECONDS,
    ) -> None:
        self.cache_folder = (cache_folder or _default_cache_folder()).resolve()
        self.device = device
        self.tr_seconds = tr_seconds
        self.hrf_lag_seconds = hrf_lag_seconds
        self._model: object | None = None
        # CC-BY-NC license warning is emitted on first ``_load()``, not
        # at construction time, so building a runner is side-effect free
        # for tests and advisory uses (e.g. reading ``model_revision``).

    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Resolve TRIBE v2 lazily; raise a friendly error if missing."""
        if self._model is not None:
            return

        try:
            from tribev2.demo_utils import TribeModel
        except ImportError as exc:
            raise MissingOptionalDependencyError(
                package="tribev2",
                install_hint=(
                    "pip install 'cortex-score[gpu-deps]'\n"
                    "    pip install -r requirements/tribev2-gpu.txt"
                ),
            ) from exc

        # TRIBE is actually present -> emit the CC-BY-NC reminder once.
        _emit_license_warning_once()

        # ffmpeg / uvx are runtime requirements of TRIBE's preprocessing.
        # Surface their absence as ``MissingExternalToolError`` so the
        # user can act, instead of letting tribev2 fail mid-subprocess.
        import shutil

        for tool, hint in (
            ("ffmpeg", "https://ffmpeg.org/download.html"),
            ("uvx", "pip install uv (or curl -LsSf https://astral.sh/uv/install.sh | sh)"),
        ):
            if shutil.which(tool) is None:
                raise MissingExternalToolError(tool=tool, install_hint=hint)

        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self._model = TribeModel.from_pretrained(
            self.model_id,
            cache_folder=self.cache_folder,
            device=self.device,
        )

    # ------------------------------------------------------------------

    def predict_video(self, path: Path) -> PredictionBundle:
        """Run TRIBE v2 inference end-to-end.

        Args:
            path: a video file. TRIBE handles its own ffmpeg / WhisperX
                preprocessing internally.

        Returns:
            A ``PredictionBundle`` with ``source="tribev2"`` and the
            recorded model revision.
        """
        if not path.exists():
            msg = f"video file not found: {path}"
            raise FileNotFoundError(msg)

        self._load()
        assert self._model is not None  # for mypy

        events = self._model.get_events_dataframe(video_path=str(path))  # type: ignore[attr-defined]
        preds, segments_obj = self._model.predict(events=events)  # type: ignore[attr-defined]

        preds_np = np.asarray(preds, dtype=np.float32)
        if preds_np.ndim != 2:
            msg = f"TRIBE v2 returned non-2D predictions with shape {preds_np.shape}"
            raise RuntimeError(msg)

        segments = _parse_segments(segments_obj)

        return PredictionBundle(
            vertex_predictions=preds_np,
            mesh="fsaverage5",
            n_vertices=int(preds_np.shape[1]),
            tr_seconds=self.tr_seconds,
            hrf_lag_seconds=self.hrf_lag_seconds,
            model_id=self.model_id,
            model_revision=self.model_revision,
            source="tribev2",
            segments=segments,
        )


def _parse_segments(segments_obj: object) -> tuple[SegmentMeta, ...]:
    """Coerce TRIBE's segments object into ``tuple[SegmentMeta, ...]``.

    TRIBE returns a DataFrame-like in current main; we accept anything
    iterable that yields per-row ``start`` / ``end`` floats so future
    upstream changes do not break the contract.
    """
    if segments_obj is None:
        return ()

    out: list[SegmentMeta] = []

    # pandas DataFrame path
    try:
        import pandas as pd

        if isinstance(segments_obj, pd.DataFrame):
            cols = {c.lower(): c for c in segments_obj.columns}
            start_col = cols.get("start") or cols.get("start_s") or cols.get("t_start")
            end_col = cols.get("end") or cols.get("end_s") or cols.get("t_end")
            if start_col is None or end_col is None:
                return ()
            for i, row in segments_obj.reset_index(drop=True).iterrows():
                out.append(
                    SegmentMeta(
                        index=int(i),
                        start_s=float(row[start_col]),
                        end_s=float(row[end_col]),
                    )
                )
            return tuple(out)
    except ImportError:
        pass

    # Iterable-of-dicts fallback
    from collections.abc import Iterable
    from typing import cast

    if not isinstance(segments_obj, Iterable):
        return ()

    iterable = cast(Iterable[object], segments_obj)

    def _pick(d: dict[str, object], *keys: str, default: float = 0.0) -> float:
        """Return the first present key's value as a float.

        Uses ``in`` rather than truthiness because a legitimate
        ``0.0`` second (start of clip) would otherwise be skipped by
        ``or``-chaining. This was the bug surfaced in pre-v0.1
        code review.
        """
        for key in keys:
            if key in d:
                return float(d[key])  # type: ignore[arg-type]
        return default

    try:
        for i, item in enumerate(iterable):
            if isinstance(item, dict):
                d = cast(dict[str, object], item)
                start = _pick(d, "start", "start_s", "t_start")
                end = _pick(d, "end", "end_s", "t_end")
                out.append(SegmentMeta(index=i, start_s=start, end_s=end))
        return tuple(out)
    except TypeError:
        return ()
