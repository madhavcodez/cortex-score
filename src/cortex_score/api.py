"""High-level API.

Three entry points, ordered from strictest to friendliest:

1. ``score_from_prediction_bundle(bundle)`` — type-safe, no kwargs.
   Use when you already have a validated ``PredictionBundle``.
2. ``score_from_predictions(preds, *, mesh, tr_seconds, ...)`` — the
   common case for CPU-only users who have a NumPy prediction tensor
   from somewhere else (their own TRIBE run, an .npy file). Forces
   them to declare the scientific assumptions up front.
3. ``score(video_path, *, runner=None)`` — full pipeline. Lazy-loads a
   TRIBE v2 runner unless one is supplied.

Plus ``CortexScorer`` — a small class for batch workflows that want to
load a runner once and score many clips.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from cortex_score.atlas import (
    load_manifest,
    load_schaefer400,
    load_yeo17,
)
from cortex_score.exceptions import MissingOptionalDependencyError, UnsupportedMeshError
from cortex_score.processing.aggregate import aggregate_to_rois
from cortex_score.processing.networks import build_network_summary
from cortex_score.processing.normalize import DEFAULT_EPS, zscore_within_atlas
from cortex_score.processing.validate import (
    coerce_float32,
    validate_predictions_against_mesh,
)
from cortex_score.runners.base import PredictionRunner
from cortex_score.schemas import (
    AtlasMeta,
    InputMeta,
    NetworkScore,
    NormalizationMeta,
    NormalizationScope,
    PredictionBundle,
    ScoreResult,
    ScoreWarning,
    SegmentMeta,
    TimingMeta,
    _detect_torch_environment,
    build_provenance,
    compute_result_id,
    default_tribev2_license_restrictions,
    utc_now,
)

_PredictionSource = Literal["tribev2", "npy", "remote", "unknown"]


@dataclass(frozen=True)
class ScoreConfig:
    """Tweakable scoring knobs.

    Stays small on purpose — anything that affects scientific output is
    recorded into ``ScoreResult`` for provenance, and any new knob
    requires bumping ``METRICS_VERSION`` or ``SCHEMA_VERSION``.

    Attributes:
        normalization_scope: how the z-score is computed. Default is
            ``"within_video"`` (the only scientifically meaningful value
            in v0.1).
        reference_id: cross-clip reference identifier when
            ``normalization_scope == "reference_distribution"``. None
            for within-video.
        epsilon: ridge for the z-score std denominator.
        include_absolute_path: when True, ``score()`` records the
            input's absolute filesystem path in ``ScoreResult.input.absolute_path``.
            Default False because that path embeds the local
            filesystem layout and username — info that does not belong
            in a shareable JSON artifact. The basename is always
            recorded in ``input.filename`` regardless.
        warnings: caller-supplied warnings to merge into
            ``ScoreResult.warnings``.
    """

    normalization_scope: NormalizationScope = "within_video"
    reference_id: str | None = None
    epsilon: float = DEFAULT_EPS
    include_absolute_path: bool = False
    warnings: tuple[ScoreWarning, ...] = field(default_factory=tuple)


def score_from_prediction_bundle(
    bundle: PredictionBundle,
    *,
    config: ScoreConfig | None = None,
    input_meta: InputMeta | None = None,
) -> ScoreResult:
    """Score a validated ``PredictionBundle``.

    This is the type-safe entry point. The bundle already carries every
    piece of metadata the scoring path needs (mesh, TR, HRF lag, model
    id + revision); no kwargs are required.

    Args:
        bundle: the validated prediction bundle.
        config: optional scoring configuration (default = within-video
            z-score, full timeseries).
        input_meta: optional ``InputMeta`` to embed in the result. If
            None, an empty ``InputMeta`` is used (matches direct
            "I have predictions, I don't have the source file" case).

    Returns:
        A ``ScoreResult`` ready to serialize to JSON.
    """
    config = config or ScoreConfig()
    input_meta = input_meta or InputMeta()

    schaefer = load_schaefer400()
    yeo = load_yeo17()
    manifest = load_manifest()

    # Defensive: re-validate even though PredictionBundle's __post_init__
    # already checked shape. This catches the case where a caller built
    # a bundle with a mesh string that does not match the bundled atlas
    # vertex count.
    validate_predictions_against_mesh(
        bundle.vertex_predictions,
        mesh=bundle.mesh,
        expected_n_vertices=schaefer.vertex_to_parcel.shape[0],
    )

    preds = coerce_float32(bundle.vertex_predictions)

    # v0.1 only emits the 5-network rollup, which is computed from
    # Yeo-17. Schaefer-400 is still loaded and its SHA is recorded in
    # the result's provenance (atlas.atlas_sha256) so the bundled atlas
    # set is fully fingerprinted, but per-ROI Schaefer-400 metrics are
    # deferred to a future v0.2 (would need a new JSON sub-schema).
    yeo_preds = aggregate_to_rois(
        preds,
        yeo.vertex_to_parcel,
        yeo.n_parcels,
    )

    z_yeo = zscore_within_atlas(yeo_preds, eps=config.epsilon)

    summaries = build_network_summary(z_yeo)

    network_groups_sha = manifest.file_shas["network_groups.json"]
    networks = tuple(
        NetworkScore(
            id=s["id"],
            label=s["label"],
            description=s["description"],
            color=s["color"],
            yeo_indices=tuple(s["yeo_indices"]),
            yeo_labels=tuple(s["yeo_labels"]),
            mean_energy=s["mean_energy"],
            peak_energy=s["peak_energy"],
            energy_timeseries=tuple(s["energy_timeseries"]),
            mean_z_timeseries=tuple(s["mean_z_timeseries"]),
            group_definition_sha256=network_groups_sha,
        )
        for s in summaries
    )

    timing = TimingMeta(
        tr_seconds=bundle.tr_seconds,
        hrf_lag_seconds=bundle.hrf_lag_seconds,
        n_segments=bundle.n_segments,
    )
    normalization = NormalizationMeta(
        scope=config.normalization_scope,
        epsilon=config.epsilon,
        reference_id=config.reference_id,
    )
    atlas = AtlasMeta(
        mesh=bundle.mesh,
        n_vertices=bundle.n_vertices,
        atlas_version=manifest.atlas_version,
        atlas_sha256=schaefer.sha256,
        yeo_atlas_sha256=yeo.sha256,
        network_groups_sha256=network_groups_sha,
        network_group_source=manifest.network_group_source,
    )

    torch_version, cuda_available, device = _detect_torch_environment()
    provenance = build_provenance(
        model_id=bundle.model_id,
        model_revision=bundle.model_revision,
        runner=_runner_class_path(bundle.source),
        torch_version=torch_version,
        cuda_available=cuda_available,
        device=device,
    )

    license_restrictions = default_tribev2_license_restrictions()

    # Build the result with an empty result_id, then stamp the audit hash
    # computed from the result's OWN canonical serialization. This makes
    # result_id cover exactly the fields ScoreResult serializes (framing,
    # schema_version, created_at, ...) and removes the drift risk of
    # hand-rebuilding a parallel payload dict. See compute_result_id().
    draft = ScoreResult(
        result_id="",
        created_at=utc_now(),
        input=input_meta,
        timing=timing,
        normalization=normalization,
        atlas=atlas,
        provenance=provenance,
        license_restrictions=license_restrictions,
        warnings=config.warnings,
        networks=networks,
    )
    return draft.model_copy(update={"result_id": compute_result_id(draft)})


def score_from_predictions(
    preds: npt.NDArray[np.floating],
    *,
    mesh: str = "fsaverage5",
    tr_seconds: float = 1.0,
    hrf_lag_seconds: float = 5.0,
    model_id: str = "facebook/tribev2",
    model_revision: str = "unknown",
    source: _PredictionSource = "npy",
    segments: tuple[SegmentMeta, ...] | None = None,
    config: ScoreConfig | None = None,
    input_meta: InputMeta | None = None,
) -> ScoreResult:
    """Friendly entry point for the CPU-only postprocessing tier.

    Forces the caller to acknowledge the scientific assumptions
    (mesh, TR, HRF lag, model identity) even when only a bare NumPy
    tensor is on hand. Construct a ``PredictionBundle`` internally and
    delegate to ``score_from_prediction_bundle``.

    Args:
        preds: shape ``(T, V)``, float-compatible. Cast to float32.
        mesh: cortical mesh the predictions live on (default
            ``"fsaverage5"``).
        tr_seconds: TRIBE's effective TR (seconds per segment row).
        hrf_lag_seconds: TRIBE's HRF lag in seconds.
        model_id: HuggingFace-style model id. Default is the TRIBE v2
            id; override if the predictions came from a different model.
        model_revision: model revision / commit / version tag.
            ``"unknown"`` is allowed but discouraged.
        source: one of ``"tribev2"``, ``"npy"``, ``"remote"``, ``"unknown"``.
        segments: optional TRIBE segment time bounds.
        config: optional ``ScoreConfig``.
        input_meta: optional ``InputMeta``.

    Returns:
        A ``ScoreResult``.
    """
    if mesh != "fsaverage5":
        raise UnsupportedMeshError(mesh=mesh, supported=("fsaverage5",))

    # Validate at the public boundary so callers see a clear error
    # before reaching PredictionBundle.__post_init__.
    if not model_id:
        raise ValueError("score_from_predictions(): model_id must be a non-empty string")
    if not model_revision:
        raise ValueError(
            "score_from_predictions(): model_revision must be a non-empty string; "
            "use 'unknown' only if you truly don't know."
        )

    preds_f32 = coerce_float32(preds)
    # Shape-check before reading shape[1] so a 1-D/0-D array fails with a
    # clear ValueError at this boundary instead of an opaque IndexError
    # inside PredictionBundle construction.
    if preds_f32.ndim != 2:
        msg = (
            f"score_from_predictions(): preds must be 2D (T, V); got shape "
            f"{preds_f32.shape}. If you have a 1D time-mean array, reshape to (1, V)."
        )
        raise ValueError(msg)

    bundle = PredictionBundle(
        vertex_predictions=preds_f32,
        mesh="fsaverage5",
        n_vertices=int(preds_f32.shape[1]),
        tr_seconds=tr_seconds,
        hrf_lag_seconds=hrf_lag_seconds,
        model_id=model_id,
        model_revision=model_revision,
        source=source,
        segments=segments or (),
    )
    return score_from_prediction_bundle(bundle, config=config, input_meta=input_meta)


def score(
    video_path: str | Path,
    *,
    runner: PredictionRunner | None = None,
    config: ScoreConfig | None = None,
) -> ScoreResult:
    """Full pipeline: video file -> ScoreResult.

    Requires a ``PredictionRunner``. If none is supplied, the default
    TRIBE v2 runner is loaded lazily; this will raise
    ``MissingOptionalDependencyError`` if the ``[gpu-deps]`` extra and
    TRIBE v2 itself have not been installed.

    Args:
        video_path: path to an .mp4 (or any container ffmpeg understands).
        runner: optional ``PredictionRunner`` instance. Construct one
            explicitly to control device, cache dir, or model revision.
        config: optional ``ScoreConfig``.

    Returns:
        A ``ScoreResult``.

    Raises:
        FileNotFoundError: if ``video_path`` does not exist.
        MissingOptionalDependencyError: if no runner is supplied and
            TRIBE v2 cannot be imported.
    """
    path = Path(video_path)
    if not path.exists():
        msg = f"video file not found: {path}"
        raise FileNotFoundError(msg)

    if runner is None:
        runner = _load_default_runner()
    _validate_runner(runner)

    bundle = runner.predict_video(path)
    cfg = config or ScoreConfig()
    input_meta = _build_input_meta(path, include_absolute_path=cfg.include_absolute_path)
    return score_from_prediction_bundle(bundle, config=cfg, input_meta=input_meta)


# ---------------------------------------------------------------------
# CortexScorer (class form for batch reuse)
# ---------------------------------------------------------------------


class CortexScorer:
    """Load a runner once, score many clips.

    The bare ``score()`` function lazy-loads a default TRIBE v2 runner
    on every call, which is correct for one-shot use but wasteful when
    scoring a batch (TRIBE weights are 12 GB and take ~30 s of cold
    start). Construct a ``CortexScorer`` once and reuse it::

        scorer = CortexScorer()
        for clip in clips:
            scorer.score(clip).save(out_dir / f"{clip.stem}.json")
    """

    def __init__(
        self,
        runner: PredictionRunner | None = None,
        *,
        config: ScoreConfig | None = None,
    ) -> None:
        self._runner = runner
        self._config = config or ScoreConfig()

    @property
    def runner(self) -> PredictionRunner:
        """Return the resident runner, constructing the default lazily."""
        if self._runner is None:
            self._runner = _load_default_runner()
        return self._runner

    def score(self, video_path: str | Path) -> ScoreResult:
        return score(video_path, runner=self.runner, config=self._config)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _load_default_runner() -> PredictionRunner:
    """Lazy import of the TRIBE v2 runner.

    Kept here (not at module top) so ``import cortex_score`` never pays
    the torch import cost.
    """
    try:
        from cortex_score.runners.tribev2 import TribeV2Runner
    except MissingOptionalDependencyError:
        raise
    except ImportError as exc:
        raise MissingOptionalDependencyError(
            package="tribev2",
            install_hint=(
                "pip install 'cortex-score[gpu-deps]'\n"
                "    pip install -r requirements/tribev2-gpu.txt"
            ),
        ) from exc
    return TribeV2Runner()


def _runner_class_path(source: str) -> str:
    """Map PredictionBundle.source to a runner class path string."""
    if source == "tribev2":
        return "cortex_score.runners.tribev2.TribeV2Runner"
    return "external"


def _validate_runner(runner: PredictionRunner) -> None:
    """Fail fast if a custom runner doesn't carry the provenance attributes.

    The ``PredictionRunner`` Protocol declares ``model_id`` and
    ``model_revision`` as required attributes, but Python's
    ``@runtime_checkable`` only verifies *methods*, not data attributes.
    A custom runner missing one of these would silently pass an
    ``isinstance(runner, PredictionRunner)`` check and then write an
    empty string into ``ScoreResult.provenance.model_id`` /
    ``model_revision``, corrupting the audit trail.

    This guard is called once per ``score()`` invocation right before
    ``runner.predict_video()`` so the error surfaces close to the
    caller's point of confusion (not three frames deep inside
    aggregation).
    """
    model_id = getattr(runner, "model_id", None)
    model_revision = getattr(runner, "model_revision", None)
    cls_name = type(runner).__name__
    if not isinstance(model_id, str) or not model_id:
        msg = (
            f"Runner {cls_name} must define a non-empty 'model_id' attribute "
            "(declared by the PredictionRunner Protocol). Provenance fields "
            "in ScoreResult depend on it."
        )
        raise ValueError(msg)
    if not isinstance(model_revision, str) or not model_revision:
        msg = (
            f"Runner {cls_name} must define a non-empty 'model_revision' "
            "attribute (declared by the PredictionRunner Protocol). Use "
            "'unknown' if a revision truly cannot be determined; never leave it empty."
        )
        raise ValueError(msg)


def _build_input_meta(path: Path, *, include_absolute_path: bool = False) -> InputMeta:
    """Build an ``InputMeta`` for a real file on disk.

    Computes SHA-256 chunked so a 100 MB clip doesn't blow memory.
    Other fields (duration, fps, resolution) are filled in when we
    have ffprobe access — left None here to keep this helper
    side-effect-free.

    Args:
        path: source video file path.
        include_absolute_path: when True, embed the resolved absolute
            path in ``InputMeta.absolute_path``. Default False to avoid
            leaking the user's filesystem layout into shareable JSON.
    """
    sha = _sha256_file(path)
    return InputMeta(
        filename=path.name,
        absolute_path=str(path.resolve()) if include_absolute_path else None,
        content_sha256=sha,
    )


def _sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# Re-exports for cortex_score.__init__.py convenience.
__all__ = [
    "CortexScorer",
    "ScoreConfig",
    "score",
    "score_from_prediction_bundle",
    "score_from_predictions",
]
