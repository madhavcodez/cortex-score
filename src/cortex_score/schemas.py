"""Public Pydantic v2 schemas (and one helper dataclass).

This module is the formal output contract. Every change that adds or
removes a field, renames anything, or relaxes a constraint MUST bump
``SCHEMA_VERSION``. Bumping ``SCHEMA_VERSION`` also invalidates the
score cache (see ``cache.keys.score_cache_key``).

Why a strict schema: downstream AI/video pipelines that integrate
``cortex-score`` need to validate the JSON they consume. Pydantic v2
gives us both runtime validation and a JSON Schema export for tooling
(see ``cortex-score schema`` CLI command).

Why ``PredictionBundle`` is a frozen dataclass and not a Pydantic model:
it owns a NumPy array that we don't want to serialize. The bundle is an
in-memory contract between a runner and the scoring path, not a
persistable artifact. The persisted artifact is ``ScoreResult``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from cortex_score.processing.metrics import METRICS_VERSION
from cortex_score.processing.networks import NetworkId

# _CORTEX_SCORE_VERSION is a module-private constant (SCREAMING_SNAKE_CASE)
# aliasing the dunder __version__. N812 misreads "lowercase -> non-lowercase"
# but the destination IS a constant; the alias is intentional.
from cortex_score.version import __version__ as _CORTEX_SCORE_VERSION  # noqa: N812

SCHEMA_VERSION: str = "1.0"
"""Top-level schema version. Bump on any breaking JSON contract change."""

SERIALIZATION_VERSION: str = "1.0"
"""Bumped when the on-the-wire encoding changes (compact mode, int16
timeseries, etc.). Recorded into the score cache key so re-serializing
invalidates older cached blobs."""

FRAMING_PRIMARY: str = (
    "Score any video for predicted cortical engagement across 5 brain networks "
    "(visual, language, faces, attention, motion). Built on Meta FAIR's TRIBE v2."
)
"""Primary headline framing — kept verbatim across docs and JSON output."""

FRAMING_SCIENTIFIC: str = (
    "cortex-score summarizes TRIBE v2 predicted cortical responses for any video "
    "across five Cortexia-defined network groups."
)
"""Secondary scientific clarification."""

FRAMING_DISCLAIMER: str = (
    "cortex-score does not measure real viewer engagement. It summarizes predicted "
    "fMRI-like responses from a pretrained brain-encoding model for an average subject."
)
"""Mandatory disclaimer recorded in every ScoreResult."""

Mesh = Literal["fsaverage5"]
"""Supported cortical meshes. Reserved as a Literal so new meshes are an
explicit, reviewed addition (not silently accepted via a raw string)."""

NormalizationScope = Literal["within_video", "reference_distribution"]
"""How the z-score was computed. Critical scientific provenance.

- ``within_video`` (default in v0.1): each ROI's mean/std come from the
  single clip's own timeline. Scores from two different clips are NOT
  directly comparable on the same numeric axis.
- ``reference_distribution`` (future): each ROI is normalized against a
  named cross-clip reference distribution. Scores are then comparable
  across clips that share the same ``reference_id``.
"""


# ---------------------------------------------------------------------
# In-memory transport objects (not in JSON output)
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentMeta:
    """One TRIBE v2 segment with its time bounds.

    TRIBE returns a list of event segments alongside the prediction
    tensor. Recording start/end seconds (when known) lets downstream
    consumers align ``mean_z_timeseries`` to real wall-clock time on
    the original video.
    """

    index: int
    start_s: float
    end_s: float


@dataclass(frozen=True)
class PredictionBundle:
    """Validated input to the postprocessing tier.

    A bare ``(T, V)`` NumPy array is not a safe scientific input: it
    doesn't say which mesh, vertex order, TR, HRF lag, model revision,
    or segment layout it came from. ``PredictionBundle`` makes those
    explicit so ``score_from_prediction_bundle`` can verify atlas
    compatibility and write honest provenance into the resulting
    ``ScoreResult``.

    Construct via ``PredictionBundle(vertex_predictions=..., ...)``
    directly or via the ergonomic ``score_from_predictions(...)``
    helper which forwards kwargs into a bundle.

    Attributes:
        vertex_predictions: shape ``(T, n_vertices)`` float32 array.
        mesh: cortical mesh the predictions live on. Currently only
            ``"fsaverage5"`` is supported.
        n_vertices: vertex count (must equal ``vertex_predictions.shape[1]``).
        tr_seconds: TRIBE's effective TR (seconds per segment row).
        hrf_lag_seconds: TRIBE's HRF lag in seconds.
        model_id: HuggingFace-style model id (e.g. ``"facebook/tribev2"``).
        model_revision: model revision / commit / version tag. ``"unknown"``
            is permitted but discouraged.
        source: where this bundle came from. ``"tribev2"`` for a runner
            invocation, ``"npy"`` for a ``.npy`` file the user loaded
            themselves, ``"remote"`` for a remote inference call,
            ``"unknown"`` as a permissive fallback.
        segments: optional list of TRIBE segment time bounds.
    """

    vertex_predictions: npt.NDArray[np.float32]
    mesh: Mesh
    n_vertices: int
    tr_seconds: float
    hrf_lag_seconds: float
    model_id: str
    model_revision: str
    source: Literal["tribev2", "npy", "remote", "unknown"] = "unknown"
    segments: tuple[SegmentMeta, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.vertex_predictions.ndim != 2:
            msg = (
                "PredictionBundle.vertex_predictions must be 2D (T, V); got "
                f"shape {self.vertex_predictions.shape}"
            )
            raise ValueError(msg)
        if self.vertex_predictions.dtype != np.float32:
            msg = (
                "PredictionBundle.vertex_predictions must be float32 (got "
                f"{self.vertex_predictions.dtype}). Use "
                "``cortex_score.processing.validate.coerce_float32`` if you "
                "have a float64 array."
            )
            raise ValueError(msg)
        if self.vertex_predictions.shape[1] != self.n_vertices:
            msg = (
                f"PredictionBundle.n_vertices={self.n_vertices} does not "
                f"match vertex_predictions.shape[1]={self.vertex_predictions.shape[1]}"
            )
            raise ValueError(msg)
        if self.tr_seconds <= 0:
            raise ValueError("tr_seconds must be > 0")
        if self.hrf_lag_seconds < 0:
            raise ValueError("hrf_lag_seconds must be >= 0")
        if not self.model_id:
            raise ValueError("model_id is required")
        if not self.model_revision:
            raise ValueError("model_revision is required (use 'unknown' if truly unknown)")

    @property
    def n_segments(self) -> int:
        return int(self.vertex_predictions.shape[0])


# ---------------------------------------------------------------------
# JSON-serializable Pydantic models (the output contract)
# ---------------------------------------------------------------------


class _StrictModel(BaseModel):
    """Common config: strict, immutable, JSON-serializable."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class InputMeta(_StrictModel):
    """Where the prediction came from on disk / wire.

    ``filename`` is always safe to share — it's just the basename of the
    input video. ``absolute_path`` is opt-in: by default it is None, so
    shared ScoreResult JSON does not leak the user's filesystem layout
    or home-directory username. To populate it, pass
    ``ScoreConfig(include_absolute_path=True)`` or construct the
    ``InputMeta`` explicitly.
    """

    filename: str | None = Field(
        default=None,
        description=("Basename of the input video (e.g. 'clip.mp4'). Safe to share."),
    )
    absolute_path: str | None = Field(
        default=None,
        description=(
            "Absolute filesystem path of the input video. Opt-in via "
            "ScoreConfig.include_absolute_path because it embeds local "
            "paths (and usernames) into a shareable artifact."
        ),
    )
    content_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-fA-F]{64}$",
        description="SHA-256 (hex) of the input bytes, if a file was provided.",
    )
    duration_s: float | None = Field(
        default=None,
        ge=0,
        description="Video duration in seconds.",
    )
    fps: float | None = Field(default=None, gt=0, description="Frames per second.")
    resolution: str | None = Field(
        default=None,
        pattern=r"^\d+x\d+$",
        description="Resolution as 'WxH' (e.g. '720x1280').",
    )


class TimingMeta(_StrictModel):
    """TRIBE-side timing constants used for this prediction."""

    tr_seconds: float = Field(gt=0, description="Effective TR (seconds per segment).")
    hrf_lag_seconds: float = Field(ge=0, description="HRF lag in seconds.")
    n_segments: int = Field(ge=1, description="Number of prediction segments.")


class NormalizationMeta(_StrictModel):
    """How the per-network z-scores were computed."""

    method: Literal["zscore"] = "zscore"
    scope: NormalizationScope = "within_video"
    epsilon: float = Field(
        gt=0,
        description="Ridge added to the std denominator to avoid division by zero.",
    )
    reference_id: str | None = Field(
        default=None,
        description=(
            "Identifier of the cross-clip reference distribution used "
            "when scope='reference_distribution'. None for within-video."
        ),
    )

    @model_validator(mode="after")
    def _check_reference_id(self) -> NormalizationMeta:
        if self.scope == "reference_distribution" and not self.reference_id:
            msg = "reference_id is required when normalization scope is 'reference_distribution'"
            raise ValueError(msg)
        return self


class AtlasMeta(_StrictModel):
    """Atlas provenance — pins the bundled data this score was computed against."""

    mesh: Mesh
    n_vertices: int = Field(ge=1)
    atlas_version: str = Field(
        description="Bundled atlas identifier (e.g. 'schaefer2018-400-yeo17-fsaverage5')."
    )
    atlas_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 (lowercase hex) of the Schaefer-400 vertex .npy used.",
    )
    yeo_atlas_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 (lowercase hex) of the Yeo-17 vertex .npy used.",
    )
    network_groups_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 (lowercase hex) of network_groups.json used.",
    )
    network_group_source: str = Field(
        description="Group definition source id (e.g. 'cortexia-network-groups-v1')."
    )


class ProvenanceMeta(_StrictModel):
    """Software-side provenance.

    These fields are enough to reproduce a score offline given the same
    input bytes.
    """

    cortex_score_version: str = Field(min_length=1)
    schema_version: str = SCHEMA_VERSION
    metrics_version: str = METRICS_VERSION
    serialization_version: str = SERIALIZATION_VERSION
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    tribev2_package_version: str | None = None
    runner: str = Field(
        min_length=1,
        description=(
            "Fully qualified class name of the runner that produced the "
            "predictions (e.g. 'cortex_score.runners.tribev2.TribeV2Runner') "
            "or 'external' when score_from_predictions was used directly."
        ),
    )
    python_version: str = Field(min_length=1)
    torch_version: str | None = None
    cuda_available: bool | None = None
    device: str | None = None


class LicenseRestriction(_StrictModel):
    """A non-MIT restriction that applies to the result."""

    component: str
    license: str
    note: str


class ScoreWarning(_StrictModel):
    """One warning emitted during scoring (preprocessing, validation, etc.)."""

    code: str
    message: str
    severity: Literal["info", "warning", "error"] = "warning"


class NetworkScore(_StrictModel):
    """One of the five 5-network rollup entries."""

    id: NetworkId
    label: str
    description: str
    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    yeo_indices: tuple[int, ...]
    yeo_labels: tuple[str, ...]
    mean_energy: float = Field(ge=0)
    peak_energy: float = Field(ge=0)
    energy_timeseries: tuple[float, ...]
    mean_z_timeseries: tuple[float, ...]
    group_definition_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 (lowercase hex) of the network_groups.json that defined this group.",
    )

    @model_validator(mode="after")
    def _labels_match_indices(self) -> NetworkScore:
        # model_validator(mode="after") sees both fields as typed
        # attributes — no Pydantic field-ordering dependency, no dict
        # lookup, no dead hasattr guard. Matches the cross-field pattern
        # used by NormalizationMeta and ScoreResult in this module.
        if len(self.yeo_labels) != len(self.yeo_indices):
            msg = (
                f"yeo_labels length {len(self.yeo_labels)} does not match "
                f"yeo_indices length {len(self.yeo_indices)}"
            )
            raise ValueError(msg)
        return self


class ScoreResult(_StrictModel):
    """Top-level result — the JSON contract of cortex-score.

    Fields are intentionally verbose: this artifact may be archived,
    audited, cited, or fed into another model years from now, so being
    self-describing matters more than being terse.
    """

    schema_version: str = SCHEMA_VERSION
    result_id: str = Field(
        description=(
            "SHA-256 audit identity. Computed as the hash of this result's "
            "model_dump(mode='json') with result_id set to '', re-serialized "
            "with sorted keys and compact separators. Reproducible from the "
            "JSON alone, so consumers can verify it. See compute_result_id()."
        ),
    )
    created_at: _dt.datetime = Field(
        description="UTC ISO-8601 timestamp at which the score was computed."
    )

    # Framing — primary headline + scientific clarification + disclaimer.
    framing: str = Field(default=FRAMING_PRIMARY)
    framing_scientific: str = Field(default=FRAMING_SCIENTIFIC)
    framing_disclaimer: str = Field(default=FRAMING_DISCLAIMER)

    input: InputMeta
    timing: TimingMeta
    normalization: NormalizationMeta
    atlas: AtlasMeta
    provenance: ProvenanceMeta
    license_restrictions: tuple[LicenseRestriction, ...] = ()
    warnings: tuple[ScoreWarning, ...] = ()

    networks: tuple[NetworkScore, ...]

    @field_validator("networks")
    @classmethod
    def _exactly_five_networks(
        cls,
        v: tuple[NetworkScore, ...],
    ) -> tuple[NetworkScore, ...]:
        if len(v) != 5:
            msg = f"ScoreResult.networks must have exactly 5 entries, got {len(v)}"
            raise ValueError(msg)
        ids = [n.id for n in v]
        expected = ["visual", "language", "faces", "attention", "motion"]
        if ids != expected:
            msg = f"ScoreResult.networks must be in order {expected}, got {ids}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def _check_timeseries_length(self) -> ScoreResult:
        for net in self.networks:
            if len(net.energy_timeseries) != self.timing.n_segments:
                msg = (
                    f"network '{net.id}' has {len(net.energy_timeseries)} "
                    f"energy points but TimingMeta.n_segments={self.timing.n_segments}"
                )
                raise ValueError(msg)
            if len(net.mean_z_timeseries) != self.timing.n_segments:
                msg = (
                    f"network '{net.id}' has {len(net.mean_z_timeseries)} "
                    f"mean_z points but TimingMeta.n_segments={self.timing.n_segments}"
                )
                raise ValueError(msg)
        return self

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to JSON with sensible defaults.

        Args:
            indent: pretty-print indent. Pass ``None`` for compact one-line.
        """
        return self.model_dump_json(indent=indent)

    def save(self, path: str | Path) -> Path:
        """Write the JSON serialization to ``path``.

        The path is resolved to an absolute location before any
        filesystem operation so the returned ``Path`` reflects what
        actually got written, and so callers passing relative paths
        don't get surprised by the current working directory.

        Callers who pass a user-supplied path should still validate it
        themselves — this method does NOT confine the output to a
        sandbox (it is a library, not a privilege boundary).
        """
        dest = Path(path).resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(self.to_json(), encoding="utf-8")
        return dest


# ---------------------------------------------------------------------
# Helpers shared by api.py
# ---------------------------------------------------------------------


def compute_result_id(result: ScoreResult) -> str:
    """Stable SHA-256 audit identity of a ``ScoreResult``.

    Defined as the SHA-256 of the result's *own* canonical JSON
    serialization with ``result_id`` blanked to the empty string:

    1. ``result.model_dump(mode="json")`` — the exact field set and value
       encoding the model serializes (so the hash can never drift away
       from ``ScoreResult``'s real fields; the previous implementation
       rebuilt a parallel dict by hand and silently disagreed with the
       serialized artifact, e.g. ``+00:00`` vs ``Z`` datetimes).
    2. ``result_id`` set to ``""``.
    3. ``json.dumps(..., sort_keys=True, separators=(",", ":"))`` — a
       canonical, key-order-independent, whitespace-free encoding.

    Recomputing this over any serialized ``ScoreResult`` (after blanking
    ``result_id``) reproduces the id, so downstream consumers can verify
    the audit hash from the JSON alone. No ``default=`` fallback is used:
    every value is already JSON-native after ``model_dump(mode="json")``,
    so a non-serializable value is a real bug that should raise loudly
    rather than be silently stringified.
    """
    cleared = result.model_copy(update={"result_id": ""})
    payload = cleared.model_dump(mode="json")
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_provenance(
    *,
    model_id: str,
    model_revision: str,
    runner: str,
    tribev2_package_version: str | None = None,
    torch_version: str | None = None,
    cuda_available: bool | None = None,
    device: str | None = None,
) -> ProvenanceMeta:
    """Construct a ``ProvenanceMeta`` with environment fields auto-filled."""
    return ProvenanceMeta(
        cortex_score_version=_CORTEX_SCORE_VERSION,
        model_id=model_id,
        model_revision=model_revision,
        tribev2_package_version=tribev2_package_version,
        runner=runner,
        python_version=platform.python_version(),
        torch_version=torch_version,
        cuda_available=cuda_available,
        device=device,
    )


def default_tribev2_license_restrictions() -> tuple[LicenseRestriction, ...]:
    """Return the CC-BY-NC-4.0 restriction string for TRIBE-derived results."""
    return (
        LicenseRestriction(
            component="TRIBE v2",
            license="CC-BY-NC-4.0",
            note=(
                "Full inference path is restricted to non-commercial use. "
                "See https://huggingface.co/facebook/tribev2 for the license text."
            ),
        ),
    )


def utc_now() -> _dt.datetime:
    """Return the current UTC time (helper so tests can monkey-patch)."""
    return _dt.datetime.now(_dt.UTC)


def _detect_torch_environment() -> tuple[str | None, bool | None, str | None]:
    """Return (torch_version, cuda_available, device) without forcing an import."""
    if "torch" not in sys.modules:
        return None, None, None
    try:
        import torch
    except ImportError:
        return None, None, None
    cuda = bool(torch.cuda.is_available())
    device = "cuda:0" if cuda else "cpu"
    return torch.__version__, cuda, device
