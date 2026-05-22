"""Public exception hierarchy.

Why a dedicated hierarchy: downstream code (especially CLI wrappers and
AI-video pipelines) wants to catch `cortex-score`-specific failures
without swallowing unrelated errors. A single `CortexScoreError` root
gives that ``except CortexScoreError`` ergonomic.

Each concrete error carries enough context that the message alone is a
useful action item — no log archaeology.
"""

from __future__ import annotations


class CortexScoreError(Exception):
    """Root of all cortex-score exceptions."""


class MissingOptionalDependencyError(CortexScoreError, ImportError):
    """Raised when an optional runtime dep (torch, tribev2, whisperx) is absent.

    Inherits from ImportError so existing ``except ImportError`` blocks
    in downstream code still catch it without modification.
    """

    def __init__(self, package: str, install_hint: str) -> None:
        msg = (
            f"Optional dependency '{package}' is not installed.\n"
            f"Install with:\n"
            f"    {install_hint}\n"
        )
        super().__init__(msg)
        self.package = package
        self.install_hint = install_hint


class MissingExternalToolError(CortexScoreError):
    """Raised when a required external binary (ffmpeg, uv) is not on PATH."""

    def __init__(self, tool: str, install_hint: str) -> None:
        msg = (
            f"External tool '{tool}' was not found on PATH.\n"
            f"Install: {install_hint}\n"
        )
        super().__init__(msg)
        self.tool = tool
        self.install_hint = install_hint


class IncompatiblePredictionShapeError(CortexScoreError, ValueError):
    """Raised when a prediction tensor's shape disagrees with the atlas mesh."""

    def __init__(
        self,
        *,
        expected_n_vertices: int,
        actual_n_vertices: int,
        mesh: str,
    ) -> None:
        msg = (
            f"Prediction tensor has {actual_n_vertices} vertices but mesh "
            f"'{mesh}' expects {expected_n_vertices}. Either re-run the "
            f"encoder with the correct mesh, or pass the matching "
            f"`mesh=` argument."
        )
        super().__init__(msg)
        self.expected_n_vertices = expected_n_vertices
        self.actual_n_vertices = actual_n_vertices
        self.mesh = mesh


class AtlasMismatchError(CortexScoreError, ValueError):
    """Raised when atlas vertex/parcel assignments are internally inconsistent.

    Surfaces only when bundled data has been tampered with — the SHA-256
    fingerprints in ``data/manifest.json`` are the first line of defense.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(f"Atlas data integrity check failed: {detail}")
        self.detail = detail


class ModelLicenseError(CortexScoreError):
    """Raised before any TRIBE inference call to record license restrictions.

    TRIBE v2 is CC-BY-NC-4.0. This is not raised in normal use; it is the
    exception type a CLI ``--strict-license`` mode raises when a caller
    has opted into hard-failing on non-commercial restrictions. The
    license string is also embedded in every ``ScoreResult``.
    """

    def __init__(self, model_id: str, license_name: str, note: str) -> None:
        msg = f"Model '{model_id}' is licensed under {license_name}. {note}"
        super().__init__(msg)
        self.model_id = model_id
        self.license_name = license_name


class PreprocessingWarning(UserWarning):
    """Emitted (not raised) when preprocessing mutates a clip in a way that
    could affect interpretation — letterboxing, aspect-ratio resampling,
    or significant frame-rate downsampling.

    Surfaces in ``ScoreResult.warnings`` as well.
    """
