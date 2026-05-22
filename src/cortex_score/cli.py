"""Typer-based CLI.

Commands:

    cortex-score doctor                  -- environment + dependency check
    cortex-score score <video>           -- full pipeline, JSON to stdout/file
    cortex-score score <v1> <v2> ...     -- batch, requires --output-dir
    cortex-score from-predictions <npy>  -- CPU-only postprocessing tier
    cortex-score schema                  -- dump JSON Schema of ScoreResult
    cortex-score cache info              -- show cache size + location
    cortex-score cache clear             -- remove cached predictions/scores

Stdout is reserved for machine-readable output (JSON / schema). All
human-readable logs go to stderr so ``cortex-score score x.mp4 | jq``
remains clean.

Typer lives in the optional ``[cli]`` extra. If it's not installed,
the entry-point shim below prints an actionable install hint and
exits with code 2 instead of raising ImportError.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

try:
    import typer
    _HAS_TYPER = True
except ImportError:  # pragma: no cover - exercised only when [cli] missing
    _HAS_TYPER = False


def _no_typer_app() -> int:
    msg = (
        "cortex-score CLI requires the 'cli' optional dependency group.\n"
        "Install with:\n"
        "    pip install 'cortex-score[cli]'\n"
    )
    sys.stderr.write(msg)
    return 2


if not _HAS_TYPER:

    def app() -> int:  # type: ignore[no-redef]
        return _no_typer_app()

else:
    # ----- Real CLI ----------------------------------------------------

    import numpy as np

    from cortex_score import (
        __version__ as _CS_VERSION,
    )
    from cortex_score.api import (
        ScoreConfig,
        score as _score_fn,
        score_from_predictions as _score_from_predictions_fn,
    )
    from cortex_score.cache import CacheStore, default_cache_dir
    from cortex_score.exceptions import (
        CortexScoreError,
        MissingOptionalDependencyError,
    )
    from cortex_score.schemas import (
        FRAMING_DISCLAIMER,
        FRAMING_PRIMARY,
        FRAMING_SCIENTIFIC,
        SCHEMA_VERSION,
        ScoreResult,
    )

    app = typer.Typer(
        name="cortex-score",
        help=(
            f"{FRAMING_PRIMARY}\n\n"
            f"{FRAMING_SCIENTIFIC}\n\n"
            f"{FRAMING_DISCLAIMER}"
        ),
        no_args_is_help=True,
        add_completion=False,
    )
    cache_app = typer.Typer(name="cache", help="Inspect / clear the on-disk cache.")
    app.add_typer(cache_app, name="cache")

    def _log(msg: str) -> None:
        """Human log -> stderr (keep stdout JSON-clean)."""
        typer.echo(msg, err=True)

    def _emit_result(result: ScoreResult, output: Path | None, compact: bool) -> None:
        indent = None if compact else 2
        if output is None:
            typer.echo(result.to_json(indent=indent))
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(result.to_json(indent=indent), encoding="utf-8")
            _log(f"wrote {output}")

    # ----- doctor -----------------------------------------------------

    @app.command()
    def doctor() -> None:
        """Check environment readiness for the full ``score`` path."""
        import shutil

        report: list[tuple[str, str, str]] = []

        # Python
        report.append(("python", sys.version.split()[0], "ok"))

        # cortex-score version
        report.append(("cortex-score", _CS_VERSION, "ok"))

        # Optional ML stack
        try:
            import torch  # type: ignore[import-not-found]

            cuda = bool(torch.cuda.is_available())
            cuda_str = "available" if cuda else "absent"
            report.append(("torch", torch.__version__, f"cuda={cuda_str}"))
        except ImportError:
            report.append(("torch", "not installed", "install [gpu-deps]"))

        try:
            import tribev2  # type: ignore[import-not-found]

            rev = getattr(tribev2, "__version__", "unknown")
            report.append(("tribev2", str(rev), "ok"))
        except ImportError:
            report.append(
                (
                    "tribev2",
                    "not installed",
                    "pip install -r requirements/tribev2-gpu.txt",
                )
            )

        # External binaries
        for tool in ("ffmpeg", "ffprobe", "uvx"):
            path = shutil.which(tool)
            report.append((tool, path or "not found", "ok" if path else "install"))

        # HF auth
        try:
            from huggingface_hub import HfFolder  # type: ignore[import-not-found]

            token = HfFolder.get_token()
            report.append(
                ("hf-token", "present" if token else "absent", "huggingface-cli login")
            )
        except ImportError:
            report.append(("huggingface-hub", "not installed", "install [gpu-deps]"))

        # Cache dir
        cache = CacheStore()
        info = cache.info()
        report.append(
            (
                "cache",
                str(info["root"]),
                f"predictions={info['predictions_count']} scores={info['scores_count']}",
            )
        )

        # Render
        typer.echo(f"{'component':<18}{'value':<40}note")
        typer.echo("-" * 78)
        for name, value, note in report:
            typer.echo(f"{name:<18}{str(value)[:38]:<40}{note}")

    # ----- score ------------------------------------------------------

    @app.command(name="score")
    def score(
        videos: list[Path] = typer.Argument(..., metavar="VIDEO...", help="One or more video files."),
        output: Path | None = typer.Option(
            None, "--output", "-o", help="Output JSON path (single-video mode only)."
        ),
        output_dir: Path | None = typer.Option(
            None, "--output-dir", help="Directory for per-video JSON (required for batch)."
        ),
        compact: bool = typer.Option(False, "--compact", help="Serialize without indent."),
        no_cache: bool = typer.Option(False, "--no-cache", help="Skip cache for this run."),
    ) -> None:
        """Full pipeline: video(s) -> ScoreResult JSON."""
        _ = no_cache  # cache integration ships in v0.1.1
        if len(videos) > 1 and output_dir is None:
            _log("multiple inputs require --output-dir")
            raise typer.Exit(code=2)

        config = ScoreConfig()
        try:
            for v in videos:
                result = _score_fn(v, config=config)
                if len(videos) == 1:
                    _emit_result(result, output, compact)
                else:
                    if output_dir is None:
                        raise AssertionError("unreachable; checked above")
                    dest = output_dir / f"{v.stem}.score.json"
                    _emit_result(result, dest, compact)
        except MissingOptionalDependencyError as exc:
            _log(str(exc))
            raise typer.Exit(code=3) from exc
        except CortexScoreError as exc:
            _log(f"error: {exc}")
            raise typer.Exit(code=1) from exc

    # ----- from-predictions -------------------------------------------

    @app.command(name="from-predictions")
    def from_predictions(
        preds_path: Path = typer.Argument(..., help="Path to (T, V) .npy prediction tensor."),
        output: Path | None = typer.Option(None, "--output", "-o"),
        mesh: str = typer.Option("fsaverage5", "--mesh"),
        tr: float = typer.Option(1.0, "--tr", help="TR in seconds."),
        hrf_lag: float = typer.Option(5.0, "--hrf-lag", help="HRF lag in seconds."),
        model_id: str = typer.Option("facebook/tribev2", "--model-id"),
        model_revision: str = typer.Option("unknown", "--model-revision"),
        compact: bool = typer.Option(False, "--compact"),
    ) -> None:
        """CPU-only postprocessing: .npy -> ScoreResult JSON."""
        try:
            preds = np.load(preds_path, allow_pickle=False)
        except Exception as exc:
            _log(f"failed to load {preds_path}: {exc}")
            raise typer.Exit(code=1) from exc

        try:
            result = _score_from_predictions_fn(
                preds,
                mesh=mesh,
                tr_seconds=tr,
                hrf_lag_seconds=hrf_lag,
                model_id=model_id,
                model_revision=model_revision,
                source="npy",
            )
        except CortexScoreError as exc:
            _log(f"error: {exc}")
            raise typer.Exit(code=1) from exc

        _emit_result(result, output, compact)

    # ----- schema -----------------------------------------------------

    @app.command()
    def schema() -> None:
        """Dump the JSON Schema of ``ScoreResult`` to stdout."""
        sch: dict[str, Any] = ScoreResult.model_json_schema()
        sch.setdefault("$id", f"https://cortex-score.dev/schema/{SCHEMA_VERSION}")
        sch.setdefault("$schema", "https://json-schema.org/draft/2020-12/schema")
        typer.echo(json.dumps(sch, indent=2))

    # ----- cache ------------------------------------------------------

    @cache_app.command("info")
    def cache_info() -> None:
        store = CacheStore()
        info = store.info()
        typer.echo(json.dumps(info, indent=2))

    @cache_app.command("clear")
    def cache_clear(
        predictions: bool = typer.Option(True, "--predictions/--no-predictions"),
        scores: bool = typer.Option(True, "--scores/--no-scores"),
    ) -> None:
        store = CacheStore()
        store.clear(predictions=predictions, scores=scores)
        typer.echo(f"cleared cache at {default_cache_dir()}")


# Allow `python -m cortex_score.cli`
if __name__ == "__main__":  # pragma: no cover
    if _HAS_TYPER:
        app()
    else:
        sys.exit(_no_typer_app())
