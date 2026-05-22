"""CLI smoke tests via typer's test runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from cortex_score.cli import app

runner = CliRunner()


def test_help_shows_framing() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Score any video for predicted cortical engagement" in result.output
    assert "does not measure real viewer engagement" in result.output


def test_schema_emits_valid_json() -> None:
    result = runner.invoke(app, ["schema"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "$defs" in parsed or "properties" in parsed


def test_cache_info_returns_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CORTEX_SCORE_CACHE_DIR", str(tmp_path))
    result = runner.invoke(app, ["cache", "info"])
    assert result.exit_code == 0
    info = json.loads(result.output)
    assert info["root"].endswith(tmp_path.name)
    assert info["scores_count"] == 0


def test_cache_clear_succeeds(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CORTEX_SCORE_CACHE_DIR", str(tmp_path))
    result = runner.invoke(app, ["cache", "clear"])
    assert result.exit_code == 0


def test_doctor_runs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """doctor should never raise; it reports environment state."""
    monkeypatch.setenv("CORTEX_SCORE_CACHE_DIR", str(tmp_path))
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "python" in result.output
    assert "cortex-score" in result.output


def test_from_predictions_writes_json(tmp_path: Path) -> None:
    preds_path = tmp_path / "p.npy"
    out_path = tmp_path / "result.json"
    np.save(preds_path, np.random.default_rng(0).standard_normal((4, 20484)).astype(np.float32))

    result = runner.invoke(
        app,
        [
            "from-predictions",
            str(preds_path),
            "-o",
            str(out_path),
            "--model-revision",
            "cli-test",
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.exists()
    parsed = json.loads(out_path.read_text(encoding="utf-8"))
    assert parsed["provenance"]["model_revision"] == "cli-test"
    assert len(parsed["networks"]) == 5


def test_from_predictions_stdout_when_no_output(tmp_path: Path) -> None:
    preds_path = tmp_path / "p.npy"
    np.save(preds_path, np.random.default_rng(1).standard_normal((3, 20484)).astype(np.float32))

    result = runner.invoke(app, ["from-predictions", str(preds_path), "--compact"])
    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert "schema_version" in parsed


def test_score_requires_output_dir_for_multiple_inputs(tmp_path: Path) -> None:
    v1 = tmp_path / "a.mp4"
    v2 = tmp_path / "b.mp4"
    v1.touch()
    v2.touch()
    result = runner.invoke(app, ["score", str(v1), str(v2)])
    assert result.exit_code == 2
    assert "--output-dir" in result.output or "output-dir" in (result.stderr or "")
