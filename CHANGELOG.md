# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Initial package scaffold (Epics 0-8):
  - Bundled atlas data (Schaefer-400 + Yeo-17 + 5-network rollup) with
    SHA-256 manifest. Total payload ~340 KB.
  - Pure-NumPy processing core: aggregate, normalize, metrics, networks,
    validate.
  - Pydantic v2 schema layer with full provenance: `PredictionBundle`,
    `ScoreResult`, `NetworkScore`, `AtlasMeta`, `ProvenanceMeta`,
    `NormalizationMeta`, `LicenseRestriction`, `ScoreWarning`.
  - Three-tier public API:
    - `score_from_prediction_bundle(bundle)` — type-safe.
    - `score_from_predictions(preds, mesh=..., tr_seconds=..., ...)` —
      ergonomic, requires explicit scientific assumptions.
    - `score(video_path, runner=None)` — full pipeline.
  - `CortexScorer` class for batch reuse.
  - Two-tier cache: prediction cache + score cache, atomic writes,
    cache_manifest.json, invalidation matrix.
  - Typer CLI under `[cli]` extra: `doctor`, `score`, `from-predictions`,
    `schema`, `cache info`, `cache clear`.
  - TRIBE v2 adapter under `[gpu-deps]` extra (TRIBE itself installed
    from `requirements/tribev2-gpu.txt`, pinned to commit
    `34f52344e5ba96660fac877393e1954e399d3ef3`).
  - 118-test suite at 88.66% coverage with property tests, schema
    snapshot, cache invalidation matrix, packaging smoke, and
    import-without-GPU gate.

### Notes
- The package source is MIT-licensed. The bundled atlases ship under
  their original licenses (Schaefer MIT, Yeo BSD-like). The full
  `score()` path uses TRIBE v2 which is **CC-BY-NC-4.0**: outputs from
  that path inherit the non-commercial restriction. See
  `LICENSE-THIRD-PARTY.md`.
