# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-06-10

First public release. The CPU-only postprocessing tier
(`score_from_predictions` / `score_from_prediction_bundle`) and the
`ScoreResult` JSON contract (`SCHEMA_VERSION = "1.0"`) are stable.

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
  - Two-tier cache infrastructure (prediction cache + score cache, atomic
    writes, cache_manifest.json, invalidation matrix). NOTE: this is
    plumbing for a future release — the scoring path does not read or write
    it yet, so `cache info` reports empty until caching is wired in v0.1.1.
  - Typer CLI under `[cli]` extra: `doctor`, `score`, `from-predictions`,
    `schema`, `cache info`, `cache clear`.

### Fixed (pre-release hardening)
- `result_id` is now the SHA-256 of the result's own canonical JSON (with
  `result_id` blanked), so it is reproducible from the serialized artifact.
  The previous hand-built hash payload disagreed with the serialized form
  (`+00:00` vs `Z` datetimes), making the documented audit hash unverifiable.
- MZ3 scalar-overlay export now emits the real NiiVue format (uint16 magic
  `0x5A4D`, `attr=8`/isSCALAR, 16-byte header, gzip). The earlier port wrote
  a header NiiVue rejected at the magic check.
- `score_from_predictions` rejects 1-D/non-finite inputs and unsupported
  meshes with clear errors at the boundary (`UnsupportedMeshError`) instead
  of an opaque `IndexError` / a raw traceback through the CLI.
- Version is read from installed distribution metadata
  (`importlib.metadata`); the build no longer ships a generated `_version.py`.

### Removed
- The no-op `score --no-cache` flag (it silently did nothing). It will
  return when the cache is wired into the scoring path.
  - TRIBE v2 adapter under `[gpu-deps]` extra (TRIBE itself installed
    from `requirements/tribev2-gpu.txt`, pinned to commit
    `34f52344e5ba96660fac877393e1954e399d3ef3`).
  - 135-test suite at ~90% coverage with property tests, schema
    snapshot, cache invalidation matrix, packaging smoke, MZ3 format
    round-trip, result_id verifiability, and import-without-GPU gate.

### Notes
- The package source is MIT-licensed. The bundled atlases ship under
  their original licenses (Schaefer MIT, Yeo BSD-like). The full
  `score()` path uses TRIBE v2 which is **CC-BY-NC-4.0**: outputs from
  that path inherit the non-commercial restriction. See
  `LICENSE-THIRD-PARTY.md`.
