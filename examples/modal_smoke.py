"""TRIBE end-to-end smoke test on Modal.

Reuses the Cortexia infrastructure (you already have the volumes,
secrets, and 12 GB TRIBE weights cached):

  Volume   clipcortex-models     /models      (HF cache, TRIBE + LLaMA weights)
  Volume   clipcortex-scratch    /scratch     (work dir)
  Secret   hf-token                           (LLaMA 3.2-3B gate access)
  Secret   r2-credentials                     (Cortexia R2 bucket access)
  GPU      A100-40GB              ~$2.26/hr   (~3 min per clip = ~$0.15)

What this script does:

  1. Build an image with the pre-built cortex-score wheel + tribev2
     from the pinned commit + the [gpu-deps] matrix.
  2. Fetch one short clean clip from Cortexia's R2 bucket.
  3. Run cortex_score.score() end-to-end (preprocess -> TRIBE forward ->
     vertex -> Yeo -> 5-network rollup -> JSON).
  4. Assert the result is well-formed AND validate the pre-v0.1 review
     fixes are live (PII-safe InputMeta, pinned model_revision in
     provenance, license restrictions present).
  5. Print the ScoreResult JSON, runtime, and estimated GPU spend.

Run::

    modal run examples/modal_smoke.py                   # default clip
    modal run examples/modal_smoke.py --clip-id <uuid>  # specific clip

Cost ceiling: $1 hard cap (timeout 10 min).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import modal

# Local repo layout: examples/ is two levels above the wheel under dist/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_WHEEL_DIR = _REPO_ROOT / "dist"

# Default smoke clip: shortest clean clip in Cortexia's curated.csv
# (letterboxed=False, duration_s=5.3, short_form_vertical).
DEFAULT_CLIP_ID = "0043e171-ca7a-41f5-a699-3c2b5057fda4"
TRIBEV2_PINNED_SHA = "34f52344e5ba96660fac877393e1954e399d3ef3"

# Where in the container the wheel and outputs live.
_CONTAINER_WHEEL_DIR = "/opt/cortex-score-wheel"
_CONTAINER_MODELS = "/models"
_CONTAINER_SCRATCH = "/scratch"

# A100-40GB posted price (Modal dashboard is the authoritative cost).
_A100_USD_PER_S = 0.000628


def _latest_local_wheel() -> Path:
    """Find the newest cortex_score-*.whl in dist/.

    Only meaningful on the client side. Inside the Modal container the
    wheel is already pip-installed into site-packages, so the path is
    irrelevant there.
    """
    wheels = sorted(_WHEEL_DIR.glob("cortex_score-*.whl"))
    if not wheels:
        msg = (
            f"No wheel found in {_WHEEL_DIR}. Build first:\n"
            "    python -m build --wheel --outdir dist"
        )
        raise FileNotFoundError(msg)
    return wheels[-1]


def _build_image_with_local_wheel() -> modal.Image:
    """Build the GPU image. CLIENT-SIDE ONLY (gated by modal.is_local()).

    Mirrors Cortexia's worker image so cached HuggingFace weights on
    the ``clipcortex-models`` volume can be reused without re-downloading
    12 GB of TRIBE+LLaMA+V-JEPA2+W2V-BERT.
    """
    wheel = _latest_local_wheel()
    remote_wheel = f"{_CONTAINER_WHEEL_DIR}/{wheel.name}"
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install(
            "ffmpeg",
            "git",
            "build-essential",
            "ca-certificates",
            "libgl1",
            "libglib2.0-0",
        )
        .pip_install(
            "pip>=24.0",
            "wheel",
            # tribev2.eventstransforms shells out to `uvx whisperx`.
            "uv>=0.5",
        )
        .pip_install(
            "torch>=2.5.1,<2.7",
            "torchvision>=0.20,<0.22",
            "torchaudio>=2.5.1,<2.7",
            extra_index_url="https://download.pytorch.org/whl/cu121",
        )
        .pip_install(
            "numpy==2.2.6",
            "pydantic>=2.7,<3",
            "platformdirs>=4,<5",
            "transformers>=4.45",
            "huggingface-hub>=0.24",
            "moviepy>=2.2.1",
            "whisperx>=3.1.1",
            "boto3>=1.35",
            "tqdm>=4.66",
        )
        .pip_install(
            # TRIBE v2 from the same commit cortex-score is built against.
            f"tribev2 @ git+https://github.com/facebookresearch/tribev2.git@{TRIBEV2_PINNED_SHA}",
        )
        .add_local_file(str(wheel), remote_wheel, copy=True)
        .run_commands(f"pip install --no-deps {remote_wheel}")
        .env(
            {
                "HF_HOME": f"{_CONTAINER_MODELS}/hf",
                "TRANSFORMERS_CACHE": f"{_CONTAINER_MODELS}/hf/transformers",
                "TORCH_HOME": f"{_CONTAINER_MODELS}/torch",
                "CORTEX_SCORE_CACHE_DIR": f"{_CONTAINER_SCRATCH}/cortex-score-cache",
                "PYTHONUNBUFFERED": "1",
            }
        )
    )


# Module-level image binding. Inside the Modal container, Modal has
# already cached and attached the real image; re-evaluating this module
# uses a stub. `modal.is_local()` returns False inside containers.
if modal.is_local():
    _image = _build_image_with_local_wheel()
else:
    _image = modal.Image.from_registry(
        "nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )

_models_volume = modal.Volume.from_name("clipcortex-models", create_if_missing=False)
_scratch_volume = modal.Volume.from_name("clipcortex-scratch", create_if_missing=False)

app = modal.App(name="cortex-score-smoke")

_SECRETS = [
    modal.Secret.from_name("hf-token"),
    modal.Secret.from_name("r2-credentials"),
]


def _fetch_clip_from_r2(clip_id: str, dest: Path) -> str:
    """Pull a curated clip from Cortexia's R2 bucket into ``dest``.

    Tries the two key conventions ``upload_curated_clips.py`` writes:
    ``clips/{clip_id}/source/{clip_id}.mp4`` and the normalized
    ``clips/{clip_id}/{hash}/clip.mp4`` fallback. Returns the R2 key
    actually used.
    """
    import os

    import boto3
    from botocore.client import Config

    # Cortexia's r2-credentials Modal secret exposes:
    #   R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_BUCKET, R2_ENDPOINT
    # (R2_ENDPOINT is the full https://...r2.cloudflarestorage.com URL).
    access_key = os.environ["R2_ACCESS_KEY_ID"]
    secret_key = os.environ["R2_SECRET_ACCESS_KEY"]
    bucket = os.environ["R2_BUCKET"]
    endpoint = os.environ["R2_ENDPOINT"]

    s3 = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )

    candidates = [
        f"clips/{clip_id}/source/{clip_id}.mp4",
    ]
    # Also try listing under clips/{clip_id}/source/ to find any extension.
    try:
        listing = s3.list_objects_v2(Bucket=bucket, Prefix=f"clips/{clip_id}/source/", MaxKeys=10)
        for obj in listing.get("Contents", []):
            k = obj["Key"]
            if k not in candidates and (k.endswith(".mp4") or k.endswith(".mov")):
                candidates.append(k)
    except Exception as exc:
        print(f"[r2] list_objects_v2 failed (continuing with default keys): {exc}", flush=True)

    last_err: Exception | None = None
    for key in candidates:
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            return key
        except Exception as exc:
            last_err = exc
            continue
    raise RuntimeError(f"clip {clip_id} not found in R2 under any of {candidates}: {last_err}")


@app.function(
    image=_image,
    gpu="A100-40GB",
    timeout=600,  # 10 minutes hard cap
    secrets=_SECRETS,
    volumes={
        _CONTAINER_MODELS: _models_volume,
        _CONTAINER_SCRATCH: _scratch_volume,
    },
)
def smoke(clip_id: str = DEFAULT_CLIP_ID) -> dict[str, object]:
    """Run cortex_score.score() on one Cortexia R2 clip."""
    import json

    # --- 1. fetch clip
    clip_path = Path(_CONTAINER_SCRATCH) / "cortex-smoke" / f"{clip_id}.mp4"
    print(f"[smoke] fetching clip {clip_id} from R2 ...", flush=True)
    r2_key = _fetch_clip_from_r2(clip_id, clip_path)
    size_mb = clip_path.stat().st_size / 1e6
    print(f"[smoke] downloaded r2://.../{r2_key} ({size_mb:.2f} MB) -> {clip_path}", flush=True)

    # --- 2. import cortex-score and confirm version
    import cortex_score
    from cortex_score import score
    from cortex_score.runners.tribev2 import TRIBEV2_PINNED_REVISION

    print(f"[smoke] cortex_score version: {cortex_score.__version__}", flush=True)
    print(f"[smoke] tribev2 pinned revision: {TRIBEV2_PINNED_REVISION}", flush=True)
    assert TRIBEV2_PINNED_REVISION == TRIBEV2_PINNED_SHA, (
        f"runner pin ({TRIBEV2_PINNED_REVISION}) does not match script pin ({TRIBEV2_PINNED_SHA})"
    )

    # --- 3. confirm GPU availability
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available in Modal container - check image build")
    print(
        f"[smoke] cuda OK: {torch.cuda.get_device_name(0)}, "
        f"vram total={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()

    # --- 4. run cortex_score.score() end-to-end
    print("[smoke] calling cortex_score.score() ...", flush=True)
    t0 = time.perf_counter()
    result = score(clip_path)
    runtime_s = time.perf_counter() - t0
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"[smoke] score() returned in {runtime_s:.1f}s, peak VRAM={peak_vram_gb:.2f} GB",
        flush=True,
    )

    # --- 5. assertions (the things only a real run can prove)
    assert len(result.networks) == 5, f"expected 5 networks, got {len(result.networks)}"
    ids = [n.id for n in result.networks]
    assert ids == ["visual", "language", "faces", "attention", "motion"], (
        f"network order off: {ids}"
    )
    for n in result.networks:
        assert n.mean_energy > 0, f"network '{n.id}' has mean_energy={n.mean_energy}"
        assert n.peak_energy >= n.mean_energy, (
            f"network '{n.id}' peak < mean ({n.peak_energy} < {n.mean_energy})"
        )

    assert result.provenance.model_revision == TRIBEV2_PINNED_SHA, (
        f"provenance.model_revision={result.provenance.model_revision} "
        f"!= TRIBEV2_PINNED_SHA={TRIBEV2_PINNED_SHA}"
    )
    assert result.input.filename == clip_path.name, (
        f"input.filename={result.input.filename} != {clip_path.name}"
    )
    assert result.input.absolute_path is None, (
        f"PII regression: absolute_path leaked = {result.input.absolute_path}"
    )
    tribe_license = next(
        (r for r in result.license_restrictions if r.component == "TRIBE v2"),
        None,
    )
    assert tribe_license is not None, "TRIBE CC-BY-NC license restriction missing"
    assert "CC-BY-NC" in tribe_license.license, (
        f"unexpected TRIBE license string: {tribe_license.license}"
    )

    # --- 6. emit
    payload = json.loads(result.to_json(indent=None))
    print("\n[smoke] ScoreResult summary:", flush=True)
    print(f"  result_id:        {payload['result_id'][:16]}...", flush=True)
    print(f"  schema_version:   {payload['schema_version']}", flush=True)
    print(f"  model:            {payload['provenance']['model_id']}", flush=True)
    print(f"  revision:         {payload['provenance']['model_revision'][:12]}", flush=True)
    print(f"  input.filename:   {payload['input']['filename']}", flush=True)
    print(f"  input.abs_path:   {payload['input']['absolute_path']}", flush=True)
    print(f"  input.sha256[:12]:{payload['input']['content_sha256'][:12]}", flush=True)
    tr = payload["timing"]["tr_seconds"]
    n_seg = payload["timing"]["n_segments"]
    print(f"  timing:           TR={tr}s n_seg={n_seg}", flush=True)
    print(f"  normalization:    scope={payload['normalization']['scope']}", flush=True)
    print(f"  atlas:            {payload['atlas']['atlas_version']}", flush=True)
    print(f"  license:          {payload['license_restrictions'][0]['license']}", flush=True)
    print("  networks (mean / peak energy):", flush=True)
    for n in payload["networks"]:
        nid = n["id"]
        me = n["mean_energy"]
        pe = n["peak_energy"]
        print(f"    {nid:>9}  mean={me:.3f}  peak={pe:.3f}", flush=True)

    return {
        "clip_id": clip_id,
        "r2_key": r2_key,
        "runtime_s": runtime_s,
        "estimated_gpu_usd": round(runtime_s * _A100_USD_PER_S, 4),
        "peak_vram_gb": peak_vram_gb,
        "cortex_score_version": cortex_score.__version__,
        "tribev2_pinned_sha": TRIBEV2_PINNED_SHA,
        "result_id": result.result_id,
        "score_result": payload,
    }


@app.local_entrypoint()
def main(clip_id: str = DEFAULT_CLIP_ID) -> None:
    """Local launcher. Reports cost + runtime, writes result JSON locally."""
    print(f"[main] dispatching clip {clip_id} to Modal A100 ...", flush=True)
    wall_t0 = time.perf_counter()
    result = smoke.remote(clip_id=clip_id)
    wall_dt = time.perf_counter() - wall_t0

    print()
    print("=" * 64)
    print("cortex-score TRIBE smoke result")
    print("=" * 64)
    print(f"  clip_id:           {result['clip_id']}")
    print(f"  r2_key:            {result['r2_key']}")
    print(f"  cortex_score:      {result['cortex_score_version']}")
    print(f"  tribev2 sha:       {str(result['tribev2_pinned_sha'])[:12]}")
    print(f"  GPU runtime:       {result['runtime_s']:.1f}s")
    print(f"  wall clock:        {wall_dt:.1f}s  (incl. image build + cold start)")
    print(f"  estimated GPU $:   ${result['estimated_gpu_usd']}")
    print(f"  peak VRAM:         {result['peak_vram_gb']:.2f} GB")
    print(f"  result_id:         {str(result['result_id'])[:16]}...")
    print("=" * 64)
    print("\n(Modal dashboard is authoritative for billing.)")

    # Save the full ScoreResult locally for inspection.
    import json

    out_path = Path("dist") / f"smoke_{result['clip_id'][:8]}.score.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result["score_result"], indent=2), encoding="utf-8")
    print(f"\nfull ScoreResult written to {out_path}")


if __name__ == "__main__":  # pragma: no cover
    sys.stderr.write("Run via: modal run examples/modal_smoke.py\n")
    sys.exit(2)
