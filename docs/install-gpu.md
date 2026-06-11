# GPU install (full `score()` pipeline)

The base `cortex-score` install is **CPU-only** — it covers
`score_from_predictions` / `score_from_prediction_bundle`, which take a
prediction tensor you already have. To run the full `score("clip.mp4")`
pipeline you also need TRIBE v2 and its GPU stack.

TRIBE v2 is **not** a declared dependency of `cortex-score`: PyPI rejects
published metadata that contains direct-URL (Git) dependencies, so it must
be installed separately. `cortex-score[gpu-deps]` declares the *compatible
environment* (torch / transformers / moviepy versions), and the pinned
TRIBE commit is installed from a requirements file.

## Requirements

- A CUDA-capable GPU (TRIBE v2 weights are ~12 GB; plan for ≥16 GB VRAM).
- `ffmpeg` and `uvx` (from [uv](https://github.com/astral-sh/uv)) on `PATH` —
  TRIBE's preprocessing shells out to them.
- A Hugging Face account with access to the gated Llama 3.2-3B weights.

## Install

```bash
# 1. Base package + the TRIBE-compatible GPU dependency matrix
pip install "cortex-score[gpu-deps]"

# 2. TRIBE v2 itself, pinned to the tested commit
pip install -r requirements/tribev2-gpu.txt

# 3. External tools (example: Debian/Ubuntu)
sudo apt-get install -y ffmpeg
curl -LsSf https://astral.sh/uv/install.sh | sh   # provides `uvx`

# 4. Authenticate for the gated model weights
huggingface-cli login
```

`requirements/tribev2-gpu.txt` pins TRIBE v2 to commit
`34f52344e5ba96660fac877393e1954e399d3ef3`, which matches
`cortex_score.runners.tribev2.TRIBEV2_PINNED_REVISION`. Bumping one
requires bumping the other and re-running the GPU smoke test.

## Verify

```bash
cortex-score doctor
```

`doctor` reports Python, `cortex-score`, torch (+ CUDA), tribev2, ffmpeg,
uvx, the Hugging Face token, and the cache directory. Every row should read
`ok` (or report what to install) before you run `score()`.

## Run

```python
from cortex_score import score

result = score("clip.mp4")
result.save("clip.score.json")
```

> TRIBE v2 is licensed **CC-BY-NC-4.0**. Scores produced through the full
> `score()` path inherit the non-commercial restriction; it is emitted as a
> runtime warning on first load and recorded in every
> `ScoreResult.license_restrictions`.
