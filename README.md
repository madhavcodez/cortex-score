# cortex-score

**Score any video for predicted cortical engagement across 5 brain networks** — visual, language, faces, attention, motion. Built on Meta FAIR's TRIBE v2 brain-encoding model.

> **What it actually does:** `cortex-score` summarizes TRIBE v2 predicted cortical responses for any video across five Cortexia-defined network groups.
>
> **What it does *not* do:** `cortex-score` does **not** measure real viewer engagement. It summarizes predicted fMRI-like responses from a pretrained brain-encoding model for an average subject.

## Install

```bash
pip install cortex-score                                  # CPU-only postprocessing tier
pip install "cortex-score[cli]"                           # + typer CLI
pip install "cortex-score[gpu-deps]"                      # + TRIBE inference dependency matrix
pip install -r requirements/tribev2-gpu.txt               # + TRIBE v2 itself (kept out of PyPI metadata)
```

## 30-second example

```python
from cortex_score import score

result = score("my_clip.mp4")
print(result.networks[0].label, result.networks[0].mean_energy)
# "Visual cortex" 1.23
result.save("my_clip.score.json")
```

CPU-only tier (no GPU required):

```python
import numpy as np
from cortex_score import score_from_predictions

preds = np.load("preds_vertex.npy")   # (T, 20484) from TRIBE v2
result = score_from_predictions(
    preds,
    mesh="fsaverage5",
    tr_seconds=1.0,
    hrf_lag_seconds=5.0,
    model_id="facebook/tribev2",
    model_revision="unknown",
)
print(result.to_json())
```

## How it works

```
video.mp4
   │
   │  preprocess (ffmpeg normalize + WhisperX events)
   ▼
TRIBE v2 inference
   │  → vertex predictions (T, ~20484) on fsaverage5
   ▼
PredictionBundle  ──┐
                    │
   aggregate ─────→ Schaefer-400 ROIs (T, 400)
   aggregate ─────→ Yeo-17 networks (T, 17)
   z-score per atlas
                    │
   5-network rollup → {visual, language, faces, attention, motion}
                    │
   metrics: mean_energy, peak_energy, temporal_volatility
                    ▼
              ScoreResult (versioned JSON with full provenance)
```

## License

`cortex-score` source code is MIT. The bundled atlas data and the downstream TRIBE v2 model carry their own licenses — TRIBE v2 is **CC-BY-NC-4.0** (non-commercial). See [`LICENSE-THIRD-PARTY.md`](LICENSE-THIRD-PARTY.md) before using in a commercial product.

## Status

Pre-release. v0.1.0 ships the CPU-only postprocessing tier first; the TRIBE inference adapter lands shortly after.
