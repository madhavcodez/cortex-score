"""cortex-score quickstart.

Three lines to score any video for predicted cortical engagement across
the 5 brain networks (visual, language, faces, attention, motion).

Run:

    pip install "cortex-score[cli,gpu-deps]"
    pip install -r requirements/tribev2-gpu.txt   # TRIBE v2 itself
    huggingface-cli login                          # gated LLaMA 3.2-3B
    python examples/quickstart.py path/to/clip.mp4

For the CPU-only tier (no GPU, no TRIBE install required) see
``examples/cpu_only_from_predictions.py``.
"""

from __future__ import annotations

import sys

from cortex_score import score


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: python quickstart.py <video.mp4>\n")
        return 2

    result = score(sys.argv[1])

    print(f"result_id: {result.result_id[:16]}...")
    print(f"model:     {result.provenance.model_id}@{result.provenance.model_revision[:12]}")
    print(f"timing:    n_segments={result.timing.n_segments} TR={result.timing.tr_seconds}s")
    print()
    print("5 networks (mean_energy / peak_energy):")
    for net in result.networks:
        print(f"  {net.id:>9}  mean={net.mean_energy:.3f}  peak={net.peak_energy:.3f}")

    out = "score_result.json"
    result.save(out)
    print(f"\nfull JSON written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
