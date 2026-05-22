"""Drop-in demo: filter a clip library by predicted attention-network engagement.

Real AI-video pipelines often want to rank clips by some property
before further processing (re-cutting, captioning, etc.). cortex-score
gives you a structured number to sort on without writing any
brain-encoding code yourself.

This script:

1. Walks a directory of .mp4 files.
2. Scores each via TRIBE v2.
3. Sorts by predicted attention-network mean energy.
4. Writes ``attention_ranking.json``.

Usage:

    pip install "cortex-score[cli,gpu-deps]"
    pip install -r requirements/tribev2-gpu.txt
    python examples/ai_video_pipeline/filter_by_attention.py clips/

NOTE: TRIBE v2 outputs are CC-BY-NC-4.0. The headline framing
("cortical engagement") is a product label over predicted cortical
responses for an average subject; it is not a measured viewer
engagement number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from cortex_score import CortexScorer


def rank_clips_by_attention(clips_dir: Path) -> list[dict[str, object]]:
    scorer = CortexScorer()  # loads TRIBE v2 once, scores many
    rows: list[dict[str, object]] = []
    for clip in sorted(clips_dir.glob("*.mp4")):
        result = scorer.score(clip)
        attention = next(n for n in result.networks if n.id == "attention")
        rows.append(
            {
                "clip": clip.name,
                "result_id": result.result_id,
                "attention_mean_energy": attention.mean_energy,
                "attention_peak_energy": attention.peak_energy,
                "visual_mean_energy": next(n.mean_energy for n in result.networks if n.id == "visual"),
            }
        )
    rows.sort(key=lambda r: r["attention_mean_energy"], reverse=True)
    return rows


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: python filter_by_attention.py <clips_dir>\n")
        return 2

    clips_dir = Path(sys.argv[1])
    if not clips_dir.is_dir():
        sys.stderr.write(f"not a directory: {clips_dir}\n")
        return 2

    ranking = rank_clips_by_attention(clips_dir)
    out_path = Path("attention_ranking.json")
    out_path.write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    print(f"ranked {len(ranking)} clips -> {out_path}")
    for row in ranking[:5]:
        print(f"  {row['clip']:<32}  attn_mean={row['attention_mean_energy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
