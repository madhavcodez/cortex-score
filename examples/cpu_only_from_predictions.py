"""CPU-only postprocessing tier.

If you already have TRIBE v2 predictions in a ``.npy`` file (computed
elsewhere — perhaps on a Modal sidecar or a teammate's GPU), the
``score_from_predictions`` path runs on a laptop in milliseconds:

    pip install cortex-score
    python examples/cpu_only_from_predictions.py preds_vertex.npy

The expected input shape is ``(n_segments, 20484)`` on fsaverage5.
"""

from __future__ import annotations

import sys

import numpy as np

from cortex_score import score_from_predictions


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: python cpu_only_from_predictions.py <preds.npy>\n")
        return 2

    preds = np.load(sys.argv[1], allow_pickle=False)
    result = score_from_predictions(
        preds,
        mesh="fsaverage5",
        tr_seconds=1.0,
        hrf_lag_seconds=5.0,
        model_id="facebook/tribev2",
        model_revision="34f52344",   # bump when you re-run TRIBE
        source="npy",
    )

    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
