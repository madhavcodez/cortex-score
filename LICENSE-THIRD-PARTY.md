# Third-party notices

`cortex-score` bundles or invokes third-party assets under their own licenses.
The MIT license in `LICENSE` covers the `cortex-score` source code only.

## TRIBE v2 (Meta FAIR)

- Upstream: https://github.com/facebookresearch/tribev2
- Model card: https://huggingface.co/facebook/tribev2
- License: **CC-BY-NC-4.0 (Creative Commons Attribution-NonCommercial 4.0)**

The full `score()` path of this library loads and runs TRIBE v2. Any output
produced by that path therefore inherits the upstream license restrictions,
which prohibit commercial use without separate permission from Meta. Review
the Hugging Face model card and the CC-BY-NC-4.0 license before using
`cortex-score` in a commercial product.

TRIBE v2 also requires access to the gated `meta-llama/Llama-3.2-3B` text
encoder; users must accept the LLaMA license on Hugging Face and run
`huggingface-cli login` before the full inference path will work.

The CPU-only `score_from_predictions` / `score_from_prediction_bundle` paths
of this library do **not** load TRIBE v2 — those paths receive an
already-computed prediction tensor and only perform aggregation,
normalization, metrics, and JSON serialization. They still operate on
TRIBE-derived outputs in normal use, so the CC-BY-NC-4.0 attribution still
applies to the resulting scores, but the dependency surface stays light.

## Schaefer 2018 cortical parcellation

- Reference: Schaefer et al., *Cerebral Cortex* 28(9): 3095-3114 (2018).
  "Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic
  Functional Connectivity MRI."
- Source: https://github.com/ThomasYeoLab/CBIG/tree/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal
- License: MIT (CBIG repository).

The `schaefer400_vertex.npy` and `labels_schaefer400.json` files in
`cortex_score/data/` are derived from the published 400-parcel /
17-network projection on the fsaverage5 cortical mesh.

## Yeo 2011 functional networks

- Reference: Yeo et al., *Journal of Neurophysiology* 106(3): 1125-1165
  (2011). "The organization of the human cerebral cortex estimated by
  intrinsic functional connectivity."
- Source: https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation_Yeo2011
- License: FreeSurfer license (BSD-like).

The `yeo17_vertex.npy` and `labels_yeo17.json` files in `cortex_score/data/`
are derived from the 17-network solution projected to fsaverage5.

## 5-network rollup (`network_groups.json`)

- Source: Cortexia project (`packages/shared-schemas/src/clipcortex_schemas/network_groups.json`).
- License: MIT (same as cortex-score).
- Group source identifier: `cortexia-network-groups-v1`.

The mapping of 17 Yeo networks into 5 dashboard groups
(visual, language, faces, attention, motion) is a product-design choice of
the Cortexia project, not a canonical neuroscience decomposition. The file
is auditable: each group records its `yeo_indices` and the bundle ships a
SHA-256 fingerprint in `data/manifest.json`.

## Downstream attribution

If you publish or distribute results computed with `cortex-score`, please
cite:

- Cite TRIBE v2 (see https://huggingface.co/facebook/tribev2 for the
  authoritative citation BibTeX).
- Cite Schaefer 2018 and Yeo 2011 if you report ROI- or network-level
  values.
- Cite `cortex-score` itself via the entry in `CITATION.cff`.
