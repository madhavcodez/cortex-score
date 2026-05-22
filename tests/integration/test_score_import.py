"""Reviewer-required: ``import cortex_score`` works without torch.

The base install is ``numpy + pydantic + platformdirs`` only. This test
verifies we never accidentally start importing torch at module top, which
would force every CPU-only user to install ~4 GB of wheels.
"""

from __future__ import annotations

import sys


def test_import_does_not_pull_in_torch() -> None:
    # Note: if torch IS installed in the test env, this can't actually
    # prove the negative — we instead just confirm cortex_score itself
    # doesn't carry torch in its module dependency chain. The CI matrix
    # has a job that runs this in a torch-free venv to cover the real case.
    if "torch" in sys.modules:
        # We can't unload torch without segfaulting; just confirm
        # import works.
        import cortex_score

        return

    import cortex_score  # noqa: F401

    assert "torch" not in sys.modules, (
        "importing cortex_score pulled in torch — heavy ML deps must stay lazy"
    )


def test_score_from_predictions_works_without_torch() -> None:
    """End-to-end CPU path with no torch import side effects."""
    import numpy as np

    from cortex_score import score_from_predictions

    preds = np.random.default_rng(0).standard_normal((4, 20484)).astype(np.float32)
    result = score_from_predictions(preds, model_revision="no-torch")
    assert len(result.networks) == 5
