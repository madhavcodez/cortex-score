"""Unit tests for the 5-network rollup."""

from __future__ import annotations

import numpy as np
import pytest

from cortex_score.processing.networks import (
    NETWORK_GROUPS,
    NETWORK_IDS,
    YEO_LABELS,
    build_network_summary,
    yeo_to_network_indices,
)


def test_exactly_five_groups() -> None:
    assert len(NETWORK_GROUPS) == 5
    assert NETWORK_IDS == ("visual", "language", "faces", "attention", "motion")


def test_yeo_labels_are_canonical_17() -> None:
    assert len(YEO_LABELS) == 17
    assert YEO_LABELS[0] == "VisCent"
    assert YEO_LABELS[-1] == "TempPar"


def test_no_overlap_between_groups() -> None:
    """Each Yeo index appears in at most one of the 5 dashboard groups."""
    seen: set[int] = set()
    for group in NETWORK_GROUPS:
        for idx in group.yeo_indices:
            assert idx not in seen, f"Yeo index {idx} is in multiple groups"
            seen.add(idx)


def test_yeo_to_network_indices_reverse_map() -> None:
    rev = yeo_to_network_indices()
    # Yeo index 0 = VisCent should map to "visual"
    assert rev[0] == "visual"
    # Yeo index 2 = SomMotA should map to "motion"
    assert rev[2] == "motion"


def test_build_network_summary_shape() -> None:
    rng = np.random.default_rng(11)
    z = rng.standard_normal((6, 17)).astype(np.float32)
    summaries = build_network_summary(z)
    assert len(summaries) == 5
    for s in summaries:
        assert len(s["energy_timeseries"]) == 6
        assert len(s["mean_z_timeseries"]) == 6
        assert s["mean_energy"] >= 0
        assert s["peak_energy"] >= 0


def test_build_network_summary_constant_input() -> None:
    z = np.ones((5, 17), dtype=np.float32)
    summaries = build_network_summary(z)
    # Each group's energy ts (|z|) is all 1.0
    for s in summaries:
        np.testing.assert_array_equal(s["energy_timeseries"], np.ones(5).tolist())
        np.testing.assert_array_equal(s["mean_z_timeseries"], np.ones(5).tolist())


def test_build_network_summary_shape_validation() -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        build_network_summary(np.zeros(17, dtype=np.float32))


def test_build_network_summary_out_of_range_yeo_index_raises() -> None:
    """A too-narrow array (fewer columns than the groups reference) must
    raise AtlasMismatchError instead of silently zero-filling networks."""
    from cortex_score.exceptions import AtlasMismatchError

    z = np.zeros((4, 5), dtype=np.float32)  # groups reference indices up to 16
    with pytest.raises(AtlasMismatchError, match="mismatch"):
        build_network_summary(z)
