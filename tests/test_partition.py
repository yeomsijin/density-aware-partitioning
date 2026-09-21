import numpy as np
import pytest

from dap import (
    counting_map,
    counting_map_range,
    density_aware_direction,
    density_aware_split,
    density_measure,
    epsilon,
)
from dap.partition import random_directions


def reference(y, p):
    y = np.asarray(y)
    e = epsilon(y)
    return sum(y == p) if e == 0 else sum((p - e <= y) & (y < p + e))


def test_half_open_and_degenerate_cases():
    y = np.array([0.0, 1.0, 2.0])
    assert counting_map(y, 0.5) == 1
    assert counting_map(y, 1.5) == 1
    assert counting_map([2, 2, 2], [1, 2, 3]).tolist() == [0, 3, 0]
    assert counting_map_range([2, 2]) == (2, 2)
    assert counting_map_range([2, 2], domain="real") == (0, 2)
    assert density_measure([0, 1, 2, 3]) == 0.25
    assert density_measure([2, 2, 2]) == 1


def test_range_includes_gaps_and_off_sample_maxima():
    y = np.array([0, 1, 2, 10, 11, 12.0])
    assert counting_map_range(y) == (0, 3)
    # Counts only at observed y miss a maximum between two nearby observations.
    y = np.array([0, 3, 6, 12.0])
    assert counting_map_range(y) == (0, 2)
    assert max(counting_map(y, y)) == 1


@pytest.mark.parametrize("seed", range(10))
def test_event_sweep_matches_independent_brute_force(seed):
    rng = np.random.default_rng(seed)
    # Integer samples / epsilon with binary-exact representation avoid ambiguous
    # double-rounding at mathematical boundaries in the independent formula.
    y = np.r_[0.0, rng.integers(0, 17, 7).astype(float), 16.0]
    e = epsilon(y)
    boundaries = np.unique(np.r_[y - e, y + e, y.min(), y.max()])
    probes = np.r_[boundaries, (boundaries[:-1] + boundaries[1:]) / 2]
    probes = probes[(probes >= y.min()) & (probes <= y.max())]
    expected = np.array([reference(y, p) for p in probes])
    np.testing.assert_array_equal(counting_map(y, probes), expected)
    assert counting_map_range(y) == (expected.min(), expected.max())


@pytest.mark.parametrize("max_trials", [0, 256])
def test_split_distribution_and_rejection_condition(max_trials):
    y = np.array([0, 1, 2, 10, 11, 12.0])
    rng = np.random.default_rng(3)
    draws = np.array(
        [density_aware_split(y, random_state=rng, max_trials=max_trials) for _ in range(1500)]
    )
    assert np.all(counting_map(y, draws) <= 1)
    assert draws.min() > 2.2 and draws.max() <= 9.8
    assert abs(draws.mean() - 6) < 0.2
    # Equal-length accepted subintervals have equal probability.
    assert abs(np.mean(draws < 6) - 0.5) < 0.05


def test_impossible_tolerance_and_constant_sentinel():
    with pytest.raises(ValueError, match="No positive-length"):
        density_aware_split([0, 1, 2], alpha=0.5, max_trials=0)
    assert density_aware_split([3, 3, 3]) == 3


def test_dad_selects_true_minimum_over_same_candidates():
    X = np.random.default_rng(12).normal(size=(30, 4))
    V = random_directions(np.random.default_rng(7), 4, 12, max_features=2)
    ranges = [np.ptp(counting_map_range(X @ v)) for v in V]
    expected = V[np.argmin(ranges)]
    got = density_aware_direction(X, n_candidates=12, max_features=2, random_state=7)
    np.testing.assert_array_equal(got, expected)
    assert np.count_nonzero(got) == 2


@pytest.mark.parametrize("y", [[], [np.nan], [np.inf], [[1, 2]]])
def test_invalid_projections(y):
    with pytest.raises(ValueError):
        density_aware_split(y)


@pytest.mark.parametrize("a,b", [(2, 0), (-2, 4), (0.5, -8)])
def test_affine_invariance(a, b):
    y = np.array([0, 0, 3, 4, 8.0])
    assert density_measure(y) == density_measure(a * y + b)
