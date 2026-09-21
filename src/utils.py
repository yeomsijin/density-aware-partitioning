"""Compatibility names backed by the shared, corrected implementation."""

import numpy as np
from dap.partition import epsilon as eps_from_range
from dap.partition import density_aware_split
from dap.forest import average_path_length as harmonic_c_isoforest
from .counting_map import window_count_range
from .directions import random_unit_vectors


def rng_from(seed_or_rng=None):
    return np.random.default_rng(seed_or_rng)


def dap_pick_threshold(y, alpha=1.0, rng=None, max_trials=256):
    return density_aware_split(y, alpha=alpha, random_state=rng, max_trials=max_trials)


def count_in_window_sorted(y_sorted, p, eps):
    if eps == 0:
        return int(np.count_nonzero(y_sorted == p))
    return int(
        np.searchsorted(y_sorted, p + eps, side="left")
        - np.searchsorted(y_sorted, p - eps, side="left")
    )
