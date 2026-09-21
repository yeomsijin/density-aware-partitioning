"""Compatibility helpers; canonical implementation is dap.partition."""

import numpy as np
from dap.partition import epsilon as eps_from_range, counting_map, counting_map_range


def count_in_window_sorted(y_sorted, p, eps, tol=0):
    if tol != 0:
        raise ValueError("tolerance widening is not part of the paper counting map")
    if eps == 0:
        return int(np.count_nonzero(y_sorted == p))
    return int(
        np.searchsorted(y_sorted, p + eps, side="left")
        - np.searchsorted(y_sorted, p - eps, side="left")
    )


def counting_map_on_grid(y, n_grid=500):
    grid = np.linspace(np.min(y), np.max(y), n_grid)
    return grid, counting_map(y, grid)


def window_count_range(y):
    return counting_map_range(y, domain="data")
