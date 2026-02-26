from __future__ import annotations
import numpy as np

def eps_from_range(y_sorted: np.ndarray) -> float:
    """Window half-width epsilon used by the counting map (Algorithm 1 helper)."""
    n = y_sorted.size
    if n <= 1:
        return 0.0
    return (float(y_sorted[-1]) - float(y_sorted[0])) / (2.0 * (n - 1))

def count_in_window_sorted(y_sorted: np.ndarray, p: float, eps: float, tol: float) -> int:
    """Count points in [p-eps, p+eps] using binary search on sorted projections."""
    L = np.searchsorted(y_sorted, p - eps - tol, side="left")
    R = np.searchsorted(y_sorted, p + eps + tol, side="right")
    return int(R - L)

def counting_map_on_grid(y: np.ndarray, n_grid: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute counting map d_Y(p) on an evenly spaced grid over projection values.
    Returns (grid_p, d_values).
    """
    y = np.asarray(y, float)
    m, M = float(np.min(y)), float(np.max(y))
    grid = np.linspace(m, M, n_grid)

    ys = np.sort(y)
    eps = eps_from_range(ys)
    tol = 1e-12 * max(1.0, M - m)

    d = np.empty_like(grid)
    for i, p in enumerate(grid):
        d[i] = count_in_window_sorted(ys, p, eps, tol)
    return grid, d

def window_count_range(y: np.ndarray) -> tuple[int, int]:
    """
    Range of window counts across points (min and max) for a given projection.
    Used as the DAD objective proxy (Algorithm 2 helper).
    """
    ys = np.sort(np.asarray(y, float))
    n = ys.size
    if n == 0:
        return 0, 0

    eps = eps_from_range(ys)
    L = 0
    R = 0
    min_c = n
    max_c = 0

    for i in range(n):
        p = ys[i]
        while L < n and p - ys[L] > eps:
            L += 1
        while R < n and ys[R] <= p + eps:
            R += 1
        cnt = R - L
        min_c = min(min_c, cnt)
        max_c = max(max_c, cnt)
    return int(min_c), int(max_c)