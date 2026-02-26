from __future__ import annotations
import numpy as np
from .counting_map import window_count_range

def random_unit_vectors(rng: np.random.Generator, d: int, k: int) -> np.ndarray:
    """Sample k random unit vectors in R^d."""
    V = rng.normal(size=(k, d))
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return V / n

def dad_direction_uniform(
    X: np.ndarray,
    rng: np.random.Generator,
    n_dir_candidates: int = 256,
) -> np.ndarray:
    """
    Density-Aware Direction (DAD):
    sample k candidate directions uniformly at random and choose the one
    minimizing the counting-map range (max-min window counts).
    """
    X = np.asarray(X, float)
    d = X.shape[1]
    V = random_unit_vectors(rng, d, n_dir_candidates)  # (k, d)
    P = X @ V.T  # (n, k)

    best_idx = 0
    best_rng = None
    for j in range(V.shape[0]):
        cmin, cmax = window_count_range(P[:, j])
        rngv = cmax - cmin
        if (best_rng is None) or (rngv < best_rng):
            best_rng = rngv
            best_idx = j
    return V[best_idx]
