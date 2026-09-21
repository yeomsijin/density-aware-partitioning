from __future__ import annotations
import numpy as np
from .utils import rng_from


def gaussian_ring_clusters(
    seed: int = 0,
    n_clusters: int = 6,
    n_per_cluster: int = 100,
    radius: float = 1.0,
    noise_std: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Example 2 dataset: isotropic Gaussian clusters in R^2 with centers on a circle.
    Centers: radius*(cos(2πj/K), sin(2πj/K)).
    """
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, n_clusters, endpoint=False)
    centers = np.c_[np.cos(angles), np.sin(angles)] * radius
    X = np.vstack([c + rng.normal(scale=noise_std, size=(n_per_cluster, 2)) for c in centers])
    return X, centers


def make_lattice_set() -> np.ndarray:
    vals = np.array([0, 1, 2, 10, 11, 12], float)
    GX, GY = np.meshgrid(vals, vals)
    return np.c_[GX.ravel(), GY.ravel()]


def make_grid_square_set(jitter: float = 0.03, seed: int = 0) -> np.ndarray:
    rng = rng_from(seed)
    X = make_lattice_set()
    if jitter and jitter > 0:
        X = X + rng.normal(scale=jitter, size=X.shape)
    return X
