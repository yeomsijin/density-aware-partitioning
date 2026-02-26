import numpy as np
from .base import BaseIForest
from ..utils import (
    random_unit_vectors, window_count_range, dap_local_mincount,
    dap_pick_threshold, harmonic_c_isoforest
)

class _ObliqueNode:
    __slots__=("leaf","left","right","v","p","size")
    def __init__(self, leaf, size, v=None, p=None, left=None, right=None):
        self.leaf=leaf; self.size=size; self.v=v; self.p=p; self.left=left; self.right=right

class ObliqueIForest(BaseIForest):
    def _rand_dir_sphere(self, d: int) -> np.ndarray:
        v = self.rng.normal(size=d)
        n = np.linalg.norm(v)
        if n == 0:
            v = np.array([1.0] + [0.0]*(d-1))
        v = v / np.linalg.norm(v)
        return v

    def _dap_direction_uniform(self, X: np.ndarray) -> np.ndarray:
        d = X.shape[1]
        V = random_unit_vectors(self.rng, d, self.n_dir_candidates)
        best_v = V[0]
        best_range = None
        for v in V:
            y = X @ v
            cmin, cmax = window_count_range(y)
            rng_val = cmax - cmin
            if best_range is None or rng_val < best_range:
                best_range, best_v = rng_val, v
        return best_v

    def _dap_direction_dense(self, X: np.ndarray) -> np.ndarray:
        d = X.shape[1]
        V = random_unit_vectors(self.rng, d, self.n_dir_candidates)
        best_v = V[0]
        best_score = -np.inf
        for v in V:
            s = dap_local_mincount(X @ v)
            if s > best_score:
                best_score, best_v = s, v
        return best_v

    def _rand_point_uniform(self, X: np.ndarray) -> np.ndarray:
        mins, maxs = X.min(axis=0), X.max(axis=0)
        return self.rng.uniform(mins, maxs)

    def _rand_point_das(self, X: np.ndarray) -> np.ndarray:
        return np.array([dap_pick_threshold(X[:, j], self.alpha, self.rng) for j in range(X.shape[1])])

    def _choose_dir_and_point(self, X: np.ndarray):
        if self.mode == "EIF":
            return self._rand_dir_sphere(X.shape[1]), self._rand_point_uniform(X)
        if self.mode == "EIF+DAS":
            return self._rand_dir_sphere(X.shape[1]), self._rand_point_das(X)
        if self.mode == "EIF+DAD":
            return self._dap_direction_uniform(X), self._rand_point_uniform(X)
        if self.mode == "EIF+DAS+DAD":
            return self._dap_direction_uniform(X), self._rand_point_das(X)
        raise ValueError(f"Unknown EIF mode: {self.mode}")

    def _build(self, X: np.ndarray, depth: int):
        n, _ = X.shape
        if self._stop(X, depth):
            return _ObliqueNode(True, n)

        v, p = self._choose_dir_and_point(X)
        proj = (X - p) @ v
        mask = proj < 0
        XL, XR = X[mask], X[~mask]
        if XL.shape[0] == 0 or XR.shape[0] == 0:
            return _ObliqueNode(True, n)

        L = self._build(XL, depth+1)
        R = self._build(XR, depth+1)
        return _ObliqueNode(False, n, v=v, p=p, left=L, right=R)

    def _path_len_one(self, x: np.ndarray, node) -> float:
        L = 0
        while node and not node.leaf:
            L += 1
            node = node.left if float((x - node.p) @ node.v) < 0 else node.right
        return L + harmonic_c_isoforest(getattr(node, "size", 1))