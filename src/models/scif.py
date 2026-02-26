import numpy as np
from .base import BaseIForest
from ..utils import random_unit_vectors, window_count_range, harmonic_c_isoforest

class _SCiNode:
    __slots__=("leaf","left","right","v","p","size")
    def __init__(self, leaf, size, v=None, p=None, left=None, right=None):
        self.leaf=leaf; self.size=size; self.v=v; self.p=p; self.left=left; self.right=right

def best_split_sd_gain(y: np.ndarray):
    y = np.asarray(y, float)
    n = y.size
    if n <= 1:
        return None, -np.inf

    ys = np.sort(y)
    s1_all = float(ys.sum())
    s2_all = float((ys*ys).sum())

    def _std(n_, s1_, s2_):
        if n_ <= 1:
            return 0.0
        mean = s1_ / n_
        var = max(s2_ / n_ - mean*mean, 0.0)
        return float(np.sqrt(var))

    sd_all = _std(n, s1_all, s2_all)
    if sd_all == 0.0:
        return None, -np.inf

    P = np.cumsum(ys)
    Q = np.cumsum(ys*ys)

    best_mid, best_gain = None, -np.inf
    for i in range(1, n):
        if ys[i-1] == ys[i]:
            continue
        nL, nR = i, n - i
        s1L, s2L = float(P[i-1]), float(Q[i-1])
        s1R, s2R = s1_all - s1L, s2_all - s2L
        sdL, sdR = _std(nL, s1L, s2L), _std(nR, s1R, s2R)
        gain = (sd_all - 0.5*(sdL + sdR)) / sd_all
        if gain > best_gain:
            best_gain = gain
            best_mid = 0.5*(ys[i-1] + ys[i])

    return best_mid, float(best_gain)

class SCiForest(BaseIForest):
    def _rand_dir_cube(self, d: int) -> np.ndarray:
        v = self.rng.uniform(-1.0, 1.0, size=d)
        n = np.linalg.norm(v)
        if n == 0:
            v = np.array([1.0] + [0.0]*(d-1))
        return v / np.linalg.norm(v)

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

    def _choose_direction(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "SCIF+DAD":
            return self._dap_direction_uniform(X)
        return self._rand_dir_cube(X.shape[1])

    def _build(self, X: np.ndarray, depth: int):
        n, _ = X.shape
        if self._stop(X, depth):
            return _SCiNode(True, n)

        v = self._choose_direction(X)
        y = X @ v

        p_scalar, gain = best_split_sd_gain(y)
        if p_scalar is None or not np.isfinite(gain):
            return _SCiNode(True, n)

        a = X.mean(axis=0)
        p = a + (float(p_scalar) - float(a @ v)) * v  # ensure p·v = p_scalar

        proj = (X - p) @ v
        mask = proj < 0
        XL, XR = X[mask], X[~mask]
        if XL.shape[0] == 0 or XR.shape[0] == 0:
            return _SCiNode(True, n)

        L = self._build(XL, depth+1)
        R = self._build(XR, depth+1)
        return _SCiNode(False, n, v=v, p=p, left=L, right=R)

    def _path_len_one(self, x: np.ndarray, node) -> float:
        L = 0
        while node and not node.leaf:
            L += 1
            node = node.left if float((x - node.p) @ node.v) < 0 else node.right
        return L + harmonic_c_isoforest(getattr(node, "size", 1))