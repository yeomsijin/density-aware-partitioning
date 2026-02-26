import numpy as np
from .base import BaseIForest
from ..utils import dap_pick_threshold, harmonic_c_isoforest

class _AxisNode:
    __slots__=("leaf","left","right","feat","thr","size")
    def __init__(self, leaf, size, feat=None, thr=None, left=None, right=None):
        self.leaf=leaf; self.size=size; self.feat=feat; self.thr=thr; self.left=left; self.right=right

class IsolationForestAD(BaseIForest):
    def _build(self, X: np.ndarray, depth: int):
        n, d = X.shape
        if self._stop(X, depth):
            return _AxisNode(True, n)

        j = int(self.rng.integers(0, d))
        y = X[:, j]
        m, M = float(y.min()), float(y.max())
        if m == M:
            return _AxisNode(True, n)

        if self.mode == "IF+DAS":
            thr = dap_pick_threshold(y, self.alpha, self.rng)
        else:
            thr = float(self.rng.uniform(m, M))

        mask = (y < thr)
        XL, XR = X[mask], X[~mask]
        if XL.shape[0] == 0 or XR.shape[0] == 0:
            return _AxisNode(True, n)

        L = self._build(XL, depth+1)
        R = self._build(XR, depth+1)
        return _AxisNode(False, n, feat=j, thr=thr, left=L, right=R)

    def _path_len_one(self, x: np.ndarray, node) -> float:
        L = 0
        while node and not node.leaf:
            L += 1
            node = node.left if x[node.feat] < node.thr else node.right
        return L + harmonic_c_isoforest(getattr(node, "size", 1))