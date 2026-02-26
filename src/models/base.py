# src/models/base.py
import numpy as np
from ..utils import rng_from, harmonic_c_isoforest

class BaseIForest:
    __slots__ = ("n_trees","sample_size","max_depth","mode","alpha","rng","trees","n_dir_candidates")
    def __init__(self, *, n_trees=100, sample_size=128, max_depth=None, mode="", alpha=1.0,
                 random_seed=0, n_dir_candidates=256):
        self.n_trees = int(n_trees)
        self.sample_size = int(sample_size)
        self.max_depth = max_depth
        self.mode = str(mode)
        self.alpha = float(alpha)
        self.rng = rng_from(random_seed)
        self.trees = []
        self.n_dir_candidates = int(n_dir_candidates)

    def _stop(self, X: np.ndarray, depth: int) -> bool:
        if X.shape[0] <= 1:
            return True
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        return bool(np.all(np.max(X, axis=0) == np.min(X, axis=0)))

    def fit(self, X: np.ndarray):
        X = np.asarray(X, float)
        n = X.shape[0]
        psi = min(self.sample_size, n)
        if self.max_depth is None:
            self.max_depth = int(np.ceil(np.log2(max(psi, 2))))
        self.trees = []
        for _ in range(self.n_trees):
            idx = self.rng.choice(n, size=psi, replace=False)
            root = self._build(X[idx], 0)
            self.trees.append((root, len(idx)))
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        scores = np.zeros(X.shape[0], float)
        cs = [max(harmonic_c_isoforest(n_sub), 1e-9) for (_, n_sub) in self.trees]
        for i, x in enumerate(X):
            lens = []
            for (root, _), c_t in zip(self.trees, cs):
                lens.append(self._path_len_one(x, root) / c_t)
            E = float(np.mean(lens)) if lens else 0.0
            scores[i] = 2.0 ** (-E)
        return scores

    # to be implemented in subclasses
    def _build(self, X: np.ndarray, depth: int):
        raise NotImplementedError

    def _path_len_one(self, x: np.ndarray, node) -> float:
        raise NotImplementedError