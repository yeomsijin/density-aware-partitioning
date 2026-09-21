"""Forest aggregation and bounded-window streaming for weighted random cut trees."""

from collections import deque
from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_array, check_is_fitted

from dap.partition import positive_int

from .wrcf import RCTree


class WeightedRandomCutForest(OutlierMixin, BaseEstimator):
    """Mean CODISP across weighted trees; larger values indicate anomalies.

    Unseen observations are inserted into independent copies with weighted rebuild.
    This is an explicit scoring extension, not fast online RRCF insertion.
    contamination controls a training-score quantile (there is no universal CODISP
    threshold). Repeated scoring does not mutate the fitted trees or RNG states.
    """

    def __init__(
        self, *, n_estimators=100, max_samples=256, alpha=2, contamination=0.1, random_state=None
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.alpha = alpha
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X, y=None):
        X = check_array(X, dtype=float, ensure_min_samples=2)
        positive_int(self.n_estimators, "n_estimators")
        positive_int(self.max_samples, "max_samples")
        if not isinstance(self.contamination, (float, int)) or not 0 < self.contamination <= 0.5:
            raise ValueError("contamination must be in (0, 0.5]")
        RCTree(alpha=self.alpha)
        self.n_features_in_ = X.shape[1]
        self.max_samples_ = min(self.max_samples, len(X))
        if self.max_samples_ < 2:
            raise ValueError("max_samples must be at least 2")
        rng = np.random.default_rng(self.random_state)
        self.estimators_ = []
        for _ in range(self.n_estimators):
            idx = rng.choice(len(X), self.max_samples_, replace=False)
            self.estimators_.append(
                RCTree(X[idx], random_state=int(rng.integers(2**32)), alpha=self.alpha)
            )
        self.offset_ = float(np.percentile(self.score_samples(X), 100 * self.contamination))
        return self

    def anomaly_score(self, X):
        """Mean CODISP; score each unseen row independently, with weighted rebuilding."""
        check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=float)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has the wrong number of features")
        scores = np.zeros(len(X))
        for tree in self.estimators_:
            for i, point in enumerate(X):
                leaf = tree.find_duplicate(point)
                if leaf is not None:
                    scores[i] += tree.codisp(leaf)
                else:
                    temporary = deepcopy(tree)
                    key = "__query__"
                    temporary.insert_point(point, key)
                    scores[i] += temporary.codisp(key)
        return scores / len(self.estimators_)

    def score_samples(self, X):
        return -self.anomaly_score(X)

    def decision_function(self, X):
        return self.score_samples(X) - self.offset_

    def predict(self, X):
        return np.where(self.decision_function(X) < 0, -1, 1)


class RollingWRCF:
    """Score a stream by rebuilding weighted forests on the most recent window.

    This correctness-first reference uses O(n_estimators * window_size) storage;
    rebuilding on each update is more expensive than ordinary incremental RRCF.
    Input rows must already be shingled if temporal context is desired.
    """

    def __init__(self, *, n_estimators=40, window_size=256, alpha=2, random_state=None):
        self.n_estimators = positive_int(n_estimators, "n_estimators")
        self.window_size = positive_int(window_size, "window_size")
        RCTree(alpha=alpha)
        self.alpha = alpha
        self._rng = np.random.default_rng(random_state)
        self._window = deque(maxlen=window_size)
        self.n_features_in_ = None
        self.estimators_ = []

    def update(self, point):
        point = np.asarray(point, dtype=float)
        if point.ndim != 1 or not point.size or not np.isfinite(point).all():
            raise ValueError("point must be a nonempty finite 1D array")
        if self.n_features_in_ is not None and point.size != self.n_features_in_:
            raise ValueError("point has the wrong number of features")
        self.n_features_in_ = point.size
        self._window.append(point.copy())
        X = np.asarray(self._window)
        self.estimators_ = [
            RCTree(X, alpha=self.alpha, random_state=int(self._rng.integers(2**32)))
            for _ in range(self.n_estimators)
        ]
        return float(np.mean([tree.codisp(len(X) - 1) for tree in self.estimators_]))
