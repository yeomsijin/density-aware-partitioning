"""Tabular IF / EIF / SCiF with composable density-aware partition rules."""

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_array, check_is_fitted

from .partition import density_aware_direction, density_aware_split, positive_int, random_directions


def average_path_length(n):
    """IF external-node correction: c(0)=c(1)=0, c(2)=1, log approximation above 2."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    return float(2 * (np.log(n - 1) + np.euler_gamma) - 2 * (n - 1) / n)


def best_sd_split(y):
    """All distinct adjacent splits, O(n log n), vectorized prefix moments."""
    ys = np.sort(np.asarray(y, dtype=float))
    if len(ys) < 2 or ys[-1] == ys[0]:
        return None, -np.inf
    # Translate/scale before moments to reduce catastrophic cancellation.
    z = (ys - ys[0]) / (ys[-1] - ys[0])
    n = len(z)
    left_n = np.arange(1, n)
    right_n = n - left_n
    sums, squares = np.cumsum(z), np.cumsum(z * z)
    left_sd = np.sqrt(np.maximum(squares[:-1] / left_n - (sums[:-1] / left_n) ** 2, 0))
    # Reverse sums avoid subtracting almost equal totals on the right tail.
    rev_sums, rev_squares = np.cumsum(z[::-1]), np.cumsum(z[::-1] ** 2)
    right_sd = np.sqrt(
        np.maximum(rev_squares[-2::-1] / right_n - (rev_sums[-2::-1] / right_n) ** 2, 0)
    )
    sd = float(np.std(z))
    gains = (sd - (left_sd + right_sd) / 2) / sd
    gains[ys[:-1] == ys[1:]] = -np.inf
    i = int(np.argmax(gains))
    threshold = ys[i] + (ys[i + 1] - ys[i]) / 2
    if threshold <= ys[i]:
        threshold = ys[i + 1]
    return float(threshold), float(gains[i])


@dataclass(slots=True)
class Node:
    size: int
    feature: int | None = None
    direction: np.ndarray | None = None
    threshold: float = 0.0
    left: object = None
    right: object = None


class DensityAwareForest(OutlierMixin, BaseEstimator):
    """IF, EIF or SCiF with optional DAS/DAD, following MLST sections 2-3.

    Parameters
    ----------
    model : {'if', 'eif', 'scif'}, default='if'
    das, dad : bool
        Enable density-aware thresholds/directions. IF supports DAS; SCiF DAD;
        EIF supports both. Unsupported combinations raise instead of being ignored.
    n_estimators : int, default=100
    max_samples : int or 'auto', default=256
    max_depth : int or None
        None uses ceil(log2(actual subsample size)). Recomputed on each fit.
    alpha : float, default=1
        Accept d_Y(p) <= alpha (Drop-in convention).
    n_directions : int, default=10
        DAD candidates k. Benchmark EIF uses 2; SCiF uses 10.
    n_hyperplanes : int, default=10
        SCiF baseline candidates tau; SCiF+DAD runs one gain search.
    max_features : int or None
        Active features per oblique candidate; None uses all. Benchmark uses 2.
    range_domain : {'real', 'data'}, default='data'
        Domain for DAD extrema; data range follows author clarification.
    contamination : 'auto' or float in (0, 0.5]
        Threshold only; does not change tree construction. Auto uses score 0.5.
    random_state : int, Generator or None
        Integer seeds restart the RNG on every fit.

    anomaly_score is larger for anomalies; score_samples is its negative.
    decision_function = score_samples - offset_; predict returns -1 or +1.
    Dense finite numeric arrays are required. Fit does not use labels.
    """

    def __init__(
        self,
        *,
        model="if",
        das=True,
        dad=False,
        n_estimators=100,
        max_samples=256,
        max_depth=None,
        alpha=1.0,
        n_directions=10,
        n_hyperplanes=10,
        max_features=None,
        range_domain="data",
        contamination="auto",
        random_state=None,
    ):
        self.model = model
        self.das = das
        self.dad = dad
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.alpha = alpha
        self.n_directions = n_directions
        self.n_hyperplanes = n_hyperplanes
        self.max_features = max_features
        self.range_domain = range_domain
        self.contamination = contamination
        self.random_state = random_state

    def _split_alpha(self):
        return self.alpha

    def _validate_parameters(self, d):
        if self.model not in ("if", "eif", "scif"):
            raise ValueError("model must be 'if', 'eif', or 'scif'")
        if not isinstance(self.das, (bool, np.bool_)) or not isinstance(self.dad, (bool, np.bool_)):
            raise ValueError("das and dad must be booleans")
        if (self.model == "if" and self.dad) or (self.model == "scif" and self.das):
            raise ValueError("IF supports DAS only; SCiF supports DAD only; EIF supports both")
        for name in ("n_estimators", "n_directions", "n_hyperplanes"):
            positive_int(getattr(self, name), name)
        if self.max_samples != "auto":
            positive_int(self.max_samples, "max_samples")
        if self.max_depth is not None:
            positive_int(self.max_depth, "max_depth")
        if self.max_features is not None:
            if positive_int(self.max_features, "max_features") > d:
                raise ValueError("max_features cannot exceed n_features")
        if self.range_domain not in ("real", "data"):
            raise ValueError("range_domain must be 'real' or 'data'")
        if not np.isfinite(self._split_alpha()) or self._split_alpha() <= 0:
            raise ValueError("alpha must give a finite positive acceptance level")
        if self.contamination != "auto":
            if (
                not isinstance(self.contamination, (int, float))
                or not 0 < self.contamination <= 0.5
            ):
                raise ValueError("contamination must be 'auto' or in (0, 0.5]")

    def fit(self, X, y=None):
        X = check_array(X, dtype=np.float64, ensure_min_samples=2)
        self._validate_parameters(X.shape[1])
        if not np.isfinite(np.ptp(X, axis=0)).all():
            raise ValueError("feature range overflows; rescale X")
        self.n_features_in_ = X.shape[1]
        self.max_samples_ = min(256 if self.max_samples == "auto" else self.max_samples, len(X))
        if self.max_samples_ < 2:
            raise ValueError("max_samples must be at least 2 for forest scoring")
        self.max_depth_ = (
            self.max_depth
            if self.max_depth is not None
            else int(np.ceil(np.log2(self.max_samples_)))
        )
        self._rng = np.random.default_rng(self.random_state)
        self.estimators_ = [
            self._build(X[self._rng.choice(len(X), self.max_samples_, replace=False)])
            for _ in range(self.n_estimators)
        ]
        self.offset_ = -0.5
        if self.contamination != "auto":
            self.offset_ = float(np.percentile(self.score_samples(X), 100 * self.contamination))
        return self

    def _direction(self, X):
        if self.dad:
            return density_aware_direction(
                X,
                n_candidates=self.n_directions,
                max_features=self.max_features,
                domain=self.range_domain,
                random_state=self._rng,
            )
        return random_directions(self._rng, X.shape[1], 1, self.max_features)[0]

    def _choose_split(self, X):
        if self.model == "if":
            j = int(self._rng.integers(X.shape[1]))
            y = X[:, j]
            if y.min() == y.max():
                return None
            p = (
                density_aware_split(y, alpha=self._split_alpha(), random_state=self._rng)
                if self.das
                else float(self._rng.uniform(y.min(), y.max()))
            )
            return j, None, p, y < p
        if self.model == "eif":
            v = self._direction(X)
            # Section 3.3's explicit procedure samples each coordinate of p.
            p = (
                np.array(
                    [
                        density_aware_split(
                            X[:, j], alpha=self._split_alpha(), random_state=self._rng
                        )
                        for j in range(X.shape[1])
                    ]
                )
                if self.das
                else self._rng.uniform(X.min(axis=0), X.max(axis=0))
            )
            threshold = float(p @ v)
            return None, v, threshold, (X @ v) < threshold
        best, best_gain = None, -np.inf
        scale = X.std(axis=0)
        valid = np.flatnonzero(scale > 0)
        if not len(valid):
            return None
        for _ in range(1 if self.dad else self.n_hyperplanes):
            if self.dad:
                # Algorithm 2 is applied directly to raw X; no unstated rescaling.
                v = self._direction(X)
            else:
                q = len(valid) if self.max_features is None else min(self.max_features, len(valid))
                features = self._rng.choice(valid, q, replace=False)
                v = np.zeros(X.shape[1])
                v[features] = self._rng.uniform(-1, 1, q) / scale[features]
                norm = np.linalg.norm(v)
                if norm == 0:
                    continue
                v /= norm
            y = X @ v
            p, gain = best_sd_split(y)
            if p is not None and gain > best_gain:
                best_gain = gain
                best = None, v, p, y < p
        return best

    def _build(self, X):
        root = Node(len(X))
        stack = [(root, X, 0)]
        while stack:
            node, data, depth = stack.pop()
            if (
                len(data) <= 1
                or depth >= self.max_depth_
                or np.all(data.max(axis=0) == data.min(axis=0))
            ):
                continue
            split = self._choose_split(data)
            if split is None:
                continue
            j, v, p, mask = split
            if not mask.any() or mask.all():
                continue
            node.feature, node.direction, node.threshold = j, v, p
            node.left, node.right = Node(int(mask.sum())), Node(int((~mask).sum()))
            stack.append((node.right, data[~mask], depth + 1))
            stack.append((node.left, data[mask], depth + 1))
        return root

    def _check_X(self, X):
        check_is_fitted(self, "estimators_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features; expected {self.n_features_in_}")
        return X

    def anomaly_score(self, X):
        """Return 2**(-mean(path length)/c(max_samples_)); larger means anomalous."""
        X = self._check_X(X)
        lengths = np.zeros(len(X))
        correction = {n: average_path_length(n) for n in range(self.max_samples_ + 1)}
        for root in self.estimators_:
            stack = [(root, np.arange(len(X)), 0)]
            while stack:
                node, idx, depth = stack.pop()
                if not len(idx):
                    continue
                if node.left is None:
                    lengths[idx] += depth + correction[node.size]
                    continue
                projected = (
                    X[idx, node.feature] if node.feature is not None else X[idx] @ node.direction
                )
                mask = projected < node.threshold
                stack.append((node.right, idx[~mask], depth + 1))
                stack.append((node.left, idx[mask], depth + 1))
        return np.exp2(-lengths / (len(self.estimators_) * average_path_length(self.max_samples_)))

    def score_samples(self, X):
        """Opposite of anomaly_score; larger means more normal."""
        return -self.anomaly_score(X)

    def decision_function(self, X):
        return self.score_samples(X) - self.offset_

    def predict(self, X):
        return np.where(self.decision_function(X) < 0, -1, 1)
