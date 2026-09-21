"""Density-aware partitioning, Yeom & Jung (2026), Definitions 1-2 / Algorithms 1-2.

Windows are half-open [p-epsilon, p+epsilon); multiplicities are retained.
The event representation (y-epsilon, y+epsilon] avoids tolerance widening.
"""

from numbers import Integral, Real

import numpy as np


def positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def projection(y):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or not y.size or not np.isfinite(y).all():
        raise ValueError("y must be a nonempty, finite one-dimensional array")
    if not np.isfinite(y.max() - y.min()):
        raise ValueError("projection range overflows; rescale the data")
    return y


def epsilon(y):
    """Return (max(y)-min(y))/(2*(n-1)), or zero for a singleton."""
    y = projection(y)
    return float((y.max() - y.min()) / (2 * (len(y) - 1))) if len(y) > 1 else 0.0


def counting_map(y, p):
    """Evaluate d_Y(p), including multiplicity and the epsilon=0 singleton rule."""
    y = projection(y)
    p = np.asarray(p, dtype=float)
    if not np.isfinite(p).all():
        raise ValueError("p must be finite")
    e = epsilon(y)
    if e == 0:
        result = np.where(p == y[0], len(y), 0)
    else:
        # A point contributes on (y-e, y+e]. Both searches exclude equality.
        result = np.searchsorted(np.sort(y - e), p, side="left") - np.searchsorted(
            np.sort(y + e), p, side="left"
        )
    return int(result) if result.ndim == 0 else result


def _events(y):
    e = epsilon(y)
    points = np.concatenate((y - e, y + e))
    changes = np.concatenate((np.ones(len(y), dtype=int), -np.ones(len(y), dtype=int)))
    order = np.argsort(points, kind="stable")
    positions, first = np.unique(points[order], return_index=True)
    changes = np.add.reduceat(changes[order], first)
    # Counts immediately to the right of each event. Tied events change together.
    return positions, np.cumsum(changes)


def counting_map_range(y, *, domain="data"):
    """Exact step-function extrema, not a grid/sample-point approximation.

    domain='real' interprets Algorithm 2's unrestricted p literally (minimum=0).
    domain='data' (default) restricts p to [min(y), max(y)], as clarified by the author.
    """
    y = projection(y)
    if domain not in ("real", "data"):
        raise ValueError("domain must be 'real' or 'data'")
    if epsilon(y) == 0:
        return (0 if domain == "real" else len(y)), len(y)
    positions, counts = _events(y)
    if domain == "real":
        return 0, int(counts.max())
    lo, hi = y.min(), y.max()
    overlap = (positions[:-1] < hi) & (positions[1:] > lo)
    values = np.concatenate((counts[:-1][overlap], np.atleast_1d(counting_map(y, [lo, hi]))))
    return int(values.min()), int(values.max())


def density_measure(X):
    """Maximum window count / n, averaged over columns for a 2D array."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return counting_map_range(X)[1] / len(X)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("X must be one- or two-dimensional")
    return float(np.mean([density_measure(X[:, j]) for j in range(X.shape[1])]))


def density_aware_split(y, *, alpha=1.0, random_state=None, max_trials=256):
    """Sample uniformly conditioned on d_Y(p) <= alpha (Algorithm 1).

    After max_trials, sample acceptable constant-count intervals by their length.
    This preserves the rejection distribution; no unchecked uniform fallback.
    Raises ValueError if no positive-length acceptance interval exists. A constant
    projection returns its value (a no-split sentinel; callers must stop the node).
    """
    y = projection(y)
    if not isinstance(alpha, Real) or not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    if isinstance(max_trials, bool) or not isinstance(max_trials, Integral) or max_trials < 0:
        raise ValueError("max_trials must be a nonnegative integer")
    rng = (
        random_state
        if isinstance(random_state, (np.random.Generator, np.random.RandomState))
        else np.random.default_rng(random_state)
    )
    lo, hi = float(y.min()), float(y.max())
    e = epsilon(y)
    if e == 0:
        return lo
    starts, ends = np.sort(y - e), np.sort(y + e)
    for _ in range(max_trials):
        p = float(rng.uniform(lo, hi))
        count = np.searchsorted(starts, p, side="left") - np.searchsorted(ends, p, side="left")
        if count <= alpha:
            return p
    positions, counts = _events(y)
    left = np.maximum(positions[:-1], lo)
    right = np.minimum(positions[1:], hi)
    valid = (right > left) & (counts[:-1] <= alpha)
    left, right = left[valid], right[valid]
    if not left.size:
        raise ValueError("No positive-length interval satisfies alpha; increase alpha")
    widths = right - left
    cumulative = np.cumsum(widths)
    draw = float(rng.uniform(0, cumulative[-1]))
    i = min(int(np.searchsorted(cumulative, draw, side="right")), len(left) - 1)
    previous = cumulative[i - 1] if i else 0.0
    p = float(left[i] + (draw - previous))
    # Use an interior float to avoid a rounded draw landing on a rejected endpoint.
    p = max(float(np.nextafter(left[i], right[i])), min(p, float(np.nextafter(right[i], left[i]))))
    if not left[i] < p < right[i] or counting_map(y, p) > alpha:
        raise FloatingPointError("Accepted interval has no usable float; rescale the data")
    return p


def random_directions(rng, n_features, n_candidates, max_features=None):
    """Sphere-uniform directions on independently selected feature subsets."""
    d = positive_int(n_features, "n_features")
    k = positive_int(n_candidates, "n_candidates")
    q = d if max_features is None else positive_int(max_features, "max_features")
    if q > d:
        raise ValueError("max_features cannot exceed n_features")
    V = np.zeros((k, d))
    for i in range(k):
        features = np.arange(d) if q == d else rng.choice(d, q, replace=False)
        v = rng.normal(size=q)
        norm = np.linalg.norm(v)
        if norm == 0:
            v[0], norm = 1.0, 1.0
        V[i, features] = v / norm
    return V


def density_aware_direction(
    X, *, n_candidates=10, max_features=None, domain="data", random_state=None
):
    """Algorithm 2; ties select the first sampled direction."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or 0 in X.shape or not np.isfinite(X).all():
        raise ValueError("X must be a nonempty finite 2D array")
    rng = np.random.default_rng(random_state)
    V = random_directions(rng, X.shape[1], n_candidates, max_features)
    best, best_range = None, np.inf
    # Process one candidate at a time, avoiding an n_samples x k allocation.
    for v in V:
        low, high = counting_map_range(X @ v, domain=domain)
        if high - low < best_range:
            best, best_range = v, high - low
    return best
