# src/utils.py
import numpy as np

def rng_from(seed_or_rng=None) -> np.random.Generator:
    return seed_or_rng if isinstance(seed_or_rng, np.random.Generator) else np.random.default_rng(seed_or_rng)

def harmonic_c_isoforest(n: int) -> float:
    if n <= 1:
        return 0.0
    H = np.log(n - 1) + 0.5772156649
    return 2.0 * H - 2.0 * (n - 1) / n

def eps_from_range(y: np.ndarray) -> float:
    y = np.asarray(y, float)
    n = y.size
    if n <= 1:
        return 0.0
    m, M = float(y.min()), float(y.max())
    return (M - m) / (2.0 * (n - 1))

def count_in_window_sorted(y_sorted: np.ndarray, p: float, eps: float) -> int:
    lo = np.searchsorted(y_sorted, p - eps, side="left")
    hi = np.searchsorted(y_sorted, p + eps, side="left")
    return max(0, int(hi - lo))

def dap_pick_threshold(y: np.ndarray, alpha: float = 1.0, rng=None, max_trials: int = 256) -> float:
    rng = rng_from(rng)
    y = np.asarray(y, float)
    m, M = float(y.min()), float(y.max())
    if m == M:
        return m
    eps = eps_from_range(y)
    y_sorted = np.sort(y)
    a = int(np.floor(alpha))
    for _ in range(max_trials):
        p = float(rng.uniform(m, M))
        if count_in_window_sorted(y_sorted, p, eps) <= a:
            return p
    return float(rng.uniform(m, M))

def random_unit_vectors(rng: np.random.Generator, d: int, k: int) -> np.ndarray:
    V = rng.normal(size=(k, d))
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n == 0.0] = 1.0
    return V / n

def window_count_range(y: np.ndarray) -> tuple[int, int]:
    ys = np.sort(np.asarray(y, float))
    n = ys.size
    if n == 0:
        return 0, 0
    eps = eps_from_range(ys)
    L = R = 0
    min_c, max_c = n, 0
    for i in range(n):
        center = ys[i]
        while L < n and center - ys[L] > eps + 1e-12:
            L += 1
        while R < n and ys[R] <= center + eps + 1e-12:
            R += 1
        cnt = R - L
        if cnt < min_c:
            min_c = cnt
        if cnt > max_c:
            max_c = cnt
    return int(min_c), int(max_c)

def dap_local_mincount(y: np.ndarray) -> int:
    ys = np.sort(np.asarray(y, float))
    n = ys.size
    if n == 0:
        return 0
    eps = eps_from_range(ys)
    best = n
    L = 0
    for R in range(n):
        while ys[R] - ys[L] > eps + 1e-12:
            L += 1
        cnt = R - L + 1
        if cnt < best:
            best = cnt
    return int(best)