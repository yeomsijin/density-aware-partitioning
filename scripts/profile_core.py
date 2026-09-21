"""Compare equivalent scoring and split-search operations, not paper speed claims."""

import argparse
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from dap import DensityAwareForest
from dap.forest import average_path_length, best_sd_split


def scalar_scores(model, X):
    out = []
    for x in X:
        paths = []
        for node in model.estimators_:
            depth = 0
            while node.left is not None:
                p = x[node.feature] if node.feature is not None else x @ node.direction
                node = node.left if p < node.threshold else node.right
                depth += 1
            paths.append(depth + average_path_length(node.size))
        out.append(2 ** (-np.mean(paths) / average_path_length(model.max_samples_)))
    return np.array(out)


def original_prefix_sd_split(y: np.ndarray):
    y = np.asarray(y, float)
    n = y.size
    if n <= 1:
        return None, -np.inf

    ys = np.sort(y)
    s1_all = float(ys.sum())
    s2_all = float((ys * ys).sum())

    def _std(n_, s1_, s2_):
        if n_ <= 1:
            return 0.0
        mean = s1_ / n_
        var = max(s2_ / n_ - mean * mean, 0.0)
        return float(np.sqrt(var))

    sd_all = _std(n, s1_all, s2_all)
    if sd_all == 0.0:
        return None, -np.inf

    P = np.cumsum(ys)
    Q = np.cumsum(ys * ys)

    best_mid, best_gain = None, -np.inf
    for i in range(1, n):
        if ys[i - 1] == ys[i]:
            continue
        nL, nR = i, n - i
        s1L, s2L = float(P[i - 1]), float(Q[i - 1])
        s1R, s2R = s1_all - s1L, s2_all - s2L
        sdL, sdR = _std(nL, s1L, s2L), _std(nR, s1R, s2R)
        gain = (sd_all - 0.5 * (sdL + sdR)) / sd_all
        if gain > best_gain:
            best_gain = gain
            best_mid = 0.5 * (ys[i - 1] + ys[i])

    return best_mid, float(best_gain)


def timed(fn, repeats=3):
    fn()
    times = []
    for _ in range(repeats):
        start = perf_counter()
        fn()
        times.append(perf_counter() - start)
    return median(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    train = rng.normal(size=(256, 4))
    query = rng.normal(size=(2000, 4))
    y = rng.normal(size=256)
    m = DensityAwareForest(n_estimators=25, max_samples=128, random_state=42).fit(train)
    np.testing.assert_allclose(
        m.anomaly_score(query), scalar_scores(m, query), rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(
        best_sd_split(y), original_prefix_sd_split(y), rtol=1e-12, atol=1e-12
    )
    scalar = timed(lambda: scalar_scores(m, query))
    batch = timed(lambda: m.anomaly_score(query))
    direct = timed(lambda: original_prefix_sd_split(y))
    prefix = timed(lambda: best_sd_split(y))
    result = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        platform=platform.platform(),
        repeats=3,
        seed=42,
        trees=25,
        train_shape=list(train.shape),
        query_shape=list(query.shape),
        subsample=128,
        score=dict(scalar_seconds=scalar, batch_seconds=batch, speedup=scalar / batch),
        sd_gain=dict(
            n=256, original_loop_seconds=direct, vectorized_seconds=prefix, speedup=direct / prefix
        ),
        note="Same-operation microbenchmarks; not paper accuracy/runtime reproduction. SD reference is the original c5bc307 prefix-sum Python loop.",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
