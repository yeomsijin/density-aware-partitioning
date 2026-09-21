import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dap import DensityAwareForest
from dap.forest import average_path_length, best_sd_split
from wif import WeightedIsolationForest

VARIANTS = [
    ("if", False, False),
    ("if", True, False),
    ("eif", False, False),
    ("eif", True, False),
    ("eif", False, True),
    ("eif", True, True),
    ("scif", False, False),
    ("scif", False, True),
]


@pytest.mark.parametrize("model,das,dad", VARIANTS)
def test_modes_fit_scores_clone_refit_pickle(model, das, dad):
    X = np.random.default_rng(0).normal(size=(40, 3))
    m = DensityAwareForest(
        model=model,
        das=das,
        dad=dad,
        n_estimators=3,
        max_samples=24,
        n_directions=3,
        max_features=2,
        random_state=17,
    )
    m.fit(X)
    expected = m.anomaly_score(X)
    assert np.isfinite(expected).all() and np.all((expected > 0) & (expected <= 1))
    np.testing.assert_array_equal(-expected, m.score_samples(X))
    np.testing.assert_array_equal(expected, m.fit(X).anomaly_score(X))
    np.testing.assert_array_equal(expected, clone(m).fit(X).anomaly_score(X))
    np.testing.assert_array_equal(expected, pickle.loads(pickle.dumps(m)).anomaly_score(X))
    assert set(m.predict(X)) <= {-1, 1}
    assert m.max_depth is None and m.max_depth_ == 5
    m.fit(X[:4])
    assert m.max_depth_ == 2
    with pytest.raises(ValueError):
        m.anomaly_score(np.zeros((2, 4)))


def test_wif_das_equivalence_and_pipeline():
    X = np.random.default_rng(1).normal(size=(30, 2))
    params = dict(n_estimators=5, max_samples=24, random_state=11)
    a = WeightedIsolationForest(alpha=2, **params).fit(X)
    b = DensityAwareForest(alpha=1, **params).fit(X)
    np.testing.assert_array_equal(a.anomaly_score(X), b.anomaly_score(X))
    np.testing.assert_array_equal(a.anomaly_score(X), clone(a).fit(X).anomaly_score(X))
    assert make_pipeline(StandardScaler(), clone(a)).fit(X).predict(X).shape == (30,)


def test_small_sample_correction_and_identical_data():
    assert [average_path_length(i) for i in range(3)] == [0, 0, 1]
    X = np.ones((8, 2))
    m = WeightedIsolationForest(n_estimators=2).fit(X)
    np.testing.assert_allclose(m.anomaly_score(X), 0.5)
    m.fit([[0], [2]])
    np.testing.assert_allclose(m.anomaly_score([[0], [2]]), 0.5)


def test_pre_fit_and_input_validation():
    with pytest.raises(NotFittedError):
        DensityAwareForest().anomaly_score([[1, 2]])
    for kwargs in [
        dict(n_estimators=0),
        dict(n_directions=0),
        dict(max_samples=1),
        dict(model="bad"),
        dict(model="if", dad=True),
        dict(model="scif", das=True),
        dict(alpha=-1),
        dict(contamination=0.9),
        dict(max_features=4),
    ]:
        with pytest.raises(ValueError):
            DensityAwareForest(**kwargs).fit(np.ones((4, 3)))
    for X in [[], [[np.nan]], [[np.inf]], [[1]], np.ones(4)]:
        with pytest.raises(ValueError):
            DensityAwareForest().fit(X)


@pytest.mark.parametrize("seed", range(5))
def test_sd_split_matches_direct_variance_calculation(seed):
    y = np.random.default_rng(seed).normal(size=25)
    ys = np.sort(y)
    mids = (ys[:-1] + ys[1:]) / 2
    gains = [(y.std() - 0.5 * (y[y < p].std() + y[y >= p].std())) / y.std() for p in mids]
    p, gain = best_sd_split(y)
    assert p == pytest.approx(mids[np.argmax(gains)])
    assert gain == pytest.approx(max(gains))
    p_shift, _ = best_sd_split(y + 1e9)
    assert p_shift - 1e9 == pytest.approx(p, abs=2e-7)


def test_scif_compares_tau_standardized_candidates():
    X = np.random.default_rng(4).normal(size=(30, 3)) * [1, 10, 100]
    m = DensityAwareForest(model="scif", das=False, n_hyperplanes=5, max_features=2)
    m._rng = np.random.default_rng(99)
    _, v, p, _ = m._choose_split(X)
    rng = np.random.default_rng(99)
    candidates = []
    for _ in range(5):
        f = rng.choice(np.arange(3), 2, replace=False)
        w = np.zeros(3)
        w[f] = rng.uniform(-1, 1, 2) / X.std(axis=0)[f]
        w /= np.linalg.norm(w)
        t, g = best_sd_split(X @ w)
        candidates.append((g, w, t))
    best = max(candidates, key=lambda row: row[0])
    np.testing.assert_allclose(v, best[1])
    assert p == best[2]


def test_batch_scoring_matches_scalar_traversal():
    X = np.random.default_rng(3).normal(size=(35, 3))
    m = DensityAwareForest(
        model="eif", dad=True, n_estimators=3, n_directions=3, random_state=4
    ).fit(X)
    expected = []
    for x in X:
        lengths = []
        for node in m.estimators_:
            depth = 0
            while node.left is not None:
                value = x[node.feature] if node.feature is not None else x @ node.direction
                node = node.left if value < node.threshold else node.right
                depth += 1
            lengths.append(depth + average_path_length(node.size))
        expected.append(2 ** (-np.mean(lengths) / average_path_length(m.max_samples_)))
    np.testing.assert_allclose(m.anomaly_score(X), expected)
