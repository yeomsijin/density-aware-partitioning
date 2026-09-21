import json

import numpy as np
import pytest
from sklearn.base import clone

from dap import counting_map
from wrcf import Leaf, RCTree, RollingWRCF, WeightedRandomCutForest, shingle


def validate(tree):
    if not tree.leaves:
        assert tree.root is None
        return

    def walk(node, parent, depth):
        assert node.u is parent
        if isinstance(node, Leaf):
            assert node.d == depth
            assert sum(leaf is node for leaf in tree.leaves.values()) == node.n
            return np.repeat(node.x[None, :], node.n, axis=0)
        left = walk(node.l, node, depth + 1)
        right = walk(node.r, node, depth + 1)
        assert np.all(left[:, node.q] <= node.p)
        assert np.all(right[:, node.q] > node.p)
        X = np.concatenate([left, right])
        assert node.n == len(X)
        np.testing.assert_array_equal(node.b, np.array([X.min(0), X.max(0)]))
        return X

    assert len(walk(tree.root, None, 0)) == len(tree.leaves)


def test_export_really_is_weighted_and_cuts_match_counts():
    assert RCTree.__module__ == "wrcf.wrcf"
    y = np.array([0, 1, 2, 10, 11, 12.0])
    for seed in range(20):
        t = RCTree(y[:, None], random_state=seed)
        assert counting_map(y, t.root.p) < 2
        validate(t)


def test_rebuild_insert_delete_duplicate_and_roundtrip():
    t = RCTree(
        [[0, 0], [0, 0], [1, 2], [3, 4]], index_labels=["a", "b", "c", "d"], random_state=8, alpha=3
    )
    assert t.leaves["a"] is t.leaves["b"]
    validate(t)
    for label, point in [("e", [4, 2]), ("f", [4, 2]), ("g", [-8, 20])]:
        t.insert_point(point, label)
        validate(t)
        assert np.isfinite(t.codisp(label))
    restored = RCTree.from_dict(json.loads(json.dumps(t.to_dict())))
    assert restored.alpha == 3
    validate(restored)
    for label in list(t.leaves):
        t.forget_point(label)
        validate(t)
    restored = RCTree.from_dict(t.to_dict())
    validate(restored)
    t.insert_point([1, 2], "new")
    validate(t)


def test_single_and_identical_input():
    for X in [[[1, 2]], [[1, 2], [1, 2], [1, 2]]]:
        t = RCTree(X)
        validate(t)
        assert t.codisp(0) == 0
    for X in [[], [[np.nan, 0]], np.ones(3)]:
        with pytest.raises(ValueError):
            RCTree(X)
    t = RCTree([[0, 1]])
    with pytest.raises(KeyError):
        t.insert_point([2, 3], 0)
    with pytest.raises(ValueError):
        t.insert_point([2], 1)
    with pytest.raises(ValueError):
        t.insert_point([np.nan, 3], 1)


def test_legacy_incremental_structure_still_works():
    t = RCTree(random_state=11, update_mode="legacy")
    X = np.random.default_rng(2).normal(size=(15, 2))
    for i, x in enumerate(X):
        t.insert_point(x, i)
        validate(t)
    for i in range(len(X)):
        t.forget_point(i)
        validate(t)


def test_forest_scores_do_not_mutate_training_trees():
    X = np.array([[0, 0], [0, 0], [1, 2], [3, 4], [5, 6.0]])
    m = WeightedRandomCutForest(n_estimators=3, max_samples=5, random_state=3).fit(X)
    before = [t.to_dict() for t in m.estimators_]
    query = np.array([[8, 9], [1, 2.0]])
    a = m.anomaly_score(query)
    np.testing.assert_array_equal(a, m.anomaly_score(query))
    np.testing.assert_array_equal(a, m.anomaly_score(query[::-1])[::-1])
    assert before == [t.to_dict() for t in m.estimators_]
    np.testing.assert_array_equal(a, clone(m).fit(X).anomaly_score(query))


def test_rolling_window_and_shingle_alignment():
    np.testing.assert_array_equal(
        list(shingle(range(6), 3)), [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]]
    )
    np.testing.assert_array_equal(list(shingle([[1, 2], [3, 4]], 2)), [[1, 2, 3, 4]])
    stream = RollingWRCF(window_size=5, n_estimators=2, random_state=3)
    for i, point in enumerate(shingle(range(12), 3)):
        assert np.isfinite(stream.update(point))
        assert all(len(t.leaves) == min(i + 1, 5) for t in stream.estimators_)
        for tree in stream.estimators_:
            validate(tree)
