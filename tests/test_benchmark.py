import numpy as np

from scripts.benchmark import configurations, run


def test_benchmark_records_all_configurations(tmp_path):
    rng = np.random.default_rng(8)
    path = tmp_path / "data.npz"
    np.savez(
        path,
        X_train=rng.normal(size=(24, 2)),
        X_test=np.r_[rng.normal(size=(5, 2)), rng.normal(5, 1, size=(5, 2))],
        y_test=np.r_[np.zeros(5), np.ones(5)],
    )
    result = run(path, [2], [0], list(configurations()))
    assert len(result["results"]) == 8
    assert {row["model"] for row in result["results"]} == set(configurations())
    assert all(0 <= row["roc_auc"] <= 1 for row in result["results"])
