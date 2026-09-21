"""Evaluate explicit train/test arrays; this does not generate the ADBench protocol.

NPZ keys: X_train, X_test, y_test (1=anomaly, 0=normal).
No implicit scaling, synthetic anomaly generation, or train/test splitting.
"""

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

from dap import DensityAwareForest


def configurations():
    return {
        "IF": dict(model="if", das=False),
        "IF+DAS": dict(model="if", das=True),
        "EIF": dict(model="eif", das=False, max_features=2),
        "EIF+DAS": dict(model="eif", das=True, max_features=2),
        "EIF+DAD": dict(model="eif", das=False, dad=True, n_directions=2, max_features=2),
        "EIF+DAS+DAD": dict(model="eif", das=True, dad=True, n_directions=2, max_features=2),
        "SCiF": dict(model="scif", das=False, n_hyperplanes=10, max_features=2),
        "SCiF+DAD": dict(model="scif", das=False, dad=True, n_directions=10, max_features=2),
    }


def run(data, tree_counts, seeds, models):
    with np.load(data, allow_pickle=False) as arrays:
        X_train, X_test, y = (arrays[key] for key in ("X_train", "X_test", "y_test"))
    if y.ndim != 1 or len(y) != len(X_test) or set(np.unique(y)) != {0, 1}:
        raise ValueError("y_test must be a 1D vector of 0/1 labels with both classes")
    rows = []
    for name in models:
        for n in tree_counts:
            for seed in seeds:
                model = DensityAwareForest(
                    **configurations()[name],
                    n_estimators=n,
                    max_samples=256,
                    max_depth=8,
                    alpha=1,
                    random_state=seed,
                )
                start = perf_counter()
                model.fit(X_train)
                fitted = perf_counter()
                scores = model.anomaly_score(X_test)
                scored = perf_counter()
                rows.append(
                    dict(
                        model=name,
                        trees=n,
                        seed=seed,
                        roc_auc=roc_auc_score(y, scores),
                        average_precision=average_precision_score(y, scores),
                        fit_seconds=fitted - start,
                        score_seconds=scored - fitted,
                    )
                )
    return dict(
        input=str(data),
        python=platform.python_version(),
        numpy=np.__version__,
        sklearn=sklearn.__version__,
        platform=platform.platform(),
        protocol="user-supplied train/test arrays; not a certified ADBench reproduction",
        results=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trees", type=int, nargs="+", default=list(range(10, 101, 10)))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    parser.add_argument(
        "--models", nargs="+", choices=list(configurations()), default=list(configurations())
    )
    args = parser.parse_args()
    result = run(args.data, args.trees, args.seeds, args.models)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
