"""Run: python examples/quickstart.py"""

import numpy as np

from density_aware_forests import DensityAwareForest, WeightedIsolationForest

X = np.random.default_rng(42).normal(size=(256, 4))
queries = np.array([[0, 0, 0, 0], [8, 8, 8, 8.0]])
models = {
    "WIF": WeightedIsolationForest(n_estimators=20, random_state=42),
    "EIF+DAS+DAD": DensityAwareForest(
        model="eif",
        das=True,
        dad=True,
        n_estimators=20,
        n_directions=2,
        max_features=2,
        random_state=42,
    ),
    "SCiF+DAD": DensityAwareForest(
        model="scif",
        das=False,
        dad=True,
        n_estimators=20,
        n_directions=10,
        max_features=2,
        random_state=42,
    ),
}
for name, model in models.items():
    model.fit(X)
    print(name, model.anomaly_score(queries), model.predict(queries))
