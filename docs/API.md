# API and migration

## One installation, one public import

```python
from density_aware_forests import (
    WeightedIsolationForest, WeightedRandomCutForest, WeightedRandomCutTree,
    DensityAwareForest, RollingWRCF, shingle,
    counting_map, counting_map_range, density_measure,
    density_aware_split, density_aware_direction,
)
```

The implementation subpackages `dap`, `wif`, and `wrcf` remain importable. There is no dependency on the unrelated baseline `rrcf` package.

## Estimators

| Parameter | Meaning |
| --- | --- |
| `n_estimators=100` | Number of independently sampled trees |
| `max_samples=256` | Per-tree sample cap, clipped to available rows; at least 2 |
| `random_state=None` | Integer for reproducible fits, Generator for stateful sampling |
| `alpha` | WIF/WRCF default 2 (`count < alpha`); DAP default 1 (`count <= alpha`) |
| `contamination` | Decision threshold calibration; never a construction weight |

WIF and DAP also expose `max_depth=None` (ceil(log2(actual sample count))). DAP accepts `max_samples="auto"` as a 256-row cap. DAP oblique parameters are `max_features=None`, `n_directions=10`, `n_hyperplanes=10` and `range_domain="data"`; see the README configuration table. A q larger than the input dimension is an error, so use `max_features=None` or 1 for one-dimensional inputs.

Input must be a finite numeric 2D array; sparse matrices, NaNs and infinities are rejected. There is no sample-weight, categorical encoding, automatic imputation, parallel fit, or automatic standardization API. Duplicate observations are retained. DataFrames can be converted to arrays; column names/reordering are not tracked, so preserve feature order at prediction.

| Method | Meaning |
| --- | --- |
| `fit(X, y=None)` | Train; return self; labels are ignored |
| `anomaly_score(X)` | Higher means more anomalous; IF score in (0,1], WRCF mean CODISP >=0 |
| `score_samples(X)` | Negative anomaly score; higher means more normal |
| `decision_function(X)` | `score_samples(X) - offset_` |
| `predict(X)` | -1 when decision_function<0, otherwise +1 |
| `fit_predict(X)` | Fit and predict via scikit-learn OutlierMixin |
| `get_params` / `set_params` | scikit-learn BaseEstimator parameter interface |

WIF/DAP `contamination="auto"` uses `offset_=-0.5`; a float in (0,0.5] uses a training-score quantile. WRCF requires a float and defaults to 0.1. Quantile ties can yield fewer flagged points than the requested fraction. Scores are not calibrated anomaly probabilities. Clone and Pipeline integration are tested; this is not a claim that every optional scikit-learn estimator check or metadata feature is supported.

## Tree and stream APIs

```python
from density_aware_forests import WeightedRandomCutTree

tree = WeightedRandomCutTree([[0.0], [1.0], [10.0]], random_state=42)
tree.insert_point([20.0], index="new")
score = tree.codisp("new")
tree.forget_point("new")
```

- Labels must be unique and hashable. Use JSON-compatible labels for `to_dict` + JSON. `from_dict` restores structure and alpha/update settings, but JSON does not checkpoint RNG state. Python pickle checkpoints the full object; only load pickles you trust.
- `precision=None` preserves values; an integer rounds both initial and inserted observations. Multiple labels may share a duplicate leaf; leaf mass reflects all of them.
- Default `update_mode="rebuild"` constructs a fresh weighted tree after every insertion/deletion. Old Leaf/Branch references become stale; use labels between updates. Empty and singleton trees are supported.
- `update_mode="legacy"` opts into uniform RRCF insertion and structural deletion. It is not a weighted online algorithm. Optional insertion `tolerance` is only supported in this mode.
- `RollingWRCF(window_size=256, n_estimators=40, alpha=2, random_state=None).update(point)` retains the latest window and returns the newly appended point's mean CODISP. It rebuilds once per update, uses bounded storage, and accepts already-shingled rows.
- `shingle(sequence, size)` yields flattened oldest-to-newest windows. For univariate time index 0, the first yielded window ends at `size-1`. Scalar and vector streams are supported with consistent shape.

WRCF forest scoring of unseen rows is deliberately conservative: it rebuilds a separate augmented tree per query and tree. It can be expensive for large test batches. Start with small subsamples/windows and measure runtime. No speedup claim is made for this operation. Low-level tree building and some inherited traversal/serialization routines are recursive; very deep adversarial trees can reach Python's recursion limit.

## Historical API migration

| Old call | New call |
| --- | --- |
| `iso.iForest(X, ntrees=100, sample_size=256)` | `WeightedIsolationForest(n_estimators=100, max_samples=256).fit(X)` |
| `forest.compute_paths(X_in=X)` | `forest.anomaly_score(X)` |
| `wrcf.RCTree(X)` | `WeightedRandomCutTree(X)` (now actually weighted exports) |
| `src.models.iforest.IsolationForestAD(mode="IF+DAS", ...)` | `DensityAwareForest(model="if", das=True, ...)` |
| `src.models.eif.ObliqueIForest(mode="EIF+DAS+DAD", ...)` | `DensityAwareForest(model="eif", das=True, dad=True, ...)` |
| `src.models.scif.SCiForest(mode="SCIF+DAD", ...)` | `DensityAwareForest(model="scif", das=False, dad=True, ...)` |
| `random_seed` | `random_state` |
| `n_dir_candidates` | `n_directions` |

The `src.models` names are narrow adapters for the supplied figure scripts, not the recommended parameter-tuning API. Internal historical Node structures and `Trees`/`trees` attributes are not preserved. New models expose `estimators_`. Exact old random sequences, incorrect boundary behavior, unchecked fallbacks and c(2) scores are intentionally not preserved; use the original Git revisions listed in the audit for historical inspection.
