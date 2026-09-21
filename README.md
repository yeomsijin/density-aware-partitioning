# Density-Aware Forests

Reusable implementations of **Weighted Isolation Forest (WIF)**, **Weighted Random Cut Forest (WRCF)**, and **Density-Aware Partitioning (DAP)** by Sijin Yeom and Jae-Hun Jung.

Density-aware splitting favors sparse gaps over dense regions. DAP exposes this rule as a standalone threshold sampler (**DAS**) and a direction selector (**DAD**), and combines them with IF, EIF and SCiF. All implementations share one counting-map definition and one installation.

| Paper | Methods | Reference |
| --- | --- | --- |
| Weighted isolation and random cut forests for anomaly detection | WIF, WRCF | [Advances in Data Analysis and Classification (2026)](https://doi.org/10.1007/s11634-026-00688-3) |
| Drop-in density-aware partitioning for tree-based anomaly detection | DAS, DAD; IF, EIF, SCiF | [Machine Learning: Science and Technology 7, 025035 (2026)](https://doi.org/10.1088/2632-2153/ae5390) |

The repository URL remains unchanged because the Drop-in paper links here. Earlier WIF/WRCF repositories remain available as historical references. See the [implementation audit](docs/PAPER_AUDIT.md), [API and migration guide](docs/API.md), and [validation scope](docs/VALIDATION.md).

## Installation

Python 3.10 or newer. From a checkout of this repository:

```bash
python -m pip install .
# Optional plotting examples and development tools:
python -m pip install -e ".[dev]"
```

The distribution is named `density-aware-forests`. No PyPI release is implied by this README. NumPy and scikit-learn are runtime dependencies; Matplotlib is optional.

## WIF in a few lines

```python
import numpy as np
from density_aware_forests import WeightedIsolationForest

X = np.random.default_rng(42).normal(size=(512, 4))
model = WeightedIsolationForest(n_estimators=100, max_samples=256, random_state=42)
model.fit(X)
scores = model.anomaly_score(X)  # larger = more anomalous
labels = model.predict(X)        # -1 = anomaly, +1 = inlier
```

The estimators expose `fit`, `anomaly_score`, `score_samples`, `decision_function`, `predict`, `get_params` and `set_params`. `score_samples` is the **negative** of `anomaly_score`, following the scikit-learn direction convention. Fit ignores `y`. Integer seeds reproduce a fit; passing a NumPy Generator consumes its state. Inputs must be finite, dense numeric arrays of shape `(n_samples, n_features)`; impute missing values explicitly before fitting.

## Drop-in threshold and direction rules

For a projection `Y` of `n` observations, define

```text
epsilon = (max(Y) - min(Y)) / (2 * (n - 1))
d_Y(p)  = number of observations in [p - epsilon, p + epsilon)
```

Duplicate values count separately. If epsilon is zero, the window is the singleton `{p}`. A constant projection cannot split a node.

```python
from density_aware_forests import density_aware_split, density_aware_direction

rng = np.random.default_rng(42)
p = density_aware_split(X[:, 0], alpha=1, random_state=rng)
v = density_aware_direction(X, n_candidates=10, max_features=2, random_state=rng)
```

DAS samples uniformly until `d_Y(p) <= alpha`. DAD chooses the candidate direction minimizing `max(d_Y) - min(d_Y)` **over the data range**. Extrema are computed exactly on the finite step-function partition induced by `y_i ± epsilon` (Drop-in Proposition 1(i)), rather than approximated at observed points or on a grid. Time per direction is `O(n log n)`; no `n × k` projection matrix is retained.

After a bounded rejection loop, DAS samples accepted intervals in proportion to their lengths, preserving the conditional uniform distribution. It never substitutes an unchecked split. If no positive-length acceptable interval exists (possible for `alpha < 1`), it raises an error.

**Alpha conventions differ between papers:** WIF/WRCF use integer `alpha >= 2` and accept `count < alpha`; Drop-in DAS uses `count <= alpha`. Therefore **WIF(alpha=2) = IF+DAS(alpha=1)** for matched parameters and seeds.

## Forest variants

```python
from density_aware_forests import DensityAwareForest

model = DensityAwareForest(
    model="eif", das=True, dad=True,
    n_estimators=100, max_samples=256,
    max_features=2, n_directions=2,
    random_state=42,
).fit(X)
```

| Variant | Configuration |
| --- | --- |
| IF | `model="if", das=False` |
| WIF / IF+DAS | `model="if", das=True` |
| EIF | `model="eif", das=False` |
| EIF+DAS | `model="eif", das=True` |
| EIF+DAD | `model="eif", das=False, dad=True` |
| EIF+DAS+DAD | `model="eif", das=True, dad=True` |
| SCiF | `model="scif", das=False` |
| SCiF+DAD | `model="scif", das=False, dad=True` |

SCiF baseline uses `n_hyperplanes` candidates with feature-standard-deviation normalization and chooses the largest standard deviation gain. SCiF+DAD selects one direction using Algorithm 2, then runs one gain search. EIF+DAS samples each coordinate of its translation point, following the explicit procedure in paper section 3.3.

The Drop-in benchmark configuration uses 256 samples, depth 8, 10–100 trees, and five seeds. SCiF uses `n_hyperplanes=10`, `max_features=2`; DAD uses 10 candidates for SCiF and 2 for EIF. Library defaults use all features unless `max_features=2` is supplied.

## WRCF and time series

```python
from density_aware_forests import WeightedRandomCutForest

# Small example: unseen-row scoring rebuilds weighted trees and is more costly.
model = WeightedRandomCutForest(
    n_estimators=10, max_samples=64, random_state=42
).fit(X[:64])
scores = model.anomaly_score(X[:64])  # mean CODISP, larger = more anomalous
```

WRCF chooses a feature with probability proportional to its range and applies the same density-aware threshold rule as WIF. It scores with collusive displacement (CODISP), not IF path length. `WeightedRandomCutTree` also exposes `insert_point`, `forget_point`, `codisp`, and `to_dict`/`from_dict`.

```python
from density_aware_forests import RollingWRCF, shingle

signal = np.sin(np.arange(80) / 5)
stream = RollingWRCF(n_estimators=5, window_size=24, random_state=42)
for t, point in enumerate(shingle(signal, size=4), start=3):
    score = stream.update(point)
```

**Streaming semantics:** the default rebuilds weighted trees on the current window, so new splits remain density-aware. This is a documented engineering extension, with a higher per-update cost than incremental RRCF; it is not a claim of reproducing the published online timing or convergence curves. The historical insertion rule never invoked weighted batch cuts. It is available explicitly as `WeightedRandomCutTree(update_mode="legacy")`, where new online cuts are uniform RRCF cuts. Unseen-row WRCF forest scoring inserts each row into an independent tree copy and rebuilds; scoring does not change fitted trees.

## Examples, experiments and checks

```bash
python examples/quickstart.py
python examples/streaming.py
python -m pytest -q
python -m build
```

[GitHub Actions](.github/workflows/tests.yml) runs automatically on pushes and pull requests. It checks Python 3.10/3.12 across Linux, macOS and Windows, plus the minimum supported NumPy/scikit-learn versions. See the [validation record](docs/VALIDATION.md) for local and remote results.

The original figure entry points now use the corrected core. Their results can differ from historical images:

```bash
python -m scripts.reproduce_fig3a --k 10 --out fig3a.png
python -m scripts.reproduce_fig3b --k 10 --out fig3b.png
python -m scripts.reproduce_fig456 --n-trees 100 --no-high-res --outdir figures
```

For a supplied NPZ containing `X_train`, `X_test`, and binary `y_test` (1 = anomaly):

```bash
python -m scripts.benchmark --data prepared-data.npz --out results.json
```

This runner records ROC-AUC, average precision, fit/scoring time, configuration and environment. It does not silently choose preprocessing or generate ADBench local anomalies. **The 46-dataset paper benchmark has not been reproduced by this update**: the original public repository contains figure demos, but not the complete experiment-generation pipeline, prepared datasets, exact seed list or results. Published performance findings are paper results, not new measurements from this refactor. See [reproduction scope](docs/PAPER_AUDIT.md#reproducibility-boundary).

## Citation and attribution

See [CITATION.cff](CITATION.cff) and [references.bib](references.bib) for both papers. Cite the weighted paper for WIF/WRCF and the Drop-in paper for DAS/DAD. The descriptions here paraphrase the authors' papers; corrections and implementation choices are listed in the audit.

Code is MIT licensed. The RCF tree implementation derives from [kLabUM/rrcf](https://github.com/kLabUM/rrcf), retaining its [MIT notice](docs/licenses/rrcf.txt). Historical WIF was based on [mgckind/iso_forest](https://github.com/mgckind/iso_forest); its [original notice](docs/licenses/iso_forest.txt) is retained. Paper text and figures have their own licenses; the PDFs and benchmark datasets are not redistributed here.
