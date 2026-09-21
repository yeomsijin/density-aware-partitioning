# Validation record

Local validation on 2026-09-22 used Python 3.12.14, NumPy 2.5.3 and scikit-learn 1.9.1 on macOS arm64. The [GitHub Actions template](ci/github-actions.yml) additionally defines Python 3.10/3.12 jobs across Linux, macOS and Windows, plus a NumPy 1.24.4 / scikit-learn 1.4.2 minimum-dependency job. The current Git credential lacks workflow-write scope, so this file is stored as an inactive template. To enable CI, place it at `.github/workflows/tests.yml` using an account/credential with that permission. No remote CI run is claimed.

## Mathematical and regression checks

The test suite checks:

- Half-open counting windows, tied events, duplicate multiplicities, constant/singleton projections and affine transforms on exactly representable examples.
- Event-sweep extrema against independent brute-force window counts at every boundary and interval midpoint; cases where observed-point-only evaluation misses an empty gap or a maximum.
- Rejection and forced interval-fallback samples satisfy the acceptance criterion and the known conditional uniform distribution for `{0,1,2,10,11,12}`.
- DAD selects the minimum-range direction among the identical seeded candidate set, using the author's data-range interpretation.
- Default WIF and IF+DAS produce exactly identical scores with matched seeds and mapped alpha values.
- All eight IF/EIF/SCiF variants, c(2), duplicate/constant input, input errors, fitted-state checks, reproducible refits, clone, Pipeline and pickle.
- Vectorized Sdgain against direct standard-deviation calculations; SCiF's tau-candidate normalization/search; batched inference against independent scalar traversal.
- WRCF cuts, routing, subtree mass, bounding boxes, parent pointers, leaf depths, duplicate labels, singleton/empty trees, insertion/deletion and JSON structure round trips.
- WRCF unseen scoring preserves fitted trees and does not depend on query order; rolling windows stay bounded and shingles align with time indices.
- The explicit-array benchmark runner exercises all eight configurations and records metrics.

## Execution checks

- **48 tests pass** both from the source checkout and from an independently installed wheel in a separate environment. `python -m pytest -q` and Ruff on the new library, tests, examples and benchmark/profile runners.
- Both `examples/quickstart.py` and `examples/streaming.py` run successfully.
- All three original figure entry points run with the corrected adapters. Figures 4–6 use reduced-resolution settings for smoke testing; generated images were visually inspected. This is not a pixel-level reproduction of the published figures.
- An eight-variant benchmark smoke run on generated train/test arrays succeeds. Those arrays are not the paper benchmark.
- Source distribution and universal Python wheel build successfully; the wheel includes the three method families and upstream license notices.

## Measured optimization

[`profile-results.json`](profile-results.json) contains the measurements and environment. Re-run:

```bash
python -m scripts.profile_core --out docs/profile-results.json
```

In the recorded small benchmark (25 IF+DAS trees, subsample 128, 2,000 query rows with four features), batched scoring took about **5.93 ms**, versus **74.56 ms** for equivalent scalar traversal, approximately **12.6x** faster. Both paths are checked against each other before timing.

For 256 projected observations, vectorized split-gain search took about **44.4 microseconds**, versus **194.0 microseconds** for the actual original `c5bc307` prefix-sum Python loop, approximately **4.4x** faster. The selected split and gain agree on the profiled data.

These are medians of three local runs after warm-up, not confidence intervals or guarantees. They compare equivalent operations, not full old-vs-new training pipelines. Exact DAD extrema do more complete work than the old approximation; corrected SCiF baseline searches more candidates. WRCF rebuilding costs more than uniform incremental insertion. No claim of universally lower end-to-end runtime follows from these microbenchmarks.

## Not validated as reproduced results

The 46 ADBench local-anomaly datasets, weighted-paper ODDS curves, GDP and NYC taxi convergence experiments have not been rerun. The original public repositories lack the full generation/preprocessing/configuration/result assets needed to certify that reproduction. See [PAPER_AUDIT.md](PAPER_AUDIT.md#reproducibility-boundary).
