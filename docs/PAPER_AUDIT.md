# Paper-to-code audit

Audited on 2026-09-22 against the author-supplied final PDFs:

- Yeom & Jung, *Weighted isolation and random cut forests for anomaly detection*, DOI [10.1007/s11634-026-00688-3](https://doi.org/10.1007/s11634-026-00688-3).
- Yeom & Jung, *Drop-in density-aware partitioning for tree-based anomaly detection*, DOI [10.1088/2632-2153/ae5390](https://doi.org/10.1088/2632-2153/ae5390).

Original repository revisions (before this update):

| Repository | Revision |
| --- | --- |
| [isolation-forest-weighted](https://github.com/yeomsijin/isolation-forest-weighted) | `ba3454a686ac1eb66964cad0100cbc29e7202482` |
| [rrcf-weighted-](https://github.com/yeomsijin/rrcf-weighted-) | `f702ea2f752ca865fad0dab34810b039bbcbe0d3` |
| [density-aware-partitioning](https://github.com/yeomsijin/density-aware-partitioning) | `c5bc307320d48534de44b3010b28931ff3e822bc` |

The audit covers the Python implementation, public exports, packaging, figure entry points and WIF notebook code cells. It does not establish that these public files generated the published numerical tables or figures.

## Findings and corrections

| Original location | Finding | Paper basis / correction |
| --- | --- | --- |
| WIF `iso_forest_w.py:make_tree` | Uniform feature selection, epsilon and `count < 2` agree with the core weighted rule. Alpha is hardcoded. | Weighted sections 2 and 5; exposed `WeightedIsolationForest(alpha=2)`. |
| WIF `c_factor`, DAP `harmonic_c_isoforest` | The logarithmic approximation was used even at n=2, giving about 0.154 rather than the exact c(2)=1; WIF also fails on n<=1. | Explicit small-n external-node correction; changes historical scores. |
| WIF notebooks | NumPy is seeded but Python's `random`, which builds the trees, is not. The visualization notebook imports unavailable `iso_forest` and includes a machine-specific Python 2 path. | One estimator RNG; runnable examples replace these workflows. |
| WRCF `wrcf/__init__.py` | Imports `rrcf.rrcf`, not the repository's `wrcf.py`. Installing baseline rrcf can silently expose unweighted code. | Local exports now resolve to the weighted implementation. |
| WRCF `_cut` | Range-weighted feature probabilities and `count < 2` match weighted batch construction. | Preserved, with configurable alpha and common DAS. |
| WRCF constructor | Deprecated `np.int`, `np.bool`, `np.asscalar`; singleton/all-identical batches call a cut with zero range. Duplicate rows are collapsed before counting. | Current NumPy support; singleton root; multiplicities retained for counting and CODISP. |
| WRCF `_insert_point_cut` | Uses ordinary RRCF uniform cuts; it never calls `_cut`. Starting an empty tree and only inserting does not produce weighted splits. | Default updates rebuild a weighted tree; legacy uniform insertion is opt-in. A weighted efficient insertion distribution is not derived in either paper. |
| WRCF `_insert_point_cut` | Allocates `bbox_hat` with the leaf's `(1,d)` shape, so assigning min then max overwrites the same row. A deterministic two-point test yields a cut at the old point and incorrect routing. | Allocate `(2,d)` explicitly; cover incremental tree invariants. |
| DAP `utils.py:window_count_range`, `counting_map.py` | Computes extrema at observed projections only; empty gaps and between-sample maxima can be missed. Closed/tolerance-widened windows differ from Definition 1. | Exact event sweep with half-open windows and grouped tied boundaries. |
| DAP `utils.py:dap_pick_threshold` | After 256 failures, returns an unchecked uniform split. | Same-law accepted-interval fallback, with errors when no usable interval exists. |
| DAP `models/scif.py` | Baseline uses one cube direction without feature-standard-deviation normalization; no tau-candidate comparison. | Section 3.2: tau candidates, normalized features, best Sdgain. DAD selects one direction before gain search. |
| DAP oblique models | Uses all features; no q parameter. | Add `max_features`; q=2 benchmark configurations are explicit. |
| DAP `BaseIForest.fit` | Mutates max_depth on first fit and advances a stored RNG, so refits depend on fit history. | Learned `max_depth_`, fresh RNG for integer seeds, input validation and fitted checks. |
| DAP Fig. 3(a) script | Repeatedly samples a "random" direction until a slope constraint holds. | Default visual comparison now uses one unconditioned random draw. |
| Packaging / README | WIF/WRCF lack a usable root package setup; DAP installs a generic `src` namespace and the README has an unclosed fence. | One wheel, unified public namespace, documented dependencies, examples, automated CI, citations and retained upstream notices. |

## Finite evaluation of the counting map

Author clarification on 2026-09-22: Algorithm 2's extrema are taken over `[min(Y), max(Y)]`. This resolves the domain left implicit in its displayed `max_p - min_p` formula. `range_domain="data"` is the default; `"real"` is only an explicitly requested alternative.

Drop-in Proposition 1(i) gives

```text
d_Y(p) = sum over y in Y of 1_(y-epsilon, y+epsilon](p).
```

Consequently the function is constant between the at most `2n` event locations `y_i ± epsilon`. Sorting these events and applying all changes at each tied location together computes the exact extrema in `O(n log n)` time and `O(n)` memory. At a boundary, the pre-event count applies; immediately to its right, the post-event count applies. We clip intervals to the data range and also evaluate the two data endpoints. This finite reduction follows from Proposition 1(i); it is not a claim that an arbitrary equally spaced epsilon grid finds every extremum. The epsilon=0 singleton rule is handled separately.

DAS uses the same representation. Given acceptable disjoint intervals of lengths L_i, choose interval i with probability L_i / sum L_i and sample uniformly within it. This has the same conditional uniform distribution as rejection sampling, except unavoidable floating-point endpoint conventions. Numerically unrepresentable fallback intervals raise an error rather than returning a rejected value. Exact binary-rational boundary tests and independent direct-window counts cover the implementation.

## Explicit interpretation choices

- **Tolerance:** WIF/WRCF accept integer `count < alpha`, alpha>=2. Drop-in accepts `count <= alpha`. Both default to at most one observation in the window.
- **Multiplicity:** preserve repeated observations in projected data, consistent with Drop-in Definition 1 and the weighted paper's treatment of duplicates in Proposition 3. Original WRCF removed duplicate rows before computing epsilon; this update changes that behavior.
- **EIF DAS:** section 3.3's introductory paragraph describes a projected threshold, while its explicit EIF+DAS procedure samples each coordinate of p. We follow that explicit procedure and the original implementation, without claiming both distributions are equivalent.
- **SCiF DAD:** run Algorithm 2 on raw candidate projections, then maximize Sdgain once. Feature-standard-deviation normalization belongs to the described SCiF baseline. No extra normalization is silently added to DAD. For a q-feature candidate, sample a uniform q-feature subset first, then a sphere direction in that subspace; this subset-selection convention is explicit because the paper does not specify every RNG detail.
- **IF constant features:** preserve section 3.1's stopping rule when the selected feature is constant, rather than silently resampling a varying feature.
- **Scores:** fix c(2), retain the conventional logarithmic harmonic approximation for n>2. Engineering changes can alter old scores even with the same numeric seed.
- **WRCF updates:** rebuilding guarantees that the current window is built with weighted cuts, at higher cost. This is an extension, not a proof of the weighted analogue of RRCF's insertion/deletion distributional invariance. `update_mode="legacy"` preserves uniform incremental insertion for comparison. A rebuild invalidates external Leaf/Branch references.
- **WRCF unseen-row score:** independent augmented-tree rebuilding; this is a documented practical extension. Duplicate queries already present in a tree use the existing leaf CODISP. The forest remains unchanged.
- **Thresholds:** IF variants use anomaly-score 0.5 for `contamination="auto"`, or a training quantile. WRCF uses an explicit contamination quantile (default 0.1); the papers do not define a universal CODISP decision threshold.

## Reproducibility boundary

The weighted PDF has no GitHub link in its 49 pages of extracted text or URI annotations. Its PDF page 47 Data Availability statement refers to dataset sources. The Drop-in PDF page 15 (printed page 14) links to this repository's `tree/main`. The repository URL is therefore retained.

The Drop-in data availability statement describes code for all figures and experiments, but the audited public revision contains only figure 3–6 entry points and supporting code. The complete ADBench local-anomaly generation/preparation pipeline, exact seeds, trained models and benchmark result files are absent. WIF/WRCF's public files likewise do not provide a complete runner for all published weighted-paper experiments.

This update includes corrected library code, figure demos, unit/regression tests, a local profiling script, and an evaluation runner for explicitly prepared train/test arrays. It does **not** certify reproduction of the 46-dataset ADBench benchmark, the weighted-paper ODDS curves, or the GDP/NYC streaming experiments. No paper-level accuracy or runtime claim should be made from the smoke tests. The profiling result compares implementations of the same operation on the same fitted trees or projections, not DAP accuracy against a baseline model.

For a full rerun, recover the authors' original data preparation and seed/configuration records, export fixed train/test arrays, and run matched baselines and variants. Do not substitute a new preprocessing protocol and label it an exact reproduction.
