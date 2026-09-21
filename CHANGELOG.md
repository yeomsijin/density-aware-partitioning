# Changelog

## 0.2.0 — integration update

- Integrate WIF, WRCF and DAP under `density_aware_forests` in one installable distribution.
- Correct counting-map extrema, half-open boundaries, degenerate projections, c(2), DAS fallback, SCiF candidate search, feature normalization and feature subsampling.
- Make integer-seeded refits reproducible; add parameter/input validation and batched IF-family scoring.
- Fix weighted tree exports, current NumPy compatibility, duplicate/singleton handling and the one-row bounding-box insertion bug.
- Add explicit weighted rebuild updates and rolling-window stream support; retain uniform incremental insertion as an opt-in legacy mode.
- Add audit, API migration, citations, provenance, tests, a CI template, examples and explicit-array benchmark runner.
- Preserve original figure commands through adapters; corrected results are not promised to match historical pixels or benchmark values.
