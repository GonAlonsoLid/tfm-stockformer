---
phase: 02-data-pipeline
plan: 01
subsystem: testing
tags: [pytest, fixtures, numpy, pandas, xfail, ohlcv, test-scaffold]

# Dependency graph
requires:
  - phase: 01-infrastructure
    provides: conftest.py with Phase 1 fixtures, project environment verified
provides:
  - "tests/conftest.py Phase 2 fixtures: sp500_ohlcv_fixture, ohlcv_wide_fixture, feature_matrix_fixture"
  - "tests/test_data_pipeline.py with 10 xfail stubs covering DATA-01 through DATA-05"
  - "Automated verify target (pytest tests/test_data_pipeline.py) for all Phase 2 implementation plans"
affects: [02-02, 02-03, 02-04, 02-05]

# Tech tracking
tech-stack:
  added: [numpy, pandas (as test fixture dependencies)]
  patterns: [xfail-stub pattern for test-first scaffolding, synthetic OHLCV fixture with random walk Close prices]

key-files:
  created:
    - tests/test_data_pipeline.py
  modified:
    - tests/conftest.py

key-decisions:
  - "xfail stubs use strict=False so the suite can collect and report without failing until implementation commits land"
  - "sp500_ohlcv_fixture uses np.random.seed(42) and cumulative exp random walk for reproducible Close prices"
  - "feature_matrix_fixture shape [280, 5] reflects 300 days minus 20-day warmup for longest TA window"

patterns-established:
  - "Wave 0 scaffolding: create test stubs before implementation so every subsequent task has an automated verify target"
  - "Synthetic fixture pattern: seed-controlled numpy/pandas DataFrames as pytest fixtures in conftest.py"

requirements-completed: [DATA-01, DATA-02, DATA-03, DATA-04, DATA-05]

# Metrics
duration: 1min
completed: 2026-03-11
---

# Phase 2 Plan 01: Test Scaffold Summary

**pytest xfail scaffold for DATA-01 through DATA-05 with synthetic S&P500 OHLCV fixtures in conftest.py, giving all Phase 2 implementation plans an automated verify target from commit zero**

## Performance

- **Duration:** ~1 min
- **Started:** 2026-03-11T09:53:19Z
- **Completed:** 2026-03-11T09:54:30Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Extended conftest.py with three Phase 2 fixtures (sp500_ohlcv_fixture, ohlcv_wide_fixture, feature_matrix_fixture) while keeping Phase 1 fixtures intact
- Created tests/test_data_pipeline.py with 10 xfail stubs covering every DATA requirement (DATA-01 through DATA-05)
- pytest collection succeeds: 10 xfailed, 0 errors, 0 collection failures

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend conftest.py with Phase 2 fixtures** - `0b51051` (feat)
2. **Task 2: Create test_data_pipeline.py with xfail stubs** - `e441f7c` (test)

## Files Created/Modified

- `tests/conftest.py` - Added sp500_ohlcv_fixture (5 tickers x 300 business days), ohlcv_wide_fixture (wide Close pivot), feature_matrix_fixture ([280, 5] ndarray)
- `tests/test_data_pipeline.py` - 10 xfail stubs: test_download_parquet_schema, test_clean_no_all_nan_rows, test_feature_columns_present, test_feature_no_all_nan_columns, test_cross_sectional_normalization, test_no_normalization_leakage, test_split_ratios, test_npz_shapes_no_nan, test_trend_indicator_binary, test_graph_embedding_shape

## Decisions Made

- Used `strict=False` on all xfail markers so they report as xfailed (not fail) until implementation replaces the stub `pytest.xfail()` call
- feature_matrix_fixture shape [280, 5] chosen to reflect 300 business days minus 20-day warmup from the longest TA indicator window

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All 10 verify targets for DATA-01 through DATA-05 are in place
- Plans 02-02 through 02-05 can each run `pytest tests/test_data_pipeline.py -x -q` to confirm their implementations pass
- Phase 1 tests (test_phase1_infra.py) remain unchanged; pre-existing torch/pywavelets failures are environment-level (packages not installed in test runner), not regressions from this plan

---
*Phase: 02-data-pipeline*
*Completed: 2026-03-11*
