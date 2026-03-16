---
phase: 02-data-pipeline
plan: 06
subsystem: data
tags: [feature-engineering, technical-analysis, pandas, numpy, ta-features]

# Dependency graph
requires:
  - phase: 02-data-pipeline
    provides: "compute_features() with 16-column TA feature baseline from plan 02-03"
provides:
  - "compute_features() producing 69 pure-pandas TA features per ticker"
  - "6 new private helpers: _atr, _stochastic, _williams_r, _cci, _donchian, _momentum"
  - "test_feature_count_for_phase3 asserting >= 60 features (Phase 3 prerequisite gate)"
affects:
  - 03-model-training
  - build_pipeline.py feature CSV output

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "All TA indicators implemented using only numpy + pandas rolling/ewm — no pandas_ta or TA-Lib"
    - "New feature groups added after existing VOL_ratio block, before return statement"
    - "pandas_ta importorskip guards retained as safety no-ops (pandas_ta not installed in Python 3.9 venv)"

key-files:
  created: []
  modified:
    - data_processing_script/sp500_pipeline/feature_engineering.py
    - tests/test_data_pipeline.py

key-decisions:
  - "69 features chosen (target ~69): hits Phase 3 prerequisite of >= 60 with headroom"
  - "pandas_ta importorskip guards retained unchanged — removing is optional but would break nothing; keeping avoids unplanned churn"
  - "test_feature_count_for_phase3 has no importorskip guard — it validates pure-pandas impl directly"

patterns-established:
  - "Feature expansion pattern: add private helpers after _cross_sectional_normalize(), extend features dict before return"
  - "Phase prerequisite gate pattern: dedicated test asserting >= N columns makes Phase 3 readiness machine-verifiable"

requirements-completed: [DATA-02]

# Metrics
duration: 2min
completed: 2026-03-11
---

# Phase 02 Plan 06: TA Feature Expansion Summary

**compute_features() expanded from 16 to 69 pure-pandas TA indicators (ATR, Stochastic, Williams %R, CCI, Donchian, OBV, MACD signal/hist, EMA/SMA, Momentum, and more) satisfying Phase 3 Alpha-360 replacement prerequisite**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-11T15:32:05Z
- **Completed:** 2026-03-11T15:34:24Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Expanded `compute_features()` from 16 to 69 columns using only numpy + pandas (no external deps)
- Added 6 new private helper functions: `_atr`, `_stochastic`, `_williams_r`, `_cci`, `_donchian`, `_momentum`
- Added `test_feature_count_for_phase3` — machine-verifiable Phase 3 prerequisite (>= 60 features required to replace Qlib Alpha-360)
- Updated `test_feature_columns_present` expected list from 16 to all 69 columns
- Zero all-NaN columns after 60-row warmup; full test suite green (11 passed, 2 skipped)

## Task Commits

Each task was committed atomically:

1. **Task 1: Expand compute_features() to ~69 pure-pandas TA features** - `9e07151` (feat)
2. **Task 2: Update DATA-02 tests for expanded feature set** - `b14adfa` (test)

**Plan metadata:** (docs commit pending)

_Note: TDD tasks — RED phase confirmed (16 cols < 60 threshold), GREEN after implementation_

## Files Created/Modified
- `data_processing_script/sp500_pipeline/feature_engineering.py` - Added 6 helpers + 53 new feature columns; docstring updated to "~69 feature columns"
- `tests/test_data_pipeline.py` - Updated test_feature_columns_present (69-col list), added test_feature_count_for_phase3

## Decisions Made
- Retained `pytest.importorskip("pandas_ta")` guard in existing tests — it's a no-op safety guard; removing it would be optional churn with no benefit
- `test_feature_count_for_phase3` has no importorskip — validates pure-pandas path directly, will run in every CI environment
- Used 69 features (plan target ~69) — hits Phase 3 >= 60 prerequisite with 15% headroom

## Deviations from Plan
None - plan executed exactly as written.

## Issues Encountered
None - implementation proceeded without any blocking issues or surprises.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Phase 3 prerequisite satisfied: `compute_features()` now returns 69 features (>= 60 required)
- `save_feature_csvs()` will produce 69 CSV files in `features/` on next `build_pipeline.py` run
- `StockDataset.infea = bonus_all.shape[-1] + 2 = 71` will be computed dynamically — no manual config edits needed
- Struc2Vec graph embeddings still need regeneration for full S&P500 universe (tracked as open blocker)

## Self-Check: PASSED
- feature_engineering.py: FOUND
- test_data_pipeline.py: FOUND
- 02-06-SUMMARY.md: FOUND
- Task 1 commit 9e07151: FOUND
- Task 2 commit b14adfa: FOUND

---
*Phase: 02-data-pipeline*
*Completed: 2026-03-11*
