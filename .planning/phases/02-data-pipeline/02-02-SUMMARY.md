---
phase: 02-data-pipeline
plan: 02
subsystem: data
tags: [yfinance, pandas, parquet, sp500, ohlcv, pipeline]

# Dependency graph
requires:
  - phase: 02-01
    provides: test scaffold with sp500_ohlcv_fixture, xfail stubs for DATA-01

provides:
  - data_processing_script/sp500_pipeline/download_ohlcv.py with get_sp500_tickers(), download_ohlcv_batched(), clean_and_align(), main()
  - data_processing_script/sp500_pipeline/__init__.py package marker
  - DATA-01 unit tests passing (test_download_parquet_schema, test_clean_no_all_nan_rows)

affects: [02-03-feature-engineering, 02-04-normalization, 02-05-npz-arrays, all downstream pipeline steps]

# Tech tracking
tech-stack:
  added: [yfinance (guarded import), pandas.read_html for Wikipedia scraping, pd.to_parquet]
  patterns: [guarded import with clear ImportError message, master-calendar alignment via reindex+ffill, chunk+retry download pattern]

key-files:
  created:
    - data_processing_script/sp500_pipeline/__init__.py
    - data_processing_script/sp500_pipeline/download_ohlcv.py
  modified:
    - tests/test_data_pipeline.py

key-decisions:
  - "download_ohlcv.py named without leading digit (not 01_download_ohlcv.py) — Python module names cannot start with a digit"
  - "yfinance import guarded with try/except so module imports without yfinance installed (tests run offline)"
  - "Master calendar derived from ticker with most dates (typically full-history large-cap) rather than hardcoded SPY"
  - "BAD_STOCK test case uses 30 consecutive NaN rows so ffill(limit=5) leaves 25 rows NaN (~8.3%) above 5% threshold"

patterns-established:
  - "Guarded optional imports: try/except with None sentinel and clear ImportError on use"
  - "Batched download with exponential backoff: chunk_size=80, 3 retries, delay*(attempt+1) sleep"
  - "Clean-and-align pattern: reindex to master calendar, ffill(limit=5), drop >5% missing"

requirements-completed: [DATA-01]

# Metrics
duration: 5min
completed: 2026-03-11
---

# Phase 2 Plan 02: OHLCV Download Pipeline Summary

**S&P 500 OHLCV downloader via yfinance with batched retry, master-calendar alignment, and Parquet-per-ticker output — DATA-01 unit tests passing**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-11T09:56:40Z
- **Completed:** 2026-03-11T10:01:00Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- `download_ohlcv.py` implemented with all four required functions: `get_sp500_tickers()`, `download_ohlcv_batched()`, `clean_and_align()`, `main()`
- CLI accepts `--data_dir`, `--start`, `--end`; saves one Parquet per ticker and `tickers.txt`
- DATA-01 unit tests (`test_download_parquet_schema`, `test_clean_no_all_nan_rows`) converted from xfail stubs to passing green tests
- yfinance import guarded so all module imports and tests work without yfinance installed

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement download_ohlcv.py** - `e64078c` (feat)
2. **Task 2: Implement DATA-01 test cases** - `83ac273` (feat)

_Note: Both tasks used TDD flow (RED baseline verified, then GREEN implementation)_

## Files Created/Modified

- `data_processing_script/sp500_pipeline/__init__.py` - Package marker (empty)
- `data_processing_script/sp500_pipeline/download_ohlcv.py` - Full OHLCV pipeline module (259 lines)
- `tests/test_data_pipeline.py` - DATA-01 stubs replaced with real test implementations

## Decisions Made

- Module named `download_ohlcv.py` not `01_download_ohlcv.py` — Python module names cannot start with a digit; the plan explicitly required this naming convention.
- yfinance import guarded with `try/except` and `None` sentinel: allows the module to import cleanly in test/offline environments; raises a descriptive `ImportError` only when download is actually attempted.
- Master calendar is derived from the ticker with the most dates (not hardcoded to SPY) to be robust if SPY is missing or has a gap-heavy period.
- Test for bad ticker uses 30 consecutive NaN rows: random scatter of 30 NaN rows is fully recovered by `ffill(limit=5)` because max run is only 3; consecutive block ensures 25 rows remain NaN (~8.3%) above the 5% threshold.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed BAD_STOCK test case NaN strategy**
- **Found during:** Task 2 (test_clean_no_all_nan_rows)
- **Issue:** Plan called for 10% NaN rows introduced via `np.random.choice(300, 30)`. With seed=0, all 30 NaN rows form runs of max length 3, which `ffill(limit=5)` fully recovers — BAD_STOCK survived cleaning instead of being dropped.
- **Fix:** Changed NaN injection to 30 consecutive rows (iloc[50:80]) so ffill(limit=5) recovers only the first 5, leaving 25 rows (~8.3%) NaN above the 5% threshold.
- **Files modified:** tests/test_data_pipeline.py
- **Verification:** `pytest tests/test_data_pipeline.py::test_clean_no_all_nan_rows` passes
- **Committed in:** 83ac273 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug in test fixture design)
**Impact on plan:** Fix ensures the test actually validates the cleaning threshold as intended. No scope creep.

## Issues Encountered

None beyond the NaN fixture issue documented above.

## User Setup Required

None - no external service configuration required at this stage. yfinance download requires internet access when `main()` is actually run, but no credentials needed.

## Next Phase Readiness

- `clean_and_align()` is ready for use by downstream feature engineering (Plan 02-03)
- `tickers.txt` pattern established: one ticker per line, written by `main()`
- Parquet schema established: DatetimeIndex, columns [Open, High, Low, Close, Volume]
- DATA-02 through DATA-05 tests remain xfail, ready for their respective implementation plans

## Self-Check: PASSED

- FOUND: data_processing_script/sp500_pipeline/__init__.py
- FOUND: data_processing_script/sp500_pipeline/download_ohlcv.py
- FOUND: .planning/phases/02-data-pipeline/02-02-SUMMARY.md
- FOUND commit: e64078c (Task 1)
- FOUND commit: 83ac273 (Task 2)

---
*Phase: 02-data-pipeline*
*Completed: 2026-03-11*
