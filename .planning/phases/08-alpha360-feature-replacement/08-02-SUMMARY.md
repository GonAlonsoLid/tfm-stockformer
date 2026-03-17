---
phase: 08-alpha360-feature-replacement
plan: "02"
subsystem: data-pipeline
tags: [alpha360, feature-engineering, pandas, parquet, csv, cross-sectional-zscore]

requires:
  - phase: 08-01
    provides: test scaffold (5 xfail stubs for ALPHA360-01..05)
  - phase: 02-data-pipeline
    provides: OHLCV Parquet files in data/Stock_SP500_2018-01-01_2024-01-01/ohlcv/

provides:
  - scripts/build_alpha360.py — single-file builder that reads OHLCV Parquets and writes 360 Alpha360-style feature CSVs
  - 360 feature CSVs (on Kaggle after running the script): CLOSE/OPEN/HIGH/LOW/VWAP/VOL x lags 1..60
affects:
  - Phase 03-model-training (retraining required after feature replacement; infea auto-detects to 362)
  - Phase 04-evaluation (IC should improve from -0.003 after retraining)

tech-stack:
  added: []
  patterns:
    - "6-field x 60-lag Alpha360-style feature pipeline: price/volume ratios with cross-sectional z-score"
    - "Zero-safe division: replace(0, NaN) before dividing to prevent silent inf"
    - "Lag buffer slice: iloc[60:] aligns output to first valid date after max lag"
    - "Context-manager-compatible tqdm fallback for environments without tqdm installed"

key-files:
  created:
    - scripts/build_alpha360.py
  modified: []

key-decisions:
  - "tqdm fallback implemented as class with __enter__/__exit__ for context manager compatibility — plain function fallback cannot be used with 'with tqdm(...) as pbar:' pattern"
  - "features_dir cleared after backup before writing new CSVs — otherwise old 3/69 dummy CSVs remain and CSV count exceeds 360"
  - "LAG_BUFFER = 60 constant used instead of literal 60 for readability; behavior identical"

patterns-established:
  - "Backup-then-clear pattern: shutil.copytree to backup dir, then os.remove each file in features_dir before writing new ones"
  - "load_wide() enforces ticker column order from tickers.txt list: pd.DataFrame(frames)[tickers]"
  - "zscore_rows() uses axis=1 (cross-sectional) not axis=0 (time-series) to avoid look-ahead leakage"

requirements-completed: [ALPHA360-01, ALPHA360-02, ALPHA360-03, ALPHA360-04, ALPHA360-05]

duration: 5min
completed: 2026-03-17
---

# Phase 8 Plan 02: Alpha360 Feature Builder Summary

**256-line Alpha360 builder (scripts/build_alpha360.py) implementing 6-field x 60-lag price/volume ratio features with cross-sectional z-score, zero-safe division, lag buffer slice, and backup-before-overwrite; all 5 ALPHA360 tests convert from XFAIL to XPASS.**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-17T00:28:27Z
- **Completed:** 2026-03-17T00:32:22Z
- **Tasks:** 1 (TDD: RED already confirmed in 08-01, GREEN implemented here)
- **Files modified:** 1 created

## Accomplishments

- Implemented full Alpha360 pipeline: 6 fields (CLOSE, OPEN, HIGH, LOW, VWAP, VOL) x 60 lags = 360 features per stock per day
- All 5 tests in tests/test_build_alpha360.py convert from XFAIL to XPASS with zero regressions in the rest of the suite
- Script is ready for production run on Kaggle: `python scripts/build_alpha360.py --config config/Multitask_Stock_SP500.conf`

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement scripts/build_alpha360.py with full feature pipeline** - `9a29c6e` (feat)

## Files Created/Modified

- `/Users/gonzaloalonsolidon/Desktop/Repos/Cursor/tfm-stockformer/scripts/build_alpha360.py` - Alpha360 feature builder: loads OHLCV Parquets, computes 360 cross-sectionally z-scored ratio features, backs up existing features/ before clearing and writing 360 new CSVs

## Decisions Made

- **tqdm fallback as class**: The plan specified a simple function fallback `def tqdm(it, **kw): return it` which cannot be used with `with tqdm(total=N) as pbar:` context manager syntax. Implemented as a minimal class with `__enter__`/`__exit__`/`update` methods for compatibility.
- **Clear features_dir after backup**: The plan's backup step copies files to a backup dir but didn't explicitly specify clearing the source dir. Without clearing, the 3 dummy CSVs (fixture) remain alongside the 360 new CSVs yielding 363 total — failing ALPHA360-01. Added explicit per-file `os.remove()` after backup.
- **LAG_BUFFER constant**: Used `LAG_BUFFER = 60` constant instead of literal `60` for readability. Functionally identical.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] tqdm fallback incompatible with context manager syntax**
- **Found during:** Task 1 (initial test run — all 5 tests remained XFAIL after script creation)
- **Issue:** Plan specified `def tqdm(it, **kw): return it` as fallback; this function cannot be used with `with tqdm(total=N, desc=X) as pbar:` pattern because a plain function doesn't support `__enter__`/`__exit__`. The `with` statement failed with AttributeError, causing the entire feature generation loop to be skipped silently (xfail absorbs the exception).
- **Fix:** Replaced function fallback with a minimal class implementing `__init__`, `__enter__`, `__exit__`, `update` methods.
- **Files modified:** scripts/build_alpha360.py
- **Verification:** Tests moved from XFAIL to XPASS after fix (4 of 5 initially)
- **Committed in:** 9a29c6e (Task 1 commit)

**2. [Rule 1 - Bug] features_dir not cleared before writing new CSVs**
- **Found during:** Task 1 (test_build_alpha360_creates_360_csvs remained XFAIL after tqdm fix — got 363 CSVs instead of 360)
- **Issue:** backup_features() copies existing files to backup dir but leaves originals in place. After writing 360 new CSVs to features_dir, the 3 original dummy CSVs remain, yielding 363 total. ALPHA360-01 asserts exactly 360.
- **Fix:** After backup, iterate `os.listdir(features_dir)` and `os.remove()` each file before the write loop.
- **Files modified:** scripts/build_alpha360.py
- **Verification:** test_build_alpha360_creates_360_csvs XPASS; all 5 tests XPASS
- **Committed in:** 9a29c6e (Task 1 commit, both fixes in same commit)

---

**Total deviations:** 2 auto-fixed (2 x Rule 1 - Bug)
**Impact on plan:** Both fixes were required for correctness. No scope creep. The plan's pseudocode had two subtle bugs; the implementation corrects them while following the plan's intent exactly.

## Issues Encountered

- Pre-existing failures in `tests/test_phase1_infra.py` (torch, pytorch-wavelets, tensorboard not installed locally) — unrelated to this plan, present before and after. Full suite excluding test_phase1_infra.py: 20 passed, 1 xfailed, 23 xpassed, 0 failed.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `scripts/build_alpha360.py` is ready to run on Kaggle (has GPU + all data)
- After running: `python scripts/build_alpha360.py --config config/Multitask_Stock_SP500.conf`
- Verify: 360 CSVs, shape (1449, 478), first date 2018-03-29, zero NaN/inf
- Then retrain: `infea` auto-detects to 362 (360 CSVs + 2 built-in), no code changes needed
- After retraining: measure IC improvement from -0.003 via `scripts/compute_ic.py`

---
*Phase: 08-alpha360-feature-replacement*
*Completed: 2026-03-17*
