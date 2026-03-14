---
phase: 04-evaluation
plan: 01
subsystem: testing
tags: [pytest, tdd, ic, icir, mae, rmse, f1, xfail, evaluation-metrics]

# Dependency graph
requires:
  - phase: 03-model-training
    provides: scripts/run_inference.py — inference output CSVs that compute_ic.py will evaluate
provides:
  - tests/test_compute_ic.py with 6 xfail-protected unit tests defining the behavioral contract for compute_ic.py

affects:
  - 04-02 (compute_ic.py implementation must satisfy these tests to reach GREEN phase)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - TDD Red phase — tests written before implementation exists; xfail(strict=False) protects CI
    - try/except ImportError -> pytest.skip() inside test bodies prevents module-level import errors

key-files:
  created:
    - tests/test_compute_ic.py
  modified: []

key-decisions:
  - "xfail(strict=False) wraps all 6 tests — consistent with project pattern from 02-01 decision; suite collects cleanly before Plan 02 lands"
  - "Imports wrapped in try/except ImportError -> pytest.skip() inside test bodies (not module level) — file is importable even before scripts/compute_ic.py exists"
  - "test_smoke_actual_output uses tmp_path fixture and xfail path — if output dir absent the test xfails gracefully without error"

patterns-established:
  - "Evaluation metric tests: assert exact mathematical properties (perfect IC=1.0, perfect MAE=0.0) not just 'no exception'"
  - "NaN handling test: construct minimal case (constant predictions on one day) to verify NaN propagation and exclusion from mean"

requirements-completed: [EVAL-01, EVAL-02]

# Metrics
duration: 1min
completed: 2026-03-14
---

# Phase 4 Plan 01: Evaluation Test Scaffold Summary

**TDD Red phase test suite for compute_ic.py — 6 xfail-protected unit tests defining IC, ICIR, MAE, RMSE, accuracy, and F1 behavioral contracts**

## Performance

- **Duration:** ~1 min
- **Started:** 2026-03-14T14:00:13Z
- **Completed:** 2026-03-14T14:01:15Z
- **Tasks:** 1 (TDD Red)
- **Files modified:** 1

## Accomplishments

- Created `tests/test_compute_ic.py` with 6 tests covering EVAL-01 and EVAL-02 requirements
- All tests protected with `xfail(strict=False)` + `try/except ImportError -> pytest.skip()` so the suite collects cleanly before Plan 02 creates `scripts/compute_ic.py`
- `pytest tests/test_compute_ic.py -v` collects exactly 6 tests, exits with code 0 (all skipped in RED phase)
- Tests assert mathematical properties (not just "no exception"): IC=1.0 for perfect predictions, MAE/RMSE=0.0 for perfect predictions, ICIR formula verified against ic_mean/std(ddof=1)

## Task Commits

Each task was committed atomically:

1. **Task 1: Write test scaffold for compute_ic.py** - `21f672f` (test)

## Files Created/Modified

- `tests/test_compute_ic.py` — 6 xfail unit tests for IC, ICIR, MAE, RMSE, accuracy, F1 metrics

## Decisions Made

- `xfail(strict=False)` pattern chosen (consistent with project decision from 02-01): suite collects without failing until implementation lands
- Imports placed inside test function bodies (not module level) so the file itself is importable before `scripts/compute_ic.py` exists
- `test_smoke_actual_output` uses `pytest.xfail()` (not `skip`) when output dir is absent — allows the xfail mark to capture the integration state correctly

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `tests/test_compute_ic.py` is the behavioral contract that Plan 02 (`04-02`) must satisfy
- Plan 02 must implement `scripts/compute_ic.py` with `compute_ic_metrics`, `compute_regression_metrics`, and `compute_classification_metrics` functions
- When Plan 02 is complete, re-running `pytest tests/test_compute_ic.py -v` should show all 6 tests PASSED (not XFAIL/SKIP)

---
*Phase: 04-evaluation*
*Completed: 2026-03-14*
