---
phase: 08-alpha360-feature-replacement
plan: "01"
subsystem: testing
tags: [pytest, xfail, alpha360, feature-engineering, tdd]

# Dependency graph
requires:
  - phase: 08-alpha360-feature-replacement
    provides: CONTEXT.md and RESEARCH.md with locked design decisions and data contracts
provides:
  - "Wave 0 test scaffold: tests/test_build_alpha360.py with 5 xfail stubs for ALPHA360-01..05"
  - "build_alpha360.main(config_path, data_dir) interface contract documented in test file"
  - "Synthetic Parquet fixture (alpha360_env) for offline testing without 478-ticker dataset"
affects: [08-02, 08-alpha360-feature-replacement]

# Tech tracking
tech-stack:
  added: []
  patterns: [xfail(strict=False) with imports inside test bodies, tmp_path synthetic Parquet fixtures]

key-files:
  created:
    - tests/test_build_alpha360.py
  modified: []

key-decisions:
  - "xfail(strict=False) with imports inside test bodies — consistent with Phase 02-01, 04-01, 05-01, 06-01 convention; keeps CI GREEN before scripts/build_alpha360.py exists"
  - "alpha360_env fixture uses 5 tickers x 80 OHLCV rows (60 lag buffer + 20 output rows) for minimal but correct fixture"
  - "build_alpha360.main(config_path, data_dir) interface locked in test file docstring — Plan 02 executor must implement this exact signature"

patterns-established:
  - "Wave 0 xfail scaffold: write failing tests before implementation; gate is suite collection success + all xfail green"

requirements-completed: [ALPHA360-01, ALPHA360-02, ALPHA360-03, ALPHA360-04, ALPHA360-05]

# Metrics
duration: 2min
completed: 2026-03-17
---

# Phase 8 Plan 01: Alpha360 Wave 0 Test Scaffold Summary

**pytest xfail stubs for all 5 ALPHA360 requirements with synthetic Parquet fixture in tmp_path, documenting the build_alpha360.main(config_path, data_dir) interface contract**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-17T00:25:46Z
- **Completed:** 2026-03-17T00:27:23Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created `tests/test_build_alpha360.py` with 5 xfail(strict=False) stubs covering ALPHA360-01..05
- Defined `alpha360_env` fixture with synthetic Parquet data: 5 tickers, 80 business-day rows, reproducible via np.random.seed(42)
- Documented the exact `build_alpha360.main(config_path, data_dir)` interface in the module docstring for Plan 02 executor
- All 5 tests xfail cleanly; pytest collects them in 0.01s; suite remains GREEN

## Task Commits

1. **Task 1: Write test scaffold with 5 xfail stubs** - `6db349e` (test)

**Plan metadata:** (pending — added in final commit)

## Files Created/Modified
- `tests/test_build_alpha360.py` — 5 xfail test stubs with alpha360_env fixture; module docstring defines the build_alpha360.main() interface contract

## Decisions Made
- xfail(strict=False) with imports inside test bodies: consistent with the pattern established in Phase 02-01 and repeated in 04-01, 05-01, 06-01; prevents module-level ImportError before scripts/build_alpha360.py exists
- Fixture uses 80 OHLCV rows (60 lag buffer + 20 output): minimal size that exercises the full lag buffer logic correctly
- tickers_file fixture creates `data/tickers.txt` (not `data/ohlcv/tickers.txt`) matching the real data layout where tickers.txt is a sibling of ohlcv/ directory

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Pre-existing failure in tests/test_phase1_infra.py::test_torch_importable (torch not installed in local dev venv) — out of scope; not caused by this plan's changes.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Wave 0 complete: test contract established for scripts/build_alpha360.py
- Plan 02 (Wave 1) can implement the script — the interface signature `main(config_path, data_dir)` is locked in the test docstring
- Running `pytest tests/test_build_alpha360.py -x` is the per-commit gate for Plan 02 tasks; tests should transition from XFAIL to XPASS as implementation lands

---
*Phase: 08-alpha360-feature-replacement*
*Completed: 2026-03-17*
