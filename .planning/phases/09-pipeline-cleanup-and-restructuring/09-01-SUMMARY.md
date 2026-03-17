---
phase: 09-pipeline-cleanup-and-restructuring
plan: "01"
subsystem: testing
tags: [tdd, xfail, wave-0, test-stubs, pipeline, download-ohlcv]
dependency_graph:
  requires: []
  provides:
    - tests/test_pipeline.py
    - tests/test_download_ohlcv.py
  affects:
    - scripts/build_pipeline.py (verified by test_pipeline.py after Plan 09-02)
    - scripts/sp500_pipeline/download_ohlcv.py (verified by test_download_ohlcv.py after Plan 09-02)
tech_stack:
  added: []
  patterns:
    - xfail(strict=False) with imports inside test bodies (project convention from 02-01)
key_files:
  created:
    - tests/test_pipeline.py
    - tests/test_download_ohlcv.py
  modified: []
decisions:
  - "xfail(strict=False) pattern maintained — consistent with all prior phases (02-01, 04-01, 05-01, 06-01, 08-01)"
  - "test_download_ohlcv.py imports from scripts.sp500_pipeline (post-move path) so tests go green after Plan 09-02 without modification"
  - "test_pipeline_dir_resolves xpassed on this machine (PIPELINE_DIR resolves to existing path) — acceptable since strict=False"
metrics:
  duration: "~85s"
  completed_date: "2026-03-17"
  tasks_completed: 2
  files_created: 2
  files_modified: 0
---

# Phase 9 Plan 01: Wave 0 Test Stubs Summary

Wave 0 xfail test scaffolding for Phase 9 pipeline restructuring — 7 acceptance gates (5 pipeline + 2 download_ohlcv) that will turn green when Plans 09-02 and 09-03 land.

## Tasks Completed

| # | Task | Commit | Files |
|---|------|--------|-------|
| 1 | Write xfail stubs for test_pipeline.py | 6d7fe75 | tests/test_pipeline.py |
| 2 | Write xfail stubs for test_download_ohlcv.py | 96df04d | tests/test_download_ohlcv.py |

## Verification Results

- `python3 -m pytest tests/test_pipeline.py -x -q` — exits 0 (4 xfailed, 1 xpassed)
- `python3 -m pytest tests/test_download_ohlcv.py tests/test_pipeline.py -x -q` — exits 0 (6 xfailed, 1 xpassed)
- `python3 -m pytest tests/ -x -q` — pre-existing torch failure unchanged (not caused by these changes)
- `grep -c "pytest.mark.xfail" tests/test_pipeline.py` — 5
- `grep -c "pytest.mark.xfail" tests/test_download_ohlcv.py` — 2

## Acceptance Gates Created

### test_pipeline.py (5 tests)

| Test | Verifies | Goes green when |
|------|----------|-----------------|
| test_pipeline_dir_resolves | PIPELINE_DIR points to real dir on disk | Plan 09-02: sp500_pipeline moved to scripts/ |
| test_config_derives_data_dir | _data_dir_from_config extracts Stock_SP500_2018-01-01_2024-01-01 | Plan 09-02: helper added to build_pipeline.py |
| test_alpha360_done_sentinel_false_when_empty | _alpha360_done returns False for empty dir | Plan 09-02: sentinel function added |
| test_alpha360_done_sentinel_false_when_partial | _alpha360_done returns False for 359 CSVs | Plan 09-02: sentinel function added |
| test_alpha360_done_sentinel_true_when_360 | _alpha360_done returns True for 360 CSVs | Plan 09-02: sentinel function added |

### test_download_ohlcv.py (2 tests)

| Test | Verifies | Goes green when |
|------|----------|-----------------|
| test_ticker_fallback_reads_from_file | tickers.txt exists → skip Wikipedia | Plan 09-02: fallback logic added to scripts/sp500_pipeline/download_ohlcv.py |
| test_ticker_fallback_calls_wikipedia_when_absent | tickers.txt absent → call Wikipedia | Plan 09-02: fallback logic added |

## Deviations from Plan

### Note: test_pipeline_dir_resolves xpassed

The test `test_pipeline_dir_resolves` unexpectedly passed (xpassed) because `PIPELINE_DIR` currently resolves to `data_processing_script/sp500_pipeline/` which exists on this machine. Since `strict=False`, this is a pass not a failure. After Plan 09-02 moves the scripts, it will pass for the right reason.

No auto-fixes were required. Plan executed exactly as written.

## Self-Check: PASSED

- tests/test_pipeline.py: FOUND
- tests/test_download_ohlcv.py: FOUND
- Commit 6d7fe75: FOUND
- Commit 96df04d: FOUND
