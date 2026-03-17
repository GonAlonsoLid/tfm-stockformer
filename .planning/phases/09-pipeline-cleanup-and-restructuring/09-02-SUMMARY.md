---
phase: 09-pipeline-cleanup-and-restructuring
plan: 02
subsystem: pipeline-orchestration
tags: [pipeline, restructuring, scripts, ticker-fallback]
dependency_graph:
  requires: [09-01]
  provides: [scripts/sp500_pipeline/, updated-PIPELINE_DIR, ticker-txt-fallback]
  affects: [scripts/build_pipeline.py, scripts/sp500_pipeline/download_ohlcv.py]
tech_stack:
  added: []
  patterns: [sentinel-based-idempotency, local-file-fallback]
key_files:
  created:
    - scripts/sp500_pipeline/__init__.py
    - scripts/sp500_pipeline/download_ohlcv.py
    - scripts/sp500_pipeline/normalize_split.py
    - scripts/sp500_pipeline/serialize_arrays.py
    - scripts/sp500_pipeline/graph_embedding.py
  modified:
    - scripts/build_pipeline.py
decisions:
  - "scripts/sp500_pipeline/ uses empty __init__.py — makes it a proper Python package consistent with project conventions"
  - "data_processing_script/sp500_pipeline/ retained intact — deletion deferred to Plan 09-04 after all tests pass"
  - "tickers.txt fallback reads from os.path.join(args.data_dir, 'tickers.txt') — matches existing write path in main()"
metrics:
  duration: ~6min
  completed_date: 2026-03-17
  tasks_completed: 2
  files_changed: 6
---

# Phase 9 Plan 2: Move sp500_pipeline to scripts/ and add ticker fallback Summary

**One-liner:** Relocated sp500_pipeline/ from data_processing_script/ to scripts/ and added tickers.txt read-before-scrape logic to eliminate Wikipedia network dependency on re-runs.

## What Was Built

- **scripts/sp500_pipeline/**: New canonical location for the four pipeline step scripts plus an `__init__.py` Python package marker
- **PIPELINE_DIR updated**: `scripts/build_pipeline.py` now resolves to `os.path.join(_SCRIPT_DIR, "sp500_pipeline")` — no longer traverses up to `data_processing_script/`
- **Ticker fallback**: `scripts/sp500_pipeline/download_ohlcv.py` main() checks `{data_dir}/tickers.txt` before calling `get_sp500_tickers()`; if file exists, tickers are read from it and Wikipedia is skipped entirely; if absent, behavior is unchanged (fetch + write file)

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Move sp500_pipeline/ to scripts/ and update PIPELINE_DIR | a8c0cd3 | scripts/sp500_pipeline/ (5 files), scripts/build_pipeline.py |
| 2 | Add ticker local-file fallback to download_ohlcv.py | 9eac1e3 | scripts/sp500_pipeline/download_ohlcv.py |

## Test Results

After Task 2:
- `test_pipeline_dir_resolves` — XPASSED (up from XFAIL before plan)
- `test_ticker_fallback_reads_from_file` — XPASSED (up from XFAIL)
- `test_ticker_fallback_calls_wikipedia_when_absent` — XPASSED (up from XFAIL)
- `test_config_derives_data_dir` — XFAIL (pending Plan 09-03)
- `test_alpha360_done_sentinel_*` (3 tests) — XFAIL (pending Plan 09-03)

## Decisions Made

- **data_processing_script/ not deleted**: Plan 09-02 is a copy operation. The source directory must remain until Plan 09-04 explicitly removes it, ensuring Plans 09-03 tests have a stable baseline to compare against.
- **Empty __init__.py**: Chosen over adding module docstring — consistent with `data_processing_script/sp500_pipeline/__init__.py` (also empty) and keeps the package marker minimal.
- **Fallback reads from args.data_dir**: The tickers.txt write path at end of main() already uses `os.path.join(args.data_dir, "tickers.txt")`; reading from the same location makes the cache coherent without any additional configuration.

## Deviations from Plan

None — plan executed exactly as written.

## Self-Check

- `scripts/sp500_pipeline/__init__.py` — FOUND
- `scripts/sp500_pipeline/download_ohlcv.py` — FOUND
- `scripts/sp500_pipeline/normalize_split.py` — FOUND
- `scripts/sp500_pipeline/serialize_arrays.py` — FOUND
- `scripts/sp500_pipeline/graph_embedding.py` — FOUND
- `scripts/build_pipeline.py` PIPELINE_DIR contains `os.path.join(_SCRIPT_DIR, "sp500_pipeline")` — CONFIRMED
- `scripts/sp500_pipeline/download_ohlcv.py` contains `skipping Wikipedia scrape` — CONFIRMED
- `data_processing_script/sp500_pipeline/` still exists — CONFIRMED
- Commit a8c0cd3 (Task 1) — FOUND
- Commit 9eac1e3 (Task 2) — FOUND

## Self-Check: PASSED
