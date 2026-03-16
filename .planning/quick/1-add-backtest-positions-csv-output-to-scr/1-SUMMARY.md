---
phase: quick
plan: 1
subsystem: backtesting
tags: [backtest, positions, csv, portfolio-audit, tdd]
dependency_graph:
  requires: [scripts/run_backtest.py, tests/test_backtest.py]
  provides: [output_dir/backtest_positions.csv]
  affects: [scripts/run_backtest.py, tests/test_backtest.py]
tech_stack:
  added: []
  patterns: [TDD red-green, 3-tuple return, long-format CSV, positional dict accumulation]
key_files:
  created: []
  modified:
    - scripts/run_backtest.py
    - tests/test_backtest.py
decisions:
  - "positions list accumulated inside run_backtest_loop using 1/top_k_n weight — keeps all logic in one place"
  - "save_outputs receives positions as explicit keyword argument — avoids positional confusion with 6-param signature"
  - "pos_df sorted by [date ASC, predicted_score DESC] — natural audit order, highest-conviction pick first per day"
metrics:
  duration: ~6 min
  completed: 2026-03-14T15:12:15Z
  tasks_completed: 2
  files_modified: 2
---

# Quick Task 1: Add Backtest Positions CSV Output Summary

**One-liner:** Extended `run_backtest_loop` to return a 3-tuple and added `backtest_positions.csv` (167 days x 10 tickers = 1670 rows) to the backtest output pipeline via TDD.

## Tasks Completed

| Task | Name | Commit | Key Files |
|------|------|--------|-----------|
| 1 (RED) | Add failing test for positions output | f3b5fc6 | tests/test_backtest.py |
| 1 (GREEN) | Extend run_backtest_loop to capture daily positions | 0da1157 | scripts/run_backtest.py |
| 2a | Add backtest_positions.csv to save_outputs | 72ea2c5 | scripts/run_backtest.py |
| 2b | Wire positions through main() and update module docstring | 0e21499 | scripts/run_backtest.py |

## What Was Built

`run_backtest_loop` now returns a 3-tuple `(portfolio_returns, spy_daily_returns, positions)` where `positions` is a list of dicts — one per (day, ticker) in the top-K selection. Each dict has keys: `date`, `ticker`, `weight` (always `1/top_k`), `predicted_score` (raw model output).

`save_outputs` writes this list as `backtest_positions.csv`, sorted by date ascending and predicted_score descending (highest-conviction pick first per day).

The module docstring and function docstrings were updated to reflect the new output file.

## Verification Results

- `python3.11 -m pytest tests/test_backtest.py -q` — 7 passed (including new `test_positions_output`)
- `wc -l output/Multitask_output_SP500_2018-2024/backtest_positions.csv` — 1671 (1670 data rows + header)
- `head -1 output/Multitask_output_SP500_2018-2024/backtest_positions.csv` — `date,ticker,weight,predicted_score`
- `python3.11 scripts/run_backtest.py --help` — CLI unchanged, exits 0

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- tests/test_backtest.py modified: FOUND
- scripts/run_backtest.py modified: FOUND
- Commit f3b5fc6: FOUND
- Commit 0da1157: FOUND
- Commit 72ea2c5: FOUND
- Commit 0e21499: FOUND
- output/Multitask_output_SP500_2018-2024/backtest_positions.csv: FOUND (1671 lines)
