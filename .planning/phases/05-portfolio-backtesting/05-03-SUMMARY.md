---
phase: 05-portfolio-backtesting
plan: "03"
subsystem: backtesting
tags: [yfinance, matplotlib, configparser, backtest, equity-curve, portfolio, SPY, benchmark]

# Dependency graph
requires:
  - phase: 05-02
    provides: "Pure functions select_top_k, build_portfolio_weights, compute_daily_return, compute_performance_metrics"
  - phase: 04-02
    provides: "regression_pred_last_step.csv output from model inference"
provides:
  - "Complete scripts/run_backtest.py CLI script with argparse, yfinance price download, backtest loop, and three output files"
  - "output/Multitask_output_SP500_2018-2024/equity_curve.png — cumulative return chart (Stockformer Top-10 vs SPY)"
  - "output/Multitask_output_SP500_2018-2024/backtest_summary.csv — one-row performance statistics"
  - "output/Multitask_output_SP500_2018-2024/backtest_daily_returns.csv — daily returns for Phase 6 Streamlit"
affects: [06-streamlit-dashboard]

# Tech tracking
tech-stack:
  added: [yfinance (guarded import), matplotlib.pyplot, configparser]
  patterns:
    - "yfinance import guarded with try/except and _YF_AVAILABLE sentinel — pure functions remain importable in offline/test environments"
    - "main() accepts optional kwargs for programmatic invocation without argparse — consistent with compute_ic.py"
    - "derive_date_index uses bdate_range + label.csv fallback for US-holiday-aware test calendar"

key-files:
  created:
    - "output/Multitask_output_SP500_2018-2024/equity_curve.png"
    - "output/Multitask_output_SP500_2018-2024/backtest_summary.csv"
    - "output/Multitask_output_SP500_2018-2024/backtest_daily_returns.csv"
  modified:
    - "scripts/run_backtest.py"

key-decisions:
  - "yfinance import guarded with try/except — same pattern intended for download_ohlcv.py; keeps pure functions importable without yfinance in test environments"
  - "derive_date_index uses bdate_range first then falls back to label.csv trading calendar for US-holiday accuracy"
  - "download_prices fetches with 10-day pre-buffer so pct_change has a valid prior-day price on day 0 of test period"
  - "save_outputs uses plt.close(fig) with no plt.show() — consistent with headless server execution"

patterns-established:
  - "Guarded optional import pattern: try/import; _AVAILABLE sentinel; raise ImportError with install hint in function body"
  - "CLI script main() with optional kwargs for programmatic smoke-test invocation without subprocess"

requirements-completed: [BACK-01, PORT-01, PORT-02, PORT-03, BACK-02, BACK-03]

# Metrics
duration: 15min
completed: 2026-03-14
---

# Phase 05 Plan 03: Backtest CLI + End-to-End Verification Summary

**Full end-to-end backtest CLI (run_backtest.py) producing equity_curve.png, backtest_summary.csv, and backtest_daily_returns.csv — human-verified chart shows Stockformer Top-10 vs SPY over 167-day test period**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-03-14T15:40:00Z
- **Completed:** 2026-03-14T16:00:00Z
- **Tasks:** 2 (1 auto + 1 checkpoint:human-verify)
- **Files modified:** 1 (scripts/run_backtest.py)

## Accomplishments

- Extended run_backtest.py with argparse CLI, yfinance price download, full backtest loop, and three output files
- Produced equity_curve.png (human-approved): two labelled lines from 1.0, DPI=150, no plt.show()
- backtest_summary.csv with 8 columns (annualized_return, sharpe_ratio, max_drawdown, alpha_annualized, beta, top_k, n_days, total_return)
- backtest_daily_returns.csv with date/portfolio_return/spy_return columns for Phase 6 Streamlit consumption
- Auto-fixed blocking deviation: guarded yfinance import so 6 test_backtest.py tests pass without yfinance installed

## Task Commits

Each task was committed atomically:

1. **Task 1: Add CLI, data loading, backtest loop, and output files to run_backtest.py** - `8cb9756` (feat)
2. **Deviation fix: Guard yfinance import** - `e285370` (fix)
3. **Task 2: Human verify checkpoint (approved)** — no separate commit; output files already existed from Task 1 execution

**Plan metadata:** _(docs commit created after SUMMARY)_

## Files Created/Modified

- `scripts/run_backtest.py` — Extended with load_predictions, derive_date_index, download_prices, run_backtest_loop, save_outputs, main(); yfinance import now guarded
- `output/Multitask_output_SP500_2018-2024/equity_curve.png` — Cumulative return chart, human-verified
- `output/Multitask_output_SP500_2018-2024/backtest_summary.csv` — One-row performance statistics table
- `output/Multitask_output_SP500_2018-2024/backtest_daily_returns.csv` — 167-row daily return time series

## Decisions Made

- yfinance guarded with try/except: pure functions (select_top_k etc.) must be importable without yfinance for tests; download_prices raises ImportError with install hint when called without yfinance
- derive_date_index tries bdate_range first, then falls back to label.csv + split_indices.json for US-holiday-aware date alignment — prevents date_index length mismatches
- prices downloaded with 10-day pre-buffer so pct_change has a valid prior-day price on day 0

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Guarded yfinance module-level import**
- **Found during:** Task 2 verification (running full test suite)
- **Issue:** `import yfinance as yf` at module level caused `ModuleNotFoundError` when tests imported pure functions in the Python 3 environment where yfinance is not installed; all 6 test_backtest.py tests failed
- **Fix:** Replaced bare `import yfinance as yf` with try/except guarded block and `_YF_AVAILABLE` sentinel; `download_prices()` raises `ImportError` with install hint if called without yfinance
- **Files modified:** scripts/run_backtest.py
- **Verification:** `python3 -m pytest tests/test_backtest.py -q` → 6 passed
- **Committed in:** e285370 (fix commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Essential for test suite correctness in offline environments. No scope creep — pure function behavior unchanged.

## Issues Encountered

- `test_phase1_infra.py` has 4 pre-existing failures (torch, pywt, tensorboard, pytorch_wavelets not installed in this Python 3 env). These pre-date plan 05-03 and are out of scope. All 6 test_backtest.py tests pass.

## User Setup Required

None — no external service configuration required. (yfinance price download runs automatically during script execution with internet access.)

## Next Phase Readiness

- backtest_daily_returns.csv is ready for Phase 6 Streamlit dashboard consumption
- backtest_summary.csv provides all metrics needed for the interactive performance display
- All BACK and PORT requirements satisfied (BACK-01, BACK-02, BACK-03, PORT-01, PORT-02, PORT-03)
- Phase 5 plans 01-03 complete; ready to proceed to Phase 6

---
*Phase: 05-portfolio-backtesting*
*Completed: 2026-03-14*
