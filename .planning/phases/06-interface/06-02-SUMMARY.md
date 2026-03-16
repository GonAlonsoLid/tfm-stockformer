---
phase: 06-interface
plan: 02
subsystem: ui
tags: [streamlit, plotly, dashboard, subprocess, equity-curve, heatmap, dark-theme]

# Dependency graph
requires:
  - phase: 06-interface
    provides: 7 xfail test stubs in tests/test_app.py, synthetic fixtures in conftest.py, .streamlit/config.toml dark theme
  - phase: 05-portfolio-backtesting
    provides: backtest CSV outputs (backtest_daily_returns.csv, backtest_summary.csv, backtest_positions.csv, regression_pred_last_step.csv)
provides:
  - app.py — complete Streamlit dashboard at project root, runnable with `streamlit run app.py`
  - build_equity_chart() — testable pure function returning go.Figure with 2 traces starting at 1.0
  - format_metrics_table() — testable pure function returning DataFrame with all 8 required columns
  - build_heatmap() — testable pure function with zmid=0, portfolio tickers from backtest_positions.csv
  - run_pipeline() — subprocess pipeline runner with PYTHONUNBUFFERED=1 live log streaming
  - load_results() — CSV loader extracting portfolio tickers via dict.fromkeys(positions_df["ticker"])
  - render_results() — renders equity curve filtered to user-selected date range, metrics table, heatmap
affects: [06-03, 06-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Streamlit import guard via get_script_run_ctx — prevents main() from executing during pytest import"
    - "cumprod().shift(1, fill_value=1.0) — equity curve starting at 1.0 on first plotted day"
    - "dict.fromkeys(positions_df['ticker'].dropna().tolist()) — order-preserving deduplication for portfolio tickers"
    - "subprocess.Popen with PYTHONUNBUFFERED=1 + iter(proc.stdout.readline,'') — live log streaming to st.empty()"

key-files:
  created:
    - app.py
  modified: []

key-decisions:
  - "cumprod().shift(1, fill_value=1.0) used for equity curve so first plotted point is always 1.0 — matches test contract and standard equity curve convention"
  - "_txn_cost and _benchmark sidebar controls are collected per UI-SPEC but informational-only in v1 (run_backtest.py has FEE hardcoded, no CLI arg)"
  - "start_date/end_date are functional — filter display_df before calling build_equity_chart() (locked decision from CONTEXT.md)"
  - "format_metrics_table() returns numeric DataFrame without fillna to preserve NumberColumn formatting (RESEARCH.md Pitfall 1)"
  - "build_heatmap() slices tickers[:k] so callers can pass a longer list with explicit k — needed to satisfy test_heatmap_top_k_filter"

patterns-established:
  - "app.py sidebar: 7 controls in 2 groups (Pipeline Settings + Backtest Parameters)"
  - "Session state keys: run_complete (bool), running (bool), results (dict|None)"
  - "Initial state: st.info() placeholder; results rendered only after successful run"

requirements-completed: [UI-01, UI-02, UI-03, UI-04]

# Metrics
duration: 10min
completed: 2026-03-16
---

# Phase 6 Plan 02: Streamlit Dashboard Summary

**Full Streamlit dashboard (app.py) with equity curve, metrics table, prediction heatmap, subprocess pipeline runner, and live log streaming — all 7 UI tests now XPASS**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-03-16T12:40:00Z
- **Completed:** 2026-03-16T12:50:00Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Created app.py (485 lines) with complete Streamlit dashboard: sidebar with 7 controls, Run Pipeline button, live subprocess log streaming, and three result visualizations
- All 7 test_app.py tests now XPASS (unexpectedly passing); full suite exits 0 with 29 passed, 18 xpassed, no failures
- Implemented all 4 locked UI requirements: UI-01 (imports/constants), UI-02 (equity curve), UI-03 (metrics table), UI-04 (heatmap)

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement app.py — full Streamlit dashboard** - `8aa9483` (feat)

## Files Created/Modified

- `app.py` — Complete 485-line Streamlit dashboard. Exports DEFAULT_CONFIG_PATH, DEFAULT_OUTPUT_DIR, DEFAULT_TOP_K, build_equity_chart, format_metrics_table, build_heatmap, run_pipeline, load_results, render_results, main. Entry point guard via get_script_run_ctx.

## Decisions Made

- `cumprod().shift(1, fill_value=1.0)` for equity curve: the first plotted y-value is 1.0 (portfolio starting value before any return is applied); subsequent values are cumulative products. This matches the test assertion `fig.data[0].y[0] == 1.0`.
- `build_heatmap()` slices `tickers[:k]` internally: allows callers to pass a list of N tickers with `k < N` to control how many rows appear on the y-axis — required to satisfy `test_heatmap_top_k_filter`.
- `_txn_cost` and `_benchmark` sidebar controls collected but unused in pipeline calls (run_backtest.py has FEE hardcoded at module level, confirmed in RESEARCH.md).
- `format_metrics_table()` returns numeric DataFrame without `fillna("—")` — Streamlit NumberColumn requires numeric dtype; filling with strings breaks formatting.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed missing plotly and streamlit dependencies**
- **Found during:** Task 1 (after writing app.py, tests still xfail on import)
- **Issue:** `plotly` and `streamlit` were not installed in the venv; `import app` failed with `ModuleNotFoundError: No module named 'plotly'`
- **Fix:** Ran `pip install "streamlit>=1.35,<=1.50" "plotly>=5.18,<7"` (same pins already in requirements.txt from Plan 06-01)
- **Files modified:** None (dependencies already in requirements.txt)
- **Verification:** `python -c "import app; print(app.DEFAULT_TOP_K)"` prints 10
- **Committed in:** 8aa9483 (Task 1 commit — fix applied inline before final commit)

**2. [Rule 1 - Bug] Fixed equity curve not starting at 1.0**
- **Found during:** Task 1 verification — `test_equity_chart_starts_at_one` XFAIL after initial implementation
- **Issue:** `(1 + returns).cumprod()` gives `y[0] = 1 + r[0]` not `1.0`; test asserts `abs(y[0] - 1.0) < 1e-6`
- **Fix:** Changed to `(1 + returns).cumprod().shift(1, fill_value=1.0)` — shifts series right by 1 with a 1.0 baseline at position 0
- **Files modified:** `app.py` (build_equity_chart function)
- **Verification:** `test_equity_chart_starts_at_one` now XPASS; `fig.data[0].y[0] == 1.0` confirmed
- **Committed in:** 8aa9483 (Task 1 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking dependency install, 1 bug fix)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered

None beyond the auto-fixed deviations above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- app.py is complete and all 7 UI tests pass; UI-01 through UI-04 satisfied
- Dashboard is runnable with `streamlit run app.py` from project root
- Requires pipeline outputs in output_dir to display results (run_inference.py + run_backtest.py must complete successfully)
- Plans 06-03 and 06-04 can proceed (if any); alternatively, Phase 6 is complete

---
*Phase: 06-interface*
*Completed: 2026-03-16*
