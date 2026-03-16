---
phase: 05-portfolio-backtesting
verified: 2026-03-14T00:00:00Z
status: human_needed
score: 8/8 must-haves verified
human_verification:
  - test: "Open output/Multitask_output_SP500_2018-2024/equity_curve.png and inspect visually"
    expected: "Two labelled lines visible: 'Stockformer Top-10' (red solid) and 'SPY' (grey dashed). Both lines start at y=1.0. X-axis shows test-period dates (2023-05-02 to ~late 2023). Chart is clearly legible at DPI=150."
    why_human: "Cannot programmatically verify visual correctness, label readability, axis formatting, or that cumulative lines actually originate at 1.0 visually — only code inspection confirms the math; the rendered PNG requires human eyes."
---

# Phase 5: Portfolio Backtesting Verification Report

**Phase Goal:** Build a production-ready portfolio backtesting pipeline that takes model predictions and produces a quantitative performance report with equity curve visualization
**Verified:** 2026-03-14
**Status:** human_needed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | pytest collects tests/test_backtest.py without import errors | VERIFIED | `python3 -m pytest tests/test_backtest.py -v` → 6 passed, 0 errors |
| 2  | Six named test functions exist, one per requirement | VERIFIED | test_top_k_selection, test_equal_weight, test_transaction_cost, test_cumulative_return, test_performance_metrics, test_alpha_beta all present with real assertions |
| 3  | select_top_k returns exactly k tickers as a pd.Index | VERIFIED | Implemented at line 38–50; test_top_k_selection passes |
| 4  | build_portfolio_weights returns 1/k for selected, 0 for others, sum=1 | VERIFIED | Implemented at line 53–71; test_equal_weight passes |
| 5  | compute_daily_return deducts turnover * 0.001 from gross return | VERIFIED | Implemented at line 74–103; test_transaction_cost passes with 0.009 net |
| 6  | compute_performance_metrics returns dict with all 7 required keys | VERIFIED | Keys: annualized_return, sharpe_ratio, max_drawdown, alpha_annualized, beta, total_return, n_days; BACK tests pass |
| 7  | CLI orchestrates load→download→loop→save, three output files exist | VERIFIED | equity_curve.png (106KB), backtest_summary.csv (1 row, 8 cols), backtest_daily_returns.csv (167 rows + header) all present |
| 8  | equity_curve.png visually shows two labelled lines from 1.0 | NEEDS HUMAN | File exists and is 106KB; chart code uses (1+pd.Series(returns)).cumprod() so math is correct; visual confirmation required |

**Score:** 7/8 automated truths verified; 1 truth needs human visual confirmation

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `tests/test_backtest.py` | xfail stubs → real assertions for 6 requirements | VERIFIED | 181 lines; 6 real test functions with specific assertions; no xfail decorators remain |
| `scripts/run_backtest.py` | Pure functions + CLI main() | VERIFIED | 627 lines; exports select_top_k, build_portfolio_weights, compute_daily_return, compute_performance_metrics, load_predictions, derive_date_index, download_prices, run_backtest_loop, save_outputs, main |
| `output/Multitask_output_SP500_2018-2024/equity_curve.png` | Cumulative return chart (portfolio vs SPY) | VERIFIED | File exists, 106,126 bytes, written 2026-03-14 15:50 |
| `output/Multitask_output_SP500_2018-2024/backtest_summary.csv` | One-row performance statistics | VERIFIED | 1 data row; columns: annualized_return, sharpe_ratio, max_drawdown, alpha_annualized, beta, top_k, n_days, total_return; n_days=167; max_drawdown=-0.2699 (negative, correct) |
| `output/Multitask_output_SP500_2018-2024/backtest_daily_returns.csv` | Daily return time series for Phase 6 | VERIFIED | 168 lines (1 header + 167 data rows); columns: date, portfolio_return, spy_return; row count matches n_days=167 |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `scripts/run_backtest.py main()` | `output_dir/regression/regression_pred_last_step.csv` | `pd.read_csv(path, header=None)` | WIRED | Line 196: `pd.read_csv(pred_path, header=None).values.astype(np.float64)` |
| `scripts/run_backtest.py main()` | `yfinance.download(tickers, start, end)` | `start = date_index[0] - pd.Timedelta(days=10)` (10-day buffer) | WIRED | Lines 332–342: `start = date_index[0] - pd.Timedelta(days=10)`, `yf.download(...)` with guarded try/except |
| `scripts/run_backtest.py` | `config/Multitask_Stock_SP500.conf` | `configparser` reads `[dataset] test_start / test_end` + `bdate_range` | WIRED | Lines 19, 220–234: `import configparser`, `cfg.read(config_path)`, `pd.bdate_range(...)` with label.csv fallback |
| `tests/test_backtest.py` | `scripts/run_backtest.py` | `from scripts.run_backtest import ...` inside test bodies | WIRED | Each test function imports the relevant function from `scripts.run_backtest` inside the function body; 6 passed |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| PORT-01 | 05-01, 05-02, 05-03 | Top-K stock selection from predicted return scores with K configurable | SATISFIED | `select_top_k(scores, k)` uses `scores.nlargest(k).index`; `--top_k` CLI arg; test_top_k_selection passes |
| PORT-02 | 05-01, 05-02, 05-03 | Equal-weight allocation with daily rebalancing | SATISFIED | `build_portfolio_weights` sets 1/k for top-k tickers; rebalanced every day in loop; test_equal_weight passes |
| PORT-03 | 05-01, 05-02, 05-03 | Transaction cost modeling applied (≈10bps round-trip) | SATISFIED | `compute_daily_return` deducts `turnover * 0.001`; test_transaction_cost verifies 0.001 cost from full cash purchase |
| BACK-01 | 05-01, 05-03 | Cumulative return curve computed and plotted against SPY benchmark | SATISFIED | `(1+pd.Series(returns)).cumprod()` used for both portfolio and SPY; equity_curve.png saved at DPI=150; test_cumulative_return verifies cumprod math |
| BACK-02 | 05-01, 05-02, 05-03 | Annualized return, Sharpe ratio, and max drawdown computed | SATISFIED | `compute_performance_metrics` returns all three; formulas match CONTEXT.md (252/n_days, ddof=1 Sharpe); backtest_summary.csv has all values; test_performance_metrics passes |
| BACK-03 | 05-01, 05-02, 05-03 | Alpha and beta versus SPY benchmark computed | SATISFIED | `linregress(spy_r, r_arr)` with x=SPY, y=portfolio; alpha annualized as intercept*252; test_alpha_beta verifies recovery within 1e-6 tolerance |

All 6 requirements satisfied. No orphaned requirements found.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `scripts/run_backtest.py` | 442 | Text `No plt.show()` | Info | This is a docstring comment, not a call. `plt.show()` is correctly absent as an executable statement. No impact. |

No blockers or warnings found. No TODO/FIXME/HACK markers. No empty implementations. No stub returns.

---

### Human Verification Required

#### 1. Equity Curve PNG Visual Inspection

**Test:** Open `/Users/gonzaloalonsolidon/Desktop/Repos/Cursor/tfm-stockformer/output/Multitask_output_SP500_2018-2024/equity_curve.png`
**Expected:**
- Two lines visible: "Stockformer Top-10" in red (#CC2529, solid) and "SPY" in dark grey (#333333, dashed)
- Both lines originate at y=1.0 (first data point)
- X-axis shows test-period dates starting 2023-05-02
- Legend visible in upper left
- Grid lines present
- Title reads "Portfolio vs SPY — Test Period"
- No plt.show() popup was triggered during generation (runs headless)

**Why human:** A 106KB PNG file exists and the rendering code matches RESEARCH.md Pattern 5 exactly — but only human eyes can confirm the chart is legible, the lines are distinguishable, labels are readable, and there are no rendering artifacts.

---

### Gaps Summary

No gaps found. All automated checks pass:

- `tests/test_backtest.py` — 6 real tests, no xfail stubs remaining, 6 passed in 0.52s
- `scripts/run_backtest.py` — 627 lines, all 9 expected functions/helpers exported and importable
- All three output files exist with correct structure (1-row summary CSV, 167-row daily CSV, 106KB PNG)
- backtest_summary.csv: max_drawdown = -0.27 (negative, correct), n_days = 167, 8 columns as specified
- backtest_daily_returns.csv: date/portfolio_return/spy_return columns, 167 data rows
- All 6 key links wired: prediction CSV reading (header=None), yfinance download with 10-day buffer, configparser + bdate_range, test-to-module imports
- All 6 requirement IDs (PORT-01 through PORT-03, BACK-01 through BACK-03) satisfied with evidence
- All phase commits verified in git history: 3f5a431, 81abbed, 5f6643a, afc3f39, 8cb9756, e285370
- Full test suite: 6 phase-5 tests pass; 4 pre-existing phase-1 failures (torch/pywt not installed) are out of scope and pre-date this phase
- No anti-patterns: no plt.show() calls, no TODO/FIXME, no stub returns, no empty handlers

One item requires human confirmation: visual correctness of equity_curve.png.

---

_Verified: 2026-03-14_
_Verifier: Claude (gsd-verifier)_
