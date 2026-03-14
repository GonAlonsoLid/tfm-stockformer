---
plan: 05-02
phase: 05-portfolio-backtesting
status: complete
completed: 2026-03-14
---

# Plan 05-02 Summary: Core Backtest Logic

## What Was Built

Implemented `scripts/run_backtest.py` — a pure-function backtest core with four exported functions. No CLI, no yfinance; designed for import by Plan 03's CLI wrapper.

## Key Files

### Created
- `scripts/run_backtest.py` — Pure-function backtest engine with 4 functions

### Modified
- `tests/test_backtest.py` — Replaced all 6 xfail stubs with real assertions

## Self-Check: PASSED

- [x] `select_top_k`, `build_portfolio_weights`, `compute_daily_return`, `compute_performance_metrics` all implemented
- [x] All 6 tests pass (0 xfail, 0 errors)
- [x] Formulas match CONTEXT.md (252/n_days annualization, ddof=1 Sharpe, linregress x=SPY y=portfolio)
- [x] No yfinance, argparse, or CLI code present

## Commits

- `81abbed` test(05-02): replace PORT stubs with real assertions (RED phase)
- `5f6643a` feat(05-02): implement top-K selection, equal-weight, and transaction cost functions
- `afc3f39` test(05-02): replace BACK stubs with real assertions — all 6 tests green

## Test Results

```
6 passed in 0.38s
```

## Decisions

1. **`nlargest(k).index`** for top-K — pandas default tie-breaking (first occurrence)
2. **`price_returns.reindex(...).fillna(0)`** — delisted tickers get 0 return, not NaN
3. **`ddof=1` Sharpe** — sample std as specified in CONTEXT.md
4. **`linregress(spy, portfolio)`** — x=SPY independent, y=portfolio dependent
5. **`intercept * 252`** — daily alpha annualized to match CONTEXT.md spec
