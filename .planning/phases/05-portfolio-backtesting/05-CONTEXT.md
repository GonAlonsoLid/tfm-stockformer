# Phase 5: Portfolio & Backtesting - Context

**Gathered:** 2026-03-14
**Status:** Ready for planning

<domain>
## Phase Boundary

Convert regression prediction scores (from Phase 4 inference CSVs) into a daily-rebalanced top-K equal-weight portfolio, run a historical backtest against SPY, compute full performance statistics, and output an equity curve chart. No new training, no live trading, no Streamlit UI — pure backtesting of the test-period predictions.

</domain>

<decisions>
## Implementation Decisions

### Script interface
- Single script: `scripts/run_backtest.py`
- Entry point: `python scripts/run_backtest.py --output_dir output/Multitask_output_SP500_2018-2024 --top_k 20`
- Flags: `--output_dir` (required), `--top_k` (optional, default K=10)
- No `--fee` or `--rf_rate` flags — these are hardcoded at 10bps and 0% respectively
- Mirrors the `compute_ic.py` pattern from Phase 4

### Script outputs
- `equity_curve.png` — cumulative return chart saved to output_dir (no plt.show())
- `backtest_summary.csv` — one row with all stats: annualized return, Sharpe, max drawdown, alpha, beta vs SPY
- `backtest_daily_returns.csv` — daily portfolio returns time series (for Phase 6 Streamlit use)
- All files saved into `output_dir`

### Close price data source
- Download SPY + portfolio stock prices from yfinance at runtime
- Covers only the test period dates (derived from the prediction CSV date range)
- No dependency on Phase 2 data directory path
- Accept `--tickers_file` flag pointing to a saved ticker list file (maps column indices to ticker symbols)
- Phase 2 must save a `sp500_tickers.txt` (or similar) for run_backtest.py to consume

### Portfolio construction
- Default K=10 (2% of S&P500 universe, matches original notebook)
- K is configurable via `--top_k` flag
- Equal-weight: each selected stock gets 1/K allocation
- Daily rebalancing: top-K re-selected every trading day from that day's prediction scores
- Transaction cost: 10bps round-trip (0.001) applied on each day's turnover

### Performance metrics
- Annualized return: (1 + total_return)^(252/n_days) - 1
- Sharpe ratio: annualized(daily_returns.mean() / daily_returns.std()), risk-free rate = 0%
- Max drawdown: max(cummax - current) / cummax over the test period
- Alpha: intercept of OLS regression of portfolio daily returns on SPY daily returns
- Beta: slope of the same OLS regression

### Chart
- Single-panel equity curve: portfolio vs SPY cumulative returns from test-period start
- Both lines start at 1.0
- Saved as `equity_curve.png` to output_dir (file only, no interactive plt.show())
- No drawdown subplot — clean thesis-ready figure

### Claude's Discretion
- Chart styling (colors, labels, legend placement, DPI, figure size)
- Exact OLS library for alpha/beta (numpy polyfit or scipy.stats.linregress)
- How to handle missing prices or delisted stocks in the test period
- Whether to annualize alpha (likely annualize via *252)
- Column names in backtest_summary.csv
- Whether backtest_daily_returns.csv includes SPY daily returns as a second column (useful for Phase 6)

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `Backtest/Backtest.ipynb`: Complete backtest class with top-K selection, equal-weight allocation, transaction cost deduction, and cumulative return computation — adapt this to a standalone script
- `Backtest/Backtest.ipynb:calculate_performance_metrics()`: Already computes total_return, annualized_return, sharpe_ratio, max_drawdown from a net value series — reuse this logic
- `scripts/compute_ic.py`: Pattern reference for standalone evaluation script structure (argparse, output_dir, CSV writing)
- `data_processing_script/sp500_pipeline/download_ohlcv.py`: Pattern for yfinance download with retry logic and batched fetching

### Established Patterns
- Output CSVs are headerless: `pd.read_csv(..., header=None)` required for regression_pred_last_step.csv
- Script pattern: argparse CLI → read from output_dir → compute → write to output_dir
- yfinance usage already established in Phase 2

### Integration Points
- Reads from `output_dir/regression/regression_pred_last_step.csv` (same file as compute_ic.py)
- Reads `--tickers_file` to map column indices → ticker symbols (Phase 2 must produce this file)
- `backtest_daily_returns.csv` consumed by Phase 6 (Streamlit) for equity curve visualization

</code_context>

<specifics>
## Specific Ideas

- Original notebook pattern (top-K selection, daily rebalancing, fee deduction) should be preserved — adapt, don't rewrite from scratch
- Run pattern: `python scripts/run_inference.py --config ... && python scripts/compute_ic.py --output_dir ... && python scripts/run_backtest.py --output_dir ...`
- Alpha/beta should be computed vs SPY (not CSI 300 as in original notebook)

</specifics>

<deferred>
## Deferred Ideas

- Risk-aware allocation (e.g., inverse-volatility weighting) — v2, out of thesis scope
- Walk-forward backtesting (multiple test windows) — explicitly out of scope per REQUIREMENTS.md
- Multi-model comparison in backtest chart — v2 (original notebook compared 10+ models; thesis only needs Stockformer vs SPY)
- Configurable fee and risk-free rate flags — deferred per Phase 5 script interface decision

</deferred>

---

*Phase: 05-portfolio-backtesting*
*Context gathered: 2026-03-14*
