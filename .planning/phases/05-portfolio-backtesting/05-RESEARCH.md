# Phase 5: Portfolio & Backtesting - Research

**Researched:** 2026-03-14
**Domain:** Quantitative backtesting — top-K portfolio construction, performance metrics, SPY benchmark comparison
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Script interface:**
- Single script: `scripts/run_backtest.py`
- Entry point: `python scripts/run_backtest.py --output_dir output/Multitask_output_SP500_2018-2024 --top_k 20`
- Flags: `--output_dir` (required), `--top_k` (optional, default K=10)
- No `--fee` or `--rf_rate` flags — these are hardcoded at 10bps and 0% respectively
- Mirrors the `compute_ic.py` pattern from Phase 4

**Script outputs:**
- `equity_curve.png` — cumulative return chart saved to output_dir (no plt.show())
- `backtest_summary.csv` — one row with all stats: annualized return, Sharpe, max drawdown, alpha, beta vs SPY
- `backtest_daily_returns.csv` — daily portfolio returns time series (for Phase 6 Streamlit use)
- All files saved into `output_dir`

**Close price data source:**
- Download SPY + portfolio stock prices from yfinance at runtime
- Covers only the test period dates (derived from the prediction CSV date range)
- No dependency on Phase 2 data directory path
- Accept `--tickers_file` flag pointing to a saved ticker list file (maps column indices to ticker symbols)
- Phase 2 must save a `sp500_tickers.txt` (or similar) for run_backtest.py to consume

**Portfolio construction:**
- Default K=10 (2% of S&P500 universe, matches original notebook)
- K is configurable via `--top_k` flag
- Equal-weight: each selected stock gets 1/K allocation
- Daily rebalancing: top-K re-selected every trading day from that day's prediction scores
- Transaction cost: 10bps round-trip (0.001) applied on each day's turnover

**Performance metrics:**
- Annualized return: (1 + total_return)^(252/n_days) - 1
- Sharpe ratio: annualized(daily_returns.mean() / daily_returns.std()), risk-free rate = 0%
- Max drawdown: max(cummax - current) / cummax over the test period
- Alpha: intercept of OLS regression of portfolio daily returns on SPY daily returns
- Beta: slope of the same OLS regression

**Chart:**
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

### Deferred Ideas (OUT OF SCOPE)
- Risk-aware allocation (e.g., inverse-volatility weighting) — v2, out of thesis scope
- Walk-forward backtesting (multiple test windows) — explicitly out of scope per REQUIREMENTS.md
- Multi-model comparison in backtest chart — v2 (original notebook compared 10+ models; thesis only needs Stockformer vs SPY)
- Configurable fee and risk-free rate flags — deferred per Phase 5 script interface decision
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| PORT-01 | Top-K stock selection from predicted return scores with K configurable via parameter | Backtest.ipynb top-K logic adapted: `factor[date].nlargest(k).index` → `real_weight.loc[top_k, date] = 1/k` |
| PORT-02 | Equal-weight allocation with daily rebalancing | Verified in notebook: weight = 1/K per stock; daily rebalancing loop over all trade dates |
| PORT-03 | Transaction cost modeling applied (~10bps round-trip) | Notebook pattern: `value_port = value_port * (1 - turnover * fee)` where fee=0.001 and turnover=sum(abs(weight_now - weight_last)) |
| BACK-01 | Cumulative return curve plotted against SPY benchmark | yfinance SPY download + cumprod of (1 + daily_pct_change); matplotlib line chart saves to PNG |
| BACK-02 | Annualized return, Sharpe ratio, max drawdown computed | Verified formulas from notebook's `calculate_performance_metrics()` — direct port |
| BACK-03 | Alpha and beta versus SPY benchmark computed | scipy.stats.linregress(spy_daily, portfolio_daily) — slope=beta, intercept=alpha |
</phase_requirements>

---

## Summary

Phase 5 converts the regression prediction matrix produced by Phase 4 (`regression_pred_last_step.csv`, shape 167×478) into a daily-rebalanced top-K equal-weight portfolio and benchmarks it against SPY. The implementation is a straight adaptation of `Backtest/Backtest.ipynb` — the core algorithm is already proven, the key engineering work is: (1) replacing Qlib/CSI-300 data sources with yfinance + the project's tickers.txt, (2) replacing the HS300 benchmark with SPY, and (3) packaging the logic into a self-contained CLI script that follows the `compute_ic.py` structural pattern.

The prediction matrix has 167 trading days and 478 stocks. The tickers.txt file in `data/Stock_SP500_2018-01-01_2024-01-01/tickers.txt` already contains the 478 tickers in column order — it is the natural `--tickers_file` argument. The prediction CSV dates must be inferred from the test-split date range stored in the config or derived from `data/Stock_SP500_2018-01-01_2024-01-01/` date boundaries; the safest approach is to read date boundaries from `split_indices.json` or from the config's `test_start`/`test_end` keys.

Performance metric formulas are locked in CONTEXT.md and match the notebook's `calculate_performance_metrics()` almost verbatim. Alpha/beta require a simple OLS regression; `scipy.stats.linregress` is the correct single-dependency choice (already imported in compute_ic.py scope via scipy). The chart is straightforward matplotlib — both series start at 1.0, one plt.savefig call, no plt.show().

**Primary recommendation:** Port the Backtest.ipynb class directly into run_backtest.py as a standalone function (not a class), follow the compute_ic.py module layout for argparse and output writing, use yfinance to download SPY and individual stock close prices for the test period dates, and use scipy.stats.linregress for alpha/beta.

---

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | (pinned in requirements.txt) | DataFrame operations, date alignment, CSV I/O | Already used throughout project; all data manipulation |
| numpy | (pinned) | Array math, return calculations | Already used throughout project |
| yfinance | (pinned) | Download SPY + stock close prices at runtime | Already established in Phase 2 for OHLCV downloads |
| scipy.stats | (via scipy, pinned) | linregress for alpha/beta OLS | Already imported in compute_ic.py; no new dependency |
| matplotlib | (pinned) | equity_curve.png chart | Already in requirements.txt; used in Backtest.ipynb |
| argparse | stdlib | CLI argument parsing | Pattern established in compute_ic.py |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| os, sys | stdlib | Path handling, error exits | Consistent with compute_ic.py pattern |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| scipy.stats.linregress | numpy.polyfit | Both work; linregress returns named fields (slope, intercept, rvalue) which is cleaner — prefer linregress |
| manual backtest loop | vectorbt / backtrader | External libs add a dependency with no thesis benefit; the notebook already has the loop — use it |
| matplotlib | seaborn | Backtest.ipynb uses seaborn for the multi-model chart, but a clean 2-line matplotlib plot is sufficient for the thesis figure |

**Installation:** No new packages required. All libraries are already in requirements.txt.

---

## Architecture Patterns

### Recommended Project Structure

The script lives alongside compute_ic.py and follows the same layout:

```
scripts/
├── run_backtest.py          # new — Phase 5 deliverable
├── compute_ic.py            # Phase 4 — structural reference
├── run_inference.py         # Phase 3
└── build_pipeline.py        # Phase 2
```

Output files written to `output_dir`:
```
output/Multitask_output_SP500_2018-2024/
├── regression/
│   └── regression_pred_last_step.csv   # input (167 days × 478 stocks)
├── equity_curve.png                     # output
├── backtest_summary.csv                 # output (1 row)
├── backtest_daily_returns.csv           # output (n_days rows)
├── evaluation_summary.csv              # Phase 4 output
└── ic_by_day.csv                       # Phase 4 output
```

### Pattern 1: Prediction Matrix to Portfolio Returns

**What:** Read the headerless regression_pred_last_step.csv (n_days × n_stocks), align with tickers and date index, then for each day select top-K tickers and compute equal-weight return.

**When to use:** Core algorithm — runs once per script invocation.

**Key insight from notebook:** The Backtest class in the notebook uses daily rebalancing (not monthly as some comments suggest — the `all_trade_dates` equals all trading days when predictions exist for every day). For Phase 5, every row in regression_pred_last_step.csv corresponds to one trading day — rebalance daily.

```python
# Load predictions — headerless, shape (n_days, n_stocks)
reg_pred = pd.read_csv(
    os.path.join(output_dir, "regression", "regression_pred_last_step.csv"),
    header=None
).values.astype(np.float64)

# tickers_file: one ticker per line, same order as CSV columns
with open(tickers_file) as f:
    tickers = [t.strip() for t in f.readlines()]

# date_index: list of trading day strings covering the test period
# shape: reg_pred.shape[0] dates
pred_df = pd.DataFrame(reg_pred, index=date_index, columns=tickers)
```

### Pattern 2: Daily Top-K Portfolio Returns

**What:** Per row (trading day), select top-K by predicted score, compute realized return using yfinance close prices, deduct transaction cost based on portfolio turnover.

**Adapted from notebook's Backtest() loop:**

```python
FEE = 0.001  # 10bps round-trip (hardcoded)

portfolio_returns = []
weight_prev = pd.Series(0.0, index=tickers)

for date in pred_df.index:
    scores = pred_df.loc[date]
    top_k = scores.nlargest(top_k_n).index
    weight_now = pd.Series(0.0, index=tickers)
    weight_now[top_k] = 1.0 / top_k_n

    # Turnover = sum of absolute weight changes (one-way turnover × 2 for round-trip)
    turnover = (weight_now - weight_prev).abs().sum()
    cost = turnover * FEE

    # Realized return: weight_now * (close[t] / close[t-1] - 1)
    daily_ret = (weight_now * price_returns.loc[date]).sum() - cost
    portfolio_returns.append(daily_ret)
    weight_prev = weight_now
```

**Pitfall:** The notebook's `Backtest()` loop tracks realized portfolio value using a "hold-then-rebalance" structure where returns accrue between rebalancing dates. For **daily** rebalancing (Phase 5 case), the simplification above is equivalent: each day we hold the top-K at equal weight, and the realized return for that day is sum(w_i * r_i) minus transaction cost.

### Pattern 3: Performance Metrics (direct port from notebook)

```python
# Cumulative return series
cum_returns = (1 + pd.Series(portfolio_returns)).cumprod()

# Annualized return
total_return = cum_returns.iloc[-1] - 1
n_days = len(portfolio_returns)
annualized_return = (1 + total_return) ** (252 / n_days) - 1

# Sharpe ratio (risk-free = 0%)
daily_ret_series = pd.Series(portfolio_returns)
sharpe = (daily_ret_series.mean() / daily_ret_series.std()) * np.sqrt(252)

# Max drawdown
max_drawdown = (cum_returns / cum_returns.cummax() - 1).min()

# Alpha and beta vs SPY
from scipy.stats import linregress
slope, intercept, _, _, _ = linregress(spy_daily_returns, portfolio_daily_returns)
beta = slope
alpha = intercept * 252  # annualized daily alpha
```

### Pattern 4: SPY Download via yfinance

```python
import yfinance as yf

# date_index is the list of test-period trading day strings
start = pd.Timestamp(date_index[0])
end = pd.Timestamp(date_index[-1]) + pd.Timedelta(days=5)  # buffer for weekends

spy_raw = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
spy_close = spy_raw["Close"].reindex(pd.DatetimeIndex(date_index), method="ffill")
spy_daily = spy_close.pct_change().fillna(0)
```

### Pattern 5: Equity Curve Chart

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(date_index, portfolio_cum, label=f"Stockformer Top-{top_k_n}", color="#CC2529", linewidth=1.5)
ax.plot(date_index, spy_cum, label="SPY", color="#333333", linewidth=1.5, linestyle="--")
ax.set_title("Portfolio vs SPY — Test Period")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return")
ax.legend(loc="upper left")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
save_path = os.path.join(output_dir, "equity_curve.png")
fig.savefig(save_path, dpi=150)
plt.close(fig)  # critical: no plt.show(), releases memory
```

### Anti-Patterns to Avoid

- **Do not use `plt.show()`:** Script runs headless (no display). Use `fig.savefig()` then `plt.close(fig)`. The notebook calls `plt.show()` — strip this in the script adaptation.
- **Do not hardcode Chinese column names from notebook:** The notebook uses '净值', '换手率', '收益率' for the internal DataFrame — replace with English names (`nav`, `turnover`, `daily_return`) in the script.
- **Do not use Qlib or HS300 benchmark:** The notebook depends on `qlib.init(provider_uri=...)` and CSI-300 — entirely replaced by yfinance + SPY in this phase.
- **Do not reread tickers_file from Phase 2 data directory:** The `--tickers_file` flag makes the script self-contained; no hardcoded paths to Phase 2 data.
- **Do not assume prediction rows are consecutive calendar days:** Use the date_index derived from yfinance or the config — trading day gaps (weekends, holidays) are real.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| OLS regression for alpha/beta | Custom least-squares | `scipy.stats.linregress` | One line, numerically stable, returns named fields |
| yfinance retry / error handling | Custom HTTP retry loop | yfinance built-in (uses requests with retries) | Already used in Phase 2 download script |
| pct_change computation | Manual (close[t]/close[t-1]-1) | `pandas.Series.pct_change()` | Handles NaN propagation automatically |
| Date alignment between predictions and prices | Custom join logic | `pd.DataFrame.reindex(date_index, method="ffill")` | Handles delisted stocks and holiday gaps cleanly |

**Key insight:** The most complex piece of this phase is already written in Backtest.ipynb. The engineering risk is in the data plumbing (prediction date ↔ price date alignment), not the math.

---

## Common Pitfalls

### Pitfall 1: Prediction Date Index is Unknown
**What goes wrong:** `regression_pred_last_step.csv` is headerless — row 0 maps to some trading day, but the CSV contains no dates. If the date index is wrong, all yfinance prices will be misaligned.
**Why it happens:** The inference script saves raw numpy arrays without date labels.
**How to avoid:** Derive the date index from the config's `test_start`/`test_end` keys using `pd.bdate_range` or from `data/Stock_SP500_2018-01-01_2024-01-01/split_indices.json`. Cross-check: the number of dates must equal `reg_pred.shape[0]` = 167. The test period date range is available in the `.conf` file under `[dataset] test_start` / `test_end`.
**Warning signs:** `assert len(dates) == reg_pred.shape[0]` fails → date range is wrong.

### Pitfall 2: Delisted / Missing Stocks in yfinance Download
**What goes wrong:** yfinance returns NaN for stocks that were delisted or added mid-period. A NaN close price propagated into weight * return gives NaN portfolio return.
**Why it happens:** S&P500 constituent changes, delistings, mergers during test period (2023-11-07 to ~2024-01-30 range).
**How to avoid:** After downloading prices, apply `df.fillna(method="ffill")` then `df.fillna(0)` for returns. For weight computation: if a stock has no price on a given day, set its weight to 0 and renormalize the remaining K stocks. Alternatively: download with `auto_adjust=True` which handles splits/dividends cleanly.
**Warning signs:** Portfolio daily return contains NaN values.

### Pitfall 3: Turnover Calculation Direction
**What goes wrong:** Using one-sided turnover (sum of positive weight changes only) instead of round-trip (sum of absolute weight changes = buys + sells). This halves the transaction cost, making the backtest look better than reality.
**Why it happens:** Common misreading of the fee formula.
**How to avoid:** `turnover = (weight_now - weight_prev).abs().sum()`. For top-K equal-weight with daily rebalancing and K=10 of 478 stocks, turnover is close to 2.0 most days (full portfolio replacement). Total cost = `turnover * FEE` where FEE = 0.001.

### Pitfall 4: First Day Return Has No Prior Price
**What goes wrong:** On the first day, `weight_prev` = 0 for all stocks, so turnover = sum(weight_now) = 1.0, which is correct (buying from cash costs 10bps × 1.0). But realized return requires the ratio `close[t] / close[t-1]` — the day before the first test day may not be in the downloaded price range.
**Why it happens:** yfinance download starts at `test_start`, which is the first prediction day. pct_change() returns NaN for the first row.
**How to avoid:** Download prices starting 5 calendar days before `test_start` to ensure `close[t-1]` exists for day 0. Set `start = pd.Timestamp(date_index[0]) - pd.Timedelta(days=10)`.

### Pitfall 5: scipy.stats.linregress Argument Order
**What goes wrong:** `linregress(x, y)` — x is the independent variable (SPY returns), y is the dependent (portfolio returns). Swapping them gives wrong slope/intercept.
**Why it happens:** Easy to transpose when reading the API.
**How to avoid:** Always: `linregress(spy_daily_returns, portfolio_daily_returns)` — SPY is x (market), portfolio is y (strategy). Beta = slope, alpha = intercept.

### Pitfall 6: Portfolio Return Denominator for First Day
**What goes wrong:** The Backtest notebook initializes `value_daily.iloc[:id_dates_trade[0]+1] = 1` to skip the period before the first signal. For daily rebalancing (every row has a signal), this simplifies — but if the first prediction row is treated as "day 0" (no prior holding), the initial portfolio has zero prior weight, which is handled correctly by `weight_prev = zeros`.

---

## Code Examples

### Deriving Date Index from Config

```python
# Source: project config pattern (config/Multitask_Stock.conf)
import configparser
import pandas as pd

cfg = configparser.ConfigParser()
cfg.read(os.path.join(project_root, "config", "Multitask_Stock.conf"))
test_start = cfg["dataset"]["test_start"]   # e.g. "2023-11-07"
test_end = cfg["dataset"]["test_end"]       # e.g. "2024-01-30"

# Generate business day date range
date_index = pd.bdate_range(start=test_start, end=test_end)
assert len(date_index) == reg_pred.shape[0], (
    f"Date range {len(date_index)} != prediction rows {reg_pred.shape[0]}"
)
```

### yfinance Batch Download (following Phase 2 pattern)

```python
# Source: data_processing_script/sp500_pipeline/download_ohlcv.py pattern
import yfinance as yf

def download_close_prices(tickers, start_date, end_date):
    """Download adjusted close prices for a list of tickers.

    Returns:
        pd.DataFrame: shape (n_days, n_tickers), NaNs for missing data
    """
    raw = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers
    return prices
```

### Write backtest_summary.csv (compute_ic.py pattern)

```python
# Source: scripts/compute_ic.py output pattern
summary = {
    "annualized_return": annualized_return,
    "sharpe_ratio": sharpe,
    "max_drawdown": max_drawdown,
    "alpha_annualized": alpha,
    "beta": beta,
    "top_k": top_k_n,
    "n_days": n_days,
    "total_return": total_return,
}
pd.DataFrame([summary]).to_csv(
    os.path.join(output_dir, "backtest_summary.csv"), index=False
)
```

### Write backtest_daily_returns.csv (with SPY column for Phase 6)

```python
# Include SPY column — useful for Phase 6 Streamlit without re-downloading
daily_df = pd.DataFrame({
    "date": [str(d.date()) for d in date_index],
    "portfolio_return": portfolio_returns,
    "spy_return": spy_daily_aligned.tolist(),
})
daily_df.to_csv(
    os.path.join(output_dir, "backtest_daily_returns.csv"), index=False
)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Qlib + Alpha-360 + HS300 benchmark | yfinance + custom TA + SPY benchmark | Phase 2-5 adaptation | Replaces Chinese-market infrastructure with US-market equivalent |
| CSI-300 close prices from local CSV | yfinance runtime download | Phase 5 decision | No local price archive required; always fresh for test period |
| Multi-model comparison chart (10+ models) | Two-line chart: Stockformer vs SPY | Phase 5 decision | Cleaner thesis figure |

**Deprecated/outdated:**
- `params.fee = 0.00` in the notebook (fee was zero in the notebook — Phase 5 corrects this to 0.001)
- `hs300_path` hardcoded CSV — replaced by yfinance SPY download
- `qlib.init(provider_uri=...)` — not used in Phase 5 at all

---

## Open Questions

1. **Config key names for test_start / test_end**
   - What we know: `config/Multitask_Stock.conf` uses ConfigParser format; Phase 3 reads from it
   - What's unclear: Exact section and key names for test period dates (might be `[dataset]`, `[backtest]`, or similar)
   - Recommendation: Read the config file at plan time and confirm the exact key names. Fallback: derive test period by reading the prediction CSV rows count (167) and matching against `pd.bdate_range` starting from a candidate date.

2. **tickers_file flag vs Phase 2 dependency**
   - What we know: `data/Stock_SP500_2018-01-01_2024-01-01/tickers.txt` already exists with 478 tickers in the correct column order
   - What's unclear: Whether Phase 2 needs to save a separate `sp500_tickers.txt` to `output_dir`, or if `--tickers_file` should default to the data directory path
   - Recommendation: Default `--tickers_file` to `data/Stock_SP500_2018-01-01_2024-01-01/tickers.txt` if the flag is not provided, since the file already exists and is in the correct order.

3. **Exact number of test-period trading days**
   - What we know: `regression_pred_last_step.csv` has 167 rows
   - What's unclear: Whether `pd.bdate_range(test_start, test_end)` yields exactly 167 business days (US market holidays are not business days in bdate_range, but they appear in the dataset)
   - Recommendation: Use NYSE calendar via `pandas_market_calendars` if the count doesn't match — but first try plain `pd.bdate_range` since 167 = approx 8 months of trading days.

---

## Validation Architecture

> `workflow.nyquist_validation` is `true` in .planning/config.json — section included.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (no config file — invoked as `python -m pytest tests/` from project root) |
| Config file | none (pytest uses auto-discovery) |
| Quick run command | `python -m pytest tests/test_backtest.py -x -q` |
| Full suite command | `python -m pytest tests/ -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PORT-01 | Top-K selection returns exactly K tickers per day | unit | `python -m pytest tests/test_backtest.py::test_top_k_selection -x` | ❌ Wave 0 |
| PORT-02 | Equal-weight portfolio: each selected stock weight = 1/K | unit | `python -m pytest tests/test_backtest.py::test_equal_weight -x` | ❌ Wave 0 |
| PORT-03 | Transaction cost deducted: 10bps * turnover per day | unit | `python -m pytest tests/test_backtest.py::test_transaction_cost -x` | ❌ Wave 0 |
| BACK-01 | Cumulative return series starts at 1.0 and matches (1+r).cumprod() | unit | `python -m pytest tests/test_backtest.py::test_cumulative_return -x` | ❌ Wave 0 |
| BACK-02 | Annualized return, Sharpe, max drawdown correct on synthetic data | unit | `python -m pytest tests/test_backtest.py::test_performance_metrics -x` | ❌ Wave 0 |
| BACK-03 | Alpha=intercept and beta=slope of linregress(spy, portfolio) | unit | `python -m pytest tests/test_backtest.py::test_alpha_beta -x` | ❌ Wave 0 |

**Note:** REQUIREMENTS.md includes `TEST-01: Unit tests for backtesting metrics (Sharpe, drawdown, alpha)` — this maps directly to BACK-02 and BACK-03. Phase 5 should satisfy TEST-01 partially; Phase 7 completes the full test pass.

### Sampling Rate
- **Per task commit:** `python -m pytest tests/test_backtest.py -x -q`
- **Per wave merge:** `python -m pytest tests/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_backtest.py` — covers PORT-01, PORT-02, PORT-03, BACK-01, BACK-02, BACK-03
- [ ] No new conftest fixtures needed — existing `conftest.py` fixtures (`sp500_ohlcv_fixture`, `feature_matrix_fixture`) are sufficient as a base

---

## Sources

### Primary (HIGH confidence)
- `Backtest/Backtest.ipynb` — direct source of Backtest class, calculate_performance_metrics(), portfolio construction algorithm
- `scripts/compute_ic.py` — verified structural pattern for argparse CLI, output_dir convention, CSV writing
- `data/Stock_SP500_2018-01-01_2024-01-01/tickers.txt` — confirmed 478 tickers, same order as CSV columns
- `output/Multitask_output_SP500_2018-2024/regression/regression_pred_last_step.csv` — confirmed shape (167, 478), headerless format

### Secondary (MEDIUM confidence)
- yfinance `auto_adjust=True` behavior — established from Phase 2 download_ohlcv.py usage; handles splits and dividends automatically
- scipy.stats.linregress argument convention — standard Python scientific computing; API stable since scipy 0.11

### Tertiary (LOW confidence)
- pd.bdate_range producing exactly 167 days for the test period — not verified against NYSE calendar; flagged as Open Question 3

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already in use in the project; no new dependencies
- Architecture: HIGH — algorithm directly ported from existing Backtest.ipynb; structural pattern from compute_ic.py
- Pitfalls: HIGH — identified from direct code inspection of the notebook and prediction CSV shape/format
- Test map: HIGH — follows established xfail/unit test pattern from Phases 2-4

**Research date:** 2026-03-14
**Valid until:** 2026-04-14 (stable domain — yfinance API changes infrequent; scipy linregress API stable)
