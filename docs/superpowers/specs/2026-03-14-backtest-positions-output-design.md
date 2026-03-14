# Design: Backtest Positions Output

**Date:** 2026-03-14
**Scope:** Add `backtest_positions.csv` to `scripts/run_backtest.py` output

---

## Problem

The backtest produces performance metrics and an equity curve, but there is no output showing which stocks were held on each day or what the model predicted for them. This makes it hard to audit portfolio decisions or understand what drove returns.

---

## Solution

Add a long-format positions CSV that records one row per selected stock per day, with the predicted score that drove the selection.

### Output file

`output_dir/backtest_positions.csv`

| Column | Type | Description |
|--------|------|-------------|
| `date` | date string (YYYY-MM-DD) | Trading day |
| `ticker` | string | Selected stock symbol |
| `weight` | float | Equal weight = 1/K |
| `predicted_score` | float | Raw regression prediction for this ticker on this day |

Rows: `n_days × top_k` (nominal — see edge case note below)
Sorted by: date ascending, predicted_score descending within each date

**Edge case:** `select_top_k` uses `nlargest(k)`, which returns fewer than k tickers if the prediction Series has fewer than k non-NaN values. In that case, the positions list will have fewer rows for that day and weights will not sum to 1.0. For Phase 5, prediction scores are always complete (no NaN), so this invariant holds. No guard is needed.

---

## Changes

### 1. `run_backtest_loop` — capture positions

Extend the return tuple from 2 to 3 elements. Inside the loop, after `select_top_k`, append one dict per selected ticker:

```python
positions = []
# inside loop, after select_top_k:
w = 1.0 / top_k_n
for ticker in top_k_index:
    positions.append({
        "date": date.strftime("%Y-%m-%d"),
        "ticker": ticker,
        "weight": w,
        "predicted_score": float(scores[ticker]),
    })
# updated return (was 2-tuple, now 3-tuple):
return portfolio_returns, spy_daily_returns, positions
```

### 2. `save_outputs` — updated signature and positions CSV

Add `positions` parameter (after `top_k_n`):

```python
def save_outputs(
    output_dir: str,
    date_index: pd.DatetimeIndex,
    portfolio_returns: list,
    spy_daily_returns: list,
    metrics: dict,
    top_k_n: int,
    positions: list,          # NEW
) -> None:
```

Inside the function, after writing `backtest_daily_returns.csv`:

```python
# --- backtest_positions.csv ---
pos_df = pd.DataFrame(positions).sort_values(
    ["date", "predicted_score"], ascending=[True, False]
)
pos_path = os.path.join(output_dir, "backtest_positions.csv")
pos_df.to_csv(pos_path, index=False)
```

Update the console print block (currently logs 3 files) to also log the new file:

```python
print(f"  {pos_path}")
```

### 3. `main()` — updated unpack and `save_outputs` call

Unpack 3 values from `run_backtest_loop` (was 2):

```python
portfolio_returns, spy_daily_returns, positions = run_backtest_loop(
    pred_df, prices, tickers, top_k_n
)
```

Pass `positions` to `save_outputs`:

```python
save_outputs(
    output_dir=args.output_dir,
    date_index=date_index,
    portfolio_returns=portfolio_returns,
    spy_daily_returns=spy_daily_returns,
    metrics=metrics,
    top_k_n=top_k_n,
    positions=positions,
)
```

### 4. Module docstring — add new file

Add `output_dir/backtest_positions.csv` to the Writes section.

---

## Tests

Add one test to `tests/test_backtest.py` — `test_positions_output`:

```python
def test_positions_output():
    from scripts.run_backtest import run_backtest_loop

    import pandas as pd
    import numpy as np

    tickers = ["A", "B", "C", "D", "E"]
    dates = pd.bdate_range("2023-01-03", periods=3)

    # pred_df: 3 days × 5 tickers, deterministic scores
    np.random.seed(7)
    pred_data = np.random.rand(3, 5)
    pred_df = pd.DataFrame(pred_data, index=dates, columns=tickers)

    # prices: include SPY column; slight upward drift
    price_cols = tickers + ["SPY"]
    prices = pd.DataFrame(
        np.ones((3, 6)) * 100 + np.arange(3)[:, None],
        index=dates,
        columns=price_cols,
    )

    _, _, positions = run_backtest_loop(pred_df, prices, tickers, top_k_n=2)

    pos_df = pd.DataFrame(positions)

    # Shape: 3 days × 2 positions = 6 rows
    assert len(pos_df) == 6, f"Expected 6 rows, got {len(pos_df)}"

    # Columns
    assert set(pos_df.columns) == {"date", "ticker", "weight", "predicted_score"}

    # Equal weight = 0.5
    assert (pos_df["weight"] == 0.5).all(), "All weights must be 0.5 for top_k=2"

    # No nulls
    assert pos_df.notna().all().all(), "No null values expected"

    # Weights sum to 1.0 per day
    for d, grp in pos_df.groupby("date"):
        assert abs(grp["weight"].sum() - 1.0) < 1e-9, f"Weights don't sum to 1.0 on {d}"
```

---

## What does NOT change

- Pure functions (`select_top_k`, `build_portfolio_weights`, `compute_daily_return`, `compute_performance_metrics`)
- CLI interface (no new flags)
- Other output files (`equity_curve.png`, `backtest_summary.csv`, `backtest_daily_returns.csv`)
- Existing tests
