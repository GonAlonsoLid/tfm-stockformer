# Backtest Positions Output Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `backtest_positions.csv` to the backtest output — a long-format CSV with one row per selected stock per day, including its weight and predicted score.

**Architecture:** Three small, focused changes to `scripts/run_backtest.py`: (1) extend `run_backtest_loop` to collect position dicts, (2) extend `save_outputs` to write them as CSV, (3) thread the data through `main()`. All pure function logic is unchanged.

**Spec:** `docs/superpowers/specs/2026-03-14-backtest-positions-output-design.md`

**Tech Stack:** pandas, existing `scripts/run_backtest.py` patterns

---

## Chunk 1: Test + `run_backtest_loop` change

### Task 1: Write failing test for positions capture

**Files:**
- Modify: `tests/test_backtest.py`

- [ ] **Step 1: Add failing test**

  Append to `tests/test_backtest.py`:

  ```python
  # ── Positions output ─────────────────────────────────────────────────────────

  def test_positions_output():
      """run_backtest_loop returns positions list with one row per (day, ticker)."""
      import numpy as np
      import pandas as pd
      from scripts.run_backtest import run_backtest_loop

      tickers = ["A", "B", "C", "D", "E"]
      dates = pd.bdate_range("2023-01-03", periods=3)

      np.random.seed(7)
      pred_df = pd.DataFrame(
          np.random.rand(3, 5), index=dates, columns=tickers
      )
      price_cols = tickers + ["SPY"]
      prices = pd.DataFrame(
          np.ones((3, 6)) * 100 + np.arange(3)[:, None],
          index=dates,
          columns=price_cols,
      )

      _, _, positions = run_backtest_loop(pred_df, prices, tickers, top_k_n=2)
      pos_df = pd.DataFrame(positions)

      assert len(pos_df) == 6, f"Expected 6 rows, got {len(pos_df)}"
      assert set(pos_df.columns) == {"date", "ticker", "weight", "predicted_score"}
      assert (pos_df["weight"] == 0.5).all(), "All weights must equal 0.5 for top_k=2"
      assert pos_df.notna().all().all(), "No null values expected"
      for d, grp in pos_df.groupby("date"):
          assert abs(grp["weight"].sum() - 1.0) < 1e-9, f"Weights don't sum to 1.0 on {d}"
  ```

- [ ] **Step 2: Verify test fails**

  ```bash
  python -m pytest tests/test_backtest.py::test_positions_output -v
  ```

  Expected: `FAILED` — `ValueError: not enough values to unpack` (loop returns 2-tuple, test expects 3).

- [ ] **Step 3: Commit failing test**

  ```bash
  git add tests/test_backtest.py
  git commit -m "test(05-gap): add failing test for positions output in run_backtest_loop"
  ```

---

### Task 2: Extend `run_backtest_loop` to capture positions

**Files:**
- Modify: `scripts/run_backtest.py` (function `run_backtest_loop`, lines ~358–425)

- [ ] **Step 1: Add `positions = []` before the loop**

  In `run_backtest_loop`, just after:
  ```python
  weight_prev = pd.Series(0.0, index=tickers)
  ```
  add:
  ```python
  positions: list = []
  ```

- [ ] **Step 2: Append position dicts inside the loop**

  After line `top_k_index = select_top_k(scores, top_k_n)`, add:
  ```python
  _w = 1.0 / top_k_n
  for _ticker in top_k_index:
      positions.append({
          "date": date.strftime("%Y-%m-%d"),
          "ticker": _ticker,
          "weight": _w,
          "predicted_score": float(scores[_ticker]),
      })
  ```

- [ ] **Step 3: Update return statement**

  Change:
  ```python
  return portfolio_returns, spy_daily_returns
  ```
  to:
  ```python
  return portfolio_returns, spy_daily_returns, positions
  ```

- [ ] **Step 4: Update docstring**

  Change the `Returns:` block of `run_backtest_loop`:
  ```
  Returns:
      (portfolio_returns, spy_daily_returns, positions): two lists of floats
      (length n_days each) and a list of dicts with keys date/ticker/weight/predicted_score.
  ```

- [ ] **Step 5: Run test — must pass**

  ```bash
  python -m pytest tests/test_backtest.py::test_positions_output -v
  ```

  Expected: `PASSED`

- [ ] **Step 6: Run full suite — must stay green**

  ```bash
  python -m pytest tests/test_backtest.py -q
  ```

  Expected: all tests pass (note: `main()` is not called by existing tests, so the unpacking change there has no immediate effect yet — that's Task 3).

- [ ] **Step 7: Commit**

  ```bash
  git add scripts/run_backtest.py
  git commit -m "feat(05-gap): extend run_backtest_loop to capture daily positions"
  ```

---

## Chunk 2: `save_outputs`, `main()`, docstring

### Task 3: Update `save_outputs` to write positions CSV

**Files:**
- Modify: `scripts/run_backtest.py` (function `save_outputs`, lines ~428–511)

- [ ] **Step 1: Add `positions` parameter**

  Change the function signature from:
  ```python
  def save_outputs(
      output_dir: str,
      date_index: pd.DatetimeIndex,
      portfolio_returns: list,
      spy_daily_returns: list,
      metrics: dict,
      top_k_n: int,
  ) -> None:
  ```
  to:
  ```python
  def save_outputs(
      output_dir: str,
      date_index: pd.DatetimeIndex,
      portfolio_returns: list,
      spy_daily_returns: list,
      metrics: dict,
      top_k_n: int,
      positions: list,
  ) -> None:
  ```

- [ ] **Step 2: Write positions CSV**

  After the `backtest_daily_returns.csv` write block (after `daily_df.to_csv(...)`), add:

  ```python
  # --- backtest_positions.csv ---
  pos_df = pd.DataFrame(positions).sort_values(
      ["date", "predicted_score"], ascending=[True, False]
  )
  pos_path = os.path.join(output_dir, "backtest_positions.csv")
  pos_df.to_csv(pos_path, index=False)
  ```

- [ ] **Step 3: Update console print block**

  In the existing print block that logs output paths, add:
  ```python
  print(f"  {pos_path}")
  ```

- [ ] **Step 4: Update `save_outputs` docstring**

  Add to the list of files written:
  ```
  output_dir/backtest_positions.csv  -- daily holdings (date, ticker, weight, predicted_score)
  ```

- [ ] **Step 5: Commit**

  ```bash
  git add scripts/run_backtest.py
  git commit -m "feat(05-gap): add backtest_positions.csv to save_outputs"
  ```

---

### Task 4: Update `main()` and module docstring

**Files:**
- Modify: `scripts/run_backtest.py` (function `main()`, ~line 601; module docstring, lines 1–18)

- [ ] **Step 1: Fix unpack in `main()`**

  Find (approximately line 601):
  ```python
  portfolio_returns, spy_daily_returns = run_backtest_loop(
  ```
  Change to:
  ```python
  portfolio_returns, spy_daily_returns, positions = run_backtest_loop(
  ```

- [ ] **Step 2: Pass positions to `save_outputs`**

  In the `save_outputs(...)` call inside `main()`, add `positions=positions` as a keyword argument.

- [ ] **Step 3: Update module docstring**

  In the `Writes:` section at the top of the file, add:
  ```
  output_dir/backtest_positions.csv  -- daily top-K holdings (date, ticker, weight, predicted_score)
  ```

- [ ] **Step 4: Run full test suite**

  ```bash
  python -m pytest tests/test_backtest.py -q
  ```

  Expected: all tests pass.

- [ ] **Step 5: Smoke test end-to-end**

  ```bash
  python scripts/run_backtest.py --output_dir output/Multitask_output_SP500_2018-2024 --top_k 10
  ```

  Expected:
  - Script exits 0
  - `output/Multitask_output_SP500_2018-2024/backtest_positions.csv` exists
  - File has 1,670 rows + header (167 days × 10 positions)
  - Verify: `wc -l output/Multitask_output_SP500_2018-2024/backtest_positions.csv` → `1671`

- [ ] **Step 6: Commit**

  ```bash
  git add scripts/run_backtest.py
  git commit -m "feat(05-gap): wire positions through main() and update module docstring"
  ```
