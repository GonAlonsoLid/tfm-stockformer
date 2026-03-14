---
phase: quick
plan: 1
type: tdd
wave: 1
depends_on: []
files_modified:
  - tests/test_backtest.py
  - scripts/run_backtest.py
autonomous: true
requirements: []
must_haves:
  truths:
    - "run_backtest_loop returns a 3-tuple (portfolio_returns, spy_daily_returns, positions)"
    - "positions list contains one dict per selected stock per day with keys date/ticker/weight/predicted_score"
    - "backtest_positions.csv is written to output_dir when run_backtest.py is executed"
    - "existing tests continue to pass after the changes"
  artifacts:
    - path: "scripts/run_backtest.py"
      provides: "Extended run_backtest_loop (3-tuple), updated save_outputs, updated main(), updated docstring"
    - path: "tests/test_backtest.py"
      provides: "test_positions_output covering shape, columns, weight values, and per-day sums"
  key_links:
    - from: "run_backtest_loop"
      to: "main()"
      via: "3-tuple unpack"
      pattern: "portfolio_returns, spy_daily_returns, positions = run_backtest_loop"
    - from: "main()"
      to: "save_outputs"
      via: "positions kwarg"
      pattern: "positions=positions"
    - from: "save_outputs"
      to: "output_dir/backtest_positions.csv"
      via: "pd.DataFrame(positions).to_csv"
      pattern: "pos_df\\.to_csv"
---

<objective>
Add `backtest_positions.csv` to the output of `scripts/run_backtest.py`.

Purpose: The backtest currently produces metrics and an equity curve but no record of which stocks were held each day or what score drove their selection. This CSV enables portfolio auditing and supports the live-trading inference workflow.
Output: `output_dir/backtest_positions.csv` — long-format, one row per selected stock per day, columns: date, ticker, weight, predicted_score.
</objective>

<execution_context>
@/Users/gonzaloalonsolidon/.claude/get-shit-done/workflows/execute-plan.md
@/Users/gonzaloalonsolidon/.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@docs/superpowers/specs/2026-03-14-backtest-positions-output-design.md
@docs/superpowers/plans/2026-03-14-backtest-positions-output.md
@scripts/run_backtest.py
@tests/test_backtest.py
</context>

<tasks>

<task type="auto" tdd="true">
  <name>Task 1: Add failing test and extend run_backtest_loop</name>
  <files>tests/test_backtest.py, scripts/run_backtest.py</files>
  <behavior>
    - test_positions_output: run_backtest_loop called with 5 tickers, top_k=2, 3 days returns a positions list of length 6
    - Columns of the resulting DataFrame are exactly {"date", "ticker", "weight", "predicted_score"}
    - All weights equal 0.5 (1/2) and sum to 1.0 per day
    - No null values in any column
  </behavior>
  <action>
    RED — append test_positions_output to tests/test_backtest.py exactly as written in the plan (docs/superpowers/plans/2026-03-14-backtest-positions-output.md Task 1 Step 1). Run the test and confirm it FAILS with "not enough values to unpack". Commit: `test(05-gap): add failing test for positions output in run_backtest_loop`.

    GREEN — modify scripts/run_backtest.py function run_backtest_loop (lines ~358–425):
    1. Add `positions: list = []` after `weight_prev = pd.Series(0.0, index=tickers)`.
    2. After `top_k_index = select_top_k(scores, top_k_n)`, append one dict per ticker:
       `_w = 1.0 / top_k_n` then `positions.append({"date": date.strftime("%Y-%m-%d"), "ticker": _ticker, "weight": _w, "predicted_score": float(scores[_ticker])})`.
    3. Change the return from `return portfolio_returns, spy_daily_returns` to `return portfolio_returns, spy_daily_returns, positions`.
    4. Update the Returns: docstring block to describe the 3-tuple.

    Run `python -m pytest tests/test_backtest.py::test_positions_output -v` — must PASS.
    Run `python -m pytest tests/test_backtest.py -q` — all tests must pass (existing tests are unaffected because main() unpack error is not exercised by them yet).
    Commit: `feat(05-gap): extend run_backtest_loop to capture daily positions`.
  </action>
  <verify>
    <automated>python -m pytest tests/test_backtest.py -q</automated>
  </verify>
  <done>test_positions_output passes; all other backtest tests still pass; run_backtest_loop returns a 3-tuple.</done>
</task>

<task type="auto">
  <name>Task 2: Update save_outputs, main(), and module docstring</name>
  <files>scripts/run_backtest.py</files>
  <action>
    Modify scripts/run_backtest.py in three places:

    1. save_outputs signature — add `positions: list` as the last parameter (after top_k_n). Inside the function body, after the block that writes backtest_daily_returns.csv, add:
       ```python
       # --- backtest_positions.csv ---
       pos_df = pd.DataFrame(positions).sort_values(
           ["date", "predicted_score"], ascending=[True, False]
       )
       pos_path = os.path.join(output_dir, "backtest_positions.csv")
       pos_df.to_csv(pos_path, index=False)
       ```
       Then add `print(f"  {pos_path}")` to the existing console print block.
       Update save_outputs docstring to list the new output file:
       `output_dir/backtest_positions.csv  -- daily holdings (date, ticker, weight, predicted_score)`.
       Commit: `feat(05-gap): add backtest_positions.csv to save_outputs`.

    2. main() unpack — find the line `portfolio_returns, spy_daily_returns = run_backtest_loop(` (approx line 601) and change it to `portfolio_returns, spy_daily_returns, positions = run_backtest_loop(`. Then in the save_outputs(...) call, add `positions=positions` as a keyword argument.

    3. Module docstring — in the `Writes:` section at the top of the file, add:
       `output_dir/backtest_positions.csv  -- daily top-K holdings (date, ticker, weight, predicted_score)`.

    Run full test suite: `python -m pytest tests/test_backtest.py -q` — must pass.

    Run smoke test: `python scripts/run_backtest.py --output_dir output/Multitask_output_SP500_2018-2024 --top_k 10`
    Verify: `wc -l output/Multitask_output_SP500_2018-2024/backtest_positions.csv` returns 1671 (1670 data rows + 1 header).

    Commit: `feat(05-gap): wire positions through main() and update module docstring`.
  </action>
  <verify>
    <automated>python -m pytest tests/test_backtest.py -q && test -f output/Multitask_output_SP500_2018-2024/backtest_positions.csv</automated>
  </verify>
  <done>All backtest tests pass; backtest_positions.csv exists with 1671 lines; script exits 0; module docstring and save_outputs docstring updated.</done>
</task>

</tasks>

<verification>
- `python -m pytest tests/test_backtest.py -q` — all tests green including test_positions_output
- `wc -l output/Multitask_output_SP500_2018-2024/backtest_positions.csv` returns 1671
- `head -2 output/Multitask_output_SP500_2018-2024/backtest_positions.csv` shows header `date,ticker,weight,predicted_score`
- `python scripts/run_backtest.py --help` still works (CLI interface unchanged)
</verification>

<success_criteria>
- backtest_positions.csv is written to the output directory on every run
- One row per (day, ticker) with correct weight (1/top_k) and the raw predicted score
- Existing tests and outputs are unaffected
- Three focused commits: failing test, loop change, save/main/docstring change
</success_criteria>

<output>
After completion, create `.planning/quick/1-add-backtest-positions-csv-output-to-scr/1-SUMMARY.md`
</output>
