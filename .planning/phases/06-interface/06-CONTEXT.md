# Phase 6: Interface - Context

**Gathered:** 2026-03-16
**Status:** Ready for planning

<domain>
## Phase Boundary

Build a Streamlit app (`app.py`) that lets the user configure and run the inference + backtest pipeline interactively, then explore the results through an equity curve chart, metrics table, and prediction heatmap. No model training — training is always pre-done separately. Entry point: `streamlit run app.py`.

</domain>

<decisions>
## Implementation Decisions

### Pipeline execution mode
- Clicking "Run" executes `run_inference.py` then `run_backtest.py` as subprocesses — NOT full model training
- Training is always done separately before using the app
- While running: show a `st.spinner()` overlay with a live scrolling log of subprocess stdout so the user can see progress
- Config path is specified by the user in the sidebar (text input field)

### App layout
- **Single page with sidebar** — classic Streamlit pattern
- Sidebar contains all controls; main area shows results
- **Initial state**: main area shows empty placeholder with instructions ("Configure settings in the sidebar and click Run to start") — no charts until a run completes
- After a successful run, results appear in the main area without page reload
- A UI design contract (UI-SPEC.md) will be generated via `/gsd:ui-phase 6` before planning — visual design decisions (colors, typography, spacing, component styling) deferred to that step

### Sidebar controls
Include all parameters relevant to backtesting that the user may need to adjust:
- **Config path** — text input for the `.conf` file path
- **Output directory** — text input for where results are saved/read
- **Top-K** — number input or slider for portfolio size (default: 10)
- **Transaction cost** — basis points for round-trip cost (default: 10bps)
- **Benchmark ticker** — text input (default: SPY)
- **Date range** — start date and end date pickers for the test window
- **Run button** — triggers the pipeline

Do NOT include: train/test split ratios, training hyperparameters, or architecture settings.

### Date range semantics
- The date range controls the **inference + backtest test window** — inference runs on selected dates; backtest evaluates that window
- The `.conf` file's training split is unchanged — only the evaluation window shifts
- If the user selects a date range with no valid checkpoint or data: show a clear error message ("No checkpoint/data found for this date range. Ensure the pipeline has been trained on this period.") — do not crash

### Charting library
- **Plotly** for all charts in the app (equity curve and heatmap) — interactive hover tooltips, zoom, pan; native `st.plotly_chart()` support
- The pre-existing static `equity_curve.png` (matplotlib) is separate and not reused in the Streamlit app

### Equity curve display
- Plotly line chart: portfolio cumulative return vs SPY cumulative return, both starting at 1.0
- Interactive: hover shows date + portfolio value + SPY value

### Metrics table
- Display all columns from `backtest_summary.csv`: annualized return, Sharpe ratio, max drawdown, alpha (annualized), beta, top-K, n_days, total return
- Use `st.dataframe()` or `st.table()` — Claude's discretion on formatting

### Prediction heatmap
- Scope: **top-K stocks only** (same K as portfolio) — not full 500-stock universe
- Color encoding: **predicted return scores** (regression output) with a diverging colormap (negative=red, positive=green)
- Library: Plotly heatmap (`px.imshow` or `go.Heatmap`)
- Axes: stocks (y-axis) × dates (x-axis)
- Data source: read from `regression_pred_last_step.csv` in the output directory, filtered to top-K tickers

### Claude's Discretion
- Exact Plotly styling (colors, fonts, margins, DPI equivalent for export)
- Whether the metrics table uses `st.dataframe()` vs `st.table()` vs custom HTML
- Exact column formatting in metrics table (e.g., percentages vs decimals)
- How to handle missing/NaN values gracefully in charts
- Whether to add a "Download results" button for CSV export
- Whether to show backtest_positions.csv as a third table/section (positions per day)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Requirements
- `.planning/REQUIREMENTS.md` §Interface — UI-01 through UI-04 define the four required UI components
- `.planning/ROADMAP.md` §Phase 6 — success criteria for what must be TRUE at phase completion

### Upstream outputs (Streamlit reads these)
- `output/Multitask_output_SP500_2018-2024/backtest_daily_returns.csv` — columns: date, portfolio_return, spy_return
- `output/Multitask_output_SP500_2018-2024/backtest_summary.csv` — one-row CSV with: annualized_return, sharpe_ratio, max_drawdown, alpha_annualized, beta, top_k, n_days, total_return
- `output/Multitask_output_SP500_2018-2024/backtest_positions.csv` — columns: date, ticker, weight, predicted_score
- `output/Multitask_output_SP500_2018-2024/regression/regression_pred_last_step.csv` — raw prediction matrix (stocks × dates, headerless)

### Scripts being invoked as subprocesses
- `scripts/run_inference.py` — inference script; takes `--config` flag
- `scripts/run_backtest.py` — backtest script; takes `--output_dir`, `--top_k` flags

No external UI specs — requirements fully captured in decisions above. UI design contract to be generated in UI-SPEC.md before planning.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `scripts/run_backtest.py`: Already structured as a subprocess-callable script with argparse; outputs all 4 CSV/PNG files to output_dir
- `scripts/run_inference.py`: Same argparse pattern; takes `--config` for the .conf file
- `scripts/compute_ic.py`: Reference for how scripts are invoked and what output_dir layout looks like
- `output/Multitask_output_SP500_2018-2024/backtest_daily_returns.csv`: Already has correct 3-column format (date, portfolio_return, spy_return) ready for Plotly line chart

### Established Patterns
- All scripts use argparse with `--output_dir` and config flags — consistent subprocess invocation pattern
- Output directory structure is flat: all CSVs/PNGs at `output_dir/` root, regression predictions at `output_dir/regression/`
- yfinance usage is guarded with try/except in run_backtest.py — app must ensure yfinance is installed

### Integration Points
- `app.py` lives at project root (required by `streamlit run app.py` success criterion)
- Subprocess invocations: `python scripts/run_inference.py --config <path>` then `python scripts/run_backtest.py --output_dir <path> --top_k <K>`
- Read results from `output_dir` after subprocess completes
- `regression_pred_last_step.csv` is headerless — use `pd.read_csv(..., header=None)` (established project convention)

</code_context>

<specifics>
## Specific Ideas

- UI design should be improved beyond basic Streamlit defaults — `/gsd:ui-phase 6` will generate a UI-SPEC.md design contract before planning to define visual polish decisions
- The app is a thesis demo tool — it should look clean and presentable, not like a default Streamlit prototype

</specifics>

<deferred>
## Deferred Ideas

- None — discussion stayed within phase scope

</deferred>

---

*Phase: 06-interface*
*Context gathered: 2026-03-16*
