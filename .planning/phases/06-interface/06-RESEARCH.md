# Phase 6: Interface - Research

**Researched:** 2026-03-16
**Domain:** Streamlit app, Plotly charting, subprocess live streaming, Python 3.9 compatibility
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Pipeline execution mode**
- Clicking "Run" executes `run_inference.py` then `run_backtest.py` as subprocesses — NOT full model training
- Training is always done separately before using the app
- While running: show a `st.spinner()` overlay with a live scrolling log of subprocess stdout
- Config path is specified by the user in the sidebar (text input field)

**App layout**
- Single page with sidebar — classic Streamlit pattern
- Sidebar contains all controls; main area shows results
- Initial state: main area shows empty placeholder with instructions — no charts until a run completes
- After a successful run, results appear in the main area without page reload

**Sidebar controls** (exactly these, in this order)
- Config path — text input, default `"config/Multitask_Stock_SP500.conf"`
- Output directory — text input, default `"output/Multitask_output_SP500_2018-2024"`
- Top-K — number input, default 10
- Transaction cost — basis points number input, default 10
- Benchmark ticker — text input, default `"SPY"`
- Start date / End date — date pickers
- Run button

Do NOT include: train/test split ratios, training hyperparameters, or architecture settings.

**Date range semantics**
- Controls the inference + backtest test window only
- .conf training split is unchanged
- Error message if no checkpoint/data for selected range (do not crash)

**Charting library**
- Plotly for all charts — equity curve and heatmap

**Equity curve**
- `go.Scatter` line chart: portfolio cumulative return vs SPY, both starting at 1.0
- Interactive hover: date + portfolio value + SPY value

**Metrics table**
- All columns from `backtest_summary.csv`: annualized_return, sharpe_ratio, max_drawdown, alpha_annualized, beta, top_k, n_days, total_return

**Prediction heatmap**
- Top-K stocks only
- Diverging colormap (negative=red, positive=green), zmid=0
- Plotly heatmap (`go.Heatmap`)
- Axes: stocks (y) × dates (x)
- Source: `regression_pred_last_step.csv`, filtered to top-K tickers by matching against `backtest_positions.csv`

### Claude's Discretion

- Exact Plotly styling (colors, fonts, margins)
- Whether metrics table uses `st.dataframe()` vs `st.table()` vs custom HTML
- Exact column formatting in metrics table (percentages vs decimals)
- How to handle missing/NaN values gracefully in charts
- Whether to add a "Download results" button for CSV export
- Whether to show `backtest_positions.csv` as a third table/section

### Deferred Ideas (OUT OF SCOPE)

None — discussion stayed within phase scope.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| UI-01 | Streamlit app with date range selector and run pipeline button | Streamlit 1.50.x sidebar date_input + button patterns confirmed; subprocess execution pattern validated |
| UI-02 | Equity curve chart showing portfolio vs SPY cumulative returns | Plotly go.Scatter pattern confirmed; backtest_daily_returns.csv confirmed at (167, 3) with correct columns |
| UI-03 | Metrics summary table (annualized return, Sharpe, max drawdown, alpha, beta) | backtest_summary.csv confirmed; st.column_config.NumberColumn format strings verified |
| UI-04 | Model prediction heatmap (stock × date grid for test period) | regression_pred_last_step.csv confirmed at (167, 478) headerless; go.Heatmap zmid pattern verified |
</phase_requirements>

---

## Summary

Phase 6 builds `app.py` at the project root — a Streamlit single-page dashboard with sidebar controls that orchestrates inference and backtesting as subprocesses, then renders three visualizations from the resulting CSVs. The domain is straightforward: Streamlit widget state management, subprocess live output streaming, and Plotly chart construction. All upstream data files already exist on disk and their shapes are verified.

The most important constraint is Python version compatibility: the project venv is Python 3.9, and Streamlit 1.50.0 is the last release to support Python 3.9 (support was dropped in 1.51.0, October 2025). The app must be pinned to `streamlit>=1.35,<=1.50` for venv compatibility. All APIs used in the UI-SPEC (st.spinner, st.empty, st.dataframe with column_config, st.plotly_chart, st.download_button) are present in 1.35+.

The live stdout streaming pattern must use `subprocess.Popen` with `stdout=PIPE`, reading line-by-line via `iter(proc.stdout.readline, b'')` and calling `st.empty().code()` on each update. The asyncio approach requires Python 3.10+ event loop integration that does not apply cleanly to Streamlit's synchronous execution model in 3.9.

**Primary recommendation:** Use `streamlit>=1.35,<=1.50` + `plotly>=5.18,<7` in requirements.txt. Build `app.py` with `st.session_state` for run state, synchronous `subprocess.Popen` for live log streaming, and `go.Scatter` / `go.Heatmap` for charts.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| streamlit | >=1.35,<=1.50 | UI framework | Only Streamlit version range compatible with Python 3.9 venv that also has column_config, spinner with show_time, download_button |
| plotly | >=5.18,<7 | Interactive charts | Decided in CONTEXT.md; native st.plotly_chart() support; go.Heatmap zmid confirmed |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas | ==2.3.3 (already pinned) | Read CSVs and format metrics table | Already installed; used to build chart DataFrames |
| numpy | ==1.24.4 (already pinned) | Array ops for heatmap normalization | Already installed |
| subprocess (stdlib) | Python 3.9+ | Spawn run_inference.py and run_backtest.py | For pipeline execution |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| streamlit<=1.50 | streamlit 1.55 | 1.55 requires Python >=3.10; project venv is 3.9 — incompatible |
| go.Heatmap | px.imshow | px.imshow works on 2D arrays but zmid + custom colorscale is cleaner with go.Heatmap |
| subprocess.Popen iter | asyncio subprocess | asyncio approach requires Python 3.10+ for clean Streamlit integration |

**Installation (add to requirements.txt):**
```bash
pip install "streamlit>=1.35,<=1.50" "plotly>=5.18,<7"
```

---

## Architecture Patterns

### Recommended Project Structure

```
app.py                            # Entry point: streamlit run app.py
.streamlit/
└── config.toml                   # Theme: dark, primaryColor=#4C9BE8
scripts/
├── run_inference.py              # Subprocess 1 — takes --config
└── run_backtest.py               # Subprocess 2 — takes --output_dir, --top_k
output/Multitask_output_SP500_2018-2024/
├── backtest_daily_returns.csv    # Input for equity curve
├── backtest_summary.csv          # Input for metrics table
├── backtest_positions.csv        # Input for top-K ticker list
└── regression/
    └── regression_pred_last_step.csv  # Input for heatmap (headerless)
```

### Pattern 1: Session State for Run Lifecycle

**What:** Use `st.session_state` to track whether a run is in progress and whether results are available.
**When to use:** Required to persist results across Streamlit reruns without re-executing the pipeline.

```python
# Source: Streamlit official session state docs
if "run_complete" not in st.session_state:
    st.session_state["run_complete"] = False
if "results" not in st.session_state:
    st.session_state["results"] = None

# After pipeline finishes:
st.session_state["run_complete"] = True
st.session_state["results"] = load_results(output_dir)
```

Key rule: never assign to a session_state key that corresponds to a widget already rendered in the current run — set flags (`run_complete`, `running`) rather than widget keys.

### Pattern 2: Live Subprocess Output (synchronous, Python 3.9 compatible)

**What:** Stream subprocess stdout line-by-line into an `st.empty()` code block.
**When to use:** For `run_inference.py` and `run_backtest.py` — both print progress to stdout.

```python
# Source: subprocess stdlib + Streamlit st.empty() pattern
import subprocess, sys

log_placeholder = st.empty()
accumulated = []

proc = subprocess.Popen(
    [sys.executable, "scripts/run_inference.py", "--config", config_path],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,   # merge stderr into stdout
    text=True,
    bufsize=1,                  # line-buffered
)
for line in iter(proc.stdout.readline, ""):
    accumulated.append(line.rstrip())
    log_placeholder.code("\n".join(accumulated[-50:]))  # keep last 50 lines
proc.wait()
return proc.returncode
```

Use `sys.executable` to ensure the subprocess runs in the same Python environment as the app. Merge stderr with `stderr=subprocess.STDOUT` so all output appears in one log. The `bufsize=1` ensures line-buffering so each print from the script appears immediately.

**Critical:** `proc.stdout.readline` blocks until a line arrives — this is intentional; Streamlit's synchronous model reruns on user interaction but the subprocess loop here runs to completion before the app re-renders. The log updates are live because `st.empty().code()` pushes incremental updates to the browser.

### Pattern 3: Equity Curve Chart

**What:** Plotly `go.Scatter` line chart from `backtest_daily_returns.csv`.

```python
# Source: verified against actual CSV schema (date, portfolio_return, spy_return)
import plotly.graph_objects as go
import pandas as pd

df = pd.read_csv(f"{output_dir}/backtest_daily_returns.csv", parse_dates=["date"])
portfolio_cum = (1 + df["portfolio_return"]).cumprod()
spy_cum = (1 + df["spy_return"]).cumprod()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["date"], y=portfolio_cum,
    name="Portfolio", line=dict(color="#4C9BE8", width=2),
    hovertemplate="%{x|%Y-%m-%d}: Portfolio %{y:.3f}x<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=df["date"], y=spy_cum,
    name="SPY", line=dict(color="#9CA3AF", width=1.5, dash="dot"),
    hovertemplate="%{x|%Y-%m-%d}: SPY %{y:.3f}x<extra></extra>",
))
fig.update_layout(
    title="Cumulative Return: Portfolio vs SPY",
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(x=0.01, y=0.99),
    xaxis=dict(tickformat="%b %Y", gridcolor="#374151"),
    yaxis=dict(ticksuffix="x", gridcolor="#374151"),
)
st.plotly_chart(fig, use_container_width=True)
```

### Pattern 4: Prediction Heatmap

**What:** `go.Heatmap` from `regression_pred_last_step.csv` (headerless, shape 167×478), filtered to top-K tickers.

```python
# Source: verified CSV shape (167, 478) and backtest_positions.csv columns
import plotly.graph_objects as go
import pandas as pd, numpy as np

# Load raw predictions (headerless)
pred_df = pd.read_csv(f"{output_dir}/regression/regression_pred_last_step.csv", header=None)

# Get top-K ticker list in column order from backtest_positions.csv
# (backtest_positions has columns: date, ticker, weight, predicted_score)
pos_df = pd.read_csv(f"{output_dir}/backtest_positions.csv")
top_k_tickers = pos_df["ticker"].unique().tolist()

# Load tickers.txt to get column-index mapping
with open("data/Stock_SP500_2018-01-01_2024-01-01/tickers.txt") as f:
    all_tickers = [t.strip() for t in f if t.strip()]

# Filter columns to top-K indices
col_indices = [all_tickers.index(t) for t in top_k_tickers if t in all_tickers]
filtered_pred = pred_df.iloc[:, col_indices]
filtered_pred.columns = top_k_tickers[:len(col_indices)]

# Symmetric color range centered at 0
max_abs = float(np.nanmax(np.abs(filtered_pred.values)))

fig = go.Figure(go.Heatmap(
    z=filtered_pred.values.T,      # shape: (K, n_days) — stocks on y-axis
    x=list(range(pred_df.shape[0])),  # date indices (replace with actual dates if available)
    y=filtered_pred.columns.tolist(),
    colorscale=[[0, "#DC2626"], [0.5, "#F9FAFB"], [1, "#16A34A"]],
    zmid=0,
    zmin=-max_abs,
    zmax=max_abs,
    colorbar=dict(title="Predicted Score"),
))
k = len(top_k_tickers)
fig.update_layout(
    title=f"Prediction Scores — Top-{k} Stocks",
    height=max(300, min(k * 24, 600)),
    xaxis=dict(tickangle=-45),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)
st.plotly_chart(fig, use_container_width=True)
```

Note: `regression_pred_last_step.csv` is headerless (confirmed shape 167×478). The x-axis date mapping requires loading `backtest_daily_returns.csv` for the date index — the 167 rows of pred correspond to 167 dates in that CSV.

### Pattern 5: Metrics Table

**What:** `st.dataframe()` with `column_config` for formatted display.

```python
# Source: Streamlit docs — st.column_config.NumberColumn format parameter
import streamlit as st
import pandas as pd

summary = pd.read_csv(f"{output_dir}/backtest_summary.csv")
summary = summary.fillna("—")   # NaN → em dash before column_config

st.dataframe(
    summary,
    use_container_width=True,
    column_config={
        "annualized_return": st.column_config.NumberColumn("Ann. Return", format="%.2f %%"),
        "total_return":      st.column_config.NumberColumn("Total Return", format="%.2f %%"),
        "max_drawdown":      st.column_config.NumberColumn("Max Drawdown", format="%.2f %%"),
        "alpha_annualized":  st.column_config.NumberColumn("Alpha (Ann.)", format="%.2f %%"),
        "sharpe_ratio":      st.column_config.NumberColumn("Sharpe", format="%.2f"),
        "beta":              st.column_config.NumberColumn("Beta", format="%.2f"),
        "top_k":             st.column_config.NumberColumn("Top-K", format="%d"),
        "n_days":            st.column_config.NumberColumn("Days", format="%d"),
    },
)
```

**Warning:** `df.fillna("—")` changes numeric columns to object dtype, which breaks `NumberColumn` formatting. Correct approach: do NOT pre-fill NaN; instead rely on Streamlit's default NaN rendering, or convert NaN-containing cells to strings after formatting.

### Pattern 6: Streamlit Theme Config

```toml
# .streamlit/config.toml
[theme]
base = "dark"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1E2130"
primaryColor = "#4C9BE8"
textColor = "#E5E7EB"
font = "sans serif"
```

### Anti-Patterns to Avoid

- **Importing model code at app.py top level:** `run_inference.py` imports torch, Stockformer model, etc. at import time — those trigger side effects (log file creation, SummaryWriter). Never `import run_inference` in app.py; always invoke as subprocess.
- **Using `sys.argv` to pass date ranges to subprocesses:** The date range in the sidebar controls the evaluation window for display purposes; the actual inference uses the `.conf` file's training split. Do not attempt to pass date range as CLI args to `run_inference.py` — that script does not accept date range flags.
- **asyncio subprocess in Python 3.9 with Streamlit:** Streamlit's event loop management does not cleanly support `asyncio.create_subprocess_exec` on Python 3.9 — stick with synchronous `subprocess.Popen` + `iter(readline, "")`.
- **`df.fillna("—")` before NumberColumn:** Converts dtype to object, breaking numeric format strings. Handle NaN after formatting, not before.
- **Hardcoding the tickers.txt path:** The app receives output_dir from the sidebar; tickers.txt lives in the data dir. Either derive it from the config path or expose it as a sidebar control with a default.
- **`st.rerun()` inside the subprocess loop:** Calling `st.rerun()` mid-loop terminates execution. Never call it while the subprocess is still running.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Percentage column formatting | Custom string formatting before st.dataframe | `st.column_config.NumberColumn(format="%.2f %%")` | Handles display, alignment, and tooltips consistently |
| Diverging colorscale centering | Manual zmin/zmax arithmetic | `go.Heatmap(zmid=0, zmin=-max_abs, zmax=max_abs)` | zmid=0 makes colorscale symmetric around 0 automatically |
| Dark theme styling | Custom CSS injection | `.streamlit/config.toml` with `base="dark"` | Theme config is the canonical Streamlit approach; CSS injection is fragile |
| Subprocess environment detection | Hardcoded python path | `sys.executable` | Ensures subprocess uses the same Python venv as the app |

---

## Common Pitfalls

### Pitfall 1: NaN in fillna Breaks column_config Formatting

**What goes wrong:** `df.fillna("—")` converts numeric columns to object dtype. When you then pass `st.column_config.NumberColumn(format="%.2f %%")`, Streamlit cannot format object strings with a numeric format string — the column renders as plain text or throws a warning.
**Why it happens:** pandas dtype promotion when mixing str and float.
**How to avoid:** Do not call `fillna("—")` on the full DataFrame before passing to `st.dataframe`. Instead, either accept Streamlit's default NaN display (shows empty cell), or multiply the float values by 100 and format as string in a separate display DataFrame.
**Warning signs:** Percentage columns show raw floats or "—" without the "%" suffix.

### Pitfall 2: Subprocess Buffering Causes No Live Output

**What goes wrong:** The live log in `st.empty()` never updates during the run — all output appears at the end.
**Why it happens:** By default, Python scripts buffer stdout. If `run_inference.py` or `run_backtest.py` are run without forcing line-buffering, `readline()` blocks until the buffer flushes (at process exit).
**How to avoid:** Pass `bufsize=1` to `subprocess.Popen` AND run the subprocess with `PYTHONUNBUFFERED=1` in the environment. Both are needed: `bufsize=1` sets the parent pipe to line-buffered; `PYTHONUNBUFFERED=1` forces the child script to flush on every print.
**Warning signs:** No log lines during run; entire output appears instantly when Run completes.

```python
import os
env = {**os.environ, "PYTHONUNBUFFERED": "1"}
proc = subprocess.Popen([sys.executable, "scripts/run_inference.py", ...],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1, env=env)
```

### Pitfall 3: Session State Results Lost on Widget Interaction

**What goes wrong:** User changes a sidebar slider after a successful run; results disappear.
**Why it happens:** Any widget interaction triggers a full script rerun from top to bottom. If results are stored as local variables (not in session_state), they are lost.
**How to avoid:** Store all loaded DataFrames and figure objects in `st.session_state` after a successful run. Only clear them when the user explicitly starts a new run.
**Warning signs:** Charts vanish when user adjusts any sidebar control.

### Pitfall 4: regression_pred_last_step.csv Column-to-Ticker Mapping

**What goes wrong:** Heatmap shows wrong ticker labels because the headerless CSV columns are not in alphabetical order — they follow the column order of `tickers.txt`.
**Why it happens:** `regression_pred_last_step.csv` is headerless; columns correspond to the index position in `tickers.txt` (established pattern from run_backtest.py which uses the same file).
**How to avoid:** Always load `tickers.txt` to build the column-to-ticker mapping. For top-K filtering, get the unique tickers from `backtest_positions.csv` then find their indices in `tickers.txt`.
**Warning signs:** Heatmap ticker labels do not match the equity curve's top holdings.

### Pitfall 5: Date Range Sidebar vs Pipeline Date Range Mismatch

**What goes wrong:** User selects a date range in the sidebar that differs from the `.conf` file's test split. The backtest output CSVs cover the config's test period, not the sidebar dates.
**Why it happens:** The date range in the sidebar controls the evaluation window for display — but `run_inference.py` does not accept date range CLI args. It uses the `.conf` test split dates.
**How to avoid:** The sidebar date range is used only to validate whether the config's test period overlaps the selected window, then to filter the display (not the pipeline). Alternatively, the sidebar date range is purely informational. Document this clearly in the UI placeholder text.
**Warning signs:** User selects 2022-2023, but equity curve shows 2023-05 to 2023-10 (the actual config test period).

### Pitfall 6: Python 3.9 / Streamlit Version Incompatibility

**What goes wrong:** `pip install streamlit` installs 1.55.0, which requires Python >=3.10. The venv is Python 3.9, so install fails or import errors occur.
**Why it happens:** Streamlit dropped Python 3.9 support in version 1.51.0 (October 2025).
**How to avoid:** Pin in requirements.txt: `streamlit>=1.35,<=1.50`. All required APIs (column_config, spinner, st.empty, st.plotly_chart, download_button, date_input) are available from 1.35 onward.
**Warning signs:** `ERROR: Package 'streamlit' requires a different Python: 3.9.6 not in '>=3.10'`.

---

## Code Examples

Verified patterns from official sources and actual project data:

### Running Both Subprocesses in Sequence

```python
# Pattern: run inference then backtest, stream both to same log
import subprocess, sys, os

def run_pipeline(config_path, output_dir, top_k):
    """Returns (success: bool, return_codes: list)."""
    log_placeholder = st.empty()
    accumulated = []
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}

    def stream(cmd):
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        for line in iter(proc.stdout.readline, ""):
            accumulated.append(line.rstrip())
            log_placeholder.code("\n".join(accumulated[-60:]))
        proc.wait()
        return proc.returncode

    rc1 = stream([sys.executable, "scripts/run_inference.py", "--config", config_path])
    rc2 = stream([sys.executable, "scripts/run_backtest.py",
                  "--output_dir", output_dir, "--top_k", str(top_k)])
    return rc1 == 0 and rc2 == 0, [rc1, rc2]
```

### Top-Level App Structure

```python
import streamlit as st
import sys, os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stockformer Dashboard", layout="wide")
st.title("Stockformer S&P500 Dashboard")

# ── Session state init ───────────────────────────────────────────────────────
for key, default in [("run_complete", False), ("results", None), ("running", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Pipeline Settings**")
    config_path = st.text_input("Config path", "config/Multitask_Stock_SP500.conf")
    output_dir = st.text_input("Output directory", "output/Multitask_output_SP500_2018-2024")
    st.markdown("---")
    st.markdown("**Backtest Parameters**")
    top_k = st.number_input("Top-K", min_value=1, max_value=100, step=1, value=10)
    txn_cost = st.number_input("Transaction cost (bps)", min_value=0, max_value=100, step=1, value=10)
    benchmark = st.text_input("Benchmark ticker", "SPY")
    start_date = st.date_input("Start date", value=date(2023, 1, 1))
    end_date = st.date_input("End date", value=date(2024, 12, 31))
    st.markdown("---")
    run_clicked = st.button("Run Pipeline", type="primary",
                            disabled=st.session_state["running"])

# ── Main area ────────────────────────────────────────────────────────────────
if not st.session_state["run_complete"] and not st.session_state["running"]:
    st.info("Configure settings to run your first analysis\n\n"
            "Configure your settings in the sidebar, then click Run Pipeline "
            "to run inference and backtesting.")

if run_clicked:
    st.session_state["running"] = True
    with st.spinner("Running inference and backtest — this may take several minutes..."):
        success, _ = run_pipeline(config_path, output_dir, top_k)
    st.session_state["running"] = False
    if success:
        st.session_state["run_complete"] = True
        st.session_state["results"] = load_results(output_dir, top_k)
    else:
        st.error("Pipeline failed. Check the log above for details.")

if st.session_state["run_complete"] and st.session_state["results"]:
    render_results(st.session_state["results"])
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `st.beta_columns()` | `st.columns()` | Streamlit ~1.0 | Old API removed; use current |
| `st.plotly_chart(**kwargs)` | `st.plotly_chart(config=...)` | Streamlit ~1.42 | **kwargs deprecated; explicit config param |
| `st.dataframe()` without column_config | `st.dataframe(column_config={...})` | Streamlit ~1.20 | column_config is now the standard formatting mechanism |
| `process.communicate()` for subprocess output | `iter(proc.stdout.readline, "")` | N/A (pattern) | communicate() buffers everything in memory; readline() streams |

**Deprecated/outdated:**
- `st.caching`: replaced by `st.cache_data` / `st.cache_resource` — not needed for this phase since no expensive pure functions are cached
- `plotly.express.imshow()` for heatmaps with custom diverging colorscales: works but `go.Heatmap` gives finer control over zmid/zmin/zmax

---

## Open Questions

1. **Date range sidebar → pipeline integration**
   - What we know: `run_inference.py` does not accept date range CLI args; it uses the `.conf` test split dates
   - What's unclear: Should the sidebar date range filter the display after the run completes, or is it purely informational? If filtering, the date column from `backtest_daily_returns.csv` can be used to slice the equity curve.
   - Recommendation: Treat sidebar dates as display filter for the equity curve and metrics (filter `backtest_daily_returns.csv` to the selected range for the chart). Add a warning if selected range falls partially outside the actual data range.

2. **tickers.txt path hardcode vs sidebar**
   - What we know: `run_backtest.py` defaults tickers_file to `data/Stock_SP500_2018-01-01_2024-01-01/tickers.txt`; heatmap needs this file
   - What's unclear: Should the app accept a custom tickers path, or hardcode the default?
   - Recommendation: Derive tickers.txt path from the same data directory as the config points to; use the same default as run_backtest.py. Do not expose as a sidebar control (CONTEXT.md does not list it as a parameter).

3. **Transaction cost bps not currently a CLI arg to run_backtest.py**
   - What we know: `run_backtest.py` has `FEE = 0.001` hardcoded; no `--fee` or `--txn_cost` CLI arg
   - What's unclear: The sidebar has a "Transaction cost (bps)" input — but there is no current mechanism to pass it to the subprocess
   - Recommendation: Either add `--fee` arg to `run_backtest.py` in Wave 0 of this phase, or note that the txn_cost sidebar field is informational-only for this version. Planner must decide and document.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest >=7.0,<8 |
| Config file | none — no pytest.ini exists (Wave 0 gap) |
| Quick run command | `pytest tests/test_app.py -x -q` |
| Full suite command | `pytest tests/ -x -q` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| UI-01 | app.py exists at project root and imports without error | smoke | `pytest tests/test_app.py::test_app_imports -x` | ❌ Wave 0 |
| UI-01 | Sidebar widget defaults match spec (config path, top_k=10, etc.) | unit | `pytest tests/test_app.py::test_sidebar_defaults -x` | ❌ Wave 0 |
| UI-02 | `build_equity_chart()` returns a go.Figure with 2 traces | unit | `pytest tests/test_app.py::test_equity_chart_shape -x` | ❌ Wave 0 |
| UI-02 | Equity curve values start at 1.0 | unit | `pytest tests/test_app.py::test_equity_chart_starts_at_one -x` | ❌ Wave 0 |
| UI-03 | `format_metrics_table()` returns DataFrame with all 8 columns | unit | `pytest tests/test_app.py::test_metrics_table_columns -x` | ❌ Wave 0 |
| UI-04 | `build_heatmap()` returns go.Figure with Heatmap trace, zmid=0 | unit | `pytest tests/test_app.py::test_heatmap_zmid -x` | ❌ Wave 0 |
| UI-04 | Heatmap filters to top-K rows only | unit | `pytest tests/test_app.py::test_heatmap_top_k_filter -x` | ❌ Wave 0 |

Note: Full Streamlit widget rendering is not unit-testable without Streamlit's test runner. Chart construction functions should be extracted to testable pure functions (e.g., `build_equity_chart(df)`, `build_heatmap(pred_df, tickers, k)`, `format_metrics_table(summary_df)`) that accept DataFrames and return Plotly figures. These can be tested with synthetic data.

### Sampling Rate

- **Per task commit:** `pytest tests/test_app.py -x -q`
- **Per wave merge:** `pytest tests/ -x -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps

- [ ] `tests/test_app.py` — covers UI-01 through UI-04 with synthetic CSV fixtures
- [ ] `tests/conftest.py` update — add `backtest_daily_returns_fixture`, `backtest_summary_fixture`, `regression_pred_fixture` synthetic data fixtures
- [ ] Framework install: `pip install "streamlit>=1.35,<=1.50" "plotly>=5.18,<7"` — neither package in venv yet
- [ ] `.streamlit/config.toml` — does not exist yet (Wave 0 setup step)

---

## Sources

### Primary (HIGH confidence)

- Verified against actual file: `output/Multitask_output_SP500_2018-2024/backtest_daily_returns.csv` — columns: date, portfolio_return, spy_return; 167 rows confirmed
- Verified against actual file: `output/Multitask_output_SP500_2018-2024/backtest_summary.csv` — columns confirmed matching UI-SPEC
- Verified against actual file: `output/Multitask_output_SP500_2018-2024/regression/regression_pred_last_step.csv` — shape (167, 478), headerless
- Verified against actual file: `scripts/run_backtest.py` — FEE hardcoded at 0.001; no --fee CLI arg; tickers_file defaults confirmed
- Verified against actual file: `scripts/run_inference.py` — only accepts --config; no date range args
- [PyPI streamlit](https://pypi.org/project/streamlit/) — confirmed latest version 1.55.0, requires Python >=3.10
- [Streamlit 2025 release notes](https://docs.streamlit.io/develop/quick-reference/release-notes/2025) — confirmed 1.51.0 dropped Python 3.9 support; 1.50.0 is last compatible version
- [Plotly go.Heatmap docs](https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Heatmap.html) — zmid parameter behavior verified
- Project venv Python version: 3.9.6 (confirmed via `venv/pyvenv.cfg`)

### Secondary (MEDIUM confidence)

- [TechOverflow — Streamlit subprocess live output pattern](https://techoverflow.net/2024/11/29/streamlit-how-to-run-subprocess-with-live-stdout-display-on-web-ui/) — PIPE + readline pattern, Nov 2024
- [Streamlit discuss — column_config percentage format](https://discuss.streamlit.io/t/formatting-floating-point-values-as-percent-in-column_config/45997) — `"%.2f %%"` format string verified
- [Streamlit docs — session state](https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state) — button state limitation confirmed (cannot set button key via session_state)

### Tertiary (LOW confidence)

- None — all critical claims verified via primary or secondary sources.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — versions verified against PyPI and project venv Python version
- Architecture: HIGH — based on verified existing scripts and actual CSV schemas
- Pitfalls: HIGH — root causes verified against actual source code (FEE hardcode, headerless CSV, subprocess buffering)
- Test patterns: MEDIUM — chart construction test pattern is standard practice but no existing test_app.py to reference

**Research date:** 2026-03-16
**Valid until:** 2026-06-16 (stable ecosystem; Streamlit version pin isolates from upstream changes)
