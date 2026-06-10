# Búsqueda de señal modular — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two causal "architecture-transplant" signals (peer/graph momentum, filtered-trend momentum) to the RQ2 signal search, evaluate them alone and combined with the existing feature-ensemble through the cost-aware walk-forward (311 weeks, 104-week holdout), and elevate the signal search to a central plank of RQ2 in the thesis text.

**Architecture:** Pure-CPU, no Stockformer retraining. Each transplant is a cross-sectional signal `f(daily_y, d) -> [N]` that uses **only** `daily_y[<d]` (the same convention as `resid_mom`). They are blended with the ensemble as `Z(ens) + Z(signal)` and run through the existing `backtest_variant(..., CONFIGS["full"])`. A causality unit test per signal enforces no look-ahead. Results feed a new two-panel LaTeX table. The bench (Lote 2) is a conditional phase, run only if no Lote-1 row clears the pre-registered bar.

**Tech Stack:** Python, numpy/pandas, statsmodels (STL, Newey-West HAC), pytest. Reuses `lib/data_panel.py`, `lib/weekly_panel.py`, `lib/decomposition.py`, `lib/portfolio.py`, `lib/neutralize.py`, and the existing `scripts/run_signal_search*.py` / `run_weekly_robustness.py` harness. Thesis in LaTeX under `MEMORIA/tfm/`.

---

## File Structure

- **Create** `lib/transplant_signals.py` — the two Lote-1 signals + shared helpers (`_trailing_returns`, `xz`). One responsibility: causal cross-sectional transplant signals.
- **Create** `tests/test_transplant_signals.py` — causality + sanity tests.
- **Create** `scripts/run_transplant_search.py` — loads weekly panel + ensemble, evaluates the candidates through the cost-aware walk-forward, writes `results/transplant_search.csv`.
- **Create** `scripts/build_transplant_table.py` — `results/transplant_search.csv` -> `MEMORIA/tfm/tablas/rq2_modular_transfer.tex`.
- **Create** `MEMORIA/tfm/tablas/rq2_modular_transfer.tex` — generated Panel B table.
- **Modify** `MEMORIA/tfm/Capitulos/06_parte2_que_funciona.tex` — elevate §6.8 to "Búsqueda de señal" with the modular-transfer subsection + `\input` the new table.
- **Modify** `MEMORIA/tfm/Capitulos/01_introduccion.tex`, `00_resumen.tex`, `08_conclusiones.tex` — RQ2 reframe to two palancas (signal search central).

**Constants (verbatim from harness):** `COST_BPS=8.0`, `GROSS=2.0`, `NAME_CAP=0.04`, `WEEKS_PER_YEAR=52`, `NW_LAGS=6`, holdout `=104` weeks. `CONFIGS["full"]=dict(neutral=True, smooth=True, costaware=True, voltarget=True, regime=True)`.

**pytest:** `pythonpath=[".","scripts"]` is configured in `pyproject.toml`; run a single test with `pytest tests/test_transplant_signals.py::<name> -v`.

---

## Phase 0 — Prep & baseline

### Task 0: Confirm environment, window params, and baseline reproduction

**Files:** none (read-only checks)

- [ ] **Step 1: Confirm the venv + deps import**

Run:
```bash
cd /Users/gonzaloalonsolidon/Desktop/Repos/Cursor/tfm-stockformer
python -c "import numpy, pandas, statsmodels, cvxpy, lightgbm, sklearn; print('ok')"
```
Expected: `ok`. If it fails, `source venv/bin/activate` first.

- [ ] **Step 2: Find the exact 311-week window params used by the long search**

Run:
```bash
ls results/_wf_preds_long_*.npz
grep -nE "INIT_TRAIN|init_train|STEP|step|HOLD|311|104" scripts/run_signal_search3.py scripts/run_signal_search4.py | head -40
```
Record the `init_train` and `step` from the cache filename `_wf_preds_long_<init_train>_<step>.npz` and the holdout length. These MUST match what produced the thesis 0.38/0.72 numbers. Use them in Task 4 (do not invent values).

- [ ] **Step 3: Confirm the cost-aware harness imports headlessly**

Run:
```bash
python -c "import sys; sys.path.insert(0,'scripts'); \
from run_weekly_robustness import backtest_variant, sharpe, CONFIGS; \
from run_signal_search import nw_tstat; \
from run_signal_search3 import get_long_ensemble; print('harness ok', list(CONFIGS))"
```
Expected: `harness ok [...]` including `'full'`. If the import names differ, record the actual names (they are the source of truth for Task 4's imports).

---

## Phase 1 — Lote 1 signals (TDD)

### Task 1: Module scaffold + shared helpers

**Files:**
- Create: `lib/transplant_signals.py`
- Test: `tests/test_transplant_signals.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_transplant_signals.py
import numpy as np
from lib import transplant_signals as ts


def test_trailing_returns_uses_only_past():
    dy = np.arange(40 * 3, dtype=float).reshape(40, 3)
    d = 30
    R = ts._trailing_returns(dy, d, window=10)
    assert R.shape == (10, 3)
    np.testing.assert_array_equal(R, dy[20:30])  # [d-window, d), strictly past


def test_trailing_returns_none_when_insufficient_history():
    dy = np.zeros((5, 3))
    assert ts._trailing_returns(dy, d=2, window=10) is None


def test_xz_zero_mean_unit_std():
    z = ts.xz(np.array([1.0, 2.0, 3.0, 4.0]))
    assert abs(float(np.mean(z))) < 1e-9
    assert abs(float(np.std(z)) - 1.0) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_transplant_signals.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lib.transplant_signals'`.

- [ ] **Step 3: Write minimal implementation**

```python
# lib/transplant_signals.py
"""Causal cross-sectional 'architecture-transplant' signals for the RQ2 signal search.

Each signal has the convention used by resid_mom: f(daily_y, d) -> [N], computed using
ONLY daily_y[:d] (strictly before the decision day d). This guarantees no look-ahead.
"""
from __future__ import annotations

import numpy as np


def _trailing_returns(daily_y: np.ndarray, d: int, window: int) -> np.ndarray | None:
    """Trailing [window, N] daily returns ending at (and excluding) decision day d.

    Returns None if there is not enough history (d - window < 0). NaNs -> 0.0.
    """
    a = d - window
    if a < 0:
        return None
    return np.nan_to_num(daily_y[a:d], nan=0.0)


def xz(v: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score (mean 0, std 1). Constant/empty -> zeros."""
    v = np.asarray(v, dtype=float)
    m = np.nanmean(v)
    s = np.nanstd(v)
    return (v - m) / s if s > 1e-12 else np.zeros_like(v)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_transplant_signals.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add lib/transplant_signals.py tests/test_transplant_signals.py
git commit -m "feat(rq2): scaffold transplant_signals module with causal helpers"
```

### Task 2: Peer/graph momentum signal (graph transplant)

**Files:**
- Modify: `lib/transplant_signals.py`
- Test: `tests/test_transplant_signals.py`

- [ ] **Step 1: Write the failing tests (causality is the critical one)**

```python
# append to tests/test_transplant_signals.py

def test_peer_graph_signal_shape_and_nan_guard():
    rng = np.random.default_rng(0)
    dy = rng.normal(0, 0.01, size=(400, 12))
    sig = ts.peer_graph_signal(dy, d=300)
    assert sig.shape == (12,)
    # not enough history -> all NaN
    assert np.all(np.isnan(ts.peer_graph_signal(dy, d=10)))


def test_peer_graph_signal_is_causal():
    rng = np.random.default_rng(1)
    dy = rng.normal(0, 0.01, size=(400, 12))
    d = 300
    s1 = ts.peer_graph_signal(dy, d)
    dy2 = dy.copy()
    dy2[d:] = rng.normal(0, 0.5, size=dy2[d:].shape)  # perturb only the FUTURE
    s2 = ts.peer_graph_signal(dy2, d)
    np.testing.assert_allclose(np.nan_to_num(s1), np.nan_to_num(s2), atol=1e-12)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_transplant_signals.py -k peer_graph -v`
Expected: FAIL with `AttributeError: module 'lib.transplant_signals' has no attribute 'peer_graph_signal'`.

- [ ] **Step 3: Implement**

```python
# append to lib/transplant_signals.py

def peer_graph_signal(daily_y: np.ndarray, d: int,
                      corr_window: int = 120, k: int = 15,
                      mom_window: int = 21) -> np.ndarray:
    """Causal peer-momentum (relational-graph transplant).

    Builds a trailing correlation graph from daily_y[d-corr_window:d] (past only),
    selects each stock's top-k |corr| peers, and returns the correlation-weighted
    mean of those peers' own recent momentum (cumulative return over the last
    mom_window days, also past only). Lead-lag / connected-stock momentum.
    """
    R = _trailing_returns(daily_y, d, corr_window)
    if R is None:
        return np.full(daily_y.shape[1], np.nan)
    N = R.shape[1]
    C = np.corrcoef(R, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)
    own_mom = np.nan_to_num(daily_y[d - mom_window:d], nan=0.0).sum(axis=0)  # [N], past
    out = np.full(N, np.nan)
    kk = min(k, N - 1)
    for i in range(N):
        order = np.argsort(np.abs(C[i]))[-kk:]   # top-k peers by |corr|
        wts = C[i, order]                        # signed correlation weights
        denom = np.abs(wts).sum()
        if denom > 1e-12:
            out[i] = float((wts * own_mom[order]).sum() / denom)
    return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_transplant_signals.py -k peer_graph -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add lib/transplant_signals.py tests/test_transplant_signals.py
git commit -m "feat(rq2): causal peer-graph momentum signal (graph transplant)"
```

### Task 3: Filtered-trend momentum signal (wavelet/dual-freq transplant)

**Files:**
- Modify: `lib/transplant_signals.py`
- Test: `tests/test_transplant_signals.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_transplant_signals.py

def test_filtered_trend_signal_shape_and_nan_guard():
    rng = np.random.default_rng(2)
    dy = rng.normal(0, 0.01, size=(400, 8))
    sig = ts.filtered_trend_signal(dy, d=300)
    assert sig.shape == (8,)
    assert np.all(np.isnan(ts.filtered_trend_signal(dy, d=5)))


def test_filtered_trend_signal_is_causal():
    rng = np.random.default_rng(3)
    dy = rng.normal(0, 0.01, size=(400, 8))
    d = 300
    s1 = ts.filtered_trend_signal(dy, d)
    dy2 = dy.copy()
    dy2[d:] = rng.normal(0, 0.5, size=dy2[d:].shape)  # perturb only the FUTURE
    s2 = ts.filtered_trend_signal(dy2, d)
    np.testing.assert_allclose(np.nan_to_num(s1), np.nan_to_num(s2), atol=1e-12)


def test_filtered_trend_signal_positive_for_uptrend():
    # a stock with steady positive drift should get a positive trend score
    dy = np.zeros((200, 2))
    dy[:, 0] = 0.002   # steady uptrend
    dy[:, 1] = -0.002  # steady downtrend
    sig = ts.filtered_trend_signal(dy, d=180, window=120, halflife=10, mom_window=21)
    assert sig[0] > sig[1]
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_transplant_signals.py -k filtered_trend -v`
Expected: FAIL with `AttributeError: ... has no attribute 'filtered_trend_signal'`.

- [ ] **Step 3: Implement**

```python
# append to lib/transplant_signals.py

def filtered_trend_signal(daily_y: np.ndarray, d: int,
                          window: int = 120, halflife: float = 10.0,
                          mom_window: int = 21) -> np.ndarray:
    """Causal filtered-trend momentum (wavelet / dual-frequency transplant).

    Extracts the low-frequency (trend) component of each stock's return series via a
    one-sided causal EWMA over daily_y[d-window:d] (strictly past), then takes the mean
    of the most recent mom_window smoothed (denoised) returns. This is the Stockformer's
    'separate trend from microstructure noise' idea reduced to a simple causal signal.
    """
    R = _trailing_returns(daily_y, d, window)
    if R is None:
        return np.full(daily_y.shape[1], np.nan)
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    sm = np.empty_like(R)
    sm[0] = R[0]
    for t in range(1, R.shape[0]):
        sm[t] = alpha * R[t] + (1.0 - alpha) * sm[t - 1]
    return sm[-mom_window:].mean(axis=0)
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_transplant_signals.py -v`
Expected: all tests pass (8 total).

- [ ] **Step 5: Commit**

```bash
git add lib/transplant_signals.py tests/test_transplant_signals.py
git commit -m "feat(rq2): causal filtered-trend momentum signal (wavelet transplant)"
```

---

## Phase 2 — Harness wiring, run, table

### Task 4: Run script over the cost-aware walk-forward

**Files:**
- Create: `scripts/run_transplant_search.py`

> Use the `init_train`/`step`/holdout recorded in Task 0 Step 2. The placeholders `INIT_TRAIN`, `STEP`, `HOLD_WEEKS` below MUST be set to those exact values so the window matches the thesis 0.38/0.72 baseline.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Lote 1 modular-transfer signals through the cost-aware walk-forward.

Evaluates: ensemble (base), peer, trend, ens+peer, ens+trend, ens+peer+trend.
Writes results/transplant_search.csv with [signal, sharpe, t, hold, turnover, ic].
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))  # so run_* scripts import as modules

from lib import data_panel as dp          # noqa: E402
from lib import weekly_panel as wp         # noqa: E402
from lib import transplant_signals as ts   # noqa: E402
from run_signal_search3 import get_long_ensemble          # noqa: E402
from run_weekly_robustness import backtest_variant, sharpe, CONFIGS  # noqa: E402
from run_signal_search import nw_tstat                    # noqa: E402

DATA_DIR = "data/Stock_SP500_2018-01-01_2026-03-16"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
INIT_TRAIN = 200   # <-- set from Task 0 Step 2 (cache filename)
STEP = 26          # <-- set from Task 0 Step 2
HOLD_WEEKS = 104   # <-- set from Task 0 Step 2


def _ev(net: pd.Series) -> dict:
    net = pd.Series(net).dropna()
    return {"sharpe": sharpe(net), "t": nw_tstat(net)}


def main():
    panel = dp.load_panel(DATA_DIR)
    week = wp.build_weekly(panel)
    ens = get_long_ensemble(week, INIT_TRAIN, STEP)
    weeks = sorted(ens)
    dy = week.daily_y
    hold_dates = {week.dates_w[w] for w in weeks[-HOLD_WEEKS:]}

    cand = {k: {} for k in
            ["ensemble", "peer", "trend", "ens_peer", "ens_trend", "ens_peer_trend"]}
    for wk in weeks:
        d = int(week.rebal_idx[wk])
        e = ts.xz(ens[wk])
        peer = ts.xz(ts.peer_graph_signal(dy, d))
        trend = ts.xz(ts.filtered_trend_signal(dy, d))
        e0, p0, t0 = np.nan_to_num(e), np.nan_to_num(peer), np.nan_to_num(trend)
        cand["ensemble"][wk] = e
        cand["peer"][wk] = peer
        cand["trend"][wk] = trend
        cand["ens_peer"][wk] = e0 + p0
        cand["ens_trend"][wk] = e0 + t0
        cand["ens_peer_trend"][wk] = e0 + p0 + t0

    rows = []
    for name, preds in cand.items():
        bt = backtest_variant(week, preds, CONFIGS["full"]).copy()
        ish = np.array([dt in hold_dates for dt in bt.index])
        full, hold = _ev(bt["net"]), _ev(bt["net"][ish])
        rows.append({"signal": name, "sharpe": full["sharpe"], "t": full["t"],
                     "hold": hold["sharpe"], "turnover": float(bt["turnover"].mean()),
                     "ic": float(bt["ic"].dropna().mean())})
        print(f"  {name:16s} Sharpe={full['sharpe']:+.2f} t={full['t']:+.2f} "
              f"hold={hold['sharpe']:+.2f} ic={rows[-1]['ic']:+.4f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "transplant_search.csv"), index=False)
    print(f"\nSaved results/transplant_search.csv ({len(rows)} signals)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-check imports without running the full backtest**

Run: `python -c "import sys; sys.path.insert(0,'scripts'); import run_transplant_search as r; print('import ok', r.INIT_TRAIN, r.STEP, r.HOLD_WEEKS)"`
Expected: `import ok <init> <step> <hold>`. Fix import names if needed (Task 0 Step 3 is the source of truth).

- [ ] **Step 3: Commit the script**

```bash
git add scripts/run_transplant_search.py
git commit -m "feat(rq2): run script for modular-transfer signal search"
```

### Task 5: Run the search and record results

**Files:** none (produces `results/transplant_search.csv`)

- [ ] **Step 1: Run it**

Run: `python scripts/run_transplant_search.py`
Expected: one printed line per signal, then `Saved results/transplant_search.csv (6 signals)`. (Reuses the cached ensemble `_wf_preds_long_*.npz`; only the new signals compute.)

- [ ] **Step 2: Inspect the numbers**

Run: `python -c "import pandas as pd; print(pd.read_csv('results/transplant_search.csv').to_string(index=False))"`
Record each row. **Pre-registered bar:** a transplant "aporta" if `hold > 0` (holdout-positive) AND `sharpe > 0.38` (beats the base). Note which (if any) clear it — this decides whether Phase 4 runs.

- [ ] **Step 3: Commit the results CSV**

```bash
git add results/transplant_search.csv
git commit -m "chore(rq2): modular-transfer signal-search results"
```

### Task 6: Generate the LaTeX table

**Files:**
- Create: `scripts/build_transplant_table.py`
- Create: `MEMORIA/tfm/tablas/rq2_modular_transfer.tex`

- [ ] **Step 1: Write the table builder**

```python
#!/usr/bin/env python3
"""results/transplant_search.csv -> MEMORIA/tfm/tablas/rq2_modular_transfer.tex.

Mirrors the row format of build_rq2_table.py (Sharpe / t_NW / Holdout / Turnover / IC).
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "results", "transplant_search.csv")
OUT = os.path.join(ROOT, "MEMORIA", "tfm", "tablas", "rq2_modular_transfer.tex")

LABEL = {
    "ensemble": "Ensemble ML (base)",
    "peer": r"Se\~nal de pares (grafo) sola",
    "trend": "Tendencia filtrada sola",
    "ens_peer": "Ensemble $+$ pares",
    "ens_trend": "Ensemble $+$ tendencia filtrada",
    "ens_peer_trend": "Ensemble $+$ pares $+$ tendencia",
}
ORDER = ["ensemble", "peer", "trend", "ens_peer", "ens_trend", "ens_peer_trend"]


def main():
    df = pd.read_csv(CSV).set_index("signal")
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Transferencia modular: se\~nales destiladas de la arquitectura del "
        r"Stockformer (grafo de pares causal, tendencia filtrada causal), solas y combinadas "
        r"con el \textit{ensemble} de \textit{features}. Misma construcci\'on cost-aware y "
        r"\textit{walk-forward} de 311 semanas (\textit{holdout}: 104 semanas) que la "
        r"Tabla~\ref{tab:rq2-signal-search}.}",
        r"\label{tab:rq2-modular}",
        r"\small",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Se\~nal & Sharpe neto & $t_{\text{NW}}$ & \textit{Holdout} & Turnover & IC \\",
        r"\midrule",
    ]
    for k in ORDER:
        if k not in df.index:
            continue
        r = df.loc[k]
        lines.append(
            f"{LABEL[k]} & ${r['sharpe']:+.2f}$ & ${r['t']:+.2f}$ & "
            f"${r['hold']:+.2f}$ & {r['turnover']:.2f} & ${r['ic']:+.4f}$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python scripts/build_transplant_table.py`
Expected: `Wrote .../tablas/rq2_modular_transfer.tex`. Open the file and confirm 6 rows with real numbers.

- [ ] **Step 3: Commit**

```bash
git add scripts/build_transplant_table.py MEMORIA/tfm/tablas/rq2_modular_transfer.tex
git commit -m "feat(rq2): modular-transfer LaTeX table generator + table"
```

---

## Phase 3 — Thesis text

### Task 7: Outcome-independent RQ2 reframe (signal search central)

**Files:**
- Modify: `MEMORIA/tfm/Capitulos/01_introduccion.tex` (RQ2 paragraph §1.2 + contribution §1.3)
- Modify: `MEMORIA/tfm/Capitulos/00_resumen.tex` (ES + EN)
- Modify: `MEMORIA/tfm/Capitulos/08_conclusiones.tex` (RQ2 answer)

- [ ] **Step 1: Read the current RQ2 + contribution paragraphs**

Run: `grep -n "RQ2\|palanca\|construcci" MEMORIA/tfm/Capitulos/01_introduccion.tex | head`
Then read the matched RQ2 paragraph and §1.3 contribution paragraph in full so the edit strings match exactly (watch for em-dashes `—`).

- [ ] **Step 2: Edit RQ2 (§1.2) to two complementary palancas**

In the RQ2 paragraph, after the existing "¿cuál de las etapas del \textit{pipeline} ... aporta Sharpe neto positivo?" question, append (verbatim):

```latex
 La respuesta tiene dos palancas complementarias: la construcción de cartera consciente
del coste y una \emph{búsqueda de señal disciplinada} que, sobre el predictor de
\textit{features} existente, identifica señales simples con contenido genuino —incluidas
señales destiladas de los sesgos inductivos del propio Stockformer (efectos de red y
filtrado frecuencial)— que lo complementan.
```

- [ ] **Step 3: Edit the contribution (§1.3) to name the signal search**

In §1.3, in the sentence that currently ends with the residual-momentum result, ensure the methodological contribution explicitly includes the disciplined signal search. Add (verbatim) after the residual-momentum sentence:

```latex
 Esa búsqueda de señal, pre-registrada y validada en un \textit{holdout} reservado, es un
eje de la contribución: localiza el contenido predictivo en factores simples y bien
elegidos —y prueba si las ideas concretas de una arquitectura que no transfiere en bloque,
destiladas a señales causales, rescatan algo de valor sobre el sustrato eficiente.
```

- [ ] **Step 4: Edit the abstract (ES + EN)** in `00_resumen.tex`

In the ES paragraph that describes RQ2, after the residual-momentum sentence add (verbatim):
```latex
 Esta búsqueda de señal, disciplinada y validada fuera de muestra, es parte central de la
respuesta a la segunda pregunta.
```
In the EN paragraph, after the residual-momentum sentence add (verbatim):
```latex
 This disciplined, out-of-sample-validated signal search is a central part of the answer to
the second question.
```

- [ ] **Step 5: Edit the conclusions (§8.1)** in `08_conclusiones.tex`

In the RQ2-answer paragraph, change the framing from "la palanca principal ... es la construcción" to acknowledge two palancas. After the construction sentence add (verbatim):
```latex
 La segunda palanca es la propia búsqueda de señal: disciplinada, pre-registrada y validada
en \textit{holdout}, localiza el contenido en factores simples y, cuando es posible, en
piezas destiladas de la arquitectura profunda.
```

- [ ] **Step 6: Commit the reframe**

```bash
git add MEMORIA/tfm/Capitulos/01_introduccion.tex MEMORIA/tfm/Capitulos/00_resumen.tex MEMORIA/tfm/Capitulos/08_conclusiones.tex
git commit -m "docs(tfm): elevar la búsqueda de señal a eje central de RQ2"
```

### Task 8: Elevate §6.8 + insert the modular-transfer subsection (outcome-dependent)

**Files:**
- Modify: `MEMORIA/tfm/Capitulos/06_parte2_que_funciona.tex`

> Fill the prose numbers from `results/transplant_search.csv` (Task 5). Two narrative branches — pick by whether any row cleared the bar.

- [ ] **Step 1: Rename §6.8 and add the framing/discipline sentence**

Change the §6.8 section title (currently `\section{Una señal con contenido: el momentum residual}`) to:
```latex
\section{Búsqueda de señal: factores clásicos y piezas destiladas de la arquitectura}\label{sec:p2-signal}
```
Right after the section intro, add (verbatim) a discipline paragraph:
```latex
\paragraph{Disciplina.} La búsqueda se realiza sobre una lista \emph{cerrada y
pre-registrada} de candidatos, fijada antes de mirar el \textit{holdout} de 104 semanas;
se reportan todos (ganen o pierdan) y cualquier ganador se defiende por su \textit{holdout}
positivo y su coherencia económica, no por un IC dentro de muestra. Dada la multiplicidad,
el listón de $t$ se interpreta a la luz de \textcite{harvey2016crosssection}.
```

- [ ] **Step 2: Add the modular-transfer subsection + table** (after the classical-factor results, before the section's closing synthesis):

```latex
\subsection{Piezas destiladas de la arquitectura}\label{sec:p2-modular}

La Parte~I mostró que el Stockformer no transfiere como un todo. Cabe la pregunta inversa:
¿alguna de sus \emph{ideas} concretas —el grafo de relaciones entre acciones, la
descomposición frecuencial— reducida a una señal simple y \emph{causal} (sin la fuga del
grafo de todo el histórico de la Sección~\ref{sec:met-leakage}), aporta sobre el predictor
\textit{shallow}? Se destilan dos: una \textbf{señal de pares} (momentum de los vecinos más
correlacionados, con grafo de correlación móvil solo de pasado) y una \textbf{señal de
tendencia filtrada} (momentum de la componente de baja frecuencia, extraída con un filtro
causal). Ambas se evalúan solas y sumadas al \textit{ensemble}, con la misma construcción
cost-aware y el mismo \textit{walk-forward}. La Tabla~\ref{tab:rq2-modular} recoge el
resultado.

\input{tablas/rq2_modular_transfer}
```

- [ ] **Step 3: Write the results paragraph — BRANCH A (something cleared the bar)**

If a transplant cleared `hold>0` and `sharpe>0.38`, add (fill `<...>` from the CSV):
```latex
La señal de <pares/tendencia> sí aporta: alcanza un Sharpe neto de <X> ($t_{NW}=<t>$) con un
\textit{holdout} positivo de <h>, y combinada con el \textit{ensemble} <mejora/iguala> el
mejor resultado del capítulo. Es evidencia de que una \emph{idea} del modelo profundo
—<el grafo de relaciones / el filtrado frecuencial}—, destilada a una señal simple y causal,
rescata contenido que el modelo completo no transfería.
```

- [ ] **Step 4: Write the results paragraph — BRANCH B (none cleared the bar)**

If none cleared it, add (fill from the CSV):
```latex
Ninguna de las dos señales destiladas supera la base de forma robusta: la de pares se queda
en un Sharpe de <X> (\textit{holdout} <h>) y la de tendencia filtrada en <Y> (\textit{holdout}
<h2>). El resultado es negativo y honesto, y \emph{refuerza} la Parte~I: ni siquiera las ideas
concretas del Stockformer, bien destiladas y sin fuga, rescatan valor sobre el sustrato
eficiente; el contenido sigue residiendo en el momentum residual y en la construcción.
```

- [ ] **Step 5: Commit**

```bash
git add MEMORIA/tfm/Capitulos/06_parte2_que_funciona.tex
git commit -m "docs(tfm): subsección de transferencia modular en la búsqueda de señal"
```

### Task 9: Rebuild the thesis and verify it compiles clean

**Files:** none (build artifact)

- [ ] **Step 1: Build**

Run: `cd MEMORIA/tfm && bash build.sh; cd ../..`

- [ ] **Step 2: Verify no real undefined refs (the script's counter can race)**

Run:
```bash
cd MEMORIA/tfm
echo "undef: $(grep -icE 'Reference .* undefined|Citation .* undefined' main.log)"
grep -c "Output written on main.pdf" main.log; cd ../..
```
Expected: `undef: 0` and the new `tab:rq2-modular` / `sec:p2-modular` references resolve. Fix any real undefined ref before continuing.

- [ ] **Step 3: Commit the build inputs if any .tex changed**

```bash
git add MEMORIA/tfm
git commit -m "build(tfm): integrar transferencia modular y reencuadre de RQ2" || echo "nothing to commit"
```

---

## Phase 4 — CONDITIONAL: pre-registered bench (Lote 2)

**Run this phase ONLY if no Lote-1 row cleared the bar in Task 5 Step 2** (`hold>0` and `sharpe>0.38`). Each bench signal is a `(daily_y, d) -> [N]` function added to `lib/transplant_signals.py`, with a causality test mirroring Task 2/3, and registered as a new key in `scripts/run_transplant_search.py`'s `cand` dict (alone and `e0 + Z(signal)`). The pre-registered list and exact formulas:

- [ ] **Bench 1 — Baja volatilidad idiosincrática (low-IVOL).** `R=_trailing_returns(dy,d,60)`; market `m=R.mean(axis=1)`; residual `resid=R-outer(m,beta)` (beta as in `resid_mom`); `ivol=resid.std(axis=0)`; **signal `= -ivol`** (low idio-vol preferred). Test: causal + `signal` higher for the lower-vol stock.
- [ ] **Bench 2 — Betting-against-beta.** `R=_trailing_returns(dy,d,120)`; `beta=` per-stock beta vs equal-weight market (same OLS as `resid_mom`); **signal `= -beta`**. Test: causal + monotone in beta.
- [ ] **Bench 3 — Momentum gestionado por volatilidad.** `mom=` cumulative return over `[d-252,d-21]`; `vol=` std of daily returns over `[d-126,d]`; **signal `= mom / (vol+1e-9)`**. Test: causal.
- [ ] **Bench 4 — Estacionalidad.** Using `panel.dates`, for decision month `M`, **signal[i] `=` mean return of stock i in prior occurrences of month M** (only dates `< d`). Test: causal (no same-month future).
- [ ] **Bench 5 — Proximidad al máximo de 52 semanas.** From cumulative price `P=cumprod(1+dy[:d])`; **signal `= P[d-1] / max(P[d-252:d]) `** (close-to-high). Test: causal.
- [ ] **Bench 6 — Trend / TS-momentum multi-horizonte.** **signal `=` mean of `sign(sum(dy[d-h:d]))` for `h in {21,63,126,252}`** (only past). Test: causal.

After adding any bench signal: re-run Task 5 (the CSV gains rows), re-run Task 6 (table), and update Task 8's results paragraph. Stop as soon as one clears the bar (holdout-positive + Sharpe > 0.38) or the list is exhausted — then report whichever is the honest outcome (Branch A or B).

---

## Self-Review (run before execution)

1. **Spec coverage:** Lote 1 (Tasks 1-3), eval frame + holdout + NW-t (Task 4-5), two-panel table (Task 6), RQ2-central reframe (Task 7), §6.8 elevation + modular subsection (Task 8), build (Task 9), pre-registered Lote 2 (Phase 4), framed-only out of scope (Phase 4 list is bounded). Covered.
2. **No placeholders:** the only `<...>` are CSV-derived result numbers in thesis prose (Task 8) and the window constants in Task 4 (sourced in Task 0) — both are explicit "fill from this exact source" instructions, not vague directives.
3. **Type consistency:** every signal is `(daily_y: [T,N], d: int) -> [N]`; `xz` and `_trailing_returns` are defined in Task 1 and used by all later signals; `backtest_variant(week, dict[wk->[N]], CONFIGS["full"]) -> df[gross,net,turnover,ic]` used consistently; `_ev` keys `sharpe`/`t` match the table builder's expected columns.
