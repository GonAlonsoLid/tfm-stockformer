# Technology Stack

**Project:** Multitask-Stockformer S&P500 Adaptation
**Researched:** 2026-03-09
**Constraint note:** External web/doc tools were unavailable during this research session.
All version information is sourced from training data (cutoff August 2025) plus the existing
codebase files. Confidence levels reflect this. Where version pinning matters, validate with
`pip index versions <package>` before committing to `requirements.txt`.

---

## Context: Brownfield Constraints

The existing stack is non-negotiable for core ML:

- **PyTorch** (2.8.0 in local venv) — do not downgrade
- **pytorch-wavelets 1.3.0** — compatibility problem with PyTorch >= 2.0; addressed below
- **NumPy / pandas / scikit-learn / scipy / statsmodels** — preserve for existing model code
- **Qlib** — being removed; replaced by custom feature engineering for US data

New libraries are additions only. Nothing in the core model layer changes.

---

## Recommended Stack

### Data Ingestion

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| yfinance | >= 0.2.38 | Download daily OHLCV for S&P500 universe | Free, no API key, SPY/constituent tickers trivially accessible, handles splits/dividends; sufficient for thesis daily data needs. PROJECT.md explicitly mandates this — no paid data. |
| pandas_datareader | >= 0.10.0 | Supplementary index composition data (SPY constituents list) | Thin wrapper over FRED/Stooq for index metadata when yfinance lookup is ambiguous. Optional; can be replaced by a hardcoded CSV of S&P500 tickers. |

**Confidence:** MEDIUM — yfinance 0.2.x API is stable and widely used for academic research through my knowledge cutoff. The 0.2.x series introduced a cleaner `yf.Ticker.history()` and `yf.download()` interface with multi-ticker support. The yfinance project has intermittently broken due to Yahoo Finance scraping changes; rate-limit / IP-block issues are a real operational concern for bulk downloads (500 tickers). Mitigation: add retry logic with exponential backoff and persist to Parquet on first download.

**What NOT to use:**
- Polygon.io, Alpha Vantage, Quandl — require API keys or paid tiers; out of scope per PROJECT.md
- Alpaca markets API — designed for live trading, not historical batch research
- pandas_datareader alone — Yahoo Finance provider in pandas_datareader is deprecated; yfinance is the maintained successor

### S&P500 Universe Management

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| requests + BeautifulSoup4 / Wikipedia scrape | any | Fetch current S&P500 constituent tickers | Wikipedia's S&P500 table is the canonical free source for constituent lists; one-time scrape to a CSV |

**Confidence:** MEDIUM — Standard pattern in academic quant finance codebases. The Wikipedia table at `https://en.wikipedia.org/wiki/List_of_S%26P_500_companies` is stable. Hardcode the date-range-appropriate list as a CSV rather than scraping live; avoids runtime dependency on external web pages during training.

**Practical note:** Ticker survivorship is a real issue. The CSI 300 version used a static universe. For thesis purposes, using a fixed S&P500 list (e.g., constituents as of a chosen start date) is methodologically sound and avoids look-ahead bias from additions/removals during the training window.

### Feature Engineering

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pandas-ta | >= 0.3.14b | Momentum, volatility, and volume technical indicators | Pure-Python, pandas-native API (`df.ta.rsi()`, `df.ta.macd()`), no C compilation required unlike TA-Lib. Covers RSI, MACD, Bollinger Bands, ATR, OBV, EMA/SMA families — all features listed in PROJECT.md active requirements. Works directly on pandas DataFrames which is the natural format after yfinance download. |
| pandas | >= 2.0.3 | Rolling windows, cross-sectional normalization, lag features | Already in stack; use `rolling()`, `pct_change()`, `rank()` for factor construction. Replaces Qlib Alpha-360/Alpha-158 for US features. |
| scikit-learn | 1.1.2 (existing) | Cross-sectional normalization, `StandardScaler`, `RobustScaler` | Already in stack; use for feature standardization before feeding to model. Pin to existing version to avoid breaking the neutralization script. |

**Confidence:** MEDIUM — pandas-ta 0.3.14b is the last stable release as of my knowledge cutoff. The library is actively maintained but has had slow release cadence. No C compilation removes the TA-Lib install friction that commonly blocks reproducibility on macOS/Windows. This is a direct replacement for the Qlib Alpha-360 factors that are calibrated for Chinese market conventions.

**What NOT to use:**
- TA-Lib (C extension) — requires binary compilation via Homebrew or conda; breaks reproducibility on fresh clones without special setup instructions; not worth the friction for a thesis project
- Qlib Alpha-360/Alpha-158 for US data — the factors are calibrated to Chinese market microstructure (T+1 rule, A-share price limits); using them on US equities introduces systematic bias
- `finta` — less comprehensive than pandas-ta, less maintained
- `ta` (ta library) — functionally fine but pandas-ta has better coverage and more Pythonic API

### Wavelet / DWT (Existing Critical Dependency)

| Technology | Version | Purpose | Why This is the Hard Problem |
|------------|---------|---------|-----|
| pytorch-wavelets | 1.3.0 (existing) | DWT signal disentanglement — core model architecture | Used for `DWT1DForward`/`DWT1DInverse` in `lib/Multitask_Stockformer_utils.py`. Not maintained since 2021. |

**Compatibility problem:** PyTorch 2.8.0 (installed in venv) is likely incompatible with pytorch-wavelets 1.3.0 without patches. This is identified in CONCERNS.md. Two resolution paths:

| Path | Effort | Risk | Recommendation |
|------|--------|------|----------------|
| Patch pytorch-wavelets in-place (fix deprecated `torch.Tensor` type checks) | Low — typically 2-5 line changes in `dwt_utils.py` | Low — surgical fix | **Preferred for thesis** |
| Reimplement DWT using `PyWavelets` (pywt) + manual torch wrapping | Medium — ~50-100 lines | Medium — behavior must match original exactly | Fallback if patching is insufficient |

**Confidence:** LOW — The exact compatibility breakage depends on the specific PyTorch 2.x version. pytorch-wavelets 1.3.0 uses `torch.Tensor` registration patterns that changed in PyTorch 2.0. The community has documented patches. Verify by running `python -c "from pytorch_wavelets import DWT1DForward"` in the local venv before proceeding.

**What NOT to do:** Do not upgrade to PyWavelets as a drop-in replacement without verifying that wavelet decomposition outputs match the original implementation — the model was trained on specific wavelet coefficients and changing the implementation silently breaks reproducibility.

### Graph Embedding (Existing Critical Dependency)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| karateclub | >= 1.3.3 | Struc2Vec graph embedding for S&P500 correlation network | The `ge` library used in the original pipeline is an unversioned, unpackaged custom library from the training server (CONCERNS.md). `karateclub` is the maintained PyPI-published package containing Struc2Vec. It produces equivalent 128-dim structural embeddings. |

**Confidence:** MEDIUM — karateclub's `Struc2Vec` class is a direct implementation of the Ribeiro et al. (2017) paper, the same algorithm used in the original `ge` library. The output format (node embeddings as numpy array) is compatible with the existing pipeline. The CONCERNS.md explicitly flags the `ge` dependency as a reproducibility blocker; `karateclub` is the standard replacement.

**Scaling note (from CONCERNS.md):** Struc2Vec on 500 nodes (S&P500) vs 300 (CSI 300) is approximately 2.8x more pairs, roughly O(n^2). Apply a correlation threshold (e.g., keep edges where |ρ| > 0.5) to prune the graph before embedding. This is documented in CONCERNS.md as a known scalability limit.

**What NOT to use:**
- The original `ge` library via `sys.path.append` — not version-controlled, not reproducible (CONCERNS.md security concern)
- `node2vec` — different algorithm; preserves graph proximity not structural equivalence; would change model semantics

### Backtesting

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| quantstats | >= 0.0.62 | Portfolio performance report generation | Pure pandas-based; produces HTML/tearsheet reports with Sharpe ratio, max drawdown, Calmar, Sortino, alpha/beta vs benchmark. Designed for post-hoc analysis of return series — exactly the use case (model outputs prediction CSVs → construct portfolio → evaluate). Zero setup friction. |
| pandas | >= 2.0.3 | Portfolio construction: top-k selection, equal-weight returns, cumulative returns | Already in stack; vectorized return calculations on prediction DataFrames |

**Confidence:** MEDIUM — quantstats is the standard lightweight tearsheet library for Python quant research. It expects a return series (pandas Series with DatetimeIndex) and produces benchmark-relative metrics against SPY. This maps directly to the existing output: `output/Multitask_output_*/regression/` prediction CSVs → portfolio returns → quantstats report.

**What NOT to use:**
- vectorbt — excellent library but significantly heavier dependency with Numba JIT compilation; overkill for a thesis backtesting module; adds complexity with no benefit given the offline, non-interactive backtest use case here
- backtrader — event-driven framework designed for live/tick-level backtesting; semantics do not match the daily batch prediction paradigm of this system; would require rewriting the consumption of prediction CSVs into an event-driven format
- Qlib backtesting — being removed; calibrated for Chinese market (T+1, A-share conventions)
- zipline-reloaded — heavy QuantopianInfrastructure heritage; difficult to set up; designed for strategy development not model evaluation

**SPY benchmark data:** Download via yfinance (`yf.download("SPY", ...)`) — same library, same pattern.

### Interactive Dashboard

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| streamlit | >= 1.32.0 | Interactive interface: period selection, prediction triggering, backtest visualization | Already mandated by PROJECT.md. No alternative considered. Fastest path from Python analytics to interactive demo; no frontend code required. Handles file upload, sliders, charts, DataFrames natively. |
| plotly | >= 5.18.0 | Interactive charts inside Streamlit (equity curves, drawdown charts, prediction scatter) | Native Streamlit integration via `st.plotly_chart()`. Plotly Express makes candlestick charts, line plots, and heatmaps trivial. Better than matplotlib for interactive exploration. |
| altair | optional | Declarative charts as alternative to Plotly | Streamlit has native `st.altair_chart()` support; lighter dependency than Plotly; useful for simple time-series line plots if Plotly feels heavy |

**Confidence:** MEDIUM — Streamlit 1.32+ is the current stable series as of my knowledge cutoff with stable `st.plotly_chart`, `st.dataframe`, `st.session_state` APIs. The `session_state` API (stable since ~1.18) is required for stateful UI (remembering run configurations across reruns).

**What NOT to use:**
- Dash (Plotly) — production-grade but requires Flask understanding and callback wiring; more setup than a thesis demo warrants
- Gradio — designed for ML model demos with single input/output; no portfolio visualization primitives
- Panel / Bokeh — heavier ecosystem, less thesis-relevant documentation and examples

### Experiment Tracking (Existing)

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| TensorBoard | existing (via torch.utils.tensorboard) | Training loss and validation metric visualization | Already in the codebase; no change needed. Keep for training runs. |

**What NOT to add:** MLflow, Weights & Biases — valid tools but add auth/setup complexity for a single-researcher thesis project. TensorBoard is already integrated and sufficient.

### Testing

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pytest | >= 7.4.0 | Unit tests for metrics, dataset loading, model forward pass | Standard Python test runner; zero config for basic test discovery. CONCERNS.md identifies "No test suite of any kind" as high priority. pytest fixtures map well to parametrized test cases for different data shapes. |
| pytest-cov | >= 4.1.0 | Coverage reporting | Single addition with pytest; provides HTML coverage report for thesis appendix |

**Confidence:** MEDIUM — pytest is the standard; no alternatives considered necessary.

### Environment Management

| Technology | Version | Purpose | Why |
|------------|---------|---------|-----|
| pip + venv | system | Environment isolation | Already in use (venv/ exists). Do not introduce conda — mixing conda and pip-installed packages in the existing venv risks conflicts. |

**`requirements.txt` fix needed (from CONCERNS.md):** The current file has `torch` commented out. The corrected file should include:

```
# CPU-only (default for local dev)
torch>=2.0.0
torchvision>=0.15.0

# Or for CUDA 11.8 on GPU server:
# --extra-index-url https://download.pytorch.org/whl/cu118
# torch>=2.0.0+cu118
```

Use a `setup.sh` or a comment block distinguishing CPU vs GPU installation — a single `requirements.txt` cannot represent both cleanly.

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Data ingestion | yfinance | Polygon.io | Paid API; out of scope per PROJECT.md |
| Data ingestion | yfinance | pandas_datareader (Yahoo) | Deprecated Yahoo provider; yfinance is the maintained successor |
| Technical indicators | pandas-ta | TA-Lib | Requires C binary compilation; breaks reproducibility on fresh clones |
| Technical indicators | pandas-ta | Qlib Alpha-360 | Calibrated for Chinese market microstructure; not valid for US equities |
| Backtesting | quantstats | vectorbt | Overkill; Numba dependency adds friction; thesis doesn't need tick-level simulation |
| Backtesting | quantstats | backtrader | Event-driven paradigm mismatches batch prediction workflow |
| Backtesting | quantstats | Qlib backtesting | Being removed; Chinese market conventions |
| Graph embedding | karateclub | Original `ge` library | Not versioned, not packaged, not reproducible (CONCERNS.md security concern) |
| Dashboard | Streamlit | Dash | More complex; overkill for single-researcher demo |
| DWT patching | Patch pytorch-wavelets in-place | Reimplement with pywt | Higher effort; risk of subtle behavior divergence |

---

## Full Dependency List for New Additions

```bash
# Data ingestion
pip install "yfinance>=0.2.38"
pip install "pandas_datareader>=0.10.0"  # optional, for supplementary index data

# Technical analysis features
pip install "pandas-ta>=0.3.14b0"

# Graph embedding (replaces unpackaged ge library)
pip install "karateclub>=1.3.3"

# Backtesting and metrics
pip install "quantstats>=0.0.62"

# Dashboard and visualization
pip install "streamlit>=1.32.0"
pip install "plotly>=5.18.0"

# Testing
pip install "pytest>=7.4.0"
pip install "pytest-cov>=4.1.0"

# Already in requirements.txt (verify versions are compatible):
# scikit-learn==1.1.2
# numpy==1.24.4  -- may need upgrade if pandas 2.x requires numpy 1.25+
# scipy==1.9.3
# matplotlib==3.7.1
# statsmodels==0.14.0
# pytorch-wavelets==1.3.0  -- needs compatibility patch for PyTorch 2.x
```

---

## Compatibility Matrix

| Existing Package | New Addition | Conflict Risk | Notes |
|-----------------|--------------|---------------|-------|
| numpy==1.24.4 | pandas>=2.0.3 | MEDIUM | pandas 2.0+ prefers numpy 1.25+; test carefully; upgrade numpy to 1.26.x if issues arise |
| pandas 2.3.3 (in venv) | pandas-ta 0.3.14b | LOW | pandas-ta 0.3.14b claims pandas 1.x compatibility but works on 2.x in practice |
| PyTorch 2.8.0 | pytorch-wavelets 1.3.0 | HIGH | Already flagged in CONCERNS.md; patch required |
| PyTorch 2.8.0 | karateclub | LOW | karateclub is numpy/networkx-based; no torch dependency |
| streamlit | plotly | LOW | Native integration; no known conflicts |

---

## Critical Path for Environment Setup

The items below must be resolved before any other pipeline work. Order matters:

1. **Verify pytorch-wavelets compatibility** with PyTorch 2.8.0 (`from pytorch_wavelets import DWT1DForward`). If broken, apply the known patch to `pytorch_wavelets/dwt/lowlevel.py` to update deprecated tensor type registration calls.

2. **Fix `requirements.txt`** to include an uncommented torch line (CPU default, GPU variant documented in comments).

3. **Add `ge` replacement** — install `karateclub` and update `Stockformer_data_preprocessing_script.py` to use `from karateclub import Struc2Vec` instead of the `sys.path.append` + `import ge` pattern.

4. **Pin numpy** to 1.26.x if pandas 2.x compatibility issues surface during yfinance integration.

5. **Fix pandas 2.x bug** in `results_data_processing.py`: replace `.applymap()` with `.map()` (CONCERNS.md known bug).

---

## Sources

All findings from training data (knowledge cutoff August 2025). External verification tools
(WebSearch, WebFetch, Context7) were unavailable during this research session.

| Claim | Confidence | Verification Path |
|-------|------------|-------------------|
| yfinance >= 0.2.38 API stability | MEDIUM | `pip index versions yfinance`; check https://github.com/ranaroussi/yfinance/releases |
| pandas-ta 0.3.14b latest stable | MEDIUM | `pip index versions pandas-ta`; check https://github.com/twopirllc/pandas-ta |
| karateclub Struc2Vec equivalence to ge | MEDIUM | Review https://karateclub.readthedocs.io/en/latest/modules/node_embedding.html#struc2vec |
| quantstats >= 0.0.62 API | MEDIUM | `pip index versions quantstats`; check https://github.com/ranaroussi/quantstats |
| streamlit >= 1.32.0 session_state stability | MEDIUM | https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state |
| pytorch-wavelets PyTorch 2.x incompatibility | HIGH | CONCERNS.md documents this explicitly; community patches documented at https://github.com/fbcotter/pytorch_wavelets/issues |
| Qlib Alpha-360 Chinese market calibration | HIGH | Qlib documentation explicitly states Alpha158/360 designed for A-share market conventions |
