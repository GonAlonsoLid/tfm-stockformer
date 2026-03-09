# Multitask-Stockformer S&P500

## What This Is

A research ML platform adapting the Multitask-Stockformer architecture — originally trained on Chinese equities (CSI 300) — to the US stock market (S&P 500). The system spans data ingestion, feature engineering, model training, portfolio construction, backtesting, and an interactive Streamlit interface for analysis and presentation. The primary deliverable is a Master's thesis demonstrating end-to-end reproducibility and quantitative evaluation.

## Core Value

A working, reproducible end-to-end pipeline that trains the Stockformer on S&P500 data, generates portfolios, and lets the user interactively explore backtest results — all in a single deployable system.

## Requirements

### Validated

<!-- Existing capabilities confirmed in the codebase. -->

- ✓ Multitask Stockformer model (dual regression + classification heads) — existing
- ✓ Wavelet DWT decomposition of input time series (low/high frequency split) — existing
- ✓ Dual encoder architecture (temporal attention + temporal conv net + sparse spatial attention) — existing
- ✓ Adaptive fusion of low/high frequency streams — existing
- ✓ Graph-based inter-stock relationships via Struc2Vec adjacency embeddings — existing
- ✓ Config-driven experiment management via `.conf` INI files — existing
- ✓ Full train/val/test loop with TensorBoard logging and checkpoint saving — existing
- ✓ Post-training inference pipeline outputting CSV predictions — existing
- ✓ Basic backtesting notebook consuming prediction CSVs — existing

### Active

<!-- Current scope for the thesis. Building toward these. -->

- [ ] S&P500 data pipeline: download daily OHLCV for S&P500 universe via yfinance, store as Parquet
- [ ] Feature engineering: momentum, volatility, moving averages, RSI, MACD, volume indicators
- [ ] S&P500 model config: new `.conf` file and data preprocessing scripts targeting US data
- [ ] Fix portability blockers: remove hardcoded `/root/autodl-tmp/` paths; make all paths config-driven
- [ ] Reproducible environment: working `requirements.txt` + setup instructions for local execution
- [ ] Portfolio construction module: top-k selection, equal-weight, risk-aware allocation
- [ ] Backtesting engine: cumulative return, Sharpe ratio, max drawdown, alpha vs SPY benchmark
- [ ] Streamlit interactive interface: time period selection, run predictions, run backtests, visualize results
- [ ] Thesis output generation: equity curve charts, benchmark comparison tables, ablation figures
- [ ] Basic test suite: unit tests for metrics, dataset loading, and model forward pass

### Out of Scope

- Real-time / live trading execution — research platform only, no brokerage integration
- Alternative data (news, sentiment, fundamentals) — price-volume features only for thesis scope
- International markets beyond S&P500 — US market is the thesis contribution
- Mobile app or production deployment — Streamlit for local/demo use only
- OAuth / user authentication — single-user research tool
- Paid market data APIs — Yahoo Finance (yfinance) sufficient for thesis data needs

## Context

The original Stockformer codebase was developed for the Chinese market (CSI 300) and trained on a remote AutoDL GPU server. All pipeline stages exist but are tightly coupled to that environment:

- **Hardcoded paths**: `/root/autodl-tmp/` appears across training script, utils, and data processing scripts — must be replaced
- **Data format**: model expects `.npz` arrays (`flow.npz`, `trend_indicator.npz`) and a Struc2Vec graph embedding — these must be regenerated for S&P500
- **Qlib dependency**: factor construction notebooks use Qlib's Alpha-360 factor library, which is calibrated for Chinese markets; US adaptation requires custom feature engineering
- **Working components**: model architecture, training loop, and inference pipeline are sound and should be preserved; refactoring scope is paths + data layer, not the model itself
- **Known bugs**: `applymap` → `map` in `results_data_processing.py` (pandas 2.x incompatibility); `wape` metric computed but discarded
- **pytorch-wavelets**: version 1.3.0 not maintained; may need patching or replacement for PyTorch ≥ 2.0

## Constraints

- **Data**: Yahoo Finance only (yfinance) — free tier, rate-limited; data quality sufficient for thesis
- **Timeline**: 3–6 months to thesis submission — scope must be achievable, defer polish
- **Compute**: No guaranteed GPU access; pipeline must be runnable on CPU (slower) with GPU opt-in via config
- **Tech stack**: PyTorch + Python 3.9+; no stack changes to core model — extend, don't replace
- **Reproducibility**: All pipeline stages must run from a fresh clone with documented setup steps
- **Interface**: Streamlit — no existing UI, no other framework required

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Preserve original model architecture | Thesis contribution is adaptation, not architectural novelty; changing the model conflates variables | — Pending |
| Use yfinance for S&P500 data | Free, no API key, sufficient daily OHLCV quality for research; avoids paid data dependency | — Pending |
| Replace Qlib Alpha-360 with custom features | Qlib Alpha-360 targets Chinese market conventions; US features built from scratch using standard TA indicators | — Pending |
| Streamlit for UI | No existing UI; Streamlit fastest path to interactive interface; thesis demo requirement not production UX | — Pending |
| Config-driven paths (no hardcodes) | Required for reproducibility across machines; the single highest-impact portability fix | — Pending |

---
*Last updated: 2026-03-09 after initialization*
