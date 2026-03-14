# Requirements: Multitask-Stockformer S&P500

**Defined:** 2026-03-10
**Core Value:** A working, reproducible end-to-end pipeline that trains the Stockformer on S&P500 data, generates portfolios, and lets the user interactively explore backtest results.

## v1 Requirements

Requirements for the thesis milestone. Each maps to a roadmap phase.

### Infrastructure

- [ ] **INFRA-01**: All hardcoded `/root/autodl-tmp/` paths replaced with config-driven paths so the pipeline runs on any machine
- [ ] **INFRA-02**: Working `requirements.txt` with pinned versions enabling a clean local install from scratch
- [x] **INFRA-03**: Setup documentation that reproduces the environment from a fresh clone

### Data Pipeline

- [x] **DATA-01**: S&P500 OHLCV data downloaded via yfinance and stored as Parquet files
- [x] **DATA-02**: Price-volume features computed: momentum (ROC), RSI, MACD, Bollinger Bands, volume ratios across 5/10/20/60-day windows
- [x] **DATA-03**: Cross-sectional z-score normalization applied per trading day across the S&P500 universe
- [x] **DATA-04**: Train/val/test split by date (no leakage); normalization statistics fit on training set only
- [x] **DATA-05**: Pipeline produces `flow.npz`, `trend_indicator.npz`, and Struc2Vec graph embeddings for S&P500

### Model Training

- [x] **MODEL-01**: New `.conf` config file targeting S&P500 data with correct paths, feature dimensions, and date ranges
- [x] **MODEL-02**: Model trains end-to-end on S&P500 data and saves checkpoints successfully

### Evaluation

- [x] **EVAL-01**: IC (Information Coefficient) and ICIR computed on test period predictions
- [x] **EVAL-02**: Existing MAE, RMSE, accuracy, and F1 metrics retained from original codebase

### Portfolio & Backtesting

- [ ] **PORT-01**: Top-K stock selection from predicted return scores with K configurable via parameter
- [ ] **PORT-02**: Equal-weight allocation with daily rebalancing
- [ ] **PORT-03**: Transaction cost modeling applied (≈10bps round-trip for US market)
- [ ] **BACK-01**: Cumulative return curve computed and plotted against SPY benchmark
- [ ] **BACK-02**: Annualized return, Sharpe ratio, and max drawdown computed
- [ ] **BACK-03**: Alpha and beta versus SPY benchmark computed

### Interface

- [ ] **UI-01**: Streamlit app with date range selector and run pipeline button
- [ ] **UI-02**: Equity curve chart showing portfolio vs SPY cumulative returns
- [ ] **UI-03**: Metrics summary table (annualized return, Sharpe, max drawdown, alpha, beta)
- [ ] **UI-04**: Model prediction heatmap (stock × date grid for test period)

### Testing

- [ ] **TEST-01**: Unit tests for backtesting metrics (Sharpe, drawdown, alpha)
- [ ] **TEST-02**: Unit test for dataset loading (correct shape, no NaN values)
- [ ] **TEST-03**: Unit test for model forward pass (smoke test with random input)

## v2 Requirements

Deferred to future work. Acknowledged but not in current roadmap.

### Ablation Studies

- **ABLA-01**: Wavelet decomposition on/off config flag for ablation
- **ABLA-02**: Graph embedding on/off config flag for ablation
- **ABLA-03**: Feature group toggle (OHLCV-only vs full feature set) for ablation

### Baseline Comparisons

- **BASE-01**: LSTM baseline trained and evaluated on S&P500 data
- **BASE-02**: GRU baseline trained and evaluated on S&P500 data
- **BASE-03**: Transformer baseline trained and evaluated on S&P500 data
- **BASE-04**: Side-by-side comparison table (IC, ICIR, Sharpe, return, drawdown)

### Advanced Visualization

- **VIZ-01**: Rolling 20-day IC time-series chart in Streamlit
- **VIZ-02**: IC distribution histogram
- **VIZ-03**: Per-sector IC breakdown (GICS sector grouping)

### Experiment Tracking

- **EXP-01**: Experiment results registry (CSV log per run with metrics)
- **EXP-02**: Seed-controlled reproducibility assertion (unit test)

## Out of Scope

| Feature | Reason |
|---------|---------|
| Real-time / live trading execution | Requires brokerage API and risk management — outside thesis scope |
| Alternative data (news, sentiment, fundamentals) | Confounds price-volume adaptation story; requires paid data |
| Short selling / long-short portfolio | Requires margin modeling; long-only top-K is sufficient for thesis |
| Walk-forward cross-validation | Compute-intensive; single time-ordered split is academic standard |
| Cloud / production deployment | Local Streamlit only; no auth, Docker, or devops |
| Hyperparameter optimization | Use original .conf defaults; one or two manual ablations at most |
| International markets beyond S&P500 | US adaptation is the thesis contribution |
| Paid market data APIs | yfinance sufficient; cost constraint |
| Intraday features | yfinance provides daily OHLCV only |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Pending |
| INFRA-02 | Phase 1 | Pending |
| INFRA-03 | Phase 1 | Complete |
| DATA-01 | Phase 2 | Complete |
| DATA-02 | Phase 2 | Complete |
| DATA-03 | Phase 2 | Complete |
| DATA-04 | Phase 2 | Complete |
| DATA-05 | Phase 2 | Complete |
| MODEL-01 | Phase 3 | Complete |
| MODEL-02 | Phase 3 | Complete |
| EVAL-01 | Phase 4 | Complete |
| EVAL-02 | Phase 4 | Complete |
| PORT-01 | Phase 5 | Pending |
| PORT-02 | Phase 5 | Pending |
| PORT-03 | Phase 5 | Pending |
| BACK-01 | Phase 5 | Pending |
| BACK-02 | Phase 5 | Pending |
| BACK-03 | Phase 5 | Pending |
| UI-01 | Phase 6 | Pending |
| UI-02 | Phase 6 | Pending |
| UI-03 | Phase 6 | Pending |
| UI-04 | Phase 6 | Pending |
| TEST-01 | Phase 7 | Pending |
| TEST-02 | Phase 7 | Pending |
| TEST-03 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 25 total
- Mapped to phases: 25
- Unmapped: 0 ✓

---
*Requirements defined: 2026-03-10*
*Last updated: 2026-03-10 after roadmap creation*
