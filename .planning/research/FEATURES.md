# Feature Landscape

**Domain:** ML quantitative trading research platform — S&P500 equity prediction and portfolio backtesting
**Researched:** 2026-03-09
**Confidence:** HIGH (domain grounded in existing codebase + established academic standards)

---

## Table Stakes

Features a thesis committee expects. Missing any of these and the project is academically incomplete.

### 1. Price-Volume Technical Feature Engineering

| Feature Group | Specific Features | Complexity | Notes |
|--------------|-------------------|------------|-------|
| Candlestick ratios | KMID=(close-open)/open, KLEN=(high-low)/open, KUP=(high-max(open,close))/open, KLOW=(min(open,close)-low)/open | Low | Normalized intraday body/shadow; direct US equivalent of Alpha158's KMID/KLEN family |
| Price momentum (cross-sectional) | ROC-5, ROC-10, ROC-20, ROC-60 (Return Over Close lookback N) | Low | Captures short/medium/long momentum; standard in US equity ML factor research |
| Moving average ratios | MA5/MA20, MA20/MA60 price vs MA | Low | Trend following; MA crossover signals |
| RSI (Relative Strength Index) | RSI-6, RSI-14 | Low | Overbought/oversold; 6-period for short-term, 14-period standard |
| MACD | MACD line (EMA12-EMA26), Signal line (EMA9 of MACD), Histogram | Medium | Momentum convergence/divergence; standard in US quant |
| Bollinger Band features | %B=(price-lower)/(upper-lower), Bandwidth=(upper-lower)/MA | Medium | Volatility normalization; captures mean-reversion signals |
| Volume features | VWAP deviation=(close-VWAP)/VWAP, Volume ratio=V/MA_V5, Volume momentum=V/V_5d_avg | Low | US market microstructure; VWAP central to institutional trading |
| Volume-price interaction | OBV (On-Balance Volume), MFI (Money Flow Index 14-period), Accumulation/Distribution Line | Medium | Cross-validates price moves with volume confirmation |
| Volatility | ATR-14 normalized by close, Historical volatility 10/20/60-day (std of log returns) | Low | Risk-adjusted feature construction; critical for portfolio construction stage |
| Lookback normalization | Each price-based feature computed at windows [5, 10, 20, 60] days | Low | Matches Alpha158 pattern of VSUMN5/10/20/60 for multi-horizon signals |

**Why table stakes:** The thesis claim is adaptation from Chinese (Alpha-360 Qlib) to US market with custom feature engineering. The committee will expect a well-defined, reproducible feature set. The existing Qlib Alpha158/360 features are price-volume ratios with multiple lookback windows — the US replacement must match this structure to make a fair comparison. Missing this means the experiment is not comparable to baseline.

**Dependency:** All downstream features (model input, graph construction, backtesting) depend on this being built first.

### 2. Feature Normalization and Preprocessing Pipeline

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Cross-sectional z-score normalization | Required for model numerical stability and fair stock ranking | Low | Apply per trading day across S&P500 universe; mirrors neutralization.py approach |
| Forward-fill missing values | S&P500 has trading halts and gaps; model cannot accept NaN | Low | pandas ffill then fill remaining with cross-sectional mean |
| Rolling window construction | Model expects T1=20 lookback window of factor sequences | Low | seq2instance pattern already in StockDataset |
| Train/val/test split (no leakage) | Critical for honest evaluation | Low | Must split by date, not randomly; apply normalization fit only on train |

**Why table stakes:** Without clean normalized features in `.npz` format, the model cannot run. This is a direct prerequisite of everything else.

### 3. Prediction Quality Metrics (Model Evaluation)

| Metric | Why Expected | Complexity | Notes |
|--------|--------------|------------|-------|
| MAE (Mean Absolute Error) — regression | Already computed in existing code | Low | Primary regression loss metric |
| RMSE — regression | Standard in time-series forecasting literature | Low | Complements MAE |
| Accuracy / F1-score — classification | Trend direction prediction (up/down); already in existing `metric()` | Low | Binary classification of return sign |
| IC (Information Coefficient) | Rank correlation between predicted and actual returns; **mandatory** in any equity alpha research | Medium | Spearman IC per day, then mean IC and ICIR=IC/std(IC) across test period |
| Rank IC | IC computed on ranked predictions (more robust to outliers) | Medium | Standard alongside regular IC in Chinese and US quant research |

**Why table stakes:** IC and ICIR are the primary metrics that quant researchers and thesis committees use to evaluate whether a signal has predictive power before backtesting. The existing codebase computes MAE and accuracy but does NOT compute IC/ICIR — this is a gap that must be filled or the thesis is academically weak.

### 4. Portfolio Construction

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Top-K long-only selection | Select top K stocks by predicted return score; already in Backtest.ipynb | Low | Existing code does top-10; parameterize K |
| Equal-weight allocation | 1/K weight per selected stock | Low | Already implemented in existing backtest |
| Daily rebalancing | Rebalance portfolio every day using new model predictions | Low | Matches daily prediction frequency |
| Transaction cost modeling | Apply bid-ask spread / commission fee (e.g. 10bps round-trip) | Low | Existing code has `fee` parameter set to 0; must set realistic US value (~0.1%) |
| SPY benchmark comparison | Compare portfolio cumulative return vs SPY (S&P500 ETF) | Low | The US equivalent of HS300 benchmark already in backtest |

**Why table stakes:** Without portfolio construction, the model's return predictions cannot be evaluated in terms of economic significance. A committee will ask "does this actually make money?" which requires backtesting a concrete strategy.

### 5. Core Backtesting Metrics

| Metric | Why Expected | Complexity | Notes |
|--------|--------------|------------|-------|
| Cumulative return curve | Fundamental output; already in existing backtest | Low | Visual equity curve plot |
| Annualized return | Already computed in existing code | Low | (1+total_return)^(252/n_days) - 1 |
| Annualized Sharpe ratio | Already computed; primary risk-adjusted metric | Low | mean_daily_return / std_daily_return * sqrt(252) |
| Maximum drawdown | Already computed | Low | max peak-to-trough decline |
| Alpha vs SPY | Excess return over S&P500 benchmark | Low | Annualized portfolio return minus SPY return |
| Beta vs SPY | Portfolio sensitivity to market | Low | Covariance(port, SPY) / Variance(SPY) |
| Annualized volatility | Std of daily returns * sqrt(252) | Low | Complements Sharpe; shows risk profile |
| Calmar ratio | Annualized return / abs(max drawdown) | Low | Common in US hedge fund evaluation |

**Why table stakes:** Sharpe, drawdown, and benchmark-relative alpha are the minimum metrics any quant finance thesis must report. The existing code computes Sharpe and drawdown but lacks alpha/beta — these must be added for US context (SPY is the natural benchmark, replacing HS300).

### 6. Streamlit Interactive Interface — Core Navigation

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Date range selector | Let user pick train/val/test periods | Low | Streamlit date_input or slider |
| Run pipeline button | Trigger feature generation, model training, inference | Medium | Long-running; needs st.spinner and progress feedback |
| Equity curve chart | Plot portfolio vs SPY cumulative returns over test period | Low | plotly line chart |
| Metrics summary table | Annualized return, Sharpe, drawdown, alpha, beta, Calmar | Low | st.dataframe or st.metric cards |
| Model prediction heatmap | Stock vs date grid of predicted returns (test period) | Medium | Plotly heatmap; shows model signal quality visually |

**Why table stakes:** PROJECT.md explicitly requires Streamlit interface for thesis presentation. Without basic navigation and visualization, the platform cannot function as a demo.

---

## Differentiators

Features that strengthen the thesis contribution beyond baseline expectations.

### A. Ablation Study Support

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Feature group toggle (OHLCV vs momentum vs volume) | Shows which feature categories contribute most to IC/return | Medium | Parameterize feature set in config; run multiple experiments |
| Wavelet decomposition vs no-wavelet comparison | Tests architecture contribution in US context | Medium | Config flag to bypass DWT; already config-driven |
| Graph embedding ablation (with/without Struc2Vec) | Tests whether inter-stock relationships matter for S&P500 | Medium | Config flag to zero out adjgat input |

**Why differentiating:** Thesis contribution is adaptation from CSI300 to S&P500. Ablations prove which components transfer. Without them, the committee can ask "how do you know the wavelet or graph parts matter in the US?" An ablation table is the standard defense.

### B. IC/ICIR Time-Series Visualization

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Rolling 20-day IC chart | Shows whether signal is consistent or decaying over time | Medium | Spearman correlation per day; rolling window |
| IC cumulative sum plot | Standard in quant research to show alpha consistency | Low | Derived from daily IC series |
| IC distribution histogram | Shows signal quality and tail behavior | Low | Normal distribution is evidence of stable alpha |

**Why differentiating:** These are standard in professional quant research papers (Barra, Qlib documentation, QuantConnect blog) but not in student theses. Including them signals quantitative sophistication to the committee.

### C. Cross-Model Comparison Table

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Stockformer vs LSTM vs GRU vs MLP baselines | Shows Stockformer improvement on S&P500 | High | Each model needs separate train/inference cycle |
| Comparison table: IC, ICIR, Sharpe, Return, Drawdown | Side-by-side performance summary across models | Low | Once baseline results exist; formatting only |

**Why differentiating:** The existing Chinese codebase already runs LSTM, GRU, TCN, Transformer, ALSTM, GATS as comparison models via Qlib. Replicating even a subset of these (LSTM + GRU + Transformer as baselines) for US data makes the thesis directly comparable to prior work.

### D. Streamlit — Analysis Depth Pages

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Per-stock prediction accuracy page | Shows which S&P500 sectors the model predicts best | Medium | Group by GICS sector; show IC by sector |
| Drawdown analysis decomposition | Shows worst drawdown periods with market context | Medium | Identify if drawdowns coincide with market crises (COVID, 2022 rate hikes) |
| Feature importance visualization | Which input features correlate most with model predictions | Medium | Compute Spearman correlation of each feature with prediction output |

**Why differentiating:** These pages transform the tool from a results-display system into an interactive research exploration platform — a stronger thesis demo that shows understanding of what drives the model.

### E. Reproducibility and Experiment Tracking

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Config-driven experiment naming | Each run tagged with date range, feature set, model config | Low | Already partially done via .conf; extend naming convention |
| Experiment results registry (CSV log) | Persist each run's metrics to a summary CSV | Low | Append row per run; enables comparison without re-running |
| Seed-controlled reproducibility assertion | Assert that two runs with same seed produce identical metrics | Low | One unit test; powerful for thesis credibility |

**Why differentiating:** Reproducibility is an explicit thesis requirement (PROJECT.md). Demonstrating it with evidence (locked seed, persisted run log) is stronger than just claiming it.

---

## Anti-Features

Things to deliberately NOT build. Scope control is critical given 3-6 month timeline.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| Real-time / live trading execution | Requires brokerage API, risk management, latency engineering — months of work outside thesis scope | Accept research-only platform; state clearly in thesis limitations section |
| Alternative data (news, sentiment, SEC filings) | Fundamentally changes the research question; confounds the price-volume feature adaptation story | Mention as future work; keep feature set purely price-volume OHLCV-derived |
| Factor neutralization (market cap, sector) | Adds a Chinese-market-specific step that requires sector data not available via yfinance; the US adaptation thesis is about the model, not factor construction methodology | Note that US price-volume features do not require neutralization at the same level; simple cross-sectional z-score is sufficient |
| Short selling / long-short portfolio | Requires margin, borrow costs, realistic execution modeling — changes the backtesting problem significantly | Build long-only top-K portfolio; note long-short as future extension |
| Walk-forward cross-validation | Adds significant compute (multiple model trains); for a thesis, single train/val/test split is the standard | Use a clear single train 60% / val 20% / test 20% time-ordered split |
| Production Streamlit deployment (cloud, auth, Docker) | Out of scope per PROJECT.md; adds devops complexity | Local execution only; document setup instructions |
| Hyperparameter optimization (grid search, Optuna) | Very compute-intensive; not essential if model already has reasonable defaults from Chinese market experiment | Use original hyperparameters from .conf; run one or two manual ablations |
| Fundamental / valuation features (P/E, P/B, ROE) | Requires paid data (Compustat, Bloomberg) or complex scraping; out of scope | Stick to yfinance OHLCV; document this constraint |
| Intraday features (tick data, order book) | yfinance only provides daily OHLCV; intraday requires expensive data | Daily granularity is sufficient for the thesis claim |
| Full Qlib baseline replication (10 models) | The Chinese backtest ran 10 models; for US context, 3 baselines (LSTM, GRU, Transformer) are sufficient to establish comparison | Limit baseline suite; prioritize quality of comparison over breadth |

---

## Feature Dependencies

```
OHLCV download (yfinance)
  └─> Price-volume feature engineering (momentum, RSI, MACD, BB, volume ratios)
        └─> Cross-sectional normalization (z-score per day)
              └─> flow.npz + trend_indicator.npz generation
              └─> Correlation matrix -> Struc2Vec graph embeddings
                    └─> Model training (MultiTask_Stockformer_train.py)
                          └─> Inference -> prediction CSVs (regression + classification)
                                └─> IC / ICIR computation (model evaluation)
                                └─> Portfolio construction (top-K selection, equal-weight)
                                      └─> Backtesting engine (returns, Sharpe, drawdown, alpha vs SPY)
                                            └─> Streamlit interface (display results, charts, metrics)
                                                  └─> Ablation study (re-run pipeline with different configs)
```

Key dependency notes:
- Feature engineering must be complete before any model work begins — the model's input dimension `d_feat` in the .conf file is determined by number of features
- Struc2Vec graph embedding depends on the correlation matrix computed from label returns (not from features) — this can be computed in parallel with feature engineering
- IC/ICIR computation depends on inference output CSVs AND the original return labels for the test period — both must be co-indexed by date and stock ticker
- Streamlit is a display layer only; all computation (training, inference, backtesting) must complete before results are shown

---

## MVP Recommendation

For thesis delivery within 3-6 months, build in this priority order:

**Must have (thesis minimum viable):**
1. Price-volume feature engineering (20-30 features across 5/10/20/60-day windows) — computed from yfinance OHLCV
2. Preprocessing pipeline producing flow.npz, trend_indicator.npz, corr_adj.npy, graph embeddings for S&P500
3. IC / ICIR metrics added to model evaluation
4. Backtesting engine with SPY benchmark: cumulative return, annualized return, Sharpe, max drawdown, alpha, beta
5. Streamlit interface with equity curve chart, metrics table, date range selector

**Should have (thesis strengthening):**
6. 2-3 baseline model comparisons (LSTM, GRU, Transformer) for the comparison table
7. IC time-series rolling chart in Streamlit
8. Ablation config flags (wavelet on/off, graph on/off)

**Defer to future work section:**
- Per-sector IC breakdown
- Cross-model comparison with 10 baselines
- Factor importance visualization

---

## Sources

- Existing codebase: `Backtest/Backtest.ipynb` — confirms existing metrics (total return, annualized return, Sharpe, max drawdown); identifies gap (no IC, no alpha/beta vs benchmark)
- Existing codebase: `data_processing_script/volume_and_price_factor_construction/3_qlib_factor_construction.ipynb` — confirms Alpha158 feature structure (KMID, KLEN, KUP, KLOW, VSUM variants with 5/10/20/30/60-day windows); informs US equivalent feature set design
- Existing codebase: `4_neutralization.py` — confirms factor neutralization (market cap + sector OLS regression) was used for Chinese market; identified as anti-feature for US adaptation scope
- Project context: `.planning/PROJECT.md` — defines out-of-scope items (real-time trading, alternative data, international markets, paid APIs); validated anti-feature decisions
- Domain knowledge: Alpha158/Alpha360 Qlib factor taxonomy (candlestick ratios, volume/price momentum, rolling statistics) — HIGH confidence, well-established academic standard
- Domain knowledge: IC/ICIR as primary quant alpha evaluation metrics — HIGH confidence, universal in academic quantitative finance research (Grinold & Kahn "Active Portfolio Management"; standard in Qlib documentation and Chinese/US quant research papers)
- Domain knowledge: Sharpe ratio, max drawdown, Calmar ratio, alpha/beta as standard backtesting metrics for US equity strategies — HIGH confidence
