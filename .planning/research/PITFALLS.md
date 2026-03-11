# Domain Pitfalls

**Domain:** ML quantitative trading research — cross-market model adaptation (CSI 300 → S&P500)
**Researched:** 2026-03-09
**Confidence:** HIGH for data leakage/backtesting methodology (established academic consensus); MEDIUM for transformer-specific financial ML pitfalls (training data + domain reasoning); LOW for pytorch-wavelets replacement specifics (requires hands-on verification)

---

## Critical Pitfalls

Mistakes that cause rewrites, invalid thesis results, or require discarding all backtest numbers.

---

### Pitfall 1: Survivorship Bias in S&P500 Universe Construction

**What goes wrong:** The thesis trains on the "current" S&P500 constituent list downloaded at research time. This list contains only companies that survived to 2026. Companies that were in the index in 2018–2023 but were removed (bankruptcy, delisting, acquisition, index rebalancing) are absent. The model sees only survivors during training, artificially inflating apparent predictability.

**Why it happens:** `yfinance` returns historical prices for tickers you ask about. If you build the ticker list from `wikipedia.org/wiki/List_of_S&P_500_companies` scraped today, you get today's members, not the historical membership. The bias is invisible — data downloads without error, training completes, and backtests look good.

**Consequences:** Backtest Sharpe ratios are overstated, drawdowns understated. Any thesis reviewer familiar with quantitative finance will flag this immediately. In severe cases it invalidates the empirical contribution entirely.

**Prevention:**
- Use a point-in-time constituent list. CRSP (subscription) has this; for free alternatives, `github.com/fja05680/sp500` maintains historical S&P500 membership CSV files going back to 2000 — use this as the ticker universe filter.
- Alternatively, explicitly acknowledge and scope the limitation: "We train on a fixed 2024 constituent list and acknowledge survivorship bias; results represent an upper bound on achievable alpha." This is thesis-acceptable if stated clearly upfront.
- Never download the ticker list from Wikipedia at runtime.

**Detection:** Check whether any stock in your universe has zero price history before 2020 or has very short histories — this indicates recent additions that may be survivorship-biased. Also check for tickers with no trading halt or delistings in the entire sample period.

**Phase:** Data pipeline phase (earliest data stage). Must be resolved before any model training.

---

### Pitfall 2: Lookahead Bias via Feature Normalization Across the Full Time Series

**What goes wrong:** When normalizing features (mean/std z-scoring, min-max scaling), the normalization statistics are computed over the entire dataset including the test set. The model trains on features that implicitly encode information about the future — specifically, what the global mean and variance of the test period will be.

**Why it happens:** The original Stockformer codebase targets a single historical run on China data where train/val/test split happens after data loading. The `StockDataset.__init__` loads all data at once as numpy arrays. It is straightforward to accidentally apply `sklearn.StandardScaler.fit_transform` to the full dataset rather than `.fit` on train only and `.transform` on val/test separately.

**Consequences:** Model appears to generalize well. Walk-forward validation shows unrealistically low MAE. Results cannot be reproduced on true out-of-sample data. This is the single most common invalidating error in ML finance papers.

**Prevention:**
- Fit all scalers exclusively on the training window. Apply the fitted scaler (no re-fitting) to val and test windows.
- For rolling window backtests: re-fit the scaler at each walk-forward step using only past data up to that step.
- Add an assertion in data preprocessing: scaler is never fitted on data whose date exceeds the train end date.
- During data preparation, log scaler statistics (mean, std per feature) to a file. Verify they match what you'd compute from the training window alone.

**Detection:** If val/test metrics are *better* than train metrics without deliberate regularization, suspect lookahead normalization. Also: add a single synthetic future row with a wildly different value — if this row affects train metrics, normalization leaks.

**Phase:** Data pipeline phase, before any feature engineering work is finalized.

---

### Pitfall 3: Temporal Train/Val/Test Split Contamination (Random Splitting)

**What goes wrong:** Financial time series have strong temporal autocorrelation. Shuffling rows before splitting (or using `sklearn.train_test_split` with `shuffle=True`) allows the model to see tomorrow's data while predicting today's. Even without explicit random shuffling, cross-validation folds that include future data in the training set contaminate results.

**Why it happens:** The Stockformer codebase appears to use a sequential split (train: first N days, val: next M days, test: last K days), which is correct structurally. The risk arises when adapting the pipeline: if a developer adds cross-validation for hyperparameter tuning using `KFold` or `GridSearchCV` without specifying `TimeSeriesSplit`, every fold will contain future data in its training portion.

**Consequences:** Validation metrics are optimistic. Hyperparameters chosen via contaminated CV are overfitted to the test set indirectly. This is particularly dangerous for the transformer's attention hyperparameters (look-back window, number of heads) which are directly responsive to temporal leakage.

**Prevention:**
- Use only `TimeSeriesSplit` or manual expanding/sliding window splits for any hyperparameter search.
- Hard-code the split dates in the config (e.g., `train_end = 2022-12-31`, `val_end = 2023-06-30`) rather than using percentage-based splits that could be applied to a shuffled dataset.
- Document the exact date boundaries of all three splits in the thesis.

**Detection:** Check that the minimum test date is strictly greater than the maximum train date, and likewise for val. Add an assertion in data loading code.

**Phase:** Data pipeline phase. Must be validated before val/test metrics are reported.

---

### Pitfall 4: Flawed Backtest — Transaction Costs and Turnover Not Modeled

**What goes wrong:** The basic backtesting notebook consumes prediction CSVs and computes cumulative returns assuming zero transaction cost and immediate execution at the close price used to generate labels. In practice, S&P500 stock-selection strategies at daily rebalancing can incur 2–10 bps per leg of a trade (spread + market impact). Top-k portfolio strategies with daily rebalancing may have 30–80% daily turnover, which destroys Sharpe ratio in realistic cost scenarios.

**Why it happens:** The original Chinese codebase made no provision for transaction costs because it was evaluating prediction accuracy (MAE, RMSE) not actual trading performance. When the thesis extends to portfolio backtesting, the temptation is to report gross (pre-cost) Sharpe ratios, which look significantly better.

**Consequences:** A gross Sharpe of 1.5 may become 0.3 after realistic costs. If the thesis reports only gross metrics, any practitioner reader will immediately discount the results. A committee member with industry background may require re-running with costs, which invalidates the narrative if numbers change dramatically.

**Prevention:**
- Model transaction costs explicitly: a flat per-trade cost of 5–10 bps round-trip is conservative and defensible for a thesis.
- Track portfolio turnover per period and report it alongside Sharpe and drawdown.
- Compare gross vs net Sharpe in the thesis — this becomes a feature (demonstrates you understand the gap) rather than a liability.
- For top-k selection: reduce rebalancing frequency to weekly if daily turnover is high.

**Detection:** Compute turnover rate = |position_today - position_yesterday| / 2 summed across all stocks. If turnover exceeds 30% daily, cost drag is material.

**Phase:** Backtesting module phase. Must be part of the portfolio construction design, not an afterthought.

---

### Pitfall 5: Regime Mismatch — Chinese Market Microstructure Applied to US Data Without Adjustment

**What goes wrong:** The Stockformer was trained and validated on CSI 300 data (2019–2021), a period of high individual-investor retail participation, government intervention in market structure, and T+1 trading rules. S&P500 data has different regime characteristics: higher institutional participation, continuous trading, options markets creating non-linear dynamics, and earnings-driven price jumps. Feature thresholds, look-back windows, and normalization ranges that worked for CSI 300 may be poorly calibrated for S&P500 without empirical validation.

**Why it happens:** The transfer is treated as a pure data swap: replace China data with US data, keep all hyperparameters. The model may technically train and produce predictions, but the learned signal may reflect China-specific microstructure that does not transfer.

**Consequences:** The model trains without errors but prediction accuracy is worse than a simple baseline (momentum, mean-reversion). The thesis contribution appears weak. Ablation analysis becomes necessary to determine whether the architecture is genuinely transferable or merely fitting noise.

**Prevention:**
- Always include a naive baseline: simple momentum (buy top-quintile prior-month return) and a linear regression benchmark. If the model cannot beat these on the US data, the architecture may not be transferring.
- Explicitly document what the thesis claims: "we transfer the architecture and retrain from scratch on US data" vs. "we transfer learned weights." These are very different claims with different validation requirements. Retraining from scratch on US data is the defensible choice for a thesis.
- Report performance across different market regimes in the test period: pre-2020, COVID-crash, post-COVID recovery, 2022 rate-rise bear market. If the model only works in one regime, the contribution is limited.

**Detection:** Compare model performance to a 1-month momentum factor. If model Sharpe < momentum Sharpe on US data, architecture transfer is failing.

**Phase:** Model training phase and evaluation phase.

---

### Pitfall 6: Struc2Vec Graph Recomputed Without Edge Filtering for ~500 Stocks

**What goes wrong:** `Stockformer_data_preprocessing_script.py` builds a dense edge list (all O(n²) pairs) before fitting Struc2Vec. For CSI 300 this is ~90,000 edges. For S&P500 with ~500 stocks this is ~250,000 edges. Struc2Vec's structural role similarity computation scales approximately O(n² × k × T) where k is the random walk depth and T is the time series length. On a CPU (the thesis constraint), this can take hours to days.

**Why it happens:** The scaling issue is documented in CONCERNS.md but easy to defer. The first time the full preprocessing pipeline is run with S&P500 data, it will simply stall or consume all memory without an informative error.

**Consequences:** The preprocessing pipeline becomes a blocker. If this is discovered late (e.g., in week 8 of 12), there may not be time to design and validate an alternative graph construction. The entire graph-based spatial attention component may need to be disabled.

**Prevention:**
- Apply an edge threshold before Struc2Vec: build the stock-stock correlation matrix first, retain only edges with |correlation| > 0.4 or top-k per node (k=20). This reduces edge count by 80–90% and makes Struc2Vec tractable.
- Alternatively: compute Struc2Vec on a random sample of 150–200 stocks, extend to full universe using nearest-neighbor embedding lookup.
- Have a fallback ready: identity adjacency matrix (no graph edges) or simple correlation-based adjacency. The spatial attention component degrades gracefully to standard attention when the adjacency matrix is identity.

**Detection:** Time the edge list construction for N=50 stocks first. Extrapolate to N=500. If extrapolation exceeds 2 hours on your CPU, edge filtering is mandatory.

**Phase:** Data pipeline phase, specifically the graph embedding step. Must be benchmarked on small N before committing to full S&P500 run.

---

### Pitfall 7: pytorch-wavelets Incompatibility Causes Silent Wrong Output Rather Than a Crash

**What goes wrong:** `pytorch-wavelets 1.3.0` was built for PyTorch 1.x. In PyTorch 2.x the internal API for custom autograd functions changed (`torch.autograd.Function.apply` semantics and `ctx.save_for_backward` shape assumptions). The library may import and run without raising an exception while computing incorrect DWT coefficients. This is harder to detect than a crash.

**Why it happens:** PyTorch 2.x maintained backward compatibility for common patterns. `pytorch-wavelets` uses private internal APIs that changed quietly. The bug manifests as numerically incorrect low/high frequency splits, not as a Python error.

**Consequences:** The wavelet disentanglement step produces garbage low/high frequency components. The model trains on corrupted input. Loss curves may look normal (the model adapts to the corrupted signal). The DWT component — a core architectural contribution — provides no benefit, but the thesis cannot detect this without explicitly validating DWT output shapes and energy partition.

**Prevention:**
- First step after environment setup: write a validation test for `disentangle()`. Apply DWT to a known sinusoidal signal, verify the low-frequency output captures the slow component and the high-frequency output captures the fast component. Compare energy: `sum(low²) + sum(high²) ≈ sum(input²)` within tolerance.
- If `pytorch-wavelets` is incompatible, replace with `pywt` (PyWavelets, actively maintained, CPU) wrapped in a `torch.nn.Module`. The 1D DWT of a batch of sequences is straightforward to implement: `pywt.dwt(x.cpu().numpy(), wavelet='haar')` with gradients handled via a custom autograd function or by treating DWT as a fixed preprocessing step.

**Detection:** Run the validation test described above before any model training. Log `energy_ratio = (low.pow(2).sum() + high.pow(2).sum()) / input.pow(2).sum()` in the first training batch. It should be close to 1.0. Values significantly different indicate incorrect DWT.

**Phase:** Environment setup phase (immediate, before any feature work begins).

---

## Moderate Pitfalls

---

### Pitfall 8: Label Construction Leaks Future Information

**What goes wrong:** If the regression target is "next-day return" defined as `(close[t+1] - close[t]) / close[t]`, this is correct. But if the target is computed after normalization using statistics that include future dates, or if the target is smoothed using a rolling window centered on the target day (rather than trailing), the label itself contains future information.

**Prevention:**
- Define labels explicitly as single-period forward returns using only t and t+1. No centering, no smoothing of labels.
- For the classification head: if the binary label is "return above median," compute the median from the training period only, not the full dataset. Apply the same threshold to val and test without re-computing.

**Phase:** Data pipeline phase, label construction step.

---

### Pitfall 9: yfinance Data Quality Issues Not Caught at Ingestion

**What goes wrong:** yfinance occasionally returns:
- Adjusted close prices for some tickers but unadjusted OHLCV for others in the same batch download
- NaN rows for trading halts, stock splits, or exchange holidays
- Duplicate rows for some tickers
- Stale data for thinly-traded periods

These errors propagate silently into feature computation (RSI, MACD), producing NaN or Inf feature values that become zero after `np.nan_to_num` in the metric function — which silently suppresses the signal.

**Prevention:**
- After each yfinance download, run validation: check NaN fraction per ticker per column (flag any ticker with >1% NaN), check for duplicate dates, check for implausible single-day returns (|return| > 30% for non-event days).
- Use adjusted close prices consistently for return computation. `yfinance` downloads adjusted prices via `auto_adjust=True` — verify this is set and consistent.
- Store raw downloaded data as Parquet before any transformation. Never re-download during training.

**Phase:** Data pipeline phase. Validation should run as part of the download script, not discovered during model training.

---

### Pitfall 10: Metric Suppression via `np.errstate` and `nan_to_num` Hides Degenerate Predictions

**What goes wrong:** The existing `metric()` function wraps all computation in `np.errstate(divide='ignore', invalid='ignore')` and replaces NaN/Inf with zero. A model that predicts a constant value for all stocks (a common failure mode in regression) will have MAPE = ∞, which gets silently replaced with 0.0. The training log shows an excellent MAPE of 0.0, masking a completely degenerate model.

**Why it happens:** This code pattern is already in the codebase (CONCERNS.md documents it). During US adaptation, if feature preprocessing is incorrect, the model may collapse to predicting the mean, which is a common local minimum for regression with L1/L2 loss.

**Consequences:** Hours of GPU compute appear to "succeed" while the model has converged to a trivial solution. The issue is only discovered when prediction CSVs show all-identical values.

**Prevention:**
- Add a collapse detection check after each epoch: if the standard deviation of predictions across all stocks is below a threshold (e.g., < 1e-4), log a `WARNING: model may be predicting constant values`.
- Replace the `nan_to_num` pattern with explicit checks: if MAPE is NaN/Inf, report it as `-1` (sentinel) rather than `0`.

**Phase:** Model training phase, training loop instrumentation.

---

### Pitfall 11: Benchmark Comparison Using SPY Total Return vs Price Return Inconsistency

**What goes wrong:** The backtest compares portfolio returns against SPY. If the portfolio uses dividend-adjusted prices (as yfinance provides with `auto_adjust=True`) but SPY benchmark is pulled as price return, the comparison is unfair. S&P500 has a dividend yield of ~1.5% annually, which over a 3-year backtest represents ~4.5% cumulative return difference. A strategy that appears to generate 3% alpha may actually have -1.5% alpha.

**Prevention:**
- Use SPY with `auto_adjust=True` for the benchmark consistently with portfolio return computation.
- Report both price-return and total-return numbers in the thesis to show awareness.

**Phase:** Backtesting module phase.

---

### Pitfall 12: Overfitting to Thesis Test Period via Repeated Model Evaluation

**What goes wrong:** Under thesis time pressure, it is tempting to tune hyperparameters, fix bugs, and re-evaluate repeatedly on the same test set. Each iteration that reports an improved test Sharpe ratio is a de facto form of overfitting — the test set is no longer out-of-sample because experimental decisions were made in response to its performance.

**Why it happens:** There is only one test set and it is evaluated often. Every time a preprocessing bug is fixed or a feature is added, the test metrics change, and the developer observes them and adjusts course accordingly.

**Prevention:**
- Define a "blind" holdout period (e.g., 2024-01-01 onward) that is never examined until final thesis evaluation.
- All development, debugging, and tuning uses train and val sets only.
- Run final test set evaluation once, at thesis submission time, after all model decisions are frozen. Report those numbers.
- If the test set has already been evaluated repeatedly, acknowledge this in the thesis limitations section.

**Phase:** Evaluation phase (but the habit must be established in the data pipeline phase).

---

## Minor Pitfalls

---

### Pitfall 13: Alpha_360 Feature Semantics Do Not Transfer to US Market

**What goes wrong:** Qlib's Alpha_360 factors include signals calibrated to Chinese market conventions: T+1 settlement, circuit breakers, sector classifications by Chinese industry standards, and alpha factors derived from A-share market microstructure. Even if these features could be recomputed for US data, their predictive content may be negligible or inverse on US equities.

**Prevention:** Do not attempt to replicate Alpha_360 for US data. Replace entirely with standard US technical indicators: 5/10/20/60-day momentum, RSI(14), MACD(12,26,9), Bollinger Bands, ATR, volume z-score. These have documented empirical support in the US market literature.

**Phase:** Feature engineering phase.

---

### Pitfall 14: Hardcoded Batch Size 12 Silently Truncates Small Test Windows

**What goes wrong:** The existing config uses `batch_size = 12`. If the test window contains fewer than 12 sequences (possible with short look-back windows or short test periods), PyTorch's DataLoader drops the last incomplete batch by default (`drop_last=True` or the data loading code simply assumes full batches). Some test days produce no predictions, silently shortening the backtest period.

**Prevention:** Set `drop_last=False` and add padding or handle the final incomplete batch explicitly. Verify the number of test predictions equals the number of test trading days.

**Phase:** Data pipeline / model training phase.

---

### Pitfall 15: Global `device` Variable Causes Silent CPU Training After GPU Init Fails

**What goes wrong:** `Stockformermodel/Multitask_Stockformer_models.py` uses a module-level `global device` variable. If GPU initialization fails partway through instantiation, the global may be set to `cuda` but subsequent layers that reference it before instantiation will use stale state or fail with `NameError`. On the thesis compute environment (CPU primary, GPU opt-in), this creates a fragile situation where training may silently run on CPU while logs indicate GPU.

**Prevention:** Add an explicit device check at training start: `print(f"Training on: {next(model.parameters()).device}")`. Verify this matches the intended device before launching a long training run.

**Phase:** Environment setup / training loop.

---

### Pitfall 16: Streamlit Session State Allows Running Stale Cached Predictions

**What goes wrong:** Streamlit caches function outputs via `@st.cache_data`. If prediction CSVs are regenerated after a model update but Streamlit's cache is not cleared, the UI displays old prediction results alongside new model metrics. The backtest appears to use the new model but actually uses stale predictions.

**Prevention:** Use versioned output file names that include the config hash or experiment timestamp. The UI should display the source file name alongside results so the user can verify recency.

**Phase:** Streamlit interface phase.

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Mitigation |
|-------------|---------------|------------|
| yfinance S&P500 download | Survivorship bias from current-day ticker list | Use historical constituent list (github.com/fja05680/sp500) |
| yfinance S&P500 download | Data quality: NaN, duplicates, split mismatches | Validate at ingestion, store raw Parquet immediately |
| Feature engineering (US) | Alpha_360 semantics inapplicable to US data | Replace entirely with standard TA indicators |
| Feature normalization | Lookahead from full-dataset scaler fit | Fit scaler on train window only, transform val/test |
| Label construction | Future median used for classification threshold | Compute threshold from training period only |
| Graph embedding (Struc2Vec) | O(n²) stall for ~500 S&P500 stocks | Benchmark at N=50 first; apply edge threshold |
| pytorch-wavelets DWT | Silent wrong output with PyTorch >= 2.0 | Validate energy partition before training |
| Train/val/test split | Random split or percentage split after shuffle | Hard-code date boundaries in config |
| Hyperparameter tuning | Test set observed repeatedly, overfitting | Hold out blind test period until final evaluation |
| Model training loop | Constant-prediction collapse hidden by nan_to_num | Log prediction std per epoch; alert if < 1e-4 |
| Portfolio backtest | Zero transaction cost overstates Sharpe | Model 5–10 bps per trade; track turnover |
| Benchmark comparison | SPY price return vs portfolio total return mismatch | Use `auto_adjust=True` for both |
| Backtest metrics | Survivorship bias inflates all performance numbers | Acknowledge or fix universe construction first |
| Thesis reporting | Gross Sharpe without cost analysis | Report both gross and net metrics |

---

## Research Gaps

The following areas were assessed from domain knowledge and codebase analysis. They were not verified against current literature due to tool access constraints and should be treated as MEDIUM confidence:

- **Walk-forward validation methodology**: the thesis may benefit from expanding-window vs rolling-window walk-forward evaluation. The correct choice depends on whether the model is expected to be retrained periodically (rolling) or trained once on all available history (expanding). This trade-off should be made explicit in the thesis methodology chapter.
- **Statistical significance of backtest results**: A single backtest over one historical period has limited statistical significance. The thesis would be strengthened by block bootstrap or permutation tests on returns. This is not common in deep learning finance papers but committee members from statistics backgrounds may require it.
- **Sector-neutral vs sector-agnostic portfolio construction**: top-k selection without sector constraints concentrates in one sector during sector momentum periods, which conflates stock selection alpha with sector beta. This is thesis-scope but worth one paragraph of discussion.

---

## Sources

Findings are based on:
- Direct analysis of the project codebase (`CONCERNS.md`, `PROJECT.md`)
- Established quantitative finance and ML research methodology:
  - Lopez de Prado, M. (2018). *Advances in Financial Machine Learning* — canonical reference for financial ML backtesting pitfalls
  - Bailey et al. (2014). "The Deflated Sharpe Ratio" — test set overfitting in finance
  - Prado (2016). "Building Diversified Portfolios that Outperform Out-of-Sample" — walk-forward validation
- Domain reasoning about CSI 300 vs S&P500 microstructure differences
- pytorch-wavelets GitHub issue tracker (last known state: unmaintained since 2021, PyTorch 2.x incompatibility reported in multiple issues)
- Confidence is HIGH for data leakage/backtesting pitfalls (academically established), MEDIUM for regime transfer pitfalls (reasoned from domain knowledge), LOW for pytorch-wavelets exact failure mode (requires hands-on validation)
