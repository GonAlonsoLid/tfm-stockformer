# Roadmap: Multitask-Stockformer S&P500

## Overview

Starting from an existing Stockformer codebase trained on Chinese equities, the project adapts the full pipeline to S&P500: fixing portability blockers, building a US data pipeline, retraining the model, adding rigorous evaluation metrics, constructing a portfolio and backtesting engine, wrapping everything in a Streamlit interface, and validating key components with tests. Each phase unblocks the next — infrastructure makes the code portable, data feeds the model, the model produces predictions, evaluation scores them, portfolio/backtest turns scores into returns, the UI makes results explorable, and tests confirm correctness.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Infrastructure** - Make the codebase portable and reproducible on any machine (completed 2026-03-10)
- [x] **Phase 2: Data Pipeline** - Build the S&P500 OHLCV ingestion, feature engineering, and preprocessing pipeline (completed 2026-03-11)
- [x] **Phase 3: Model Training** - Configure and train the Stockformer on S&P500 data end-to-end (completed 2026-03-12)
- [x] **Phase 4: Evaluation** - Compute IC/ICIR and retain all original metrics on test predictions (completed 2026-03-14)
- [ ] **Phase 5: Portfolio & Backtesting** - Turn prediction scores into portfolios and compute performance statistics
- [ ] **Phase 6: Interface** - Streamlit app for interactive exploration of predictions and backtest results
- [ ] **Phase 7: Testing** - Unit test suite covering metrics, data loading, and model forward pass

## Phase Details

### Phase 1: Infrastructure
**Goal**: The codebase runs on any machine without manual path editing, and a developer can reproduce the environment from a fresh clone
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03
**Success Criteria** (what must be TRUE):
  1. Running the training script on a machine that has no `/root/autodl-tmp/` directory does not raise a path error
  2. `pip install -r requirements.txt` on a clean Python 3.9+ environment completes without conflicts
  3. A developer following the setup documentation can run a smoke test (import model, load config) within 30 minutes of cloning
**Plans**: 2 plans
Plans:
- [ ] 01-01-PLAN.md — Fix hardcoded paths (INFRA-01) + fix requirements.txt and create smoke test (INFRA-02)
- [ ] 01-02-PLAN.md — Write SETUP.md onboarding guide + human verify smoke test (INFRA-03)

### Phase 2: Data Pipeline
**Goal**: Raw S&P500 price-volume data is downloaded, cleaned, feature-engineered, normalized, split, and serialized into the exact array format the model expects
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04, DATA-05
**Success Criteria** (what must be TRUE):
  1. Running the pipeline script produces Parquet files containing daily OHLCV for S&P500 constituents with no missing trading days
  2. The output feature matrix contains momentum, RSI, MACD, Bollinger Bands, and volume ratio columns for all configured windows
  3. Training-set normalization statistics are computed first and applied separately to val/test splits (no leakage)
  4. `flow.npz`, `trend_indicator.npz`, and the Struc2Vec graph embedding file exist and have the correct shapes for model ingestion
  5. Feature engineering produces >= 60 universal TA indicators to replace Alpha-360 richness for Phase 3 (gap closure: plan 02-06)
**Plans**: 6 plans
Plans:
- [x] 02-01-PLAN.md — Test scaffold: Wave 0 stubs + conftest Phase 2 fixtures (DATA-01..DATA-05) (completed 2026-03-11)
- [x] 02-02-PLAN.md — Download + clean S&P500 OHLCV via yfinance → Parquet (DATA-01) (completed 2026-03-11)
- [x] 02-03-PLAN.md — Feature engineering: TA indicators + label.csv (DATA-02) (completed 2026-03-11)
- [x] 02-04-PLAN.md — Normalize/split + serialize flow.npz + trend_indicator.npz (DATA-03, DATA-04, DATA-05) (completed 2026-03-11)
- [x] 02-05-PLAN.md — Graph embedding + orchestrator script + requirements.txt update (DATA-05) (completed 2026-03-11)
- [x] 02-06-PLAN.md — Expand TA features from 16 to ~69 (gap closure: Phase 3 prerequisite, DATA-02) (completed 2026-03-11)

### Phase 3: Model Training
**Goal**: The Stockformer trains end-to-end on S&P500 data, saves checkpoints, and produces prediction CSVs on the test period
**Depends on**: Phase 2 (including 02-06 feature expansion)
**Requirements**: MODEL-01, MODEL-02
**Success Criteria** (what must be TRUE):
  1. A `.conf` config file for S&P500 exists with correct feature dimensions, date ranges, and path references pointing to Phase 2 outputs
  2. Running the training script completes without error, produces TensorBoard logs, and saves at least one checkpoint
  3. Running the inference script on the saved checkpoint produces a prediction CSV covering the test period
**Plans**: 4 plans
Plans:
- [ ] 03-01-PLAN.md — Wave 0 test scaffold: xfail stubs for MODEL-01 and MODEL-02 (MODEL-01, MODEL-02)
- [ ] 03-02-PLAN.md — Create config/Multitask_Stock_SP500.conf wiring Phase 2 outputs (MODEL-01)
- [ ] 03-03-PLAN.md — Create scripts/run_inference.py standalone inference script (MODEL-02)
- [ ] 03-04-PLAN.md — Human verify: training smoke test (2 epochs) + inference CSV confirmation (MODEL-01, MODEL-02)

### Phase 4: Evaluation
**Goal**: Model predictions are scored with IC/ICIR in addition to the original MAE/RMSE/accuracy/F1 metrics
**Depends on**: Phase 3
**Requirements**: EVAL-01, EVAL-02
**Success Criteria** (what must be TRUE):
  1. After inference, a script computes and prints IC (Spearman rank correlation between predicted and realized returns) and ICIR (IC mean / IC std) for the test period
  2. MAE, RMSE, accuracy, and F1 are still computed and reported without regression from the original codebase behavior
**Plans**: 2 plans
Plans:
- [ ] 04-01-PLAN.md — Wave 0 test scaffold: unit tests for compute_ic.py functions (EVAL-01, EVAL-02)
- [ ] 04-02-PLAN.md — Implement scripts/compute_ic.py + human verify (EVAL-01, EVAL-02)

### Phase 5: Portfolio & Backtesting
**Goal**: Prediction scores are converted into a daily-rebalanced top-K portfolio and evaluated against the SPY benchmark with full performance statistics
**Depends on**: Phase 4
**Requirements**: PORT-01, PORT-02, PORT-03, BACK-01, BACK-02, BACK-03
**Success Criteria** (what must be TRUE):
  1. A portfolio module selects the top-K stocks by predicted return score each day, where K is configurable via a single parameter
  2. The backtest applies equal-weight allocation with daily rebalancing and deducts ~10bps round-trip transaction cost
  3. A cumulative return chart is produced comparing the portfolio to SPY over the test period
  4. A summary table shows annualized return, Sharpe ratio, max drawdown, alpha, and beta vs SPY
**Plans**: 3 plans
Plans:
- [ ] 05-01-PLAN.md — Wave 0 test scaffold: xfail stubs for all 6 requirements (PORT-01..BACK-03)
- [ ] 05-02-PLAN.md — Core backtest functions: top-K selection, portfolio weights, transaction cost, performance metrics (PORT-01, PORT-02, PORT-03, BACK-02, BACK-03)
- [ ] 05-03-PLAN.md — CLI wiring: yfinance download, backtest loop, equity curve chart, output files + human verify (BACK-01)

### Phase 6: Interface
**Goal**: A Streamlit app lets the user select a date range, trigger the pipeline, and explore predictions and backtest results interactively
**Depends on**: Phase 5
**Requirements**: UI-01, UI-02, UI-03, UI-04
**Success Criteria** (what must be TRUE):
  1. Launching `streamlit run app.py` opens a browser interface with a date range selector and a button to run predictions and backtesting
  2. After running, an equity curve chart shows the portfolio vs SPY cumulative returns for the selected period
  3. A metrics table displays annualized return, Sharpe ratio, max drawdown, alpha, and beta on the same page
  4. A heatmap visualization shows model prediction scores per stock per day for the test period
**Plans**: TBD

### Phase 7: Testing
**Goal**: Key components have automated unit tests that catch regressions in metrics, data loading, and model behavior
**Depends on**: Phase 6
**Requirements**: TEST-01, TEST-02, TEST-03
**Success Criteria** (what must be TRUE):
  1. Running `pytest` produces a passing test suite with no errors for backtesting metric functions (Sharpe, drawdown, alpha)
  2. A dataset loading test asserts correct tensor shape and zero NaN values in the processed S&P500 arrays
  3. A model forward-pass smoke test with random input of the correct shape completes without error and returns the expected output shape
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Infrastructure | 2/2 | Complete   | 2026-03-10 |
| 2. Data Pipeline | 6/6 | Complete   | 2026-03-11 |
| 3. Model Training | 4/4 | Complete   | 2026-03-12 |
| 4. Evaluation | 2/2 | Complete   | 2026-03-14 |
| 5. Portfolio & Backtesting | 0/TBD | Not started | - |
| 6. Interface | 0/TBD | Not started | - |
| 7. Testing | 0/TBD | Not started | - |
