# External Integrations

**Analysis Date:** 2026-03-09

## APIs & External Services

**Quantitative Finance Platform:**
- Microsoft Qlib - Stock factor generation, baseline model training, and backtesting
  - SDK/Client: `qlib` (Python package, not in requirements.txt — installed separately)
  - Auth: None (local data provider)
  - Initialization: `qlib.init(provider_uri='<path_to_qlib_data>')`
  - Used in: `Backtest/Backtest.ipynb`, `data_processing_script/volume_and_price_factor_construction/3_qlib_factor_construction.ipynb`
  - Factor sets used: `Alpha158` (158 features), `Alpha360` (360 features)
  - Baseline models accessed via qlib: LGBModel, CatBoostModel, XGBModel, TCN, LSTM, GRU, TransformerModel, LocalformerModel, ALSTM, GATs

**Graph Embedding Library:**
- Struc2Vec (custom `ge` library) - Structural graph embedding of stock correlation network
  - SDK/Client: custom library at `/root/autodl-tmp/Stockformer/Stockformer_run/GraphEmbedding`
  - Auth: None (local library, added via `sys.path.append`)
  - Used in: `data_processing_script/stockformer_input_data_processing/Stockformer_data_preprocessing_script.py`
  - Output: 128-dimensional embeddings saved as `128_corr_struc2vec_adjgat.npy`

## Data Storage

**Databases:**
- None — all data stored as flat files (CSV, NPZ, NPY)

**File Storage:**
- Local filesystem only
- Data directory structure (on training server): `/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/data/Stock_CN_2021-06-04_2024-01-30/`
  - `flow.npz` - Stock return time-series (main input features)
  - `trend_indicator.npz` - Binary up/down trend indicator per stock per day
  - `corr_adj.npy` - Stock correlation adjacency matrix
  - `128_corr_struc2vec_adjgat.npy` - Struc2Vec graph embeddings (GAT positional encoding)
  - `label.csv` - Raw stock return labels
  - `Alpha_360_2021-06-04_2024-01-30/` - Directory of per-factor CSVs (Alpha360 features)
- Config-driven paths in `config/Multitask_Stock.conf` under `[file]` section

**Caching:**
- None — no explicit caching layer

## Authentication & Identity

**Auth Provider:**
- None — no user authentication anywhere in the codebase

## Monitoring & Observability

**Experiment Tracking:**
- TensorBoard - Training loss and validation metrics (accuracy, MAE, RMSE, MAPE) logged per epoch
  - Writer initialized in `MultiTask_Stockformer_train.py`: `SummaryWriter(new_folder)`
  - Log directory: `runs/Multitask_Stockformer/Stock_CN_2021-06-04_2024-01-30/version{N}/`
  - Scalars logged: `training loss`, `Val/Average_Accuracy`, `Val/Average_MAE`, `Val/Average_RMSE`, `Val/Average_MAPE`

**Logs:**
- Plain text log files written via `log_string()` in `lib/Multitask_Stockformer_utils.py`
- Log path configured in `config/Multitask_Stock.conf`: `log = ./log/STOCK/log_Multitask_2021-06-04_2024-01-30`
- Best model epoch and metric printed to log on improvement

**Error Tracking:**
- None — no external error tracking service

## CI/CD & Deployment

**Hosting:**
- AutoDL cloud GPU server (Linux-64) — original training environment
- No containerization or deployment manifests detected

**CI Pipeline:**
- None detected

## Environment Configuration

**Required configurations:**
- INI config file path passed as `--config` CLI argument to `MultiTask_Stockformer_train.py`
- Qlib data provider URI hardcoded in notebooks (not externalized):
  - `'/root/autodl-tmp/Stockformer/Stockformer_run/SOTA_Model_Comparison/qlib_data/ch_data'` (Backtest notebook)
  - `'~/.qlib/qlib_data/ch_data'` (factor construction notebook)
- GraphEmbedding library path hardcoded in preprocessing script:
  - `'/root/autodl-tmp/Stockformer/Stockformer_run/GraphEmbedding'`
- TensorBoard output path hardcoded in training script:
  - `'/root/autodl-tmp/Stockformer/Stockformer_run/Stockformer_code/runs/Multitask_Stockformer/Stock_CN_...'`

**Secrets location:**
- No secrets detected — all data is local, no API keys or credentials required

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Data Pipeline Overview

The project uses a sequential offline data pipeline (no live data feeds):

1. Raw stock data (CSVs) → `data_processing_script/volume_and_price_factor_construction/1_stock_data_consolidation.ipynb`
2. Preprocessing + Alpha158/360 factor generation via qlib → `3_qlib_factor_construction.ipynb`
3. Factor neutralization (market cap + industry) → `4_neutralization.py`
4. Graph construction + Struc2Vec embedding → `Stockformer_data_preprocessing_script.py`
5. Model training → `MultiTask_Stockformer_train.py` with `config/Multitask_Stock.conf`
6. Result post-processing → `data_processing_script/stockformer_input_data_processing/results_data_processing.py`
7. Backtesting vs. qlib baselines → `Backtest/Backtest.ipynb`

---

*Integration audit: 2026-03-09*
