# Phase 2: Data Pipeline - Research

**Researched:** 2026-03-10
**Domain:** Financial time-series data acquisition, technical indicator engineering, normalization, graph embedding (Struc2Vec), NumPy array serialization for PyTorch model ingestion
**Confidence:** MEDIUM-HIGH

---

## Summary

Phase 2 replaces the original Chinese-market Qlib-based factor pipeline with a fully custom S&P500 pipeline built on yfinance + pandas. The model's `StockDataset` class expects two primary `.npz` arrays (`flow.npz` = return labels, `trend_indicator.npz` = binary up/down), plus a directory of per-factor CSV files (the "Alpha" bonus features) and a Struc2Vec graph embedding `.npy` file. All of these are consumed at training time; the shapes and column ordering must exactly match what `StockDataset.__init__` and `disentangle()` assume.

The original pipeline used Qlib Alpha158 (158-feature Chinese market factors). For S&P500 we replace this with custom TA features: momentum (ROC), RSI, MACD, Bollinger Bands, and volume ratios computed with `pandas-ta` across 5/10/20/60-day windows. Normalization must be cross-sectional z-score (across stocks per trading day), fit only on the training split to avoid leakage. The train/val/test split is date-ordered at 75%/12.5%/12.5%.

The biggest risk in this phase is the Struc2Vec graph embedding step: the `shenweichen/GraphEmbedding` library is not on PyPI and must be installed from GitHub. For 500+ stocks the O(n^2) edge list is large but manageable with a correlation threshold filter (keep only |corr| > 0.3). A secondary risk is yfinance rate limiting at 500-ticker download scale — batch downloads with chunks of ~80 tickers and retry logic are required.

**Primary recommendation:** Build a single `scripts/build_pipeline.py` script with clearly ordered steps (download → clean → engineer → normalize → split → serialize → graph), driven by a config file. Do not use notebooks — they cannot be automated or tested. Each step writes intermediate files so reruns can skip completed steps.

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| DATA-01 | S&P500 OHLCV data downloaded via yfinance and stored as Parquet files | yfinance 1.2.0 supports batch download; Wikipedia S&P500 constituent list via `pd.read_html`; Parquet via `pandas.to_parquet` |
| DATA-02 | Price-volume features computed: momentum (ROC), RSI, MACD, Bollinger Bands, volume ratios across 5/10/20/60-day windows | `pandas-ta` library provides all required indicators; custom volume ratio is `volume / volume.rolling(N).mean()` |
| DATA-03 | Cross-sectional z-score normalization applied per trading day across the S&P500 universe | Compute mean/std per date row across stock columns; apply with `(x - mean) / std` using pandas `groupby` or vectorized operations |
| DATA-04 | Train/val/test split by date (no leakage); normalization statistics fit on training set only | Date-sorted split at 75%/12.5%/12.5% time boundaries; fit `StandardScaler` (or manual mean/std) on train rows only; apply to val/test |
| DATA-05 | Pipeline produces `flow.npz`, `trend_indicator.npz`, and Struc2Vec graph embeddings for S&P500 | `Stockformer_data_preprocessing_script.py` already produces these; adapt it for S&P500 ticker universe; Struc2Vec via `shenweichen/GraphEmbedding` GitHub install |
</phase_requirements>

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| yfinance | 1.2.0 (latest) | Download S&P500 OHLCV from Yahoo Finance API | Project decision: free, sufficient for daily data |
| pandas | 2.3.3 (already installed) | Data wrangling, Parquet I/O, rolling window calculations | Project standard; already in requirements.txt |
| numpy | 1.24.4 (already installed) | Array operations, .npz serialization | Project standard; model ingestion format |
| pandas-ta | latest (not yet installed) | RSI, MACD, Bollinger Bands, ROC, volume ratio indicators | Pure-Python, pandas-native, no C compiler needed unlike TA-Lib |
| networkx | 3.2.1 (already installed) | Build stock correlation graph for Struc2Vec | Already in requirements.txt; used in existing preprocessing script |
| scikit-learn | 1.1.2 (already installed) | `StandardScaler` for normalization statistics | Already in requirements.txt |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| GraphEmbedding (ge) | GitHub master (shenweichen) | Struc2Vec node embeddings for stock correlation graph | Only for DATA-05; install via `pip install git+https://github.com/shenweichen/GraphEmbedding.git` |
| pyarrow | latest | Parquet file engine for pandas | Install alongside yfinance; needed for `df.to_parquet()` |
| tqdm | 4.62.3 (already installed) | Progress bars during 500-ticker download loop | Already in requirements.txt |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pandas-ta | TA-Lib | TA-Lib requires system C library compile step (complex CI/local setup); pandas-ta is pure Python |
| pandas-ta | ta (technical-analysis-library) | Both are viable; pandas-ta has cleaner API for our indicator set |
| yfinance | Alpha Vantage, Polygon.io | Paid tier or very low free-tier limits; project decision is yfinance-only |
| GraphEmbedding (ge) | dgllge (PyPI) | dgllge page failed to load; use GitHub source to guarantee shenweichen API compatibility with existing script |

**Installation (new packages only):**
```bash
pip install yfinance pyarrow pandas-ta
pip install git+https://github.com/shenweichen/GraphEmbedding.git
```

Then add to `requirements.txt`:
```
yfinance>=0.2.50
pyarrow>=12.0
pandas-ta>=0.3.14b
# GraphEmbedding: pip install git+https://github.com/shenweichen/GraphEmbedding.git
```

---

## Architecture Patterns

### Recommended Project Structure

```
data_processing_script/
└── sp500_pipeline/
    ├── 01_download_ohlcv.py       # DATA-01: yfinance download → Parquet
    ├── 02_feature_engineering.py  # DATA-02: TA indicators per stock
    ├── 03_normalize_split.py      # DATA-03+04: z-score norm, train/val/test
    ├── 04_serialize_arrays.py     # DATA-05: flow.npz, trend_indicator.npz
    └── 05_graph_embedding.py      # DATA-05: Struc2Vec → 128_corr_struc2vec_adjgat.npy

scripts/
└── build_pipeline.py              # Orchestrates all five steps end-to-end

data/
└── Stock_SP500_<start>_<end>/
    ├── ohlcv/                     # One Parquet file per ticker
    ├── features/                  # One CSV per feature (for StockDataset Alpha format)
    ├── flow.npz                   # Return labels [T, N]
    ├── trend_indicator.npz        # Binary labels [T, N]
    ├── corr_adj.npy               # Correlation matrix [N, N]
    ├── data.edgelist              # Weighted edge list for Struc2Vec
    └── 128_corr_struc2vec_adjgat.npy  # Embedding [N, 128]
```

### Pattern 1: S&P500 Constituent List Acquisition

**What:** Fetch current S&P500 tickers from Wikipedia using pandas; filter to tickers that have data for the full date range.
**When to use:** At the start of the download step; the list is dynamic (constituents change) so use the Wikipedia snapshot as a practical approximation.

```python
# Source: pandas read_html from Wikipedia (standard pattern, verified by WebSearch)
import pandas as pd

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    tickers = table['Symbol'].str.replace('.', '-', regex=False).tolist()
    return tickers  # Yahoo Finance uses '-' not '.' (e.g., BRK-B not BRK.B)
```

**Note:** The Wikipedia list is the current S&P500 snapshot, not point-in-time. For a thesis this is acceptable (the project is not claiming survivor-bias-free backtesting). The requirements specify "S&P500 constituents" which this satisfies.

### Pattern 2: Batched yfinance Download with Retry

**What:** Download OHLCV for 500 tickers in chunks of ~80 to avoid rate limiting (429 errors).
**When to use:** DATA-01 download step.

```python
# Source: WebSearch findings (multiple community sources, MEDIUM confidence)
import yfinance as yf
import time

def download_ohlcv_batched(tickers, start, end, chunk_size=80, delay=2.0):
    """Download OHLCV in chunks with retry. Returns dict of {ticker: DataFrame}."""
    results = {}
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        for attempt in range(3):
            try:
                data = yf.download(
                    chunk, start=start, end=end,
                    auto_adjust=True, progress=False
                )
                # data is MultiIndex: (field, ticker) for multiple tickers
                for ticker in chunk:
                    if ticker in data.columns.get_level_values(1):
                        results[ticker] = data.xs(ticker, axis=1, level=1).dropna(how='all')
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(delay * (attempt + 1))
                else:
                    print(f"Failed chunk starting {chunk[0]}: {e}")
        time.sleep(delay)
    return results
```

### Pattern 3: Technical Indicator Engineering with pandas-ta

**What:** Compute RSI, MACD, Bollinger Bands, ROC (momentum), and volume ratio for each window using pandas-ta applied per ticker.
**When to use:** DATA-02 feature engineering step.

```python
# Source: pandas-ta documentation patterns (MEDIUM confidence — not Context7 verified)
import pandas_ta as ta

def compute_features(df_ohlcv, windows=(5, 10, 20, 60)):
    """
    df_ohlcv: DataFrame with columns [Open, High, Low, Close, Volume]
    Returns: DataFrame with all feature columns
    """
    features = df_ohlcv[['Close', 'Volume']].copy()

    for w in windows:
        # Momentum (Rate of Change)
        features[f'ROC_{w}'] = ta.roc(df_ohlcv['Close'], length=w)
        # RSI
        features[f'RSI_{w}'] = ta.rsi(df_ohlcv['Close'], length=w)
        # MACD (standard uses 12/26/9; use w as fast period for multi-window)
        macd = ta.macd(df_ohlcv['Close'], fast=w, slow=w*2, signal=9)
        if macd is not None:
            features[f'MACD_{w}'] = macd.iloc[:, 0]  # MACD line
        # Bollinger Bands
        bbands = ta.bbands(df_ohlcv['Close'], length=w)
        if bbands is not None:
            features[f'BB_upper_{w}'] = bbands.iloc[:, 0]
            features[f'BB_lower_{w}'] = bbands.iloc[:, 2]
            features[f'BB_width_{w}'] = bbands.iloc[:, 0] - bbands.iloc[:, 2]
        # Volume ratio
        features[f'VOL_ratio_{w}'] = df_ohlcv['Volume'] / df_ohlcv['Volume'].rolling(w).mean()

    return features
```

**MACD window interpretation:** The requirement says "across 5/10/20/60-day windows." For RSI and ROC this is straightforward (lookback = window). For MACD and Bollinger Bands, the window serves as the primary period. Use the standard (12/26/9) MACD as the canonical MACD feature and treat the window variation for the other indicators.

### Pattern 4: Cross-Sectional Z-Score Normalization (No Leakage)

**What:** Per trading day, z-score normalize across all stocks. Compute mean/std from training rows only, then apply to all splits.
**When to use:** DATA-03 + DATA-04 normalization step.

```python
# Source: Standard ML practice, verified by WebSearch (MEDIUM confidence)
import numpy as np

def cross_sectional_normalize(feature_matrix, train_end_idx):
    """
    feature_matrix: np.ndarray [T, N] — rows=dates, cols=stocks
    train_end_idx: integer — last index of training set (exclusive)
    Returns: normalized matrix, (train_mean, train_std) for record
    """
    train_slice = feature_matrix[:train_end_idx]
    # Compute stats per date row across stocks (axis=1)
    train_mean = train_slice.mean(axis=1, keepdims=True)
    train_std = train_slice.std(axis=1, keepdims=True)
    # Avoid division by zero
    train_std = np.where(train_std < 1e-8, 1.0, train_std)

    # Apply training stats to all splits
    normalized = (feature_matrix - train_mean[:feature_matrix.shape[0]]) / train_std[:feature_matrix.shape[0]]
    # For val/test, we cannot use per-date stats from future dates.
    # Instead, use the LAST training day's stats as a proxy — or compute
    # stats from training-period dates and broadcast to val/test.
    # See Pitfall 2 for the correct approach.
    return normalized
```

**Critical note:** True cross-sectional z-score = normalize across stocks for each date. The training-set normalization statistics are per-date values (one mean, one std per date). For val/test dates, we do not have "training statistics" because the dates themselves are new. The correct approach is: compute per-date statistics using ONLY training dates, and for val/test dates compute per-date statistics from the val/test data itself (cross-sectional does not look forward — it looks across stocks at the same point in time). What must NOT happen is fitting a scaler on all dates including val/test dates.

### Pattern 5: StockDataset-Compatible Array Format

**What:** The model's `StockDataset` expects specific array shapes loaded from `.npz` files.
**When to use:** DATA-05 serialization step.

```python
# Source: lib/Multitask_Stockformer_utils.py — direct codebase analysis (HIGH confidence)

# flow.npz['result'] — shape: [T, N]
#   T = total trading days, N = number of stocks
#   Values: daily return (forward 1-day return, e.g., (close_t+1 - close_t) / close_t)
#   This is what StockDataset loads as 'Traffic' and uses as regression target

# trend_indicator.npz['result'] — shape: [T, N]
#   Values: binary (1 if return > 0, else 0)
#   This is what StockDataset loads as 'indicator' and uses as classification target

# Alpha feature CSVs — directory of files, each [T, N] (dates x stocks)
#   Each CSV = one feature factor; index = dates, columns = stock tickers
#   StockDataset reads all CSVs and stacks on axis=2 to form [T, N, F] bonus array

# 128_corr_struc2vec_adjgat.npy — shape: [N, 128]
#   Graph embedding of stocks, used by GAT attention in model
#   Loaded by lib/graph_utils.py loadGraph()

import numpy as np

def save_model_arrays(returns_df, data_dir):
    """
    returns_df: DataFrame [dates x tickers] of 1-day forward returns
    Saves flow.npz and trend_indicator.npz in the format StockDataset expects.
    """
    data = returns_df.values  # [T, N]
    np.savez(f'{data_dir}/flow.npz', result=data)
    trend = (data > 0).astype(int)
    np.savez(f'{data_dir}/trend_indicator.npz', result=trend)
```

### Anti-Patterns to Avoid

- **Running notebooks as the pipeline:** The original code uses Jupyter notebooks for factor construction. These cannot be automated, tested, or called from `build_pipeline.py`. Convert all logic to `.py` scripts.
- **Computing normalization stats on the full dataset then splitting:** This is data leakage. Always split dates first, compute stats on train rows only.
- **Building the full O(n^2) edge list without thresholding for 500 stocks:** 500 stocks produce 124,750 edges; this works but Struc2Vec training will be slow (hours). Apply `|corr| > 0.3` threshold first to reduce edge count.
- **Downloading one ticker at a time in a loop:** Triggers Yahoo Finance rate limiting immediately. Use `yf.download()` with batch tickers.
- **Saving feature CSVs with stocks as rows and dates as columns:** `StockDataset` expects index=dates, columns=stocks. Verify orientation before saving.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| RSI computation | Manual Wilder's smoothing | `pandas_ta.rsi()` | Wilder's smoothing is not standard EWM — off-by-one errors common |
| MACD line | Manual EWM subtraction | `pandas_ta.macd()` | Signal line period and initial NaN handling are fiddly |
| Bollinger Bands | Manual rolling mean ± 2*std | `pandas_ta.bbands()` | Handles NaN periods and ddof conventions correctly |
| Parquet file I/O | Custom binary format | `pandas.to_parquet()` + pyarrow | Standard, compressed, fast; already in pandas |
| Correlation matrix | Manual loops | `numpy.corrcoef()` or `pandas.DataFrame.corr()` | Already used in existing `Stockformer_data_preprocessing_script.py` |
| Node embeddings | Custom graph neural network | `Struc2Vec` from GraphEmbedding | The model expects exactly this embedding format; it is already wired into the training script |

**Key insight:** The factor engineering for Chinese stocks used Qlib Alpha158 (158 features, proprietary Chinese data). We replace this with our own feature set but keep the output format identical — one CSV per feature, indexed by date with stock tickers as columns. The model does not care what the features mean, only their shape.

---

## Common Pitfalls

### Pitfall 1: yfinance Rate Limiting on 500 Tickers
**What goes wrong:** Calling `yf.download()` with all 500 tickers at once, or in a per-ticker loop, triggers HTTP 429 (Too Many Requests). The download silently returns empty DataFrames for failed tickers.
**Why it happens:** Yahoo Finance has tightened API limits in 2025; the free tier is hit-based with no clear documented limit.
**How to avoid:** Use chunks of ~80 tickers, add `time.sleep(2)` between chunks, implement 3-retry logic with exponential backoff.
**Warning signs:** DataFrames that come back with NaN-only columns, or total ticker count after download is less than input count.

### Pitfall 2: Data Leakage in Cross-Sectional Normalization
**What goes wrong:** Computing `scaler.fit(entire_feature_matrix)` then splitting. The val/test statistics leak into the normalization of training data.
**Why it happens:** Misreading "cross-sectional" (across stocks at one time point) as "global" (across all time and stocks).
**How to avoid:** Cross-sectional z-score is computed per date row. For training dates: compute mean/std across stocks for that date. For val/test dates: same — compute across stocks for that date. The "no leakage" constraint means: do not use future dates' statistics when normalizing earlier dates. Per-date cross-sectional normalization inherently has no time leakage because it only looks across the stock dimension, not across time.
**Warning signs:** Val/test metrics look unrealistically good (near-perfect); normalization stats differ significantly if computed on train-only vs full dataset.

### Pitfall 3: Stock-Date Matrix Alignment
**What goes wrong:** Different stocks have different trading history lengths (IPOs, delistings). The final matrix must be rectangular [T, N] with no holes.
**Why it happens:** Some S&P500 constituents IPO'd during the analysis period; yfinance returns shorter series for them.
**How to avoid:** Forward-fill up to 5 days (handles weekends/holidays already excluded), then drop stocks with more than 5% missing after forward-fill. Align all stock series to a master trading calendar (derive from the most liquid ticker, e.g., SPY).
**Warning signs:** The `flow.npz` array has NaN values; `StockDataset` loads them and `disentangle()` fails with NaN-contaminated DWT output.

### Pitfall 4: Struc2Vec Memory/Time on 500 Stocks
**What goes wrong:** Building the full correlation edge list for 500 stocks produces 124,750 edges; Struc2Vec with workers=4 may take 2-8 hours on a laptop CPU.
**Why it happens:** Struc2Vec DTW distance computation scales O(n^2 * T) where T is time series length.
**How to avoid:** Apply absolute correlation threshold `|corr| > 0.3` before building the edge list; this typically reduces edges by 60-80%. Set `workers=4` or higher if machine allows. Pre-compute and cache the embedding — it only needs to be done once.
**Warning signs:** Process runs for >30 minutes without progress output; memory usage exceeds 8GB.

### Pitfall 5: Feature CSV Format Mismatch
**What goes wrong:** Saving feature CSVs with stocks as the index and dates as columns (transposed). `StockDataset` concatenates CSVs on axis=2 expecting `[T, stocks, 1]` per CSV.
**Why it happens:** pandas convention for "per stock time series" is often (date index, stock columns) but it's easy to accidentally transpose.
**How to avoid:** Always verify: `df.index` = DatetimeIndex of trading days; `df.columns` = stock ticker list. Shape should be `[T, N]` for each saved CSV.
**Warning signs:** `StockDataset.__init__` raises an axis mismatch error when concatenating; or `bonus_all` shape is `[N, T, F]` instead of `[T, N, F]`.

### Pitfall 6: GraphEmbedding Library Not in requirements.txt
**What goes wrong:** `Stockformer_data_preprocessing_script.py` imports `from ge import Struc2Vec` but `ge` is not pip-installable by name. A fresh clone will fail.
**Why it happens:** The original code used a local clone of GraphEmbedding on the AutoDL server at a hardcoded path.
**How to avoid:** Install via `pip install git+https://github.com/shenweichen/GraphEmbedding.git`. Document this in `requirements.txt` as a comment and in `SETUP.md`. The `--ge_path` CLI argument in the existing script can be removed once pip-installed.
**Warning signs:** `ModuleNotFoundError: No module named 'ge'` when running the preprocessing script.

---

## Code Examples

Verified patterns from codebase analysis and official sources:

### StockDataset Array Contract (HIGH confidence — direct code analysis)
```python
# From lib/Multitask_Stockformer_utils.py (lines 108-119)
# The model expects EXACTLY this structure:

Traffic = np.load(args.traffic_file)['result']   # shape: [T, N_stocks]
indicator = np.load(args.indicator_file)['result']  # shape: [T, N_stocks]

# Alpha feature directory: each CSV has shape [T, N_stocks]
# index = datetime, columns = stock tickers
# All CSVs stacked -> bonus_all shape: [T, N_stocks, N_features]

# seq2instance creates sliding windows:
# X shape: [n_samples, T1=20, N_stocks]   (input)
# Y shape: [n_samples, T2=2,  N_stocks]   (target)
# T1=20 is the lookback window (20 trading days)
# T2=2 is the prediction horizon (2 trading days)
```

### Config Parameters for S&P500 Dataset (HIGH confidence)
```ini
# New config file: config/Multitask_SP500.conf
[file]
traffic    = ./data/Stock_SP500_<start>_<end>/flow.npz
indicator  = ./data/Stock_SP500_<start>_<end>/trend_indicator.npz
adj        = ./data/Stock_SP500_<start>_<end>/corr_adj.npy
adjgat     = ./data/Stock_SP500_<start>_<end>/128_corr_struc2vec_adjgat.npy
alpha_360_dir = ./data/Stock_SP500_<start>_<end>/features
model      = ./cpt/STOCK/saved_model_Multitask_SP500_<start>_<end>
log        = ./log/STOCK/log_Multitask_SP500_<start>_<end>
output_dir = ./output/Multitask_output_SP500_<start>_<end>
tensorboard_dir = ./runs/Multitask_Stockformer/Stock_SP500_<start>_<end>

[data]
dataset    = STOCK
T1         = 20
T2         = 2
train_ratio = 0.75
val_ratio   = 0.125
test_ratio  = 0.125

# N_stocks will be set by the actual filtered S&P500 universe (~480-500)
```

### Existing Preprocessing Script (Adapted) — HIGH confidence
```python
# data_processing_script/stockformer_input_data_processing/Stockformer_data_preprocessing_script.py
# Already handles: flow.npz, trend_indicator.npz, corr_adj.npy, data.edgelist, Struc2Vec
# It reads from label.csv (dates x stocks return matrix)
# We only need to produce label.csv in the correct format from our pipeline

# The --data_dir argument now points to the S&P500 data directory:
# python data_processing_script/stockformer_input_data_processing/Stockformer_data_preprocessing_script.py \
#   --data_dir ./data/Stock_SP500_2018-01-01_2024-01-01
```

### Forward Return Computation (HIGH confidence)
```python
# Returns label = 1-day forward return (used as regression target)
# From existing codebase pattern: label.csv contains these values
close_prices = ohlcv_wide['Close']  # [T, N] pivot: index=date, columns=ticker
forward_returns = close_prices.shift(-1) / close_prices - 1
# Drop last row (no forward return available)
label_df = forward_returns.iloc[:-1]
label_df.fillna(0, inplace=True)
label_df.to_csv(f'{data_dir}/label.csv')
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Qlib Alpha158 (158 features, Qlib pipeline) | Custom TA features via pandas-ta | Phase 2 (this phase) | Must produce matching CSV format for StockDataset |
| Hardcoded `/root/autodl-tmp/` paths in preprocessing scripts | Config-driven paths via `--data_dir` CLI arg | Phase 1 (completed) | Scripts already updated; no further path work needed |
| `ge` library from local server path | `pip install git+https://github.com/shenweichen/GraphEmbedding.git` | Phase 2 (this phase) | First-time setup requires internet; then fully reproducible |
| Chinese A-share universe (~300 stocks) | S&P500 universe (~480-505 active constituents) | Phase 2 (this phase) | N dimension changes; config and array shapes must reflect actual count |

**Deprecated/outdated:**
- Qlib: Originally used for Chinese market data; removed entirely; do not add as dependency
- `data_processing_script/volume_and_price_factor_construction/` notebooks 1-5: Chinese market pipeline; replaced by new S&P500 scripts; leave in place but do not extend

---

## Open Questions

1. **Exact date range for S&P500 training data**
   - What we know: Requirements say "S&P500 constituents" with no specific date range; original Chinese model used 2021-06-04 to 2024-01-30 (2.5 years); thesis needs enough data for meaningful evaluation
   - What's unclear: Whether to use a similar 2.5-3 year window (e.g., 2020-01-01 to 2024-01-01) or a longer window back to 2015-2018
   - Recommendation: Use 2018-01-01 to 2024-01-01 (6 years) — provides ~1500 trading days; train/val/test split gives ~1125/188/187 days; sufficient for Stockformer's 20-day lookback window

2. **Number of S&P500 stocks to include after quality filtering**
   - What we know: Wikipedia lists ~503 tickers; some have short history, some have name/symbol changes; after filtering for data completeness, expect ~450-490
   - What's unclear: Whether a minimum history threshold (e.g., 3 years) should be applied
   - Recommendation: Require full coverage for the entire date range; drop stocks with >5% missing days; document final universe size in a `tickers.txt` file

3. **Struc2Vec embed_size alignment with model**
   - What we know: Existing script uses `model.train(embed_size=128)` and saves `128_corr_struc2vec_adjgat.npy` with shape [N, 128]; `graph_utils.py` `loadGraph()` reads this file; model `param.dims = 128` in config
   - What's unclear: Whether N (number of stocks) matters for the GAT adjacency loading — if N changes from Chinese universe (~300) to S&P500 (~480), the model's GAT layer expects N nodes
   - Recommendation: The embedding shape [N, 128] should match the actual filtered stock count; confirm GAT layer input dimension in `Multitask_Stockformer_models.py` before running training (Phase 3 concern)

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest >= 7.0 |
| Config file | `pytest.ini` (does not exist yet — create in Wave 0) |
| Quick run command | `pytest tests/test_phase2_pipeline.py -x -q` |
| Full suite command | `pytest tests/ -q` |

### Phase Requirements -> Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DATA-01 | Download produces Parquet files with OHLCV columns and no fully-empty rows | unit (mock yfinance) | `pytest tests/test_phase2_pipeline.py::test_parquet_ohlcv_schema -x` | Wave 0 |
| DATA-02 | Feature matrix contains RSI, MACD, Bollinger Bands, ROC, volume ratio columns for all windows | unit | `pytest tests/test_phase2_pipeline.py::test_feature_columns_present -x` | Wave 0 |
| DATA-03 | Cross-sectional z-score: per-date row has mean~0, std~1 across stocks | unit | `pytest tests/test_phase2_pipeline.py::test_cross_sectional_normalization -x` | Wave 0 |
| DATA-04 | Training stats computed before val/test; val/test normalization does not see future data | unit | `pytest tests/test_phase2_pipeline.py::test_no_normalization_leakage -x` | Wave 0 |
| DATA-05 | flow.npz and trend_indicator.npz have correct shape [T, N] and no NaN values | unit | `pytest tests/test_phase2_pipeline.py::test_npz_shapes_no_nan -x` | Wave 0 |
| DATA-05 | Struc2Vec embedding file exists and has shape [N, 128] | smoke | `pytest tests/test_phase2_pipeline.py::test_graph_embedding_shape -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_phase2_pipeline.py -x -q`
- **Per wave merge:** `pytest tests/ -q`
- **Phase gate:** Full suite green before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_phase2_pipeline.py` — covers DATA-01 through DATA-05
- [ ] `pytest.ini` — standardize test discovery settings
- [ ] `tests/conftest.py` — add fixtures for synthetic OHLCV DataFrames (already exists but needs Phase 2 fixtures)

---

## Sources

### Primary (HIGH confidence)
- Direct codebase analysis: `lib/Multitask_Stockformer_utils.py` lines 104-184 — StockDataset array contract
- Direct codebase analysis: `data_processing_script/stockformer_input_data_processing/Stockformer_data_preprocessing_script.py` — existing flow.npz/trend_indicator.npz/Struc2Vec pipeline
- Direct codebase analysis: `config/Multitask_Stock.conf` — config key names and data split ratios
- Direct codebase analysis: `requirements.txt` — installed package versions

### Secondary (MEDIUM confidence)
- [yfinance PyPI](https://pypi.org/project/yfinance/) — version 1.2.0 (Feb 2026), confirmed operational
- [GitHub shenweichen/GraphEmbedding](https://github.com/shenweichen/GraphEmbedding) — Struc2Vec source, install via setup.py from GitHub clone
- [pandas-ta GitHub](https://github.com/Data-Analisis/Technical-Analysis-Indicators---Pandas) — indicator API patterns
- [WebSearch: yfinance rate limiting 2025](https://github.com/ranaroussi/yfinance/issues/2614) — chunk size ~80, retry pattern confirmed by community
- [WebSearch: cross-sectional normalization no leakage](https://towardsdatascience.com/avoiding-data-leakage-in-timeseries-101-25ea13fcb15f/) — split-first, normalize-second pattern

### Tertiary (LOW confidence)
- WebSearch: dgllge PyPI package — page failed to load; do not use; fall back to GitHub GraphEmbedding install
- WebSearch: yfinance historical data access changes — some reports of premium restriction; current version 1.2.0 may have resolved; verify during implementation

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — packages verified against installed venv and PyPI
- Architecture: HIGH — derived directly from StockDataset codebase analysis
- Feature engineering patterns: MEDIUM — pandas-ta API not verified via Context7 (library not indexed)
- Pitfalls: MEDIUM-HIGH — rate limiting and leakage verified by multiple WebSearch sources; Struc2Vec scaling from codebase CONCERNS.md (HIGH)
- Struc2Vec install method: MEDIUM — GitHub setup.py install; not pip-verified

**Research date:** 2026-03-10
**Valid until:** 2026-04-10 (yfinance API status may change; pandas-ta API is stable)
