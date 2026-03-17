# Design: Alpha360 Feature Replacement

**Date:** 2026-03-17
**Status:** Approved
**Scope:** Block 1 only — replace TA features with real Alpha360 price-ratio features

---

## Problem

Current IC = -0.003 (essentially zero). The model has no cross-sectional predictive power.
The 69 custom TA features (RSI, MACD, Bollinger Bands, etc.) describe each stock's
individual time-series history but carry little signal for ranking one stock against another.

## Solution

Replace the 69 TA feature CSVs with 360 Alpha360-style features: normalized price and
volume ratios at 60 lag intervals. These are designed for cross-sectional stock ranking
and are the standard feature set used in quantitative equity research.

---

## Feature Definition

6 fields × 60 lags = 360 features per stock per day.

| Field | Formula | CSV prefix |
|-------|---------|-----------|
| CLOSE ratio | CLOSE[t] / CLOSE[t-d] | `CLOSE_d{d}` |
| OPEN ratio | OPEN[t] / CLOSE[t-d] | `OPEN_d{d}` |
| HIGH ratio | HIGH[t] / CLOSE[t-d] | `HIGH_d{d}` |
| LOW ratio | LOW[t] / CLOSE[t-d] | `LOW_d{d}` |
| VWAP ratio | VWAP[t] / CLOSE[t-d] | `VWAP_d{d}` |
| VOLUME ratio | VOLUME[t] / VOLUME[t-d] | `VOL_d{d}` |

Where d = 1, 2, 3, ..., 60.

VWAP is approximated as (HIGH + LOW + CLOSE) / 3 from daily OHLC data.

---

## Implementation

### New script: `scripts/build_alpha360.py`

**Input:** Existing OHLCV Parquet files in
`data/Stock_SP500_2018-01-01_2024-01-01/ohlcv/`
(already downloaded; reading from Parquet avoids network dependency and guarantees
calendar alignment and ticker count N=478 consistent with the rest of the pipeline).

**Output:** 360 CSV files in
`data/Stock_SP500_2018-01-01_2024-01-01/features/`
each shaped `[T_features × 478]` where `T_features` is the number of trading days
after consuming the 60-row lag buffer — the first row of each CSV must match the
first row date of the existing feature CSVs (currently `2018-03-29`).

**No other file changes:** The model uses
`infea = len(os.listdir(alpha_360_dir)) + 2`
which auto-detects the new feature count (infea: 71 → 362) without any code changes.
Note: the `+2` is a pre-existing convention in `StockDataset` that must be preserved
exactly as-is; changing it would alter the model's first layer input dimension.

### Processing steps

1. Load all OHLCV Parquet files from `ohlcv/` directory into a single aligned DataFrame
   (rows = trading days, columns = tickers); the Parquet files already cover a pre-2018
   buffer period sufficient for 60-lag computation
2. For VOLUME ratio: replace zero denominator values with NaN before dividing to avoid
   `inf` results (division by zero yields `inf`, not `NaN`, so `fillna` alone is insufficient)
3. For each field and each lag d (1..60): compute ratio across all stocks
4. Cross-sectionally z-score normalize per day: `(x - mean(x)) / std(x)` across all stocks.
   This normalization uses only same-day cross-sectional values, so no look-ahead or
   train/test contamination occurs regardless of when in the pipeline it is applied.
5. Replace `NaN` and `inf` with 0.0 (the cross-sectional mean after z-scoring)
6. Slice the output to start from the same first date as the existing feature CSVs
   (i.e., drop the first 60 rows used as the lag buffer)
7. Back up the existing `features/` contents, then write 360 new CSVs

### Output format

Each CSV must match the format expected by `StockDataset.bonus_seq2instance`:
- Rows: trading days (same date range as existing feature CSVs, starting `2018-03-29`)
- Columns: tickers in the same order as `tickers.txt`
- Values: float, z-score normalized, no NaN or inf

---

## Validation

After regenerating features, verify the files and then retrain before measuring IC.
Running inference with the old checkpoint is not valid — the checkpoint was trained
with `infea=71`; a model instantiated with the new `infea=362` will fail to load it.

```bash
# Step 1: Regenerate features
python scripts/build_alpha360.py --config config/Multitask_Stock_SP500.conf

# Step 2: Verify output (file count, shape, no NaN)
python -c "
import os, pandas as pd, numpy as np
feat_dir = 'data/Stock_SP500_2018-01-01_2024-01-01/features'
files = os.listdir(feat_dir)
print(f'Feature count: {len(files)}')  # expect 360
df = pd.read_csv(os.path.join(feat_dir, files[0]), index_col=0)
print(f'Shape: {df.shape}')            # expect (T_features, 478)
print(f'NaN count: {df.isna().sum().sum()}')  # expect 0
"

# Step 3: Retrain from scratch on Kaggle with new features
# python MultiTask_Stockformer_train.py --config config/Multitask_Stock_SP500.conf

# Step 4: Run inference and measure IC (after retraining)
# python scripts/run_inference.py --config config/Multitask_Stock_SP500.conf
# python scripts/compute_ic.py --output_dir output/Multitask_output_SP500_2018-2024
# Expected: IC mean > 0.01 on test set
```

---

## What does NOT change

- Model architecture (`Stockformermodel/`)
- Training script (`MultiTask_Stockformer_train.py`)
- Config file (`config/Multitask_Stock_SP500.conf`)
- Backtest script (`scripts/run_backtest.py`)
- `flow.npz`, `trend_indicator.npz`, graph files

---

## Success Criteria

1. `features/` directory contains exactly 360 CSV files after running `build_alpha360.py`
2. Each CSV has the same number of rows and columns as the existing feature CSVs
   (i.e., first row date = `2018-03-29`, columns = 478 tickers)
3. No NaN or inf values in any feature CSV
4. After full retraining from random initialization on Kaggle, IC mean > 0.01 measured
   on the test set via `scripts/compute_ic.py` (vs current IC = -0.003)
