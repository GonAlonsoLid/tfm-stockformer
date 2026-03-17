# Phase 8: Alpha360 Feature Replacement - Research

**Researched:** 2026-03-17
**Domain:** Feature engineering / pandas data transformation / CSV pipeline
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Feature Definition (LOCKED)**
- 6 fields × 60 lags = 360 features per stock per day
- Column naming convention:
  - `CLOSE_d{d}` = CLOSE[t] / CLOSE[t-d]
  - `OPEN_d{d}`  = OPEN[t] / CLOSE[t-d]
  - `HIGH_d{d}`  = HIGH[t] / CLOSE[t-d]
  - `LOW_d{d}`   = LOW[t] / CLOSE[t-d]
  - `VWAP_d{d}`  = VWAP[t] / CLOSE[t-d]
  - `VOL_d{d}`   = VOLUME[t] / VOLUME[t-d]
  - Where d = 1, 2, 3, ..., 60
- VWAP approximation: `(HIGH + LOW + CLOSE) / 3`

**Input Source (LOCKED)**
- Read existing OHLCV Parquet files from `data/Stock_SP500_2018-01-01_2024-01-01/ohlcv/`
- No network dependency — Parquet files already downloaded
- Guarantees calendar alignment and ticker count N=478

**Output Format (LOCKED)**
- Write 360 CSV files to `data/Stock_SP500_2018-01-01_2024-01-01/features/`
- Each CSV shape: `[T_features × 478]` (rows=trading days, cols=tickers)
- First row date must be `2018-03-29` (same as existing feature CSVs)
- Column order: tickers in the same order as `tickers.txt`
- Values: float, z-score normalized, no NaN or inf

**Processing Pipeline (LOCKED)**
1. Load all OHLCV Parquets into a single aligned DataFrame (rows=trading days, cols=tickers)
2. VOLUME zero denominator: replace zero denominator values with `NaN` BEFORE dividing
3. For each field and each lag d (1..60): compute ratio across all stocks
4. Cross-sectional z-score per day: `(x - mean(x)) / std(x)` across all stocks
5. Replace `NaN` and `inf` with `0.0`
6. Slice output: drop first 60 rows (lag buffer), start from `2018-03-29`
7. Back up existing `features/` contents, then write 360 new CSVs

**Script Interface (LOCKED)**
- New file: `scripts/build_alpha360.py`
- Invocation: `python scripts/build_alpha360.py --config config/Multitask_Stock_SP500.conf`
- Config provides paths to data directory

**Downstream Compatibility (LOCKED — must NOT change)**
- `infea = len(os.listdir(alpha_360_dir)) + 2` auto-detects new feature count (71 → 362)
- The `+2` is a pre-existing convention in `StockDataset` — **do not change it**
- No changes to: model architecture, training script, config, backtest script,
  `flow.npz`, `trend_indicator.npz`, graph files

**Validation Sequence (LOCKED)**
1. Run `python scripts/build_alpha360.py --config config/Multitask_Stock_SP500.conf`
2. Verify: feature count == 360, shape == (T_features, 478), NaN count == 0
3. Retrain on Kaggle (out of scope for this phase — next step after verification)

### Claude's Discretion
- Logging verbosity and progress reporting during build (tqdm, print, logging module)
- Backup directory naming convention (e.g., `features_backup_YYYYMMDD/`)
- Internal code structure and helper functions within `build_alpha360.py`
- Error handling for missing Parquet files or malformed data
- Whether to process features in field-major or lag-major order (both produce same result)

### Deferred Ideas (OUT OF SCOPE)
- Model retraining (done on Kaggle after this phase completes)
- IC measurement and benchmark comparison (post-retraining step)
- Additional feature engineering blocks (Block 2, 3 etc. — Block 1 only)
- Integration with live inference pipeline
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| ALPHA360-01 | Build `scripts/build_alpha360.py` that reads OHLCV Parquets and writes 360 CSV files | Verified: Parquet schema, date range, ticker alignment |
| ALPHA360-02 | 360 feature CSVs pass validation: count=360, shape=(1449, 478), NaN=0 | Verified: existing CSVs are (1449, 478) starting 2018-03-29 |
| ALPHA360-03 | First row of each CSV is dated `2018-03-29` (lag buffer = 60 rows exactly) | Verified: index[60] of OHLCV = 2018-03-29 |
| ALPHA360-04 | Backup existing 69-feature `features/` directory before overwriting | Confirmed: 69 CSVs exist currently |
| ALPHA360-05 | Downstream `infea` auto-detects to 362 with no code changes | Verified: `infea = len(os.listdir(path)) + 2` in `lib/Multitask_Stockformer_utils.py` line 147 |
</phase_requirements>

---

## Summary

Phase 8 is a focused data transformation task: build one new Python script (`scripts/build_alpha360.py`) that reads the existing OHLCV Parquet files and writes 360 new feature CSVs, replacing the current 69 TA feature CSVs. No model code, config, or any other pipeline artifact changes.

The core technical work is a nested loop over 6 OHLCV fields and 60 lag values, computing price/volume ratios across all 478 tickers simultaneously using pandas `.shift(d)`, then applying cross-sectional z-score normalization per trading day. The most subtle correctness requirement is zero-safe division for the VOLUME ratio: `df_volume.shift(d).replace(0, np.nan)` before dividing to prevent silent `inf` values that `fillna` would leave in place.

All data contracts are verified against the live filesystem: 478 Parquet files covering 2018-01-02 to 2023-12-29 (1509 trading days), 69 existing CSVs of shape (1449, 478), first row date confirmed as 2018-03-29 (exactly index 60 of the OHLCV series). The downstream `infea` computation is confirmed in `lib/Multitask_Stockformer_utils.py` line 147 as `bonus_all.shape[-1] + 2`.

**Primary recommendation:** Implement `build_alpha360.py` as a single-file script that loads all Parquets into wide DataFrames, iterates 6 × 60 = 360 feature computations with tqdm progress, and writes CSVs in a single pass after backing up the existing `features/` directory.

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas | >=1.3 (project uses 2.x) | Wide-format DataFrame operations, `.shift()`, z-score | Already in requirements.txt; native for row-wise ops |
| numpy | >=1.21 | `np.nan`, `np.isinf`, `fillna` complement | Already in requirements.txt |
| configparser | stdlib | Parse `.conf` file to get `alpha_360_dir` path | Same pattern used by training script |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tqdm | any | Progress bar over 360 feature iterations | Claude's discretion — recommended for UX |
| shutil | stdlib | Backup `features/` directory (`shutil.copytree`) | Backup step |
| os / pathlib | stdlib | Directory creation, file listing | Throughout |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pandas wide DataFrame per field | polars | pandas already installed and sufficient for this one-time batch job |
| `df.shift(d)` per field matrix | explicit loop over tickers | pandas vectorized is correct and faster; no reason to loop per ticker |
| configparser | argparse only | Config already read this way elsewhere in the project; stays consistent |

**Installation:** No new packages required — everything is already in `requirements.txt`.

---

## Architecture Patterns

### Recommended Project Structure

```
scripts/
└── build_alpha360.py    # New script — all logic self-contained
```

No new modules, no new directories, no changes to existing files.

### Pattern 1: Load All OHLCV Into Wide DataFrames

**What:** Load each per-ticker Parquet into a wide DataFrame (rows=dates, cols=tickers) for each OHLCV field. One wide DataFrame per field (OPEN, HIGH, LOW, CLOSE, VOLUME).

**When to use:** Required — the cross-sectional z-score operates across all stocks on the same day, so a row-per-day, col-per-ticker layout is mandatory.

**Example:**

```python
# Source: verified against live data at data/Stock_SP500_2018-01-01_2024-01-01/ohlcv/
import pandas as pd
import os

def load_wide(ohlcv_dir: str, tickers: list[str], field: str) -> pd.DataFrame:
    """Returns DataFrame[dates x tickers] for one OHLCV field."""
    frames = {}
    for ticker in tickers:
        path = os.path.join(ohlcv_dir, f"{ticker}.parquet")
        df = pd.read_parquet(path)
        frames[ticker] = df[field]
    return pd.DataFrame(frames)[tickers]  # enforce ticker column order
```

### Pattern 2: Zero-Safe Ratio Computation

**What:** Replace zero denominator values with `NaN` before dividing to prevent silent `inf` in VOLUME ratio.

**When to use:** Mandatory for `VOL_d{d}` features. Best practice to apply for all fields for defensive coding.

**Example:**

```python
# Zero-safe shift and divide
denominator = df.shift(d)
denominator = denominator.replace(0, float('nan'))  # or: denominator[denominator == 0] = float('nan')
ratio = df / denominator
```

### Pattern 3: Cross-Sectional Z-Score Per Row

**What:** Normalize each row (trading day) by subtracting row mean and dividing by row standard deviation across all tickers.

**When to use:** After computing each raw ratio, before writing to CSV.

**Example:**

```python
# Cross-sectional z-score (axis=1 = across tickers per day)
mean = ratio.mean(axis=1)
std = ratio.std(axis=1)
z_scored = ratio.sub(mean, axis=0).div(std, axis=0)
# Replace NaN and inf with 0.0 (cross-sectional mean after z-scoring)
z_scored = z_scored.replace([float('inf'), float('-inf')], float('nan'))
z_scored = z_scored.fillna(0.0)
```

### Pattern 4: Config Parsing (Consistent With Project)

**What:** Read `alpha_360_dir` and infer `ohlcv_dir` from the `.conf` file using `configparser`.

**When to use:** The script accepts `--config` argument matching all other scripts in `scripts/`.

**Example:**

```python
import configparser, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

cfg = configparser.ConfigParser()
cfg.read(args.config)
alpha_360_dir = cfg["file"]["alpha_360_dir"]
# ohlcv_dir is alpha_360_dir/../ohlcv/
```

### Anti-Patterns to Avoid

- **Do not use `fillna(0)` before `replace(0, nan)`:** `fillna` does not touch `inf` values produced by zero division. The zero-denominator must be replaced with `nan` before the division, not after.
- **Do not normalize across time (time-series z-score):** The spec requires cross-sectional z-score (across stocks, same day). Using `axis=0` instead of `axis=1` would be a look-ahead leak.
- **Do not hard-code paths:** Use the config file for `alpha_360_dir`; derive `ohlcv_dir` from it.
- **Do not `os.listdir()` for ticker order:** Use `tickers.txt` to control column order. `os.listdir()` is OS-dependent and may return files in arbitrary order, which would break downstream alignment.
- **Do not change `infea` or the `+2` constant:** `lib/Multitask_Stockformer_utils.py` line 147 computes `infea = bonus_all.shape[-1] + 2`. This is auto-detected from file count. Changing it would require retraining from a different initial state.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cross-sectional z-score | Custom loop over rows | `df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)` | pandas row-wise broadcasting; no loop needed |
| Wide DataFrame assembly | Manual dict accumulation with alignment | `pd.DataFrame({ticker: series for ...})[ordered_tickers]` | Single-line construction with guaranteed column order |
| Directory backup | Manual file-by-file copy | `shutil.copytree(src, dst)` | Atomic, preserves metadata |
| Config parsing | Custom regex | `configparser.ConfigParser()` | Already the project pattern |

**Key insight:** The ratio computation and z-scoring are trivially vectorized with pandas. The entire 360-feature build is effectively 6 nested loops over 60 lags — each iteration is 3 pandas operations (shift, divide, z-score). No custom numerical code is needed.

---

## Common Pitfalls

### Pitfall 1: Zero Denominator in VOLUME Ratio Produces Silent `inf`

**What goes wrong:** `df_vol / df_vol.shift(d)` where `df_vol == 0` on the denominator produces `inf`, not `NaN`. Subsequent `fillna(0.0)` ignores `inf` values, leaving them in the output CSV. The validation check `df.isna().sum().sum() == 0` passes even though `inf` values are present.

**Why it happens:** In pandas/numpy, `x / 0.0 = inf` (IEEE 754). `fillna` only replaces `NaN`, not `inf`.

**How to avoid:** Replace zero denominator with `nan` before dividing:
```python
denom = df_vol.shift(d)
denom = denom.replace(0, float('nan'))
ratio = df_vol / denom
```

**Warning signs:** `np.isinf(df.values).any()` returns True after supposedly cleaning.

### Pitfall 2: Ticker Column Order Mismatch

**What goes wrong:** `os.listdir(ohlcv_dir)` returns tickers in arbitrary filesystem order (varies by OS/filesystem). If wide DataFrames are assembled without enforcing the `tickers.txt` order, the feature CSVs will have columns in a different order than the existing CSVs, breaking downstream alignment with `flow.npz`, `trend_indicator.npz`, graph files, and `label.csv`.

**Why it happens:** `os.listdir` is not sorted on all platforms. The existing CSVs have columns in `tickers.txt` order.

**How to avoid:** Read `tickers.txt` first; use that list as the canonical column order when assembling wide DataFrames and writing CSV columns.

**Warning signs:** Mismatch between feature CSV column header and existing `label.csv` column header.

### Pitfall 3: Lag Slice Off-by-One Produces Wrong Start Date

**What goes wrong:** After computing ratios with lags 1..60, the first 60 rows are `NaN`-only (no valid lag-60 value until row 60). If the slice drops only 59 rows or drops 61 rows, the first date of the output CSVs differs from `2018-03-29`.

**Why it happens:** Python slicing and `.shift(d)` semantics: `shift(60)` makes rows 0..59 NaN; row 60 (index 0-based) is the first valid row.

**How to avoid:** Verify that `df.iloc[60:].index[0] == '2018-03-29'` before writing any CSVs. The OHLCV data starts at 2018-01-02; index 60 of the date range is confirmed as 2018-03-29 (verified live).

**Warning signs:** `df.index[0]` of a written CSV is not `2018-03-29`.

### Pitfall 4: `os.listdir` Count Includes `.DS_Store` on macOS

**What goes wrong:** When running on macOS, `os.listdir(features_dir)` may include `.DS_Store`, making the count 361 instead of 360. The downstream `infea` computation (`len(os.listdir(alpha_360_dir)) + 2`) would become 363 instead of 362.

**Why it happens:** macOS Finder creates `.DS_Store` files invisibly.

**How to avoid:** When writing feature CSVs, write exactly 360 `.csv` files. The validation step and `StockDataset` both call `os.listdir()` — if `.DS_Store` exists the count is wrong. Either filter in `StockDataset` (don't touch it — LOCKED) or ensure the backup step moves all non-CSV files out, or document that validation should be run on Linux/Kaggle. The training environment is Kaggle (Linux), so `.DS_Store` is not present there.

**Warning signs:** Local validation shows 361 files; `infea` differs locally vs on Kaggle.

### Pitfall 5: z-score with Zero Standard Deviation

**What goes wrong:** On a given day, if all 478 stocks have the same ratio value (e.g., all CLOSE[t]/CLOSE[t-1] = 0.0 due to data issues), `std = 0` and the z-score is `0/0 = NaN`. These `NaN` values survive into the CSV unless caught explicitly.

**Why it happens:** Degenerate days with no cross-sectional variation.

**How to avoid:** The `fillna(0.0)` step after `replace(inf, nan)` already handles this — division by zero std produces `NaN`, which `fillna(0.0)` converts to 0.0. No extra handling needed as long as `fillna` is applied after the z-score step.

---

## Code Examples

Verified patterns from live codebase inspection:

### Full Pipeline Skeleton

```python
# Source: verified against live data contracts and lib/Multitask_Stockformer_utils.py
import argparse
import configparser
import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime

FIELDS = ["Close", "Open", "High", "Low"]  # price fields (all normalized by Close[t-d])
LAGS = range(1, 61)  # d = 1..60

def load_tickers(data_dir: str) -> list[str]:
    with open(os.path.join(data_dir, "tickers.txt")) as f:
        return [line.strip() for line in f if line.strip()]

def load_wide(ohlcv_dir: str, tickers: list[str], field: str) -> pd.DataFrame:
    frames = {t: pd.read_parquet(os.path.join(ohlcv_dir, f"{t}.parquet"))[field]
              for t in tickers}
    return pd.DataFrame(frames)[tickers]

def zscore_rows(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    return df.sub(mean, axis=0).div(std, axis=0)

def safe_ratio(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    denom = denominator.replace(0, float("nan"))
    ratio = numerator / denom
    ratio = ratio.replace([float("inf"), float("-inf")], float("nan"))
    z = zscore_rows(ratio)
    return z.fillna(0.0)
```

### Existing CSV Format Reference

```
# Verified: data/Stock_SP500_2018-01-01_2024-01-01/features/ATR_10.csv
# Shape: (1449, 478)
# Index: Date column, starting 2018-03-29
# Columns: tickers in tickers.txt order
Date,MMM,AOS,ABT,...
2018-03-29,0.2426,...
2018-04-02,...
```

### `infea` Auto-Detection in StockDataset

```python
# Source: lib/Multitask_Stockformer_utils.py lines 110-147
path = args.alpha_360_dir
files = os.listdir(path)
data_list = []
for file in files:
    file_path = os.path.join(path, file)
    df = pd.read_csv(file_path, index_col=0)
    arr = np.expand_dims(df.values, axis=2)
    data_list.append(arr)
bonus_all = np.concatenate(data_list, axis=2)
...
self.infea = bonus_all.shape[-1] + 2  # 360 files -> infea = 362
```

### Validation Snippet

```python
# Source: docs/superpowers/specs/2026-03-17-alpha360-features-design.md
import os, pandas as pd, numpy as np
feat_dir = "data/Stock_SP500_2018-01-01_2024-01-01/features"
files = [f for f in os.listdir(feat_dir) if f.endswith(".csv")]
print(f"Feature count: {len(files)}")          # expect 360
df = pd.read_csv(os.path.join(feat_dir, files[0]), index_col=0)
print(f"Shape: {df.shape}")                    # expect (1449, 478)
print(f"First date: {df.index[0]}")            # expect 2018-03-29
print(f"NaN count: {df.isna().sum().sum()}")   # expect 0
print(f"Inf count: {np.isinf(df.values).sum()}")  # expect 0
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| 69 TA features (RSI, MACD, BB, ROC, VOL ratios) | 360 Alpha360-style price/volume ratios | Phase 8 | IC expected to improve from -0.003 to >0.01 |
| `infea=71` (auto-detected) | `infea=362` (auto-detected, no code change) | Phase 8 | First model layer input dimension changes; retraining required |

**Deprecated/outdated after this phase:**
- All 69 existing TA feature CSVs: replaced by the 360 new CSVs. They are backed up, not deleted.
- Old checkpoint at `cpt/STOCK/saved_model_Multitask_SP500_2018-2024`: incompatible with new `infea=362`; cannot be used for inference.

---

## Open Questions

1. **VWAP field naming in the spec says `VWAP_d{d}` but VWAP is derived, not directly in Parquet**
   - What we know: Parquet columns are `['Open', 'High', 'Low', 'Close', 'Volume']` (confirmed live). VWAP = `(High + Low + Close) / 3`.
   - What's unclear: Nothing — the CONTEXT.md explicitly specifies this approximation.
   - Recommendation: Compute VWAP inline as `(df_high + df_low + df_close) / 3` during the feature loop.

2. **macOS `.DS_Store` in features/ on local development machine**
   - What we know: Training and final validation runs on Kaggle (Linux). Local machine is macOS.
   - What's unclear: Whether the developer validates locally before pushing to Kaggle.
   - Recommendation: Validation script should filter to `*.csv` only: `[f for f in os.listdir(...) if f.endswith('.csv')]`. Document this difference in script comments.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | pytest (project standard, confirmed in tests/) |
| Config file | `pytest.ini` or `pyproject.toml` — check project root |
| Quick run command | `pytest tests/test_build_alpha360.py -x` |
| Full suite command | `pytest tests/ -x` |

### Phase Requirements to Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ALPHA360-01 | Script runs without error, produces 360 CSVs | integration smoke | `pytest tests/test_build_alpha360.py::test_build_alpha360_creates_360_csvs -x` | Wave 0 |
| ALPHA360-02 | Each CSV has shape (T_features, 478), zero NaN, zero inf | unit | `pytest tests/test_build_alpha360.py::test_feature_csv_shape_and_no_nan -x` | Wave 0 |
| ALPHA360-03 | First row date is 2018-03-29, confirms lag buffer = 60 | unit | `pytest tests/test_build_alpha360.py::test_first_row_date -x` | Wave 0 |
| ALPHA360-04 | Backup directory created with 69 original CSVs | unit | `pytest tests/test_build_alpha360.py::test_backup_created -x` | Wave 0 |
| ALPHA360-05 | Column order matches tickers.txt | unit | `pytest tests/test_build_alpha360.py::test_column_order_matches_tickers -x` | Wave 0 |

### Sampling Rate
- **Per task commit:** `pytest tests/test_build_alpha360.py -x`
- **Per wave merge:** `pytest tests/ -x`
- **Phase gate:** Full suite green + manual validation snippet (`feature count=360, shape=(1449,478), NaN=0`) before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `tests/test_build_alpha360.py` — covers ALPHA360-01 through ALPHA360-05; uses `tmp_path` fixtures with synthetic Parquet data
- [ ] The test file does not yet exist — must be created in Wave 0 before implementation

*(No framework install needed — pytest already present in project test suite.)*

---

## Key Data Contracts (Verified Live)

| Artifact | Location | Verified Value |
|----------|----------|---------------|
| OHLCV Parquet count | `ohlcv/*.parquet` | 478 files |
| OHLCV date range | AAPL.parquet index | 2018-01-02 to 2023-12-29 (1509 rows) |
| OHLCV column names | Parquet columns | `['Open', 'High', 'Low', 'Close', 'Volume']` |
| Existing feature CSV shape | `features/ATR_10.csv` | (1449, 478) |
| Existing feature CSV first date | `features/ATR_10.csv` index[0] | 2018-03-29 |
| Lag buffer confirmation | OHLCV index[60] | 2018-03-29 (exact match) |
| Ticker count in tickers.txt | `tickers.txt` word count | 478 |
| `infea` formula | `lib/Multitask_Stockformer_utils.py:147` | `bonus_all.shape[-1] + 2` |
| Existing feature CSV count | `features/` listdir | 69 |

---

## Sources

### Primary (HIGH confidence)
- Live filesystem inspection — OHLCV Parquet schema, date range, shape confirmed by running `python3 -c` on actual data
- `lib/Multitask_Stockformer_utils.py` — `infea` formula at line 147 read directly
- `data/Stock_SP500_2018-01-01_2024-01-01/features/ATR_10.csv` — existing CSV format confirmed
- `docs/superpowers/specs/2026-03-17-alpha360-features-design.md` — approved design spec
- `.planning/phases/08-alpha360-feature-replacement/08-CONTEXT.md` — locked decisions

### Secondary (MEDIUM confidence)
- Qlib Alpha360 documentation (design inspiration): the feature formula matches the standard Qlib Alpha360 approach; not directly verified against Qlib source but consistent with published research conventions
- pandas `.shift()` + zero-division behavior: well-established pandas/numpy IEEE 754 behavior; no version risk for project pandas 2.x

### Tertiary (LOW confidence)
- IC improvement expectation (>0.01 after retraining): based on design spec motivation section; actual IC improvement depends on model retraining (out of scope for this phase and not verifiable here)

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already installed; no new dependencies
- Architecture: HIGH — verified against live data contracts, StockDataset source, and existing script patterns
- Pitfalls: HIGH — zero-division behavior verified against pandas/numpy spec; ticker ordering verified against existing CSV column headers; lag buffer confirmed computationally

**Research date:** 2026-03-17
**Valid until:** 2026-04-17 (stable domain — pandas/numpy APIs, project data files do not change)
