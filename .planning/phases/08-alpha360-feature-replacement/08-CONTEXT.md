# Phase 8: Alpha360 Feature Replacement - Context

**Gathered:** 2026-03-17
**Status:** Ready for planning
**Source:** PRD Express Path (docs/superpowers/specs/2026-03-17-alpha360-features-design.md)

<domain>
## Phase Boundary

Build `scripts/build_alpha360.py` to replace the existing 69 TA feature CSVs with 360
Alpha360-style price-ratio features. The script reads existing OHLCV Parquet files and
writes 360 new CSVs to `data/Stock_SP500_2018-01-01_2024-01-01/features/`.

**This phase does NOT retrain the model** — retraining is a separate step done on Kaggle
after the features are verified. The phase is complete when 360 valid CSVs exist in
`features/` and pass the validation check (count, shape, no NaN/inf).

**Motivation:** Current IC = -0.003 (essentially zero). The 69 TA features have near-zero
cross-sectional predictive power. Alpha360-style features (price/volume ratios across lags)
are the standard feature set for cross-sectional stock ranking.

</domain>

<decisions>
## Implementation Decisions

### Feature Definition (LOCKED)
- 6 fields × 60 lags = **360 features** per stock per day
- Column naming convention:
  - `CLOSE_d{d}` = CLOSE[t] / CLOSE[t-d]
  - `OPEN_d{d}`  = OPEN[t] / CLOSE[t-d]
  - `HIGH_d{d}`  = HIGH[t] / CLOSE[t-d]
  - `LOW_d{d}`   = LOW[t] / CLOSE[t-d]
  - `VWAP_d{d}`  = VWAP[t] / CLOSE[t-d]
  - `VOL_d{d}`   = VOLUME[t] / VOLUME[t-d]
  - Where d = 1, 2, 3, ..., 60
- VWAP approximation: `(HIGH + LOW + CLOSE) / 3`

### Input Source (LOCKED)
- Read existing OHLCV Parquet files from `data/Stock_SP500_2018-01-01_2024-01-01/ohlcv/`
- No network dependency — Parquet files already downloaded
- Guarantees calendar alignment and ticker count N=478

### Output Format (LOCKED)
- Write 360 CSV files to `data/Stock_SP500_2018-01-01_2024-01-01/features/`
- Each CSV shape: `[T_features × 478]` (rows=trading days, cols=tickers)
- First row date must be `2018-03-29` (same as existing feature CSVs)
- Column order: tickers in the same order as `tickers.txt`
- Values: float, z-score normalized, no NaN or inf

### Processing Pipeline (LOCKED)
1. Load all OHLCV Parquets into a single aligned DataFrame (rows=trading days, cols=tickers)
2. **VOLUME zero denominator**: replace zero denominator values with `NaN` BEFORE dividing
   (division by zero yields `inf`, not `NaN`; `fillna` alone is insufficient)
3. For each field and each lag d (1..60): compute ratio across all stocks
4. **Cross-sectional z-score per day**: `(x - mean(x)) / std(x)` across all stocks
   — uses only same-day values, no look-ahead contamination
5. Replace `NaN` and `inf` with `0.0` (the cross-sectional mean after z-scoring)
6. Slice output: drop first 60 rows (lag buffer), start from `2018-03-29`
7. Back up existing `features/` contents, then write 360 new CSVs

### Script Interface (LOCKED)
- New file: `scripts/build_alpha360.py`
- Invocation: `python scripts/build_alpha360.py --config config/Multitask_Stock_SP500.conf`
- Config provides paths to data directory

### Downstream Compatibility (LOCKED — must NOT change)
- `infea = len(os.listdir(alpha_360_dir)) + 2` auto-detects new feature count (71 → 362)
- The `+2` is a pre-existing convention in `StockDataset` — **do not change it**
- No changes to: model architecture, training script, config, backtest script,
  `flow.npz`, `trend_indicator.npz`, graph files

### Validation Sequence (LOCKED)
1. Run `python scripts/build_alpha360.py --config config/Multitask_Stock_SP500.conf`
2. Verify: feature count == 360, shape == (T_features, 478), NaN count == 0
3. Retrain on Kaggle (out of scope for this phase — next step after verification)
4. After retraining: IC mean > 0.01 on test set (measured via `scripts/compute_ic.py`)

**Note:** Running inference with the old checkpoint is INVALID after this phase.
The old checkpoint was trained with `infea=71`; new model requires `infea=362`.

### Claude's Discretion
- Logging verbosity and progress reporting during build (tqdm, print, logging module)
- Backup directory naming convention (e.g., `features_backup_YYYYMMDD/`)
- Internal code structure and helper functions within `build_alpha360.py`
- Error handling for missing Parquet files or malformed data
- Whether to process features in field-major or lag-major order (both produce same result)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Design Specification
- `docs/superpowers/specs/2026-03-17-alpha360-features-design.md` — Full approved design: feature formulas, processing pipeline, validation steps, success criteria

### Existing Data Contracts
- `data/Stock_SP500_2018-01-01_2024-01-01/ohlcv/` — Input Parquet files (source data)
- `data/Stock_SP500_2018-01-01_2024-01-01/features/` — Output directory (existing 69 CSVs to be replaced)
- `data/Stock_SP500_2018-01-01_2024-01-01/tickers.txt` — Ticker order for CSV columns

### Model Integration Point
- `StockDataset` in model code — reads `infea = len(os.listdir(alpha_360_dir)) + 2`; changing `+2` breaks the model

### Project Config
- `config/Multitask_Stock_SP500.conf` — Provides `alpha_360_dir` and data paths used by the pipeline

</canonical_refs>

<specifics>
## Specific Ideas

- The existing feature CSVs start at `2018-03-29` — this date boundary must be preserved exactly
- The OHLCV Parquet files already contain a pre-2018 buffer sufficient for 60-lag computation
- Ticker count N=478 must be preserved (consistent with graph files and other pipeline artifacts)
- The z-score normalization is cross-sectional (across stocks on the same day), NOT time-series normalization
- Volume zero handling is subtle: `df / df.shift(d)` where `df==0` on denominator produces `inf`, which `fillna(0)` ignores — must set `denominator[denominator==0] = np.nan` before dividing

</specifics>

<deferred>
## Deferred Ideas

- Model retraining (done on Kaggle after this phase completes)
- IC measurement and benchmark comparison (post-retraining step)
- Additional feature engineering blocks (Block 2, 3 etc. from original design discussion — out of scope, Block 1 only)
- Integration with live inference pipeline

</deferred>

---

*Phase: 08-alpha360-feature-replacement*
*Context gathered: 2026-03-17 via PRD Express Path*
