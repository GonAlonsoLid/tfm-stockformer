---
phase: 02-data-pipeline
verified: 2026-03-11T19:02:23Z
status: passed
score: 7/7 must-haves verified
re_verification: false
---

# Phase 2: Data Pipeline Verification Report

**Phase Goal:** Build a complete, reproducible data pipeline that downloads S&P 500 OHLCV data, engineers technical features, normalizes and splits data for training, serializes arrays into the format StockDataset expects, and computes graph embeddings for the model's relational input.
**Verified:** 2026-03-11T19:02:23Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | S&P 500 OHLCV download step exists and is importable with all required functions | VERIFIED | `download_ohlcv.py` — 259 lines; `get_sp500_tickers`, `download_ohlcv_batched`, `clean_and_align`, `main` all present and importable |
| 2 | Feature engineering produces >= 60 TA features per ticker with no all-NaN columns after warmup | VERIFIED | `compute_features()` returns 69 columns; `test_feature_count_for_phase3` passes; 0 all-NaN columns verified in live run |
| 3 | Cross-sectional normalization is applied to feature CSVs at save time (not post-hoc) | VERIFIED | `save_feature_csvs()` calls `_cross_sectional_normalize()` before `to_csv()`; `test_cross_sectional_normalization` reloads saved CSV and asserts row mean < 1e-8, row std within 0.05 of 1.0 |
| 4 | Train/val/test split is 75/12.5/12.5 by date with no normalization leakage | VERIFIED | `split_by_date()` verified by `test_split_ratios` (750/125/125 on 1000 rows); cross-sectional normalization proven row-independent by `test_no_normalization_leakage` |
| 5 | `flow.npz` and `trend_indicator.npz` have shape [T, N], no NaN, binary trend values | VERIFIED | `save_model_arrays()` uses `np.savez(..., result=data)`; `test_npz_shapes_no_nan` and `test_trend_indicator_binary` both pass |
| 6 | Graph embedding step exists with `build_correlation_graph()` / `run_struc2vec()` producing [N, 128] output | VERIFIED | `graph_embedding.py` — 148 lines; `build_correlation_graph` verified by two passing tests; `run_struc2vec` guards on `ge` import and saves `128_corr_struc2vec_adjgat.npy`; `test_graph_embedding_shape` passes |
| 7 | End-to-end orchestrator (`build_pipeline.py`) chains all 5 steps with idempotent skip logic | VERIFIED | `build_pipeline.py` — 130 lines; `--help` prints correctly; 5-step STEPS list with sentinel files; `subprocess.run(..., check=True)` per step |

**Score:** 7/7 truths verified

---

### Required Artifacts

| Artifact | Plan | Min Lines | Actual Lines | Status | Details |
|----------|------|-----------|--------------|--------|---------|
| `tests/conftest.py` | 02-01 | — | 59 | VERIFIED | `sp500_ohlcv_fixture`, `ohlcv_wide_fixture`, `feature_matrix_fixture` all defined |
| `tests/test_data_pipeline.py` | 02-01 | 80 | 264 | VERIFIED | 13 tests collected; 11 pass, 2 skip (pandas_ta not installed — expected) |
| `data_processing_script/sp500_pipeline/__init__.py` | 02-02 | — | exists | VERIFIED | Package marker present |
| `data_processing_script/sp500_pipeline/download_ohlcv.py` | 02-02 | 100 | 259 | VERIFIED | All 4 functions present and importable |
| `data_processing_script/sp500_pipeline/feature_engineering.py` | 02-03/02-06 | 180 | 426 | VERIFIED | All 5 public functions present; 69 features computed |
| `data_processing_script/sp500_pipeline/normalize_split.py` | 02-04 | — | 56 | VERIFIED | `cross_sectional_normalize`, `split_by_date`, `main` all present |
| `data_processing_script/sp500_pipeline/serialize_arrays.py` | 02-04 | — | 35 | VERIFIED | `save_model_arrays`, `main` present |
| `data_processing_script/sp500_pipeline/graph_embedding.py` | 02-05 | — | 148 | VERIFIED | `build_correlation_graph`, `run_struc2vec`, `main` present |
| `scripts/build_pipeline.py` | 02-05 | 60 | 130 | VERIFIED | 5-step orchestrator with sentinel skip logic |
| `requirements.txt` | 02-05 | — | updated | VERIFIED | Lines 18-21: `yfinance>=0.2.50`, `pyarrow>=12.0`, `pandas-ta>=0.3.14b`, GraphEmbedding comment |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `download_ohlcv.py` | `ohlcv/*.parquet` | `df.to_parquet()` | WIRED | Line 246: `df.to_parquet(out_path)` in `main()` |
| `download_ohlcv.py` | `tickers.txt` | `open().write()` | WIRED | Line 251-252: `open(tickers_txt_path, "w")` writes tickers |
| `feature_engineering.py` | `features/*.csv` | `_cross_sectional_normalize()` then `normalized_df.to_csv()` | WIRED | Lines 346-348: normalize then `to_csv`; verified by passing test |
| `feature_engineering.py` | `label.csv` | `forward_return.to_csv()` | WIRED | Line 388: `forward_return.to_csv(out_path)` in `compute_label_csv()` |
| `serialize_arrays.py` | `flow.npz` + `trend_indicator.npz` | `np.savez(..., result=data)` | WIRED | Lines 15-17: two `np.savez` calls with `result=` key |
| `normalize_split.py` | `split_indices.json` | `json.dump({'train_end': ..., 'val_end': ...})` | WIRED | Lines 48-50: `json.dump(splits, f)` writes both keys |
| `graph_embedding.py` | `128_corr_struc2vec_adjgat.npy` | `np.save()` | WIRED | Line 112: `np.save(out_path, embedding_array)` |
| `scripts/build_pipeline.py` | `download_ohlcv.py` | `subprocess.run()` | WIRED | Line 84: `subprocess.run(cmd, check=True)` where cmd targets pipeline scripts |

---

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
|-------------|---------------|-------------|--------|----------|
| DATA-01 | 02-01, 02-02 | S&P 500 OHLCV downloaded via yfinance, stored as Parquet | SATISFIED | `download_ohlcv.py` with `get_sp500_tickers()`, `download_ohlcv_batched()`, `clean_and_align()`; `test_download_parquet_schema` and `test_clean_no_all_nan_rows` pass |
| DATA-02 | 02-01, 02-03, 02-06 | Price-volume features: ROC, RSI, MACD, BB, VOL ratios across windows | SATISFIED | `compute_features()` returns 69 columns covering all named indicators; `test_feature_count_for_phase3` passes (>= 60) |
| DATA-03 | 02-01, 02-03 | Cross-sectional z-score normalization per trading day | SATISFIED | `save_feature_csvs()` applies `_cross_sectional_normalize()` before writing; `test_cross_sectional_normalization` validates saved CSV contents |
| DATA-04 | 02-01, 02-04 | Train/val/test split by date, no leakage, stats fit on training only | SATISFIED | `split_by_date()` at 75/12.5/12.5; cross-sectional normalization is row-independent (no time leakage); `test_no_normalization_leakage` and `test_split_ratios` pass |
| DATA-05 | 02-01, 02-04, 02-05 | `flow.npz`, `trend_indicator.npz`, Struc2Vec graph embeddings | SATISFIED | `save_model_arrays()` produces both `.npz` files with `result=` key; `graph_embedding.py` produces `128_corr_struc2vec_adjgat.npy` [N, 128]; all related tests pass |

All 5 Phase 2 requirements are SATISFIED. No orphaned requirements detected.

---

### Anti-Patterns Found

None. Scan of all 6 pipeline source files and `build_pipeline.py` found zero TODO/FIXME/HACK/placeholder comments. No stub returns (`return null`, `return {}`, `return []`) detected. All functions contain substantive implementations.

---

### Human Verification Required

#### 1. End-to-end pipeline run with real data

**Test:** Run `python scripts/build_pipeline.py --data_dir ./data/Stock_SP500_2018-01-01_2024-01-01` with yfinance and GraphEmbedding installed.
**Expected:** All 5 steps complete without error; `ohlcv/` directory contains ~450-490 Parquet files; `features/` contains 69 CSV files; `flow.npz`, `trend_indicator.npz`, and `128_corr_struc2vec_adjgat.npy` exist with correct shapes.
**Why human:** Requires live Wikipedia fetch, yfinance network access, and GraphEmbedding (Struc2Vec) installed — none available in this environment.

#### 2. pandas_ta optional guard

**Test:** `pytest tests/test_data_pipeline.py::test_feature_columns_present tests/test_data_pipeline.py::test_feature_no_all_nan_columns -v` in an environment with `pandas_ta` installed.
**Expected:** Both tests pass (currently they skip due to `pytest.importorskip("pandas_ta")`). Note: `compute_features()` does not actually use `pandas_ta` at runtime — the guard is vestigial from an earlier design. The tests work without `pandas_ta`; the `importorskip` guard causes unnecessary skips.
**Why human:** Requires a Python environment with pandas_ta installed to confirm the tests pass rather than erroneously skip. This is an informational note, not a blocker — the feature count test (`test_feature_count_for_phase3`) already confirms the same logic passes without the guard.

---

### Notable Observations

1. **Test suite completeness:** 11/13 tests pass; 2 skip due to `pytest.importorskip("pandas_ta")` in `test_feature_columns_present` and `test_feature_no_all_nan_columns`. The `importorskip` guard is vestigial — `compute_features()` is now fully pure-pandas and does not call `pandas_ta` at all. The guard causes unnecessary skips but does not block the goal.

2. **Phase 1 regression check:** The full `tests/` suite shows 4 failing Phase 1 tests (`test_torch_importable`, `test_pywavelets_importable`, `test_tensorboard_importable`, `test_pytorch_wavelets_importable`). These are pre-existing environment failures (torch/pywt not installed locally) that predate Phase 2 — no regressions introduced.

3. **Feature count confirmed:** `compute_features()` outputs exactly 69 columns — matching the plan's target of ~69 pure-pandas TA indicators. Zero all-NaN columns after the 60-row warmup period on a 300-row test input.

4. **Normalization wiring correctly placed:** Normalization is applied inside `save_feature_csvs()`, not in a separate post-processing step. `normalize_split.py main()` correctly writes only `split_indices.json` and does not re-normalize feature CSVs.

---

### Gaps Summary

No gaps. All 5 requirements and 7 observable truths are verified. The phase goal is achieved.

---

_Verified: 2026-03-11T19:02:23Z_
_Verifier: Claude (gsd-verifier)_
