"""Phase 2 data pipeline tests — DATA-01 through DATA-05."""
import pytest
import numpy as np
import pandas as pd


# ── DATA-01: OHLCV download ─────────────────────────────────────────────────

@pytest.mark.xfail(reason="DATA-01 not yet implemented", strict=False)
def test_download_parquet_schema(sp500_ohlcv_fixture):
    """Downloaded Parquet files have OHLCV columns and no fully-empty rows."""
    # Will import and call 01_download_ohlcv.download_ohlcv_batched in real test
    pytest.xfail("DATA-01 stub")


@pytest.mark.xfail(reason="DATA-01 not yet implemented", strict=False)
def test_clean_no_all_nan_rows(sp500_ohlcv_fixture):
    """After cleaning, no stock has >5% missing trading days."""
    pytest.xfail("DATA-01 stub")


# ── DATA-02: Feature engineering ────────────────────────────────────────────

@pytest.mark.xfail(reason="DATA-02 not yet implemented", strict=False)
def test_feature_columns_present(ohlcv_wide_fixture):
    """Feature matrix contains RSI, MACD, BB, ROC, VOL_ratio for windows 5/10/20/60."""
    pytest.xfail("DATA-02 stub")


@pytest.mark.xfail(reason="DATA-02 not yet implemented", strict=False)
def test_feature_no_all_nan_columns(ohlcv_wide_fixture):
    """No feature column is entirely NaN after warmup period is dropped."""
    pytest.xfail("DATA-02 stub")


# ── DATA-03: Cross-sectional normalization ───────────────────────────────────

@pytest.mark.xfail(reason="DATA-03 not yet implemented", strict=False)
def test_cross_sectional_normalization(feature_matrix_fixture):
    """Per-date row has mean≈0 and std≈1 across stocks after normalization."""
    pytest.xfail("DATA-03 stub")


# ── DATA-04: Train/val/test split no leakage ─────────────────────────────────

@pytest.mark.xfail(reason="DATA-04 not yet implemented", strict=False)
def test_no_normalization_leakage(feature_matrix_fixture):
    """Val/test normalization does not use statistics from val/test rows."""
    pytest.xfail("DATA-04 stub")


@pytest.mark.xfail(reason="DATA-04 not yet implemented", strict=False)
def test_split_ratios(feature_matrix_fixture):
    """Train is 75%, val is 12.5%, test is 12.5% of total time steps."""
    pytest.xfail("DATA-04 stub")


# ── DATA-05: NPZ arrays + graph embedding ────────────────────────────────────

@pytest.mark.xfail(reason="DATA-05 not yet implemented", strict=False)
def test_npz_shapes_no_nan(tmp_path):
    """flow.npz and trend_indicator.npz have shape [T, N] and no NaN values."""
    pytest.xfail("DATA-05 stub")


@pytest.mark.xfail(reason="DATA-05 not yet implemented", strict=False)
def test_trend_indicator_binary(tmp_path):
    """trend_indicator.npz contains only 0 and 1 values."""
    pytest.xfail("DATA-05 stub")


@pytest.mark.xfail(reason="DATA-05 not yet implemented", strict=False)
def test_graph_embedding_shape(tmp_path):
    """Struc2Vec embedding file exists with shape [N, 128]."""
    pytest.xfail("DATA-05 stub")
