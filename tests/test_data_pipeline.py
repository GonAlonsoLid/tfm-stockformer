"""Phase 2 data pipeline tests — DATA-01 through DATA-05."""
import pytest
import numpy as np
import pandas as pd


# ── DATA-01: OHLCV download ─────────────────────────────────────────────────

def test_download_parquet_schema(sp500_ohlcv_fixture):
    """Downloaded Parquet files have OHLCV columns and no fully-empty rows."""
    from data_processing_script.sp500_pipeline.download_ohlcv import clean_and_align
    cleaned = clean_and_align(sp500_ohlcv_fixture, max_missing_pct=0.05)
    for ticker, df in cleaned.items():
        assert set(["Open", "High", "Low", "Close", "Volume"]).issubset(df.columns)
        assert not df.dropna(how="all").empty


def test_clean_no_all_nan_rows(sp500_ohlcv_fixture):
    """After cleaning, tickers with >5% missing days are dropped; those with <5% are kept."""
    from data_processing_script.sp500_pipeline.download_ohlcv import clean_and_align

    np.random.seed(0)
    dates = pd.bdate_range(start="2020-01-02", periods=300)

    # Build fixture extended with two extra tickers
    ticker_data = dict(sp500_ohlcv_fixture)

    # Ticker with >5% unrecoverable NaN rows — should be dropped.
    # Use a block of 20 consecutive NaN rows so ffill(limit=5) cannot recover them.
    bad_ticker = "BAD_STOCK"
    bad_df = pd.DataFrame(
        {
            "Open": np.random.rand(300) * 100,
            "High": np.random.rand(300) * 110,
            "Low": np.random.rand(300) * 90,
            "Close": np.random.rand(300) * 100,
            "Volume": np.random.randint(1_000_000, 5_000_000, size=300).astype(float),
        },
        index=dates,
    )
    # 20 consecutive NaN rows at positions 50-69: ffill(limit=5) leaves rows 56-69 (14 rows)
    # still NaN → ~4.7% > 5%? Let's use 30 consecutive rows 50-79 to be safe (leaving ~25 unfilled ≈ 8%)
    bad_df.iloc[50:80] = np.nan  # 30 consecutive rows; ffill recovers only first 5 → 25 rows remain NaN (~8.3%)
    ticker_data[bad_ticker] = bad_df

    # Ticker with 3% NaN rows — should survive after ffill
    good_ticker = "GOOD_STOCK"
    good_df = pd.DataFrame(
        {
            "Open": np.random.rand(300) * 100,
            "High": np.random.rand(300) * 110,
            "Low": np.random.rand(300) * 90,
            "Close": np.random.rand(300) * 100,
            "Volume": np.random.randint(1_000_000, 5_000_000, size=300).astype(float),
        },
        index=dates,
    )
    # Introduce 3% NaN rows (9 rows) — ffill(limit=5) should recover them if isolated
    # Place them as single isolated NaN rows so ffill can fill them
    nan_rows_good = np.arange(10, 300, 33)[:9]  # 9 isolated rows
    good_df.iloc[nan_rows_good] = np.nan
    ticker_data[good_ticker] = good_df

    cleaned = clean_and_align(ticker_data, max_missing_pct=0.05)

    assert bad_ticker not in cleaned, (
        f"{bad_ticker} (10% NaN rows) should have been dropped by clean_and_align"
    )
    assert good_ticker in cleaned, (
        f"{good_ticker} (3% NaN rows, isolated) should have survived clean_and_align"
    )


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
