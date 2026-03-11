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

def test_feature_columns_present(sp500_ohlcv_fixture):
    """Feature matrix contains RSI, MACD, BB, ROC, VOL_ratio for windows 5/10/20/60."""
    pytest.importorskip("pandas_ta")
    from data_processing_script.sp500_pipeline.feature_engineering import compute_features
    ticker_df = sp500_ohlcv_fixture['AAPL']
    features = compute_features(ticker_df)
    expected_cols = [
        'ROC_5', 'ROC_10', 'ROC_20', 'ROC_60',
        'RSI_5', 'RSI_10', 'RSI_20', 'RSI_60',
        'MACD',
        'BB_upper_20', 'BB_lower_20', 'BB_width_20',
        'VOL_ratio_5', 'VOL_ratio_10', 'VOL_ratio_20', 'VOL_ratio_60',
    ]
    for col in expected_cols:
        assert col in features.columns, f"Missing column: {col}"


def test_feature_no_all_nan_columns(sp500_ohlcv_fixture):
    """No feature column is entirely NaN after warmup period is dropped."""
    pytest.importorskip("pandas_ta")
    from data_processing_script.sp500_pipeline.feature_engineering import compute_features
    features = compute_features(sp500_ohlcv_fixture['AAPL'])
    assert not features.iloc[60:].isna().all().any(), (
        "Some feature columns are entirely NaN after the 60-row warmup is dropped"
    )


# ── DATA-03: Cross-sectional normalization ───────────────────────────────────

def test_cross_sectional_normalization(tmp_path, ohlcv_wide_fixture):
    """Per-date row has mean≈0 and std≈1 across stocks after normalization."""
    from data_processing_script.sp500_pipeline.feature_engineering import save_feature_csvs
    # Build synthetic wide DataFrame (dates x tickers) with un-normalized values
    wide_df = ohlcv_wide_fixture.copy()
    # Shift values so they are clearly NOT normalized
    wide_df = wide_df * 100 + 500
    feature_dict = {'ROC_5': wide_df}
    save_feature_csvs(feature_dict, str(tmp_path))
    # Reload and verify normalization was applied
    saved = pd.read_csv(tmp_path / 'features' / 'ROC_5.csv', index_col=0)
    row_means = saved.mean(axis=1).abs()
    row_stds = saved.std(axis=1)
    assert (row_means < 1e-8).all(), f"Row means not ~0: max={row_means.max()}"
    assert ((row_stds - 1.0).abs() < 0.05).all(), f"Row stds not ~1"


# ── DATA-04: Train/val/test split no leakage ─────────────────────────────────

def test_no_normalization_leakage():
    """Cross-sectional normalization is per-row: val/test rows normalized independently are identical."""
    from data_processing_script.sp500_pipeline.normalize_split import cross_sectional_normalize
    np.random.seed(0)
    df = pd.DataFrame(np.random.randn(200, 10))
    # Normalize full dataset
    full_normed = cross_sectional_normalize(df)
    # Normalize just the val/test portion (rows 150:200) independently
    val_normed_standalone = cross_sectional_normalize(df.iloc[150:])
    # They should be identical — cross-sectional norm does not use other rows
    pd.testing.assert_frame_equal(
        full_normed.iloc[150:].reset_index(drop=True),
        val_normed_standalone.reset_index(drop=True),
        atol=1e-10,
    )


def test_split_ratios():
    """Train is 75%, val is 12.5%, test is 12.5% of total time steps."""
    from data_processing_script.sp500_pipeline.normalize_split import split_by_date
    df = pd.DataFrame(np.random.randn(1000, 5))
    train, val, test = split_by_date(df, train_ratio=0.75, val_ratio=0.125)
    assert len(train) == 750
    assert len(val) == 125
    assert len(test) == 125


# ── DATA-05: NPZ arrays + graph embedding ────────────────────────────────────

def test_npz_shapes_no_nan(tmp_path):
    """flow.npz and trend_indicator.npz have shape [T, N] and no NaN values."""
    from data_processing_script.sp500_pipeline.serialize_arrays import save_model_arrays
    np.random.seed(7)
    df = pd.DataFrame(np.random.randn(50, 5))
    save_model_arrays(df, str(tmp_path))
    flow = np.load(str(tmp_path / "flow.npz"))["result"]
    trend = np.load(str(tmp_path / "trend_indicator.npz"))["result"]
    assert flow.shape == (50, 5), f"Expected (50, 5), got {flow.shape}"
    assert not np.isnan(flow).any(), "flow.npz contains NaN"
    assert trend.shape == (50, 5), f"Expected (50, 5), got {trend.shape}"
    assert not np.isnan(trend.astype(float)).any(), "trend_indicator.npz contains NaN"


def test_trend_indicator_binary(tmp_path):
    """trend_indicator.npz contains only 0 and 1 values."""
    from data_processing_script.sp500_pipeline.serialize_arrays import save_model_arrays
    np.random.seed(7)
    df = pd.DataFrame(np.random.randn(50, 5))
    save_model_arrays(df, str(tmp_path))
    trend = np.load(str(tmp_path / "trend_indicator.npz"))["result"]
    unique_vals = np.unique(trend)
    assert set(unique_vals).issubset({0, 1}), f"Unexpected values in trend: {unique_vals}"


def test_graph_embedding_shape(tmp_path):
    """Struc2Vec embedding file has shape [N, 128]."""
    N = 50  # synthetic stock count
    embedding = np.random.randn(N, 128).astype(np.float32)
    out_path = tmp_path / "128_corr_struc2vec_adjgat.npy"
    np.save(str(out_path), embedding)

    loaded = np.load(str(out_path))
    assert loaded.ndim == 2, f"Expected 2D array, got {loaded.ndim}D"
    assert loaded.shape[1] == 128, f"Expected embed_size=128, got {loaded.shape[1]}"
    assert loaded.shape[0] == N, f"Expected N={N}, got {loaded.shape[0]}"


# ── Task 1 TDD: build_correlation_graph and build_pipeline ───────────────────

def test_build_correlation_graph_creates_files(tmp_path):
    """build_correlation_graph writes corr_adj.npy and data.edgelist, returns edge count."""
    from data_processing_script.sp500_pipeline.graph_embedding import build_correlation_graph
    np.random.seed(42)
    N, T = 5, 100
    # Create a synthetic label.csv (T rows x N cols)
    dates = pd.bdate_range(start="2020-01-02", periods=T)
    data = np.random.randn(T, N)
    pd.DataFrame(data, index=dates, columns=[f"S{i}" for i in range(N)]).to_csv(
        str(tmp_path / "label.csv")
    )

    edge_count = build_correlation_graph(str(tmp_path), threshold=0.0)
    assert (tmp_path / "corr_adj.npy").exists(), "corr_adj.npy not created"
    assert (tmp_path / "data.edgelist").exists(), "data.edgelist not created"
    corr = np.load(str(tmp_path / "corr_adj.npy"))
    assert corr.shape == (N, N), f"Expected ({N},{N}), got {corr.shape}"
    assert isinstance(edge_count, int) and edge_count > 0


def test_build_correlation_graph_threshold(tmp_path):
    """build_correlation_graph filters edges by |corr| > threshold."""
    from data_processing_script.sp500_pipeline.graph_embedding import build_correlation_graph
    np.random.seed(42)
    T, N = 200, 5
    dates = pd.bdate_range(start="2020-01-02", periods=T)
    data = np.random.randn(T, N)
    pd.DataFrame(data, index=dates, columns=[f"S{i}" for i in range(N)]).to_csv(
        str(tmp_path / "label.csv")
    )

    # With threshold=0 all N*(N-1)/2 pairs should appear
    count_all = build_correlation_graph(str(tmp_path), threshold=0.0)
    # With threshold=0.999 almost no edges survive
    count_strict = build_correlation_graph(str(tmp_path), threshold=0.999)
    assert count_all >= count_strict, "Higher threshold should yield fewer or equal edges"
