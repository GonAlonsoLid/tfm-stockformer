"""Phase 8 Alpha360 feature replacement tests — ALPHA360-01 through ALPHA360-05.

Wave 0 test scaffold: all tests are xfail stubs.
Implementation contract for scripts/build_alpha360.py:

    build_alpha360.main(config_path: str, data_dir: str) -> None

    - config_path: path to .conf file with [file] alpha_360_dir pointing to the features dir
    - data_dir: base data directory; the function derives:
        ohlcv_dir    = data_dir / "ohlcv"
        tickers_file = data_dir / "tickers.txt"
        features_dir = from config [file] alpha_360_dir  (already points to data_dir/features)

    The function must:
    1. Back up existing CSVs in alpha_360_dir to data_dir/features_backup_{YYYYMMDD}/
    2. Load OHLCV Parquets, compute 360 Alpha360-style features (6 fields x 60 lags)
    3. Apply cross-sectional z-score per day, replace NaN/inf with 0.0
    4. Write 360 CSV files to alpha_360_dir; columns = tickers in tickers.txt order
    5. First row date = index[60] of the OHLCV date range (60-row lag buffer)
"""
import pytest
import os
import numpy as np
import pandas as pd


# ── Shared fixture ────────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def alpha360_env(tmp_path):
    """Create a minimal environment mirroring the real data layout.

    Structure:
        tmp_path/
            config/Multitask_Stock_SP500.conf
            data/ohlcv/{ticker}.parquet  (5 tickers, 80 OHLCV rows each)
            data/features/               (3 dummy CSVs simulating existing 69)
            data/tickers.txt             (5 tickers: AAPL, MSFT, GOOG, AMZN, META)

    Parquet fixture: 80 rows (60 lag buffer + 20 output rows), bdate_range from 2018-01-02.
    """
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    np.random.seed(42)

    # Create directory structure
    ohlcv_dir = tmp_path / "data" / "ohlcv"
    features_dir = tmp_path / "data" / "features"
    config_dir = tmp_path / "config"
    for d in [ohlcv_dir, features_dir, config_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create synthetic OHLCV Parquet files
    dates = pd.bdate_range(start="2018-01-02", periods=80)
    for ticker in tickers:
        close = np.random.rand(80) * 50 + 75      # ~75-125
        df = pd.DataFrame(
            {
                "Open": close * (1 + np.random.rand(80) * 0.02 - 0.01),
                "High": close * (1 + np.random.rand(80) * 0.03),
                "Low": close * (1 - np.random.rand(80) * 0.03),
                "Close": close,
                "Volume": np.random.randint(500_000, 2_000_000, size=80).astype(float),
            },
            index=dates,
        )
        df.to_parquet(str(ohlcv_dir / f"{ticker}.parquet"))

    # Create tickers.txt
    tickers_file = tmp_path / "data" / "tickers.txt"
    tickers_file.write_text("\n".join(tickers) + "\n")

    # Create 3 dummy existing feature CSVs (simulates the 69-CSV state to be backed up)
    output_dates = dates[60:]  # 20 rows — the output date range
    for name in ["dummy_01.csv", "dummy_02.csv", "dummy_03.csv"]:
        dummy = pd.DataFrame(
            np.random.randn(20, 5),
            index=output_dates,
            columns=tickers,
        )
        dummy.index.name = "Date"
        dummy.to_csv(str(features_dir / name))

    # Create minimal config file
    config_path = config_dir / "Multitask_Stock_SP500.conf"
    config_path.write_text(
        f"[file]\nalpha_360_dir = {str(features_dir)}\n"
    )

    return {
        "tmp_path": tmp_path,
        "ohlcv_dir": ohlcv_dir,
        "features_dir": features_dir,
        "tickers_file": tickers_file,
        "config_path": config_path,
        "tickers": tickers,
    }


# ── ALPHA360-01: Script runs and produces 360 CSVs ───────────────────────────

def test_build_alpha360_creates_360_csvs(alpha360_env):
    """ALPHA360-01: script runs and produces exactly 360 CSV files in features_dir."""
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
    import build_alpha360

    build_alpha360.main(
        config_path=str(alpha360_env["config_path"]),
        data_dir=str(alpha360_env["tmp_path"] / "data"),
    )

    csv_files = [f for f in os.listdir(alpha360_env["features_dir"]) if f.endswith(".csv")]
    assert len(csv_files) == 360, (
        f"Expected 360 CSV files in features_dir; got {len(csv_files)}"
    )


# ── ALPHA360-02: Shape, NaN, and inf validation ───────────────────────────────

def test_feature_csv_shape_and_no_nan(alpha360_env):
    """ALPHA360-02: each CSV has correct shape (T_features, N_tickers), zero NaN, zero inf."""
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
    import build_alpha360

    build_alpha360.main(
        config_path=str(alpha360_env["config_path"]),
        data_dir=str(alpha360_env["tmp_path"] / "data"),
    )

    features_dir = alpha360_env["features_dir"]
    csv_files = [f for f in os.listdir(features_dir) if f.endswith(".csv")]
    df = pd.read_csv(os.path.join(features_dir, csv_files[0]), index_col=0)

    assert df.shape[1] == 5, (
        f"Expected 5 ticker columns (fixture has 5 tickers); got {df.shape[1]}"
    )
    assert df.isna().sum().sum() == 0, "NaN values found in feature CSV"
    assert not np.isinf(df.values).any(), "Inf values found in feature CSV"


# ── ALPHA360-03: First row date = index[60] of OHLCV (60-row lag buffer) ─────

def test_first_row_date(alpha360_env):
    """ALPHA360-03: first date in each feature CSV is index[60] of the OHLCV date range.

    Fixture uses bdate_range(start='2018-01-02', periods=80); index[60] is the
    61st business day from 2018-01-02.
    """
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
    import build_alpha360

    build_alpha360.main(
        config_path=str(alpha360_env["config_path"]),
        data_dir=str(alpha360_env["tmp_path"] / "data"),
    )

    features_dir = alpha360_env["features_dir"]
    csv_files = [f for f in os.listdir(features_dir) if f.endswith(".csv")]
    df = pd.read_csv(os.path.join(features_dir, csv_files[0]), index_col=0)

    # In fixture: bdate_range(start='2018-01-02', periods=80); index[60] is the 61st business day
    ohlcv_dates = pd.bdate_range(start="2018-01-02", periods=80)
    expected_first_date = str(ohlcv_dates[60].date())

    assert str(df.index[0]) == expected_first_date, (
        f"Expected first date {expected_first_date}, got {df.index[0]}"
    )


# ── ALPHA360-04: Backup directory created with original CSVs ─────────────────

def test_backup_created(alpha360_env):
    """ALPHA360-04: backup directory created in data_dir, containing the 3 original dummy CSVs."""
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
    import build_alpha360

    build_alpha360.main(
        config_path=str(alpha360_env["config_path"]),
        data_dir=str(alpha360_env["tmp_path"] / "data"),
    )

    data_dir = alpha360_env["tmp_path"] / "data"
    backup_dirs = [
        d
        for d in os.listdir(data_dir)
        if d.startswith("features_backup") and os.path.isdir(data_dir / d)
    ]
    assert len(backup_dirs) >= 1, "No backup directory created"

    backup_path = data_dir / backup_dirs[0]
    backup_csvs = [f for f in os.listdir(backup_path) if f.endswith(".csv")]
    assert len(backup_csvs) == 3, (
        f"Expected 3 original CSVs in backup, got {len(backup_csvs)}"
    )


# ── ALPHA360-05: Column order matches tickers.txt ────────────────────────────

def test_column_order_matches_tickers(alpha360_env):
    """ALPHA360-05: column order in every CSV matches the order in tickers.txt."""
    import sys
    import pathlib

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "scripts"))
    import build_alpha360

    build_alpha360.main(
        config_path=str(alpha360_env["config_path"]),
        data_dir=str(alpha360_env["tmp_path"] / "data"),
    )

    features_dir = alpha360_env["features_dir"]
    with open(alpha360_env["tickers_file"]) as f:
        expected_tickers = [line.strip() for line in f if line.strip()]

    csv_files = [f for f in os.listdir(features_dir) if f.endswith(".csv")]
    for csv_file in csv_files[:3]:  # spot-check first 3
        df = pd.read_csv(os.path.join(features_dir, csv_file), index_col=0)
        assert list(df.columns) == expected_tickers, (
            f"{csv_file}: column order {list(df.columns)} != tickers.txt {expected_tickers}"
        )
