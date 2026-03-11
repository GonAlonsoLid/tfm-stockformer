"""Shared fixtures for Phase 1 infrastructure tests."""
import pytest
import os
import sys
import configparser
import numpy as np
import pandas as pd

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

@pytest.fixture
def project_root():
    return PROJECT_ROOT

@pytest.fixture
def config(project_root):
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(project_root, 'config', 'Multitask_Stock.conf'))
    return cfg


# ── Phase 2 Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def sp500_ohlcv_fixture():
    """Returns a dict mapping 5 fake tickers to OHLCV DataFrames (300 business days)."""
    np.random.seed(42)
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN", "META"]
    dates = pd.bdate_range(start="2020-01-02", periods=300)
    result = {}
    for ticker in tickers:
        close = 100.0 * np.exp(np.cumsum(np.random.randn(300) * 0.01))
        open_ = close * (1 + np.random.randn(300) * 0.005)
        high = np.maximum(open_, close) * (1 + np.abs(np.random.randn(300) * 0.005))
        low = np.minimum(open_, close) * (1 - np.abs(np.random.randn(300) * 0.005))
        volume = np.random.randint(int(1e6), int(5e6), size=300)
        result[ticker] = pd.DataFrame(
            {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=dates,
        )
    return result


@pytest.fixture
def ohlcv_wide_fixture(sp500_ohlcv_fixture):
    """Returns a wide DataFrame with DatetimeIndex rows and ticker columns (Close prices)."""
    return pd.DataFrame(
        {ticker: df["Close"] for ticker, df in sp500_ohlcv_fixture.items()}
    )


@pytest.fixture
def feature_matrix_fixture():
    """Returns a synthetic normalized feature matrix of shape [280, 5]."""
    np.random.seed(99)
    return np.random.randn(280, 5).astype(np.float32)
