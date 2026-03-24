"""
Tests for scripts/sp500_pipeline/download_ohlcv.py — Phase 9 restructuring.

Tests verify the ticker local-file fallback logic added in Plan 09-02.
All tests are xfail(strict=False) until the module is moved and the fallback is added.
Import convention: imports inside test bodies to prevent module-level ImportError.
"""
import os
import pytest


def test_ticker_fallback_reads_from_file(tmp_path, monkeypatch):
    """When tickers.txt exists, main() reads tickers from file and skips Wikipedia."""
    # Write a known tickers.txt
    tickers_file = tmp_path / "tickers.txt"
    tickers_file.write_text("AAPL\nMSFT\nGOOGL\n")

    # Import after path is confirmed to exist
    from scripts.sp500_pipeline import download_ohlcv as mod

    wikipedia_called = {"called": False}

    def fake_get_tickers():
        wikipedia_called["called"] = True
        return ["FAKE1", "FAKE2"]

    def fake_download_batched(tickers, start, end, **kwargs):
        # Return minimal dict so main() can continue
        return {t: None for t in tickers}

    def fake_clean_and_align(data, **kwargs):
        return data

    monkeypatch.setattr(mod, "get_sp500_tickers", fake_get_tickers)
    monkeypatch.setattr(mod, "download_ohlcv_batched", fake_download_batched)
    monkeypatch.setattr(mod, "clean_and_align", fake_clean_and_align)

    import sys
    monkeypatch.setattr(
        sys, "argv",
        ["download_ohlcv.py", "--data_dir", str(tmp_path),
         "--start", "2018-01-01", "--end", "2024-01-01"]
    )

    # Prevent file I/O for parquet saves
    import unittest.mock as mock
    with mock.patch("pandas.DataFrame.to_parquet"):
        try:
            mod.main()
        except Exception:
            pass  # partial run is fine for this check

    assert not wikipedia_called["called"], (
        "get_sp500_tickers() was called even though tickers.txt existed"
    )


def test_ticker_fallback_calls_wikipedia_when_absent(tmp_path, monkeypatch):
    """When tickers.txt is absent, main() calls get_sp500_tickers()."""
    # No tickers.txt in tmp_path

    from scripts.sp500_pipeline import download_ohlcv as mod

    wikipedia_called = {"called": False}

    def fake_get_tickers():
        wikipedia_called["called"] = True
        return ["AAPL", "MSFT"]

    def fake_download_batched(tickers, start, end, **kwargs):
        return {t: None for t in tickers}

    def fake_clean_and_align(data, **kwargs):
        return data

    monkeypatch.setattr(mod, "get_sp500_tickers", fake_get_tickers)
    monkeypatch.setattr(mod, "download_ohlcv_batched", fake_download_batched)
    monkeypatch.setattr(mod, "clean_and_align", fake_clean_and_align)

    import sys
    monkeypatch.setattr(
        sys, "argv",
        ["download_ohlcv.py", "--data_dir", str(tmp_path),
         "--start", "2018-01-01", "--end", "2024-01-01"]
    )

    import unittest.mock as mock
    with mock.patch("pandas.DataFrame.to_parquet"):
        try:
            mod.main()
        except Exception:
            pass

    assert wikipedia_called["called"], (
        "get_sp500_tickers() was NOT called even though tickers.txt was absent"
    )
