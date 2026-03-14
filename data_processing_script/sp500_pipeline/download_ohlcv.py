"""S&P 500 OHLCV download and clean pipeline step.

Fetches the S&P 500 constituent list from Wikipedia, downloads daily OHLCV
data via yfinance in batched chunks with retry logic, aligns all tickers to
a master trading calendar, drops tickers with excessive missing data, and
saves one Parquet file per ticker.

Usage:
    python data_processing_script/sp500_pipeline/download_ohlcv.py \
        --data_dir ./data/Stock_SP500_2018-01-01_2024-01-01 \
        --start 2018-01-01 \
        --end 2024-01-01
"""

import argparse
import logging
import os
import time
from typing import Dict, List, Optional

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger(__name__)


def get_sp500_tickers() -> List[str]:
    """Fetch the current S&P 500 constituent list from Wikipedia.

    Returns:
        List of ticker symbols with '.' replaced by '-' (e.g., BRK-B).
    """
    import urllib.request
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        html = resp.read()
    table = pd.read_html(html)[0]
    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers


def download_ohlcv_batched(
    tickers: List[str],
    start: str,
    end: str,
    chunk_size: int = 80,
    delay: float = 2.0,
) -> Dict[str, pd.DataFrame]:
    """Download daily OHLCV data for a list of tickers in batched chunks.

    Downloads in chunks of ``chunk_size`` tickers, retrying up to 3 times
    on failure with exponential back-off. Returns a dict mapping each
    successfully downloaded ticker to its OHLCV DataFrame.

    Args:
        tickers: List of ticker symbols to download.
        start: Start date string in ISO format (e.g., '2018-01-01').
        end: End date string in ISO format (e.g., '2024-01-01').
        chunk_size: Number of tickers to download per yfinance call.
        delay: Base delay in seconds between chunks and for back-off.

    Returns:
        Dict mapping ticker symbol to a DataFrame with columns
        [Open, High, Low, Close, Volume] and a DatetimeIndex.

    Raises:
        ImportError: If yfinance is not installed.
    """
    if yf is None:
        raise ImportError(
            "yfinance is required to download OHLCV data. "
            "Install it with: pip install yfinance"
        )

    result: Dict[str, pd.DataFrame] = {}

    chunks = [tickers[i : i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    for chunk_idx, chunk in enumerate(chunks):
        logger.info(
            "Downloading chunk %d/%d (%d tickers)...",
            chunk_idx + 1,
            len(chunks),
            len(chunk),
        )

        data = None
        for attempt in range(3):
            try:
                data = yf.download(
                    chunk,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                )
                break
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Attempt %d failed for chunk %d: %s", attempt + 1, chunk_idx + 1, exc
                )
                time.sleep(delay * (attempt + 1))

        if data is None or data.empty:
            logger.warning("No data returned for chunk %d — skipping.", chunk_idx + 1)
            time.sleep(delay)
            continue

        for ticker in chunk:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    # MultiIndex: (field, ticker)
                    ticker_df = data.xs(ticker, axis=1, level=1)
                else:
                    # Single-ticker download returns flat columns
                    ticker_df = data.copy()

                # Keep only OHLCV columns
                ohlcv_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in ticker_df.columns]
                if not ohlcv_cols:
                    logger.warning("No OHLCV columns for ticker %s — skipping.", ticker)
                    continue

                ticker_df = ticker_df[ohlcv_cols].dropna(how="all")
                if ticker_df.empty:
                    logger.warning("Empty DataFrame for ticker %s — skipping.", ticker)
                    continue

                result[ticker] = ticker_df

            except KeyError:
                logger.warning("Ticker %s not found in downloaded data — skipping.", ticker)

        time.sleep(delay)

    logger.info("Downloaded %d/%d tickers successfully.", len(result), len(tickers))
    return result


def clean_and_align(
    ticker_data_dict: Dict[str, pd.DataFrame],
    max_missing_pct: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    """Align all tickers to a master trading calendar and drop sparse ones.

    Derives the master calendar from the ticker with the most dates (typically
    SPY or a large-cap with complete history), reindexes every ticker to that
    calendar, forward-fills gaps up to 5 consecutive days, then drops any
    ticker whose fraction of remaining NaN rows exceeds ``max_missing_pct``.

    Args:
        ticker_data_dict: Dict mapping ticker symbols to raw OHLCV DataFrames.
        max_missing_pct: Maximum fraction of trading days allowed to have any
            NaN value after forward-filling. Tickers exceeding this threshold
            are dropped.

    Returns:
        Dict mapping ticker symbols to cleaned, aligned OHLCV DataFrames.
    """
    if not ticker_data_dict:
        return {}

    # Build master calendar from the ticker with the most dates
    master_ticker = max(ticker_data_dict, key=lambda t: len(ticker_data_dict[t]))
    master_calendar = ticker_data_dict[master_ticker].index

    cleaned: Dict[str, pd.DataFrame] = {}
    for ticker, df in ticker_data_dict.items():
        # Reindex to master calendar
        aligned = df.reindex(master_calendar)

        # Forward-fill up to 5 consecutive missing values
        aligned = aligned.ffill(limit=5)

        # Drop ticker if fraction of rows with any NaN exceeds threshold
        missing_fraction = aligned.isna().any(axis=1).mean()
        if missing_fraction > max_missing_pct:
            logger.debug(
                "Dropping ticker %s: %.1f%% missing days (threshold %.1f%%).",
                ticker,
                missing_fraction * 100,
                max_missing_pct * 100,
            )
            continue

        cleaned[ticker] = aligned

    logger.info(
        "Retained %d/%d tickers after cleaning (max_missing_pct=%.0f%%).",
        len(cleaned),
        len(ticker_data_dict),
        max_missing_pct * 100,
    )
    return cleaned


def main() -> None:
    """CLI entry point for the OHLCV download and clean step.

    Downloads S&P 500 OHLCV data for the specified date range, cleans and
    aligns to a master trading calendar, and saves one Parquet file per
    ticker in ``{data_dir}/ohlcv/``. Also writes ``{data_dir}/tickers.txt``.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Download and clean S&P 500 OHLCV data via yfinance."
    )
    parser.add_argument(
        "--data_dir",
        default="./data/Stock_SP500_2018-01-01_2024-01-01",
        help="Root directory for output data (default: ./data/Stock_SP500_2018-01-01_2024-01-01)",
    )
    parser.add_argument(
        "--start",
        default="2018-01-01",
        help="Start date in ISO format (default: 2018-01-01)",
    )
    parser.add_argument(
        "--end",
        default="2024-01-01",
        help="End date in ISO format (default: 2024-01-01)",
    )
    args = parser.parse_args()

    # Fetch ticker list
    logger.info("Fetching S&P 500 ticker list from Wikipedia...")
    tickers = get_sp500_tickers()
    logger.info("Found %d tickers.", len(tickers))

    # Download OHLCV data
    logger.info("Downloading OHLCV data from %s to %s...", args.start, args.end)
    raw_data = download_ohlcv_batched(tickers, start=args.start, end=args.end)

    # Clean and align
    logger.info("Cleaning and aligning to master trading calendar...")
    clean_data = clean_and_align(raw_data)

    # Save output
    ohlcv_dir = os.path.join(args.data_dir, "ohlcv")
    os.makedirs(ohlcv_dir, exist_ok=True)

    saved_tickers = []
    for ticker, df in clean_data.items():
        out_path = os.path.join(ohlcv_dir, f"{ticker}.parquet")
        df.to_parquet(out_path)
        saved_tickers.append(ticker)

    # Write tickers.txt
    tickers_txt_path = os.path.join(args.data_dir, "tickers.txt")
    with open(tickers_txt_path, "w") as f:
        f.write("\n".join(saved_tickers) + "\n")

    print(f"Saved {len(saved_tickers)} tickers to {ohlcv_dir}/")
    print(f"Ticker list written to {tickers_txt_path}")


if __name__ == "__main__":
    main()
