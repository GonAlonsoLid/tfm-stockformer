#!/usr/bin/env python3
"""Build point-in-time fundamental factors from SEC EDGAR XBRL (Tier B).

For each S&P 500 ticker, downloads companyfacts (cached), extracts the annual
(10-K) series of a focused concept set, and derives documented cross-sectional
factors. CRITICAL anti-leakage rule: a fundamental enters date t only if its
SEC `filed` date <= t (point-in-time via pandas.merge_asof).

Factors (strongest documented families: value, profitability, investment,
issuance — Gu-Kelly-Xiu 2020, Novy-Marx 2013, Fama-French 2015):
    roa, gross_profitability, asset_growth, net_issuance,
    earnings_yield, sales_to_price, book_to_market

Output: data/<dataset>/fundamentals.npz with X_fund [T, N, F], feature_names,
aligned to the panel calendar (panel.dates) and ticker order.

Usage:
    python scripts/build_fundamentals.py --data_dir data/Stock_SP500_2018-01-01_2026-03-16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib import data_panel as dp  # noqa: E402

UA = {"User-Agent": "TFM Stockformer Research contacto@ejemplo.com"}
CONCEPTS = {  # canonical -> list of acceptable us-gaap/dei tags (first found wins)
    "assets": ["Assets"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "gross_profit": ["GrossProfit"],
    "cogs": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "shares": ["CommonStockSharesOutstanding", "WeightedAverageNumberOfDilutedSharesOutstanding"],
}
FACTORS = ["roa", "gross_profitability", "asset_growth", "net_issuance",
           "earnings_yield", "sales_to_price", "book_to_market"]


def _get(url: str) -> bytes:
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


def ticker_cik_map() -> dict:
    ct = json.loads(_get("https://www.sec.gov/files/company_tickers.json"))
    return {v["ticker"]: v["cik_str"] for v in ct.values()}


def fetch_extract(ticker: str, cik: int, cache_dir: str) -> dict | None:
    """Download companyfacts and extract annual (10-K) series for needed concepts."""
    cache = os.path.join(cache_dir, f"{ticker}.json")
    if os.path.isfile(cache):
        with open(cache) as f:
            return json.load(f)
    try:
        cf = json.loads(_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"))
    except Exception as e:
        print(f"  {ticker}: fetch failed ({e})")
        return None
    facts = cf.get("facts", {})
    pools = {**facts.get("us-gaap", {}), **facts.get("dei", {})}
    out = {}
    for canon, tags in CONCEPTS.items():
        series = []
        for tag in tags:
            if tag not in pools:
                continue
            for unit_items in pools[tag].get("units", {}).values():
                for it in unit_items:
                    # annual figures from 10-K filings; keep filing date for PIT
                    if it.get("form", "").startswith("10-K") and it.get("fp") in ("FY", None) and "filed" in it:
                        series.append({"end": it["end"], "filed": it["filed"], "val": it["val"]})
            if series:
                break
        if series:
            out[canon] = series
    with open(cache, "w") as f:
        json.dump(out, f)
    return out


def annual_table(extract: dict) -> pd.DataFrame:
    """One row per fiscal year: latest-filed value per concept, with filed date."""
    frames = {}
    for canon, series in extract.items():
        df = pd.DataFrame(series)
        df["end"] = pd.to_datetime(df["end"])
        df["filed"] = pd.to_datetime(df["filed"])
        # keep last-filed value per fiscal-year-end
        df = df.sort_values("filed").drop_duplicates("end", keep="last")
        frames[canon] = df.set_index("end")[["val", "filed"]].rename(
            columns={"val": canon, "filed": f"{canon}_filed"})
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames.values(), axis=1).sort_index()
    return out


def factor_table(extract: dict) -> pd.DataFrame:
    """Per fiscal year: the fundamental factors + the max filing date (PIT key)."""
    t = annual_table(extract)
    if t.empty or "assets" not in t:
        return pd.DataFrame()
    nan = pd.Series(np.nan, index=t.index)
    col = lambda name: t[name] if name in t.columns else nan
    f = pd.DataFrame(index=t.index)
    gp = t["gross_profit"] if "gross_profit" in t.columns else (col("revenue") - col("cogs"))
    f["roa"] = col("net_income") / t["assets"]
    f["gross_profitability"] = gp / t["assets"]
    f["asset_growth"] = t["assets"] / t["assets"].shift(1) - 1
    f["net_issuance"] = col("shares") / col("shares").shift(1) - 1
    f["_net_income"] = col("net_income")
    f["_revenue"] = col("revenue")
    f["_equity"] = col("equity")
    f["_shares"] = col("shares")
    # PIT date = latest filing date among the concepts used this year
    filed_cols = [c for c in t.columns if c.endswith("_filed")]
    f["filed"] = t[filed_cols].max(axis=1)
    return f.dropna(subset=["filed"]).reset_index(drop=True).sort_values("filed")


def build(data_dir: str) -> None:
    panel = dp.load_panel(data_dir)        # for dates + ticker order
    dates, tickers = pd.DatetimeIndex(panel.dates), panel.tickers
    T, N = len(dates), len(tickers)
    cache_dir = os.path.join(data_dir, "edgar_cache")
    os.makedirs(cache_dir, exist_ok=True)

    print("Mapping tickers -> CIK ...")
    t2c = ticker_cik_map()

    # price (Close) per ticker for market-cap ratios
    ohlcv_dir = os.path.join(data_dir, "ohlcv")

    X = np.full((T, N, len(FACTORS)), np.nan)
    n_ok = 0
    for j, tk in enumerate(tickers):
        if tk not in t2c:
            continue
        extract = fetch_extract(tk, t2c[tk], cache_dir)
        time.sleep(0.12)  # ~8 req/s, under SEC's 10/s limit
        if not extract:
            continue
        ft = factor_table(extract)
        if ft.empty:
            continue
        # point-in-time merge onto the daily calendar by filed date
        ft = ft.sort_values("filed")
        cal = pd.DataFrame({"date": dates})
        merged = pd.merge_asof(cal, ft, left_on="date", right_on="filed", direction="backward")

        # price-based ratios (need market cap = Close * shares)
        close = _load_close(ohlcv_dir, tk, dates)
        mktcap = close * merged["_shares"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            merged["earnings_yield"] = merged["_net_income"].to_numpy() / mktcap
            merged["sales_to_price"] = merged["_revenue"].to_numpy() / mktcap
            merged["book_to_market"] = merged["_equity"].to_numpy() / mktcap

        for fi, fac in enumerate(FACTORS):
            if fac in merged:
                X[:, j, fi] = merged[fac].to_numpy()
        n_ok += 1
        if n_ok % 50 == 0:
            print(f"  processed {n_ok} tickers ...")

    print(f"Built fundamentals for {n_ok}/{N} tickers.")
    # cross-sectional rank-normalization per date (Gaussian-rank -> z-score)
    Xz = _cross_sectional_zscore(X)
    out = os.path.join(data_dir, "fundamentals.npz")
    np.savez_compressed(out, X_fund=Xz.astype(np.float32),
                        feature_names=np.array([f"FUND_{f}" for f in FACTORS]))
    cov = np.mean(~np.isnan(X).all(axis=2))  # fraction of (date,stock) with any factor
    print(f"Saved {out}  shape={Xz.shape}  coverage={cov:.1%}")


def _load_close(ohlcv_dir: str, ticker: str, dates: pd.DatetimeIndex) -> np.ndarray:
    p = os.path.join(ohlcv_dir, f"{ticker}.parquet")
    if not os.path.isfile(p):
        return np.full(len(dates), np.nan)
    df = pd.read_parquet(p)
    s = df["Close"] if "Close" in df else df.iloc[:, 0]
    s.index = pd.to_datetime(s.index)
    return s.reindex(dates, method="ffill").to_numpy()


def _cross_sectional_zscore(X: np.ndarray) -> np.ndarray:
    out = np.full_like(X, np.nan)
    T, N, F = X.shape
    for t in range(T):
        for f in range(F):
            col = X[t, :, f]
            m = ~np.isnan(col)
            if m.sum() > 5:
                # winsorize 1/99 then z-score cross-sectionally
                v = col[m]
                lo, hi = np.percentile(v, [1, 99])
                v = np.clip(v, lo, hi)
                out[t, m, f] = (v - v.mean()) / (v.std() + 1e-9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/Stock_SP500_2018-01-01_2026-03-16")
    args = ap.parse_args()
    build(args.data_dir)


if __name__ == "__main__":
    main()
