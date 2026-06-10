#!/usr/bin/env python3
"""Fetch GICS-style sector per ticker via yfinance -> data/<dir>/sector_map.json.

Best-effort: failures/missing become "Unknown". Used by the sector-peer momentum
signal (industry momentum). Run once; the result is cached.
"""
import json
import os
import sys

import yfinance as yf

DATA_DIR = "data/Stock_SP500_2018-01-01_2026-03-16"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, DATA_DIR, "sector_map.json")


def main():
    with open(os.path.join(ROOT, DATA_DIR, "tickers.txt")) as f:
        tickers = [ln.strip() for ln in f if ln.strip()]
    out = {}
    for i, t in enumerate(tickers):
        sec = "Unknown"
        try:
            info = yf.Ticker(t).info
            sec = info.get("sector") or "Unknown"
        except Exception:
            sec = "Unknown"
        out[t] = sec
        if i % 25 == 0:
            n_known = sum(1 for v in out.values() if v != "Unknown")
            print(f"  {i:3d}/{len(tickers)}  {t:6s} -> {sec}   ({n_known} known)", flush=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=0)
    n_known = sum(1 for v in out.values() if v != "Unknown")
    n_sectors = len(set(v for v in out.values() if v != "Unknown"))
    print(f"\nSaved {OUT}: {n_known}/{len(tickers)} known, {n_sectors} distinct sectors")


if __name__ == "__main__":
    main()
