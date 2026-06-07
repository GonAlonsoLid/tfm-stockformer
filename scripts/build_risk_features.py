#!/usr/bin/env python3
"""Build realized-vol / IVOL risk features (Tier C) -> realized_vol.npz.

Price-only, full-coverage cross-sectional features computed from the daily
return panel over a trailing window (no look-ahead: day t uses returns < t).
Cross-sectionally winsorized + z-scored, aligned to the panel grid. Stored with
the X_fund/feature_names keys so data_panel.attach_fundamentals can load it.

Usage:
    python scripts/build_risk_features.py --window 60
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lib import data_panel as dp  # noqa: E402
from lib import risk_features as rf  # noqa: E402


def _xs_zscore(X):
    out = np.full_like(X, np.nan)
    T, N, F = X.shape
    for t in range(T):
        for f in range(F):
            col = X[t, :, f]
            mask = ~np.isnan(col)
            if mask.sum() > 5:
                v = col[mask]
                lo, hi = np.percentile(v, [1, 99])
                v = np.clip(v, lo, hi)
                out[t, mask, f] = (v - v.mean()) / (v.std() + 1e-9)
    return out


def build(data_dir: str, window: int) -> None:
    panel = dp.load_panel(data_dir)
    y = panel.y                       # [T, N] daily returns (forward-1d proxy)
    market = np.nanmean(y, axis=1)    # equal-weight market
    T, N = y.shape
    F = len(rf.FEATURES)
    X = np.full((T, N, F), np.nan)
    for t in range(window, T):
        R = y[t - window:t]
        m = market[t - window:t]
        if np.isnan(R).all():
            continue
        feats = rf.window_features(np.nan_to_num(R, nan=0.0), np.nan_to_num(m, nan=0.0))
        for fi, name in enumerate(rf.FEATURES):
            X[t, :, fi] = feats[name]
    Xz = _xs_zscore(X)
    out = os.path.join(data_dir, "realized_vol.npz")
    np.savez_compressed(out, X_fund=Xz.astype(np.float32),
                        feature_names=np.array([f"RISK_{n}" for n in rf.FEATURES]))
    cov = np.mean(~np.isnan(X).all(axis=2))
    print(f"Saved {out}  shape={Xz.shape}  features={['RISK_'+n for n in rf.FEATURES]}  coverage={cov:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/Stock_SP500_2018-01-01_2026-03-16")
    ap.add_argument("--window", type=int, default=60)
    args = ap.parse_args()
    build(args.data_dir, args.window)


if __name__ == "__main__":
    main()
