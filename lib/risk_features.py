"""Realized-vol / IVOL cross-sectional risk features (Tier C, price-only).

These are computed purely from the daily return panel, so they have full
coverage and no staleness (unlike the annual fundamentals). Documented
cross-sectional anomalies: low idiosyncratic vol outperforms (Ang et al. 2006),
low-beta outperforms (Frazzini-Pedersen 2014), high max-daily-return
underperforms / lottery effect (Bali et al. 2011).
"""
from __future__ import annotations

import numpy as np

FEATURES = ["rvol", "ivol", "beta", "maxret"]


def window_features(R: np.ndarray, m: np.ndarray) -> dict:
    """Per-stock risk features over one trailing window.

    R : [W, N] daily returns; m : [W] market (e.g. equal-weight) returns.
    Returns dict of [N] arrays: rvol, ivol, beta, maxret.
    """
    R = np.asarray(R, dtype=float)
    m = np.asarray(m, dtype=float).reshape(-1)
    mc = m - m.mean()
    var = float((mc ** 2).mean())

    rvol = R.std(axis=0)
    maxret = R.max(axis=0)
    Rc = R - R.mean(axis=0, keepdims=True)
    if var <= 1e-18:
        beta = np.zeros(R.shape[1])
    else:
        beta = (Rc * mc.reshape(-1, 1)).mean(axis=0) / var
    resid = Rc - beta.reshape(1, -1) * mc.reshape(-1, 1)
    ivol = resid.std(axis=0)
    return {"rvol": rvol, "ivol": ivol, "beta": beta, "maxret": maxret}
