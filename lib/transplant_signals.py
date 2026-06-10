"""Causal cross-sectional 'architecture-transplant' signals for the RQ2 signal search.

Each signal has the convention used by resid_mom: f(daily_y, d) -> [N], computed using
ONLY daily_y[:d] (strictly before the decision day d). This guarantees no look-ahead.
"""
from __future__ import annotations

import numpy as np


def _trailing_returns(daily_y: np.ndarray, d: int, window: int) -> np.ndarray | None:
    """Trailing [window, N] daily returns ending at (and excluding) decision day d.

    Returns None if there is not enough history (d - window < 0). NaNs -> 0.0.
    """
    a = d - window
    if a < 0:
        return None
    return np.nan_to_num(daily_y[a:d], nan=0.0)


def xz(v: np.ndarray) -> np.ndarray:
    """Cross-sectional z-score (mean 0, std 1). Constant/empty -> zeros."""
    v = np.asarray(v, dtype=float)
    m = np.nanmean(v)
    s = np.nanstd(v)
    return (v - m) / s if s > 1e-12 else np.zeros_like(v)


def peer_graph_signal(daily_y: np.ndarray, d: int,
                      corr_window: int = 120, k: int = 15,
                      mom_window: int = 21) -> np.ndarray:
    """Causal peer-momentum (relational-graph transplant).

    Builds a trailing correlation graph from daily_y[d-corr_window:d] (past only),
    selects each stock's top-k |corr| peers, and returns the correlation-weighted
    mean of those peers' own recent momentum (cumulative return over the last
    mom_window days, also past only). Lead-lag / connected-stock momentum.
    """
    R = _trailing_returns(daily_y, d, corr_window)
    own_R = _trailing_returns(daily_y, d, mom_window)
    if R is None or own_R is None:
        return np.full(daily_y.shape[1], np.nan)
    N = R.shape[1]
    C = np.corrcoef(R, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)
    own_mom = own_R.sum(axis=0)  # [N] cumulative past momentum (guarded, strictly < d)
    out = np.full(N, np.nan)
    kk = min(k, N - 1)
    for i in range(N):
        order = np.argsort(np.abs(C[i]))[-kk:]   # top-k peers by |corr|
        wts = C[i, order]                        # signed correlation weights
        denom = np.abs(wts).sum()
        if denom > 1e-12:
            out[i] = float((wts * own_mom[order]).sum() / denom)
    return out


def filtered_trend_signal(daily_y: np.ndarray, d: int,
                          window: int = 120, halflife: float = 10.0,
                          mom_window: int = 21) -> np.ndarray:
    """Causal filtered-trend momentum (wavelet / dual-frequency transplant).

    Extracts the low-frequency (trend) component of each stock's return series via a
    one-sided causal EWMA over daily_y[d-window:d] (strictly past), then takes the mean
    of the most recent mom_window smoothed (denoised) returns.
    """
    R = _trailing_returns(daily_y, d, window)
    if R is None:
        return np.full(daily_y.shape[1], np.nan)
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    sm = np.empty_like(R)
    sm[0] = R[0]
    for t in range(1, R.shape[0]):
        sm[t] = alpha * R[t] + (1.0 - alpha) * sm[t - 1]
    take = min(mom_window, R.shape[0])  # guard: never silently use more than `window` rows
    return sm[-take:].mean(axis=0)


# ── Pre-registered bench (Lote 2): established factors, all causal f(daily_y, d) -> [N] ──

def _market_beta(R: np.ndarray):
    """Per-stock beta vs the equal-weight market over R[days, N]. Returns (mkt[days], beta[N])."""
    mkt = R.mean(axis=1)
    mc = mkt - mkt.mean()
    var = float((mc ** 2).mean()) + 1e-12
    Rc = R - R.mean(axis=0, keepdims=True)
    beta = (Rc * mc.reshape(-1, 1)).mean(axis=0) / var
    return mkt, beta


def low_ivol_signal(daily_y: np.ndarray, d: int, window: int = 60) -> np.ndarray:
    """Low idiosyncratic volatility (Ang et al. 2006): -std of residual-vs-market returns."""
    R = _trailing_returns(daily_y, d, window)
    if R is None:
        return np.full(daily_y.shape[1], np.nan)
    mkt, beta = _market_beta(R)
    resid = R - np.outer(mkt, beta)
    return -resid.std(axis=0)


def bab_signal(daily_y: np.ndarray, d: int, window: int = 120) -> np.ndarray:
    """Betting-against-beta (Frazzini-Pedersen 2014): -market beta (low beta preferred)."""
    R = _trailing_returns(daily_y, d, window)
    if R is None:
        return np.full(daily_y.shape[1], np.nan)
    _, beta = _market_beta(R)
    return -beta


def volmanaged_momentum_signal(daily_y: np.ndarray, d: int,
                               mom_lo: int = 252, mom_hi: int = 21,
                               vol_window: int = 126) -> np.ndarray:
    """Volatility-managed momentum (Barroso-Santa-Clara 2015): 12-1 momentum / recent vol."""
    momR = _trailing_returns(daily_y, d, mom_lo)
    volR = _trailing_returns(daily_y, d, vol_window)
    if momR is None or volR is None:
        return np.full(daily_y.shape[1], np.nan)
    mom = momR[:-mom_hi].sum(axis=0) if mom_hi > 0 else momR.sum(axis=0)
    vol = volR.std(axis=0)
    return mom / (vol + 1e-9)


def fiftytwo_week_high_signal(daily_y: np.ndarray, d: int, window: int = 252) -> np.ndarray:
    """Proximity to the 52-week high (George-Hwang 2004): price / trailing max price."""
    R = _trailing_returns(daily_y, d, window)
    if R is None:
        return np.full(daily_y.shape[1], np.nan)
    P = np.cumprod(1.0 + R, axis=0)
    return P[-1] / (P.max(axis=0) + 1e-12)


def ts_momentum_signal(daily_y: np.ndarray, d: int,
                       horizons: tuple = (21, 63, 126, 252)) -> np.ndarray:
    """Time-series/trend momentum (Moskowitz-Ooi-Pedersen 2012): mean sign of cum return."""
    N = daily_y.shape[1]
    if d - max(horizons) < 0:
        return np.full(N, np.nan)
    sigs = [np.sign(_trailing_returns(daily_y, d, h).sum(axis=0)) for h in horizons]
    return np.mean(sigs, axis=0)


def seasonality_signal(daily_y: np.ndarray, d: int, dates, min_obs: int = 20) -> np.ndarray:
    """Same-calendar-month historical mean return (Heston-Sadka 2008). Causal: uses dates[:d]."""
    N = daily_y.shape[1]
    if d < 252:
        return np.full(N, np.nan)
    m = dates[d].month
    months = np.array([dt.month for dt in dates[:d]])
    mask = months == m
    if mask.sum() < min_obs:
        return np.full(N, np.nan)
    R = np.nan_to_num(daily_y[:d][mask], nan=0.0)
    return R.mean(axis=0)


# ── Architecture completion (round 2): the other wavelet band + the asymmetric graph ──

def _causal_ewma(R: np.ndarray, halflife: float) -> np.ndarray:
    """One-sided (past-weighted) EWMA over the time axis of R[window, N]."""
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    sm = np.empty_like(R)
    sm[0] = R[0]
    for t in range(1, R.shape[0]):
        sm[t] = alpha * R[t] + (1.0 - alpha) * sm[t - 1]
    return sm


def high_freq_reversal_signal(daily_y: np.ndarray, d: int, window: int = 120,
                              halflife: float = 10.0, rev_window: int = 5) -> np.ndarray:
    """Reversal on the HIGH-frequency band (the other half of the wavelet transplant).

    Splits the trailing return series into a causal EWMA trend (low freq) and the residual
    (high freq = microstructure noise / short-term moves), then bets on short-term reversal of
    that residual: -sum of the most recent rev_window residual returns. Causal: daily_y[:d].
    """
    R = _trailing_returns(daily_y, d, window)
    if R is None:
        return np.full(daily_y.shape[1], np.nan)
    resid = R - _causal_ewma(R, halflife)   # high-frequency component
    take = min(rev_window, R.shape[0])
    return -resid[-take:].sum(axis=0)


def lead_lag_signal(daily_y: np.ndarray, d: int, window: int = 120, k: int = 15,
                    lead_window: int = 5) -> np.ndarray:
    """Directional lead-lag (asymmetric spatial-attention transplant).

    Estimates the lag-1 cross-correlation L[i, j] = mean_t z_i(t) z_j(t-1) over the trailing
    window: stocks j whose YESTERDAY return co-moves with i's TODAY return are i's 'leaders'.
    The signal for i is the lead-weighted recent return of its top-k leaders. Unlike the
    symmetric correlation peer signal, this is directional. Causal: only daily_y[:d].
    """
    R = _trailing_returns(daily_y, d, window)
    if R is None:
        return np.full(daily_y.shape[1], np.nan)
    N = R.shape[1]
    mu = R.mean(axis=0, keepdims=True)
    sd = R.std(axis=0, keepdims=True) + 1e-12
    Z = (R - mu) / sd                                   # [W, N] time-standardized
    L = (Z[1:].T @ Z[:-1]) / (Z.shape[0] - 1)           # [N, N]: j(t-1) leads i(t)
    L = np.nan_to_num(L, nan=0.0)
    np.fill_diagonal(L, 0.0)
    lead_mom = np.nan_to_num(daily_y[d - lead_window:d], nan=0.0).sum(axis=0)  # leaders' recent move
    out = np.full(N, np.nan)
    kk = min(k, N - 1)
    for i in range(N):
        order = np.argsort(np.abs(L[i]))[-kk:]
        wts = L[i, order]
        denom = np.abs(wts).sum()
        if denom > 1e-12:
            out[i] = float((wts * lead_mom[order]).sum() / denom)
    return out


def sector_peer_momentum_signal(daily_y: np.ndarray, d: int, sector_ids,
                                mom_window: int = 21) -> np.ndarray:
    """Industry/sector-peer momentum (Moskowitz-Grinblatt 1999): for each stock, the mean
    recent momentum of the OTHER stocks in its sector. sector_ids is an [N] array of labels.
    The 'right' graph the Stockformer's learned graph approximates. Causal: daily_y[:d]."""
    own_R = _trailing_returns(daily_y, d, mom_window)
    if own_R is None:
        return np.full(daily_y.shape[1], np.nan)
    mom = own_R.sum(axis=0)                              # [N] each stock's recent momentum
    sector_ids = np.asarray(sector_ids)
    out = np.full(len(mom), np.nan)
    for s in np.unique(sector_ids):
        idx = np.where(sector_ids == s)[0]
        if len(idx) < 2:
            continue
        out[idx] = (mom[idx].sum() - mom[idx]) / (len(idx) - 1)  # peer mean, excluding self
    return out
