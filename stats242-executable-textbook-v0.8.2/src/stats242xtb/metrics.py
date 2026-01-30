from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

TRADING_DAYS = 252

def _to_series(x) -> pd.Series:
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)

def equity_curve(log_returns: pd.Series, *, start: float = 1.0) -> pd.Series:
    r = pd.to_numeric(log_returns, errors="coerce").fillna(0.0)
    eq = start * np.exp(r.cumsum())
    eq.name = "equity"
    return eq

def drawdown(equity: pd.Series) -> pd.Series:
    e = pd.to_numeric(equity, errors="coerce").astype(float)
    peak = e.cummax()
    dd = e / peak - 1.0
    dd.name = "drawdown"
    return dd

def max_drawdown(equity: pd.Series) -> float:
    dd = drawdown(equity)
    return float(dd.min()) if len(dd) else float("nan")

def perf_stats(log_returns: pd.Series, *, ann_factor: int = TRADING_DAYS) -> Dict[str, float]:
    r = pd.to_numeric(log_returns, errors="coerce").dropna()
    if len(r) == 0:
        return {"n": 0, "mean": float("nan"), "vol": float("nan"), "sharpe": float("nan"),
                "skew": float("nan"), "kurt": float("nan")}
    mu = float(r.mean())
    sig = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    sharpe = float((mu / sig) * np.sqrt(ann_factor)) if (sig and sig > 0) else float("nan")
    return {
        "n": int(len(r)),
        "mean": mu,
        "vol": sig,
        "sharpe": sharpe,
        "skew": float(stats.skew(r.to_numpy(), bias=False)) if len(r) > 2 else float("nan"),
        "kurt": float(stats.kurtosis(r.to_numpy(), fisher=True, bias=False)) if len(r) > 3 else float("nan"),
    }

def turnover_from_weights(W: pd.DataFrame) -> pd.Series:
    if W is None or len(W) == 0:
        return pd.Series(dtype=float)
    dW = W.diff().abs().sum(axis=1)
    # standard definition: half-L1 change
    t = 0.5 * dW
    t.iloc[0] = 0.0
    t.name = "turnover"
    return t

def apply_costs(log_returns: pd.Series, turnover: pd.Series, cost_bps: float, *, exposure_mult: float = 1.0) -> pd.Series:
    r = pd.to_numeric(log_returns, errors="coerce").fillna(0.0)
    t = pd.to_numeric(turnover, errors="coerce").reindex(r.index).fillna(0.0)
    c = (cost_bps * 1e-4) * exposure_mult * t
    out = (r - c)
    out.name = f"net_{cost_bps:g}bps"
    return out

def cost_sweep(log_returns: pd.Series, turnover: pd.Series, cost_bps_list: Sequence[float], *, exposure_mult: float = 1.0) -> Dict[str, Dict[str, float]]:
    res: Dict[str, Dict[str, float]] = {}
    for bps in cost_bps_list:
        net = apply_costs(log_returns, turnover, float(bps), exposure_mult=exposure_mult)
        res[str(float(bps))] = perf_stats(net)
    return res

def trades_from_position(pos: pd.Series, log_pnl: pd.Series) -> Dict[str, float]:
    """Derive trade-level stats from a {-1,0,1} position series and per-period PnL (log units).
    This is an approximation intended for course labs (not a broker-grade fill simulator).
    """
    p = pd.to_numeric(pos, errors="coerce").fillna(0.0)
    r = pd.to_numeric(log_pnl, errors="coerce").reindex(p.index).fillna(0.0)
    trades: List[float] = []
    holds: List[int] = []
    cur = 0.0
    h = 0
    in_trade = False

    for i in range(len(p)):
        if i == 0:
            prev = 0.0
        else:
            prev = float(p.iloc[i-1])
        now = float(p.iloc[i])

        if not in_trade and now != 0.0:
            in_trade = True
            cur = float(r.iloc[i])
            h = 1
        elif in_trade:
            # if position continues with same sign
            if now == prev and now != 0.0:
                cur += float(r.iloc[i])
                h += 1
            else:
                # trade ended or flipped
                trades.append(cur)
                holds.append(h)
                in_trade = False
                cur = 0.0
                h = 0
                if now != 0.0:
                    in_trade = True
                    cur = float(r.iloc[i])
                    h = 1

    if in_trade:
        trades.append(cur)
        holds.append(h)

    if len(trades) == 0:
        return {"n_trades": 0, "win_rate": float("nan"), "avg_trade": float("nan"),
                "profit_factor": float("nan"), "avg_hold": float("nan")}
    t = np.asarray(trades, dtype=float)
    wins = t[t > 0]
    losses = -t[t < 0]
    pf = float(wins.sum() / losses.sum()) if losses.sum() > 0 else float("inf")
    return {
        "n_trades": int(len(t)),
        "win_rate": float((t > 0).mean()),
        "avg_trade": float(t.mean()),
        "profit_factor": pf,
        "avg_hold": float(np.mean(holds)) if holds else float("nan"),
    }

def weight_equal(cols: Sequence[str]) -> pd.Series:
    n = len(cols)
    return pd.Series(np.ones(n) / n, index=list(cols))

def weight_inv_vol(Rw: pd.DataFrame) -> pd.Series:
    s = Rw.std(ddof=1)
    s = s.replace(0.0, np.nan)
    w = (1.0 / s).fillna(0.0)
    if w.sum() == 0:
        return weight_equal(Rw.columns)
    w = w / w.sum()
    return w

def weight_min_var_long_only(Rw: pd.DataFrame, *, shrink: float = 0.0) -> pd.Series:
    R = Rw.dropna()
    cols = list(Rw.columns)
    if R.shape[0] < 30:
        return weight_equal(cols)
    cov = np.cov(R.to_numpy(), rowvar=False)
    if shrink and shrink > 0:
        d = np.diag(np.diag(cov))
        cov = (1 - shrink) * cov + shrink * d
    n = cov.shape[0]

    def obj(w):
        return float(w @ cov @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500})
    w = res.x if res.success else w0
    return pd.Series(w, index=cols)

def weight_mean_var_long_only(Rw: pd.DataFrame, *, risk_aversion: float = 10.0, shrink: float = 0.0) -> pd.Series:
    R = Rw.dropna()
    cols = list(Rw.columns)
    if R.shape[0] < 30:
        return weight_equal(cols)
    mu = R.mean().to_numpy()
    cov = np.cov(R.to_numpy(), rowvar=False)
    if shrink and shrink > 0:
        d = np.diag(np.diag(cov))
        cov = (1 - shrink) * cov + shrink * d
    n = len(cols)

    # maximize mu'w - (gamma/2) w'cov w  <=> minimize -(mu'w) + (gamma/2) w'cov w
    def obj(w):
        return float(-(mu @ w) + 0.5 * risk_aversion * (w @ cov @ w))

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 700})
    w = res.x if res.success else w0
    return pd.Series(w, index=cols)

def weight_risk_parity_long_only(Rw: pd.DataFrame, *, shrink: float = 0.0, iters: int = 2000) -> pd.Series:
    R = Rw.dropna()
    cols = list(Rw.columns)
    if R.shape[0] < 30:
        return weight_equal(cols)
    cov = np.cov(R.to_numpy(), rowvar=False)
    if shrink and shrink > 0:
        d = np.diag(np.diag(cov))
        cov = (1 - shrink) * cov + shrink * d
    n = cov.shape[0]
    w = np.ones(n) / n

    # simple multiplicative update targeting equal risk contributions
    for _ in range(iters):
        port_var = float(w @ cov @ w)
        if port_var <= 0:
            break
        mrc = cov @ w                 # marginal risk contrib
        rc = w * mrc                  # total risk contrib
        target = port_var / n
        # avoid division by zero
        adj = np.where(rc > 1e-16, target / rc, 1.0)
        w = w * adj
        w = np.clip(w, 1e-12, None)
        w = w / w.sum()
    return pd.Series(w, index=cols)
