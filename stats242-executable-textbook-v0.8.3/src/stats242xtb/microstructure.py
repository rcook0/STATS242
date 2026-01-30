from __future__ import annotations
import numpy as np
import pandas as pd

def roll_spread_from_trades(trade_prices: pd.Series) -> dict:
    p = pd.to_numeric(trade_prices, errors="coerce").dropna()
    dp = p.diff().dropna()
    if len(dp) < 3:
        raise ValueError("Need more trade points for Roll estimator.")
    cov1 = float(np.cov(dp[1:].to_numpy(), dp[:-1].to_numpy(), bias=False)[0, 1])
    s = float("nan") if cov1 >= 0 else float(2.0 * np.sqrt(-cov1))
    return {"lag1_autocov": cov1, "roll_spread": s}

def tick_rule_sign(trade_prices: pd.Series) -> pd.Series:
    p = pd.to_numeric(trade_prices, errors="coerce").dropna()
    dp = p.diff()
    sign = dp.apply(lambda v: 1 if v > 0 else (-1 if v < 0 else 0))
    sign = sign.replace(0, np.nan).ffill().fillna(0).astype(int)
    return sign



def quoted_spread(quotes: pd.DataFrame) -> pd.Series:
    bid = pd.to_numeric(quotes["bid"], errors="coerce")
    ask = pd.to_numeric(quotes["ask"], errors="coerce")
    spr = (ask - bid)
    spr.name = "quoted_spread"
    return spr

def midprice(quotes: pd.DataFrame) -> pd.Series:
    bid = pd.to_numeric(quotes["bid"], errors="coerce")
    ask = pd.to_numeric(quotes["ask"], errors="coerce")
    mid = (bid + ask) / 2.0
    mid.name = "mid"
    return mid

def effective_spread(trades: pd.DataFrame, mid_at_trade: pd.Series) -> pd.Series:
    p = pd.to_numeric(trades["price"], errors="coerce")
    mid = pd.to_numeric(mid_at_trade, errors="coerce")
    eff = 2.0 * (p - mid).abs()
    eff.name = "effective_spread"
    return eff

def realized_spread(trades: pd.DataFrame, mid_future: pd.Series, sign: pd.Series) -> pd.Series:
    p = pd.to_numeric(trades["price"], errors="coerce")
    mf = pd.to_numeric(mid_future, errors="coerce")
    s = pd.to_numeric(sign, errors="coerce")
    rs = 2.0 * s * (p - mf)
    rs.name = "realized_spread"
    return rs

def realized_variance(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) < 3:
        return float("nan")
    r = np.diff(np.log(x.to_numpy()))
    return float(np.sum(r * r))

def signature_plot(mid: pd.Series, grids: list[str]) -> pd.Series:
    out = {}
    m = pd.to_numeric(mid, errors="coerce").dropna()
    for g in grids:
        try:
            s = m.resample(g).last().dropna()
        except Exception:
            continue
        out[g] = realized_variance(s)
    return pd.Series(out, name="realized_variance")

def kyle_lambda(trades: pd.DataFrame, quotes: pd.DataFrame, horizon: str = "10s") -> pd.DataFrame:
    t = trades.copy()
    if "size" not in t.columns:
        t["size"] = 1.0
    p = pd.to_numeric(t["price"], errors="coerce").dropna()
    t = t.loc[p.index].copy()
    t["price"] = p

    q = quotes.copy()
    q["mid"] = midprice(q)
    q = q[["mid"]].dropna()

    mid_t = q["mid"].reindex(t.index, method="ffill")
    mid_f = q["mid"].reindex(t.index + pd.Timedelta(horizon), method="ffill")

    sign = tick_rule_sign(t["price"])
    x = sign * pd.to_numeric(t["size"], errors="coerce").fillna(0.0)
    y = (mid_f - mid_t)

    df = pd.DataFrame({"x_signed_size": x, "y_dmid": y, "sign": sign}).dropna()
    return df

def queue_birth_death_rates(lob: pd.DataFrame) -> dict:
    df = lob.copy()
    dt = pd.to_datetime(df["ts"], utc=True, errors="coerce").diff().dt.total_seconds().dropna()
    total_time = float(dt.sum()) if len(dt) else float("nan")

    out = {"total_time_s": total_time}
    for side, col in [("bid", "bid_size"), ("ask", "ask_size")]:
        q = pd.to_numeric(df[col], errors="coerce")
        dq = q.diff().iloc[1:]
        pos = dq.clip(lower=0).fillna(0.0)
        neg = (-dq.clip(upper=0)).fillna(0.0)
        lam = float(pos.sum() / total_time) if total_time and total_time > 0 else float("nan")
        mu = float(neg.sum() / total_time) if total_time and total_time > 0 else float("nan")
        out[f"{side}_lambda_arrival_per_s"] = lam
        out[f"{side}_mu_departure_per_s"] = mu
        out[f"{side}_rho_lambda_over_mu"] = float(lam / mu) if mu and mu > 0 else float("nan")
    return out

def bd_depletion_prob(q0: float, lam: float, mu: float, horizon_s: float, sims: int = 2000, rng_seed: int = 123) -> float:
    if not np.isfinite(q0) or q0 <= 0:
        return 1.0
    if not np.isfinite(lam) or not np.isfinite(mu) or lam < 0 or mu < 0:
        return float("nan")
    if horizon_s <= 0:
        return 0.0
    q0i = int(max(1, round(float(q0))))
    rng = np.random.default_rng(rng_seed)
    hits = 0
    rate = lam + mu
    if rate <= 0:
        return 0.0
    for _ in range(int(sims)):
        t = 0.0
        q = q0i
        while t < horizon_s and q > 0:
            t += float(rng.exponential(1.0 / rate))
            if t >= horizon_s:
                break
            if rng.random() < (lam / rate if rate > 0 else 0.0):
                q += 1
            else:
                q -= 1
        if q <= 0 and t <= horizon_s:
            hits += 1
    return float(hits / float(sims))
