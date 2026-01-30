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
