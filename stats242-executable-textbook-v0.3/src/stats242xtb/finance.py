from __future__ import annotations
import numpy as np
import pandas as pd

def log_returns_from_close(close: pd.Series) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    r = np.log(close).diff()
    return r.dropna()

def simple_returns_from_close(close: pd.Series) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    r = close.pct_change()
    return r.dropna()

def to_period_returns(df: pd.DataFrame, ts_col: str, price_cols: list[str], rule: str = "1D") -> pd.DataFrame:
    """Resample close prices to a regular grid, then compute log returns."""
    x = df[[ts_col] + price_cols].copy()
    x = x.set_index(ts_col).sort_index()
    x = x.resample(rule).last().dropna(how="any")
    out = {}
    for c in price_cols:
        out[c] = log_returns_from_close(x[c])
    return pd.DataFrame(out).dropna()

def zscore(x: pd.Series, win: int = 60) -> pd.Series:
    m = x.rolling(win).mean()
    s = x.rolling(win).std(ddof=0)
    return (x - m) / s
