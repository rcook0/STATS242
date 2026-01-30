from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

def jarque_bera(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy()
    n = len(x)
    if n < 8:
        raise ValueError("Need at least ~8 points for JB to be meaningful.")
    sk = stats.skew(x, bias=False)
    ku = stats.kurtosis(x, fisher=False, bias=False)
    jb = n * (sk**2 / 6.0 + ((ku - 3.0)**2) / 24.0)
    p = 1.0 - stats.chi2.cdf(jb, df=2)
    return {"n": int(n), "skew": float(sk), "kurtosis": float(ku), "JB": float(jb), "p_value": float(p)}

def ljung_box(x: pd.Series, lags: int = 20) -> pd.DataFrame:
    x = pd.to_numeric(x, errors="coerce").dropna()
    return acorr_ljungbox(x, lags=lags, return_df=True)

def acf_simple(x: pd.Series, max_lag: int = 20) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce").dropna().to_numpy()
    x = x - x.mean()
    denom = float(np.dot(x, x))
    out = []
    for k in range(1, max_lag + 1):
        out.append(float(np.dot(x[k:], x[:-k])) / denom)
    return pd.Series(out, index=np.arange(1, max_lag + 1), name="acf")

def variance_ratio(x: pd.Series, q: int = 5) -> dict:
    """Lo–MacKinlay-ish variance ratio (simple implementation)."""
    r = pd.to_numeric(x, errors="coerce").dropna().to_numpy()
    if len(r) < 5*q + 5:
        raise ValueError("Not enough data for variance ratio.")
    # treat r as returns; VR compares var(sum_{i=1..q} r) / (q * var(r))
    r1 = r - r.mean()
    var1 = float(np.var(r1, ddof=1))
    rq = np.array([r[i-q+1:i+1].sum() for i in range(q-1, len(r))])
    rq = rq - rq.mean()
    varq = float(np.var(rq, ddof=1))
    vr = varq / (q * var1) if var1 > 0 else float("nan")
    return {"q": int(q), "vr": float(vr), "var1": float(var1), "varq": float(varq)}
