from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm

@dataclass
class RegressionSummary:
    params: Dict[str, float]
    bse: Dict[str, float]
    tvalues: Dict[str, float]
    pvalues: Dict[str, float]
    conf_int: Dict[str, tuple[float, float]]
    r2: float
    nobs: int
    cov_type: str
    cov_kwds: Dict[str, Any]

def _as_2d(X: Any) -> np.ndarray:
    if isinstance(X, pd.Series):
        return X.to_numpy().reshape(-1, 1)
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X

def fit_ols(y: Any, X: Any, *, add_const: bool = True, cov_type: str = "OLS", cov_kwds: Optional[Dict[str, Any]] = None):
    yv = pd.to_numeric(pd.Series(y), errors="coerce")
    X_arr = _as_2d(X)
    # Critical: align indices so concat() does not create all-NaN rows when y has a non-0-based index
    Xv = pd.DataFrame(X_arr, index=yv.index)
    if len(Xv) != len(yv):
        raise ValueError(f"fit_ols: length mismatch: len(y)={len(yv)} len(X)={len(Xv)}")
    df = pd.concat([yv.rename("y"), Xv], axis=1).dropna()
    y2 = df.iloc[:, 0].to_numpy()
    X2 = df.iloc[:, 1:].to_numpy()
    if add_const:
        X2 = sm.add_constant(X2, has_constant="add")
    base = sm.OLS(y2, X2).fit()
    cov_kwds = cov_kwds or {}

    if cov_type.upper() == "OLS":
        return base

    ct = cov_type.upper()
    if ct == "HAC":
        return base.get_robustcov_results(cov_type="HAC", **cov_kwds)
    return base.get_robustcov_results(cov_type=ct, **cov_kwds)

def summarize(fit, *, names: list[str] | None = None, cov_type: str = "OLS", cov_kwds: Optional[Dict[str, Any]] = None) -> RegressionSummary:
    cov_kwds = cov_kwds or {}
    params = fit.params
    bse = fit.bse
    tvalues = fit.tvalues
    pvalues = fit.pvalues
    ci = fit.conf_int()

    if names is None:
        names = ["const"] + [f"x{i}" for i in range(1, len(params))]

    def _to_dict(arr) -> Dict[str, float]:
        return {names[i]: float(arr[i]) for i in range(len(arr))}

    conf = {names[i]: (float(ci[i, 0]), float(ci[i, 1])) for i in range(ci.shape[0])}

    return RegressionSummary(
        params=_to_dict(params),
        bse=_to_dict(bse),
        tvalues=_to_dict(tvalues),
        pvalues=_to_dict(pvalues),
        conf_int=conf,
        r2=float(getattr(fit, "rsquared", float("nan"))),
        nobs=int(getattr(fit, "nobs", 0)),
        cov_type=str(cov_type),
        cov_kwds=dict(cov_kwds),
    )

def summary_md(s: RegressionSummary) -> str:
    lines = []
    lines.append("| term | coef | se | t | p | 95% CI |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for k in s.params.keys():
        lo, hi = s.conf_int[k]
        lines.append(f"| {k} | {s.params[k]:.6g} | {s.bse[k]:.6g} | {s.tvalues[k]:.4g} | {s.pvalues[k]:.4g} | [{lo:.6g}, {hi:.6g}] |")
    lines.append("")
    lines.append(f"- R²: {s.r2:.4g}")
    lines.append(f"- n: {s.nobs}")
    lines.append(f"- covariance: **{s.cov_type}** {s.cov_kwds if s.cov_kwds else ''}")
    return "\n".join(lines)
