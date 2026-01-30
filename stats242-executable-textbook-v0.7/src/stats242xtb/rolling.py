from __future__ import annotations
from typing import Any, Dict, Optional
import pandas as pd
from .regression import fit_ols, summarize

def rolling_ols(y: pd.Series, x: pd.Series, *, window: int, step: int = 1, cov_type: str = "HAC", cov_kwds: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    df = pd.DataFrame({"y": y, "x": x}).dropna().reset_index(drop=True)

    rows = []
    for end in range(window, len(df) + 1, step):
        sl = df.iloc[end-window:end]
        fit = fit_ols(sl["y"], sl["x"], add_const=True, cov_type=cov_type, cov_kwds=cov_kwds or {})
        s = summarize(fit, names=["const", "beta"], cov_type=cov_type, cov_kwds=cov_kwds or {})
        beta = s.params.get("beta", s.params.get("x1", float("nan")))
        beta_se = s.bse.get("beta", s.bse.get("x1", float("nan")))
        lo, hi = s.conf_int.get("beta", s.conf_int.get("x1", (float("nan"), float("nan"))))
        rows.append({"end": end, "beta": beta, "beta_se": beta_se, "beta_ci_lo": lo, "beta_ci_hi": hi, "r2": s.r2})
    return pd.DataFrame(rows)
