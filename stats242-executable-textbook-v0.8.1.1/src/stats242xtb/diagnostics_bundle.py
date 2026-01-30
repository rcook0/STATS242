from __future__ import annotations
import os
import json
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
import pandas as pd
import statsmodels.api as sm

from .util import ensure_dir
from . import diagnostics as diag
from . import plots

def run_residual_diagnostics(
    resid: pd.Series,
    out_dir: str,
    *,
    lags: int = 20,
    X_for_white: Optional[np.ndarray] = None,
    prefix: str = "resid",
) -> Tuple[Dict[str, Any], List[str]]:
    ensure_dir(out_dir)
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))
    art_dir = ensure_dir(os.path.join(out_dir, "artifacts"))

    r = pd.to_numeric(pd.Series(resid), errors="coerce").dropna()
    figs: List[str] = []

    figs.append(plots.plot_series(r, "Residuals (time order)", os.path.join(fig_dir, f"{prefix}_series.png")))
    figs.append(plots.plot_hist(r, "Residuals histogram (density)", os.path.join(fig_dir, f"{prefix}_hist.png"), bins=60))
    figs.append(plots.plot_qq(r, "Residuals QQ plot vs Normal", os.path.join(fig_dir, f"{prefix}_qq.png")))
    acf = diag.acf_simple(r, max_lag=min(lags, 50))
    figs.append(plots.plot_acf_bar(acf, "Residual ACF", os.path.join(fig_dir, f"{prefix}_acf.png")))

    out: Dict[str, Any] = {}
    out["jarque_bera"] = diag.jarque_bera(r)
    out["ljung_box_resid"] = diag.ljung_box(r, lags=lags).to_dict(orient="index")
    out["ljung_box_resid_sq"] = diag.ljung_box(r**2, lags=lags).to_dict(orient="index")

    try:
        from statsmodels.stats.diagnostic import het_arch
        arch = het_arch(r.to_numpy(), nlags=min(lags, 20))
        out["arch_lm"] = {"stat": float(arch[0]), "p_value": float(arch[1]), "nlags": int(min(lags, 20))}
    except Exception as e:
        out["arch_lm"] = {"error": str(e)}

    if X_for_white is not None:
        try:
            from statsmodels.stats.diagnostic import het_white
            X = np.asarray(X_for_white)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            X = sm.add_constant(X, has_constant="add")
            white = het_white(r.to_numpy(), X)
            out["white_test"] = {"stat": float(white[0]), "p_value": float(white[1])}
        except Exception as e:
            out["white_test"] = {"error": str(e)}

    diag_path = os.path.join(art_dir, f"{prefix}_diagnostics.json")
    with open(diag_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    out["_paths"] = {"diagnostics_json": diag_path}
    return out, figs

def diagnostics_md(diag_dict: Dict[str, Any], out_dir: str, figs: List[str]) -> str:
    import pandas as pd
    from .util import relpath
    lines = []
    lines.append("**Residual diagnostics bundle**")
    lines.append("")
    for p in figs:
        lines.append(f"![]({relpath(out_dir, p)})")
    lines.append("")
    jb = diag_dict.get("jarque_bera", {})
    lines.append(f"- JB p-value: {jb.get('p_value')}")
    arch = diag_dict.get("arch_lm", {})
    if isinstance(arch, dict) and "p_value" in arch:
        lines.append(f"- ARCH LM p-value: {arch.get('p_value')}")
    white = diag_dict.get("white_test", {})
    if isinstance(white, dict) and "p_value" in white:
        lines.append(f"- White p-value: {white.get('p_value')}")
    lines.append("")
    try:
        lb = pd.DataFrame.from_dict(diag_dict.get("ljung_box_resid", {}), orient="index")
        lines.append("Ljung–Box (residuals)")
        lines.append("")
        lines.append(lb.head(10).to_string())
        lines.append("")
    except Exception:
        pass
    return "\n".join(lines)
