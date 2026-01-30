from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

from ..util import ensure_dir, relpath
from ..provenance import write_run_meta
from ..report import write_report
from ..model_choices import model_card
from .. import plots
from ..regression import fit_ols, summarize

def _md_imgs(out_dir: str, paths: list[str]) -> str:
    return "\n".join([f"![]({relpath(out_dir, p)})" for p in paths])

def principia_ch04_inference(*, out_dir: str, n: int = 1000, sims: int = 500, phi: float = 0.4, alpha_true: float = 0.0, beta_true: float = 1.0, cov_lags: int = 5) -> tuple[str, str]:
    ensure_dir(out_dir)
    write_run_meta(out_dir=out_dir, chapter="principia_ch04_inference", params={"n": n, "sims": sims, "phi": phi, "alpha_true": alpha_true, "beta_true": beta_true, "cov_lags": cov_lags}, inputs={})
    fig_dir = ensure_dir(os.path.join(out_dir, "figures"))
    art_dir = ensure_dir(os.path.join(out_dir, "artifacts"))

    rng = np.random.default_rng(0)

    t_alpha_ols, t_alpha_hac = [], []
    p_alpha_ols, p_alpha_hac = [], []

    for _ in range(sims):
        x = rng.normal(size=n)
        eps = np.zeros(n)
        u = rng.normal(size=n)
        for i in range(1, n):
            eps[i] = phi * eps[i-1] + u[i]
        y = alpha_true + beta_true * x + eps

        fit_plain = fit_ols(y, x, cov_type="OLS")
        s_plain = summarize(fit_plain, names=["alpha","beta"], cov_type="OLS")
        t_alpha_ols.append(s_plain.tvalues["alpha"])
        p_alpha_ols.append(s_plain.pvalues["alpha"])

        fit_hac = fit_ols(y, x, cov_type="HAC", cov_kwds={"maxlags": int(cov_lags), "kernel": "bartlett", "use_correction": True})
        s_hac = summarize(fit_hac, names=["alpha","beta"], cov_type="HAC", cov_kwds={"maxlags": int(cov_lags), "kernel": "bartlett", "use_correction": True})
        t_alpha_hac.append(s_hac.tvalues["alpha"])
        p_alpha_hac.append(s_hac.pvalues["alpha"])

    t_alpha_ols = pd.Series(t_alpha_ols, name="t_alpha_ols")
    t_alpha_hac = pd.Series(t_alpha_hac, name="t_alpha_hac")
    p_alpha_ols = pd.Series(p_alpha_ols, name="p_alpha_ols")
    p_alpha_hac = pd.Series(p_alpha_hac, name="p_alpha_hac")

    fpr_ols = float((p_alpha_ols < 0.05).mean())
    fpr_hac = float((p_alpha_hac < 0.05).mean())

    figs = [
        plots.plot_hist(t_alpha_ols, "t(alpha) under OLS (serial corr errors)", os.path.join(fig_dir, "t_alpha_ols.png"), bins=60),
        plots.plot_hist(t_alpha_hac, "t(alpha) under HAC (Newey–West)", os.path.join(fig_dir, "t_alpha_hac.png"), bins=60),
        plots.plot_hist(p_alpha_ols, "p(alpha) under OLS", os.path.join(fig_dir, "p_alpha_ols.png"), bins=60),
        plots.plot_hist(p_alpha_hac, "p(alpha) under HAC", os.path.join(fig_dir, "p_alpha_hac.png"), bins=60),
    ]

    diag = {
        "n": int(n),
        "sims": int(sims),
        "phi": float(phi),
        "alpha_true": float(alpha_true),
        "beta_true": float(beta_true),
        "cov_lags": int(cov_lags),
        "fpr_alpha_p_lt_0_05_ols": fpr_ols,
        "fpr_alpha_p_lt_0_05_hac": fpr_hac,
    }
    with open(os.path.join(art_dir, "principia_ch04_summary.json"), "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    mc = model_card("Principia — Ch04 Inference", {
        "DGP": "y_t = α + β x_t + ε_t",
        "Error process": "ε_t = φ ε_{t-1} + u_t, u_t ~ N(0,1)",
        "Tested claim": "α = 0 (false-positive rate at 5%)",
        "Inference": f"Compare OLS vs HAC (Newey–West) with maxlags={cov_lags}",
    }, notes=[
        "Key idea: HAC targets the long-run variance (spectral density at zero) rather than assuming iid residuals.",
        "With autocorrelated residuals, naive OLS SEs often understate uncertainty.",
    ])

    math = """<div class="note">
In HAC/Newey–West, we estimate a long-run variance of the score:
\[
\widehat{\Omega} = \Gamma_0 + \sum_{\ell=1}^L w_\ell(\Gamma_\ell + \Gamma_\ell^\top),
\]
where $\Gamma_\ell$ is the lag-$\ell$ covariance of the moment condition and $w_\ell$ is a kernel weight (Bartlett by default).
</div>
"""

    sections = [
        ("Model card", mc),
        ("Why OLS can lie", math),
        ("Simulation outputs", _md_imgs(out_dir, figs)),
        ("False-positive rate (α)", f"- OLS: **{fpr_ols:.3f}**\n- HAC: **{fpr_hac:.3f}**\n\n(ideal under correct size: ~0.05)"),
        ("Artifacts", f"- `artifacts/principia_ch04_summary.json`"),
    ]
    return write_report(out_dir, "Principia — Chapter 04: Inference under serial correlation", sections, mathjax=True)
