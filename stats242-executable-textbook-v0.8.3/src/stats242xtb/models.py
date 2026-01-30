from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def hedge_ratio_ols(y: pd.Series, x: pd.Series) -> float:
    y = pd.to_numeric(y, errors="coerce").dropna()
    x = pd.to_numeric(x, errors="coerce").dropna()
    idx = y.index.intersection(x.index)
    yv = y.loc[idx].to_numpy()
    xv = x.loc[idx].to_numpy()
    X = np.vstack([xv, np.ones_like(xv)]).T
    beta, alpha = np.linalg.lstsq(X, yv, rcond=None)[0]
    return float(beta)

def fit_ar1(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(s) < 30:
        raise ValueError("Need >= 30 points to fit AR(1) reasonably.")
    x = s[:-1]
    y = s[1:]
    X = np.vstack([x, np.ones_like(x)]).T
    phi, c = np.linalg.lstsq(X, y, rcond=None)[0]
    mu = float("nan") if abs(1 - phi) < 1e-8 else float(c / (1 - phi))
    half_life = float(np.log(0.5) / np.log(phi)) if 0 < phi < 1 else float("nan")
    return {"phi": float(phi), "c": float(c), "mu": mu, "half_life_steps": half_life}

def acd11_fit(durations: np.ndarray) -> dict:
    """Fit ACD(1,1) with exponential innovations via MLE."""
    x = np.asarray(durations, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if len(x) < 200:
        raise ValueError("Need at least ~200 durations for a meaningful ACD fit.")

    x_mean = float(np.mean(x))
    theta0 = np.array([0.1 * x_mean, 0.1, 0.8])

    def nll(theta):
        omega, alpha, beta = theta
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 0.999:
            return 1e30
        psi = np.empty_like(x)
        psi[0] = x_mean
        for i in range(1, len(x)):
            psi[i] = omega + alpha * x[i-1] + beta * psi[i-1]
            if psi[i] <= 0:
                return 1e30
        ll = -np.sum(np.log(psi) + x / psi)
        return -float(ll)

    bounds = [(1e-12, None), (0.0, 0.999), (0.0, 0.999)]
    res = minimize(nll, theta0, method="L-BFGS-B", bounds=bounds)
    omega, alpha, beta = res.x.tolist()

    psi = np.empty_like(x)
    psi[0] = x_mean
    for i in range(1, len(x)):
        psi[i] = omega + alpha * x[i-1] + beta * psi[i-1]
    e = x / psi
    return {
        "omega": float(omega),
        "alpha": float(alpha),
        "beta": float(beta),
        "alpha_plus_beta": float(alpha + beta),
        "success": bool(res.success),
        "message": str(res.message),
        "n": int(len(x)),
        "mean_duration": float(x_mean),
        "mean_e": float(np.mean(e)),
        "var_e": float(np.var(e, ddof=1)),
    }

def long_only_min_var_weights(returns: pd.DataFrame) -> dict:
    R = returns.dropna()
    if R.shape[0] < 50:
        raise ValueError("Need >= 50 return rows for MV demo.")
    cov = np.cov(R.to_numpy(), rowvar=False)
    n = cov.shape[0]

    def obj(w):
        return float(w @ cov @ w)

    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500})
    w = res.x
    return {"success": bool(res.success), "message": str(res.message), "weights": {c: float(w[i]) for i, c in enumerate(R.columns)}}
