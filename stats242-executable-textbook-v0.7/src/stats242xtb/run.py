from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd

from .util import ensure_dir, read_csv_ts, relpath
from .report import write_report
from .model_choices import model_card
from . import finance as fin
from . import diagnostics as diag
from . import microstructure as ms
from . import models
from . import plots

from .regression import fit_ols, summarize, summary_md
from .diagnostics_bundle import run_residual_diagnostics, diagnostics_md
from .rolling import rolling_ols

from .principia.ch04_inference import principia_ch04_inference

def _md_imgs(out_dir: str, paths: list[str]) -> str:
    return "\n".join([f"![]({relpath(out_dir, p)})" for p in paths])

def chapter02(*, out_dir: str, prices: str, lags: int = 20, returns: str = "log") -> tuple[str, str]:
    df = read_csv_ts(prices)
    r = fin.log_returns_from_close(df["close"]) if returns == "log" else fin.simple_returns_from_close(df["close"])

    jb = diag.jarque_bera(r)
    lb_r = diag.ljung_box(r, lags=lags)
    lb_r2 = diag.ljung_box(r**2, lags=lags)
    acf_r = diag.acf_simple(r, max_lag=min(lags, 50))
    acf_r2 = diag.acf_simple(r**2, max_lag=min(lags, 50))

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_series(r, f"{returns} returns (time series)", os.path.join(fig_dir, "returns_series.png")),
        plots.plot_hist(r, f"{returns} returns (histogram, density)", os.path.join(fig_dir, "returns_hist.png")),
        plots.plot_qq(r, f"QQ plot vs Normal ({returns} returns)", os.path.join(fig_dir, "returns_qq.png")),
        plots.plot_acf_bar(acf_r, "ACF: returns", os.path.join(fig_dir, "acf_returns.png")),
        plots.plot_acf_bar(acf_r2, "ACF: squared returns (vol clustering)", os.path.join(fig_dir, "acf_returns_sq.png")),
    ]

    mc = model_card("Chapter 02", {
        "Return definition": returns,
        "Normality test": "Jarque–Bera (χ², df=2)",
        "Autocorrelation test": f"Ljung–Box up to lag {lags}",
        "Sampling": "As-is from input; no resample in this runner",
    }, notes=[
        "Finance almost always rejects normality at scale; the point is *how* it deviates.",
    ])

    sections = [
        ("Model card", mc),
        ("What you ran", f"- prices: `{prices}`\n- n returns: {len(r)}\n- lags: {lags}"),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Jarque–Bera", "```json\n" + json.dumps(jb, indent=2) + "\n```"),
        ("Ljung–Box on returns", lb_r.to_string()),
        ("Ljung–Box on squared returns", lb_r2.to_string()),
    ]
    return write_report(out_dir, "Chapter 02 — Returns, tails, and diagnostics", sections)

def chapter03(*, out_dir: str, prices: str, lags: int = 20, returns: str = "log", vr_q: int = 5) -> tuple[str, str]:
    df = read_csv_ts(prices)
    r = fin.log_returns_from_close(df["close"]) if returns == "log" else fin.simple_returns_from_close(df["close"])
    acf_r = diag.acf_simple(r, max_lag=min(lags, 50))
    acf_r2 = diag.acf_simple(r**2, max_lag=min(lags, 50))
    r_arr = r.to_numpy()
    q = int(vr_q)
    rq = np.array([r_arr[i-q+1:i+1].sum() for i in range(q-1, len(r_arr))])
    vr = float(np.var(rq, ddof=1) / (q * np.var(r_arr, ddof=1)))

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_acf_bar(acf_r, "ACF: returns", os.path.join(fig_dir, "acf_returns.png")),
        plots.plot_acf_bar(acf_r2, "ACF: squared returns", os.path.join(fig_dir, "acf_returns_sq.png")),
    ]

    mc = model_card("Chapter 03", {
        "Return definition": returns,
        "Variance ratio": f"q={vr_q}, VR = Var(sum_q r) / (q Var(r))",
        "Autocorrelation view": f"ACF up to {min(lags,50)}",
    }, notes=[
        "VR is a blunt instrument; it mostly tells you if the process is obviously trending/reverting at that horizon."
    ])

    sections = [
        ("Model card", mc),
        ("What you ran", f"- prices: `{prices}`\n- n returns: {len(r)}\n- VR q: {vr_q}"),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Variance ratio", "```json\n" + json.dumps({"q": q, "vr": vr}, indent=2) + "\n```"),
    ]
    return write_report(out_dir, "Chapter 03 — Random walk sanity checks", sections)

def chapter04(*, out_dir: str, returns: str, cov_type: str = "HAC", hac_lags: int = 5, hac_kernel: str = "bartlett", rolling_window: int = 0, rolling_step: int = 5) -> tuple[str, str]:
    df = read_csv_ts(returns)
    for col in ["strategy", "market"]:
        if col not in df.columns:
            raise ValueError("returns CSV must include 'strategy' and 'market' columns")
    rf = df["rf"] if "rf" in df.columns else 0.0
    y = pd.to_numeric(df["strategy"], errors="coerce") - pd.to_numeric(rf, errors="coerce")
    x = pd.to_numeric(df["market"], errors="coerce") - pd.to_numeric(rf, errors="coerce")
    xy = pd.DataFrame({"y": y, "x": x}).dropna()
    y = xy["y"]; x = xy["x"]

    cov_kwds = {}
    ct = cov_type.upper()
    if ct == "HAC":
        cov_kwds = {"maxlags": int(hac_lags), "kernel": str(hac_kernel), "use_correction": True}

    fit = fit_ols(y, x, cov_type=ct, cov_kwds=cov_kwds)
    summ = summarize(fit, names=["alpha","beta"], cov_type=ct, cov_kwds=cov_kwds)

    resid = pd.Series(getattr(fit, "resid", []))
    diag_dict, diag_figs = run_residual_diagnostics(resid, out_dir=out_dir, lags=20, X_for_white=x.to_numpy(), prefix="capm")
    diag_block = diagnostics_md(diag_dict, out_dir, diag_figs)

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_scatter_fit(x, y, "CAPM: (strategy-rf) vs (market-rf)", os.path.join(fig_dir, "capm_scatter.png")),
    ]

    roll_figs = []
    if rolling_window and rolling_window > 10:
        roll_tbl = rolling_ols(y, x, window=int(rolling_window), step=int(max(1, rolling_step)), cov_type=ct, cov_kwds=cov_kwds)
        try:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(roll_tbl["beta"].to_numpy(), label="beta")
            ax.plot(roll_tbl["beta_ci_lo"].to_numpy(), linestyle="--", label="CI lo")
            ax.plot(roll_tbl["beta_ci_hi"].to_numpy(), linestyle="--", label="CI hi")
            ax.set_title(f"Rolling beta (window={rolling_window}, step={rolling_step})")
            ax.grid(True, alpha=0.3)
            p = os.path.join(fig_dir, "rolling_beta.png")
            fig.savefig(p, dpi=150, bbox_inches="tight")
            plt.close(fig)
            roll_figs.append(p)
        except Exception:
            pass

    mc = model_card("Chapter 04", {
        "Regression": "OLS: (strategy-rf) = alpha + beta*(market-rf) + eps",
        "Covariance / SEs": f"{ct} {cov_kwds if cov_kwds else ''}",
        "Diagnostics": "Residual bundle: Ljung–Box, ARCH LM, White (if available), JB/QQ + plots",
        "Rolling stability": f"window={rolling_window}, step={rolling_step} (0=off)",
    }, notes=[
        "Use HAC when returns are serially correlated or overlapping; HC when only heteroskedasticity is the concern.",
    ])

    sections = [
        ("Model card", mc),
        ("Regression table", summary_md(summ)),
        ("Core plots", _md_imgs(out_dir, figs + roll_figs)),
        ("Residual diagnostics", diag_block),
        ("Artifacts", "- `artifacts/capm_diagnostics.json` (created by diagnostics bundle)"),
    ]
    return write_report(out_dir, "Chapter 04 — CAPM regression with robust inference", sections)

def chapter05(*, out_dir: str, prices_multi: str, J: int = 60, K: int = 20, top_frac: float = 0.25, overlap: bool = True) -> tuple[str, str]:
    df = read_csv_ts(prices_multi)
    asset_cols = [c for c in df.columns if c != "ts"]
    if len(asset_cols) < 3:
        raise ValueError("prices_multi needs >= 3 asset columns")
    R = fin.to_period_returns(df, "ts", asset_cols, rule="1D")
    if len(R) < J + K + 10:
        raise ValueError("Not enough history after resample for momentum demo.")

    past = R.rolling(J).sum().shift(1)
    n_assets = len(asset_cols)
    n_top = max(1, int(np.floor(top_frac * n_assets)))
    pnl = []
    steps = range(J+1, len(R)-K) if overlap else range(J+1, len(R)-K, K)

    for t in steps:
        scores = past.iloc[t].dropna()
        if len(scores) < n_assets:
            continue
        winners = scores.sort_values(ascending=False).head(n_top).index
        losers = scores.sort_values(ascending=True).head(n_top).index
        hold = R.iloc[t:t+K]
        w_ret = hold[winners].mean(axis=1).sum()
        l_ret = hold[losers].mean(axis=1).sum()
        pnl.append(w_ret - l_ret)
    pnl = pd.Series(pnl, name="wml")

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_hist(pnl, "Winners-minus-losers PnL (per holding window)", os.path.join(fig_dir, "wml_hist.png"), bins=50),
        plots.plot_cum_pnl(pnl, "Cumulative WML (window samples)", os.path.join(fig_dir, "wml_cum.png")),
    ]

    mc = model_card("Chapter 05", {
        "Universe": f"{n_assets} assets from prices_multi columns",
        "Return horizon": "Daily log returns",
        "Signal": f"Past J={J} day cumulative return",
        "Portfolio": f"Top/bottom {top_frac:.2f} fraction (n={n_top})",
        "Holding": f"K={K} days",
        "Windowing": "Overlapping" if overlap else "Non-overlapping",
        "Costs": "Not included (toy)",
    }, notes=[
        "Momentum is very cost/constraints sensitive; treat this runner as a measurement template."
    ])

    summary = {
        "J": int(J), "K": int(K), "top_frac": float(top_frac),
        "n_assets": int(n_assets), "n_samples": int(len(pnl)),
        "mean_pnl": float(np.mean(pnl)) if len(pnl) else float("nan"),
        "std_pnl": float(np.std(pnl, ddof=1)) if len(pnl) > 1 else float("nan"),
        "sharpe_like": float(np.mean(pnl) / np.std(pnl, ddof=1)) if len(pnl) > 2 else float("nan"),
    }

    sections = [
        ("Model card", mc),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Summary", "```json\n" + json.dumps(summary, indent=2) + "\n```"),
    ]
    return write_report(out_dir, "Chapter 05 — Momentum J/K sorts (toy)", sections)

def chapter06(*, out_dir: str, prices_multi: str, pair: str = "A,B", z_win: int = 60, z_entry: float = 1.5, adf_maxlag: int = 1) -> tuple[str, str]:
    df = read_csv_ts(prices_multi)
    a, b = [s.strip() for s in pair.split(",")]
    if a not in df.columns or b not in df.columns:
        raise ValueError(f"Pair columns not found. Need '{a}' and '{b}' in prices_multi.")
    x = df.set_index("ts").sort_index()
    px = x[[a, b]].resample("1D").last().dropna()
    logA = np.log(px[a]); logB = np.log(px[b])
    beta = models.hedge_ratio_ols(logA, logB)
    spread = logA - beta * logB
    spread.name = "spread"

    from statsmodels.tsa.stattools import adfuller
    adf_stat, adf_p, *_ = adfuller(spread.dropna().to_numpy(), maxlag=adf_maxlag, regression="c", autolag=None)
    ar1 = models.fit_ar1(spread)

    z = fin.zscore(spread, win=z_win).dropna()

    pos = 0
    pnl = []
    for i in range(1, len(z)):
        z_now = z.iloc[i]
        if pos == 0:
            if z_now < -z_entry: pos = 1
            elif z_now > z_entry: pos = -1
        else:
            if (pos == 1 and z_now >= 0) or (pos == -1 and z_now <= 0):
                pos = 0
        ds = float(spread.loc[z.index[i]] - spread.loc[z.index[i-1]])
        pnl.append(pos * ds)
    pnl = pd.Series(pnl, index=z.index[1:], name="pnl_spread_units")

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_series(spread.dropna(), f"Spread: log({a}) - beta*log({b}), beta={beta:.3f}", os.path.join(fig_dir, "spread.png")),
        plots.plot_series(z, f"Z-score (window={z_win})", os.path.join(fig_dir, "zscore.png")),
        plots.plot_cum_pnl(pnl, "Toy strategy cumulative PnL (spread units)", os.path.join(fig_dir, "pnl_cum.png")),
    ]

    mc = model_card("Chapter 06", {
        "Resample": "Daily last",
        "Hedge ratio": "OLS on log prices (y=logA, x=logB)",
        "Spread": "logA - beta*logB",
        "Stationarity test": f"ADF maxlag={adf_maxlag}, no autolag",
        "OU proxy": "AR(1) fit on spread → implied half-life",
        "Signal": f"Rolling z-score window={z_win}",
        "Rule": f"Enter at |z|>{z_entry}, exit at z crossing 0",
        "PnL convention": "pos * Δspread (spread units), no costs",
    })

    summary = {
        "pair": [a, b],
        "beta": float(beta),
        "adf_stat": float(adf_stat),
        "adf_p_value": float(adf_p),
        "ar1": ar1,
        "n_days": int(len(px)),
        "n_samples": int(len(pnl)),
        "mean_pnl": float(np.mean(pnl)) if len(pnl) else float("nan"),
    }

    sections = [
        ("Model card", mc),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Summary", "```json\n" + json.dumps(summary, indent=2) + "\n```"),
    ]
    return write_report(out_dir, "Chapter 06 — Pairs / OU intuition (toy)", sections)

def chapter07(*, out_dir: str, prices_multi: str) -> tuple[str, str]:
    df = read_csv_ts(prices_multi)
    asset_cols = [c for c in df.columns if c != "ts"]
    R = fin.to_period_returns(df, "ts", asset_cols, rule="1D")
    w = models.long_only_min_var_weights(R)

    fig_dir = os.path.join(out_dir, "figures")
    figs = [plots.plot_bar(w["weights"], "Long-only minimum variance weights (toy)", os.path.join(fig_dir, "weights.png"))]

    mc = model_card("Chapter 07", {
        "Returns": "Daily log returns",
        "Objective": "Minimize w'Σw",
        "Constraints": "w>=0, sum(w)=1",
        "Estimator": "Sample covariance (no shrinkage)",
    })

    sections = [
        ("Model card", mc),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Weights", "```json\n" + json.dumps(w, indent=2) + "\n```"),
    ]
    return write_report(out_dir, "Chapter 07 — Mean-variance under constraints (toy)", sections)

def chapter08(*, out_dir: str, events: str, seasonality: str = "hour_of_day") -> tuple[str, str]:
    df = read_csv_ts(events)
    ts = df["ts"].sort_values()
    dt = ts.diff().dt.total_seconds().dropna()
    dt = dt[dt > 0]

    if seasonality == "hour_of_day":
        hours = ts.iloc[1:].dt.hour
        by_hour = pd.DataFrame({"dt": dt.to_numpy(), "h": hours.to_numpy()}).groupby("h")["dt"].mean()
        dt_adj = dt / hours.map(by_hour).to_numpy()
    else:
        dt_adj = dt

    fit = models.acd11_fit(dt_adj.to_numpy())

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_hist(pd.Series(dt), "Durations (raw seconds)", os.path.join(fig_dir, "dur_hist_raw.png"), bins=80),
        plots.plot_hist(pd.Series(dt_adj), "Durations (adjusted)", os.path.join(fig_dir, "dur_hist_adj.png"), bins=80),
    ]

    mc = model_card("Chapter 08", {
        "Durations": "Δt between events (seconds)",
        "Seasonality adjustment": seasonality,
        "Model": "ACD(1,1) with Exp(1) innovations, MLE",
        "Constraints": "omega>0, alpha>=0, beta>=0, alpha+beta<1",
    })

    sections = [
        ("Model card", mc),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Fit", "```json\n" + json.dumps(fit, indent=2) + "\n```"),
    ]
    return write_report(out_dir, "Chapter 08 — Durations & ACD (toy)", sections)

def chapter09(*, out_dir: str, quotes: str, trades: str, grid: str = "1min") -> tuple[str, str]:
    q = read_csv_ts(quotes).set_index("ts").sort_index()
    t = read_csv_ts(trades).set_index("ts").sort_index()
    q["mid"] = (pd.to_numeric(q["bid"], errors="coerce") + pd.to_numeric(q["ask"], errors="coerce")) / 2.0

    q1 = q[["mid"]].resample(grid).last().dropna()
    t1 = t[["price"]].resample(grid).last().dropna()
    idx = q1.index.intersection(t1.index)
    q1 = q1.loc[idx]; t1 = t1.loc[idx]

    r_mid = fin.log_returns_from_close(q1["mid"])
    r_trd = fin.log_returns_from_close(t1["price"])
    acf_mid = diag.acf_simple(r_mid, max_lag=30)
    acf_trd = diag.acf_simple(r_trd, max_lag=30)

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_acf_bar(acf_mid, "ACF: mid returns", os.path.join(fig_dir, "acf_mid.png")),
        plots.plot_acf_bar(acf_trd, "ACF: trade returns", os.path.join(fig_dir, "acf_trade.png")),
    ]

    mc = model_card("Chapter 09", {
        "Resample grid": grid,
        "Midprice": "(bid+ask)/2",
        "Returns": "log returns",
        "Autocorr view": "ACF up to 30",
    })

    sections = [
        ("Model card", mc),
        ("Charts", _md_imgs(out_dir, figs)),
    ]
    return write_report(out_dir, "Chapter 09 — Microstructure noise: mid vs trade", sections)

def chapter10(*, out_dir: str, trades: str) -> tuple[str, str]:
    df = read_csv_ts(trades)
    roll = ms.roll_spread_from_trades(df["price"])
    dp = pd.to_numeric(df["price"], errors="coerce").diff().dropna()

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_series(pd.to_numeric(df["price"], errors="coerce"), "Trade price", os.path.join(fig_dir, "trade_price.png")),
        plots.plot_hist(dp, "ΔP histogram", os.path.join(fig_dir, "dp_hist.png"), bins=80),
    ]

    mc = model_card("Chapter 10", {
        "Estimator": "Roll: 2 * sqrt(-Cov(Δp_t, Δp_{t-1}))",
        "Assumption": "Bid–ask bounce dominates serial covariance; no inventory effects etc.",
    })

    sections = [
        ("Model card", mc),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Roll", "```json\n" + json.dumps(roll, indent=2) + "\n```"),
    ]
    return write_report(out_dir, "Chapter 10 — Roll spread baseline", sections)

def chapter11(*, out_dir: str, quotes: str, trades: str, sign_method: str = "tick_rule") -> tuple[str, str]:
    q = read_csv_ts(quotes).set_index("ts").sort_index()
    t = read_csv_ts(trades).set_index("ts").sort_index()
    q["mid"] = (pd.to_numeric(q["bid"], errors="coerce") + pd.to_numeric(q["ask"], errors="coerce")) / 2.0
    tp = pd.to_numeric(t["price"], errors="coerce").dropna()

    sign = ms.tick_rule_sign(tp)
    t = t.loc[tp.index].copy()
    t["sign"] = sign

    mid_at_trade = q["mid"].reindex(t.index, method="ffill")
    mid_change = mid_at_trade.diff().shift(-1)
    df2 = pd.DataFrame({"sign": t["sign"], "dmid_next": mid_change}).dropna()

    resp_buy = float(df2.loc[df2["sign"] > 0, "dmid_next"].mean())
    resp_sell = float(df2.loc[df2["sign"] < 0, "dmid_next"].mean())

    fig_dir = os.path.join(out_dir, "figures")
    figs = [plots.plot_hist(df2["dmid_next"], "Next mid change after trade", os.path.join(fig_dir, "dmid_hist.png"), bins=80)]

    mc = model_card("Chapter 11", {
        "Trade sign": sign_method,
        "Mid at trade": "last quote mid before trade (ffill)",
        "Response": "E[Δmid_next | sign]",
    })

    summary = {"E[dMid_next | Buy]": resp_buy, "E[dMid_next | Sell]": resp_sell, "n": int(len(df2))}
    sections = [
        ("Model card", mc),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Summary", "```json\n" + json.dumps(summary, indent=2) + "\n```"),
    ]
    return write_report(out_dir, "Chapter 11 — Trade-sign response (toy)", sections)

def chapter12(*, out_dir: str, lob: str, bins: int = 10) -> tuple[str, str]:
    df = read_csv_ts(lob).set_index("ts").sort_index()
    mid = pd.to_numeric(df["mid"], errors="coerce")
    bid_sz = pd.to_numeric(df["bid_size"], errors="coerce")
    ask_sz = pd.to_numeric(df["ask_size"], errors="coerce")
    imb = (bid_sz - ask_sz) / (bid_sz + ask_sz)
    nxt = mid.diff().shift(-1)
    y = (nxt > 0).astype(int)
    tmp = pd.DataFrame({"imb": imb, "up": y}).dropna()
    tmp["bin"] = pd.qcut(tmp["imb"], bins, duplicates="drop")
    p_up = tmp.groupby("bin")["up"].mean()

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_series(imb.dropna(), "Imbalance (best level)", os.path.join(fig_dir, "imb_series.png")),
        plots.plot_bar({str(k): float(v) for k, v in p_up.items()}, "P(up) by imbalance bin", os.path.join(fig_dir, "pup_by_bin.png")),
    ]

    mc = model_card("Chapter 12", {
        "Feature": "imb = (bid_size-ask_size)/(bid_size+ask_size)",
        "Label": "1(next mid move up), using Δmid_{t+1}",
        "Binning": f"qcut into {bins} quantile bins",
    })

    sections = [
        ("Model card", mc),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Table", p_up.to_string()),
    ]
    return write_report(out_dir, "Chapter 12 — LOB imbalance baseline (toy)", sections)

def chapter13(*, out_dir: str, total_shares: float = 1_000_000, N: int = 20, risk_aversion: float = 0.15, sigma: float = 0.01, impact: float = 1e-6, sims: int = 2000, schedule: str = "exp") -> tuple[str, str]:
    t = np.arange(N)
    if schedule == "exp":
        w = np.exp(-risk_aversion * t)
    else:
        w = np.ones(N)
    w = w / w.sum()
    sched = total_shares * w

    rng = np.random.default_rng(0)
    shortfalls = []
    for _ in range(sims):
        dP = sigma * rng.normal(size=N)
        P = np.cumsum(dP)
        tmp_impact = impact * (sched**2)
        sf = float(np.sum(sched * P) + np.sum(tmp_impact))
        shortfalls.append(sf)
    sf = pd.Series(shortfalls, name="shortfall")

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_hist(sf, "Implementation shortfall (toy)", os.path.join(fig_dir, "shortfall_hist.png"), bins=60),
        plots.plot_series(pd.Series(sched), "Schedule (shares per slice)", os.path.join(fig_dir, "schedule.png")),
    ]

    mc = model_card("Chapter 13", {
        "Schedule": schedule,
        "Price risk": f"Gaussian increments sigma={sigma}",
        "Impact proxy": "temporary cost ∝ impact * rate^2",
        "Objective": "Simulate distribution of implementation shortfall",
        "Calibration": "None (toy)",
    })

    summary = {
        "total_shares": float(total_shares),
        "N_slices": int(N),
        "risk_aversion": float(risk_aversion),
        "sigma": float(sigma),
        "impact_coeff": float(impact),
        "sims": int(sims),
        "shortfall_mean": float(sf.mean()),
        "shortfall_std": float(sf.std(ddof=1)),
        "shortfall_p95": float(sf.quantile(0.95)),
    }

    sections = [
        ("Model card", mc),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Summary", "```json\n" + json.dumps(summary, indent=2) + "\n```"),
    ]
    return write_report(out_dir, "Chapter 13 — Execution & shortfall (toy AC)", sections)

def chapter14(*, out_dir: str) -> tuple[str, str]:
    mc = model_card("Chapter 14", {"Type": "Checklist / workflow"}, notes=["Not everything needs to be a model; some things need to be a habit."])
    text = """<div class="note">
Strategies die from engineering reality more often than from missing equations.
</div>

**Research hygiene**
- Define the information set (no lookahead, correct timestamps, calendar/corporate actions).
- Separate signal research from execution (spread, slippage, rejects).
- Track edge in **bps after costs**, not raw return.

**Robustness**
- Walk-forward / rolling windows.
- Sensitivity: costs, lags, filters, universe.
- Placebo tests: shuffled labels, randomized entry timing, synthetic controls.

**Capacity & monitoring**
- Turnover, participation rate, expected impact.
- Live drift monitors: fill quality, slippage vs expectation, regime shifts.

**Postmortems**
- For every failure, write the exact chain: hypothesis → measurement → implementation → outcome.
"""
    sections = [("Model card", mc), ("Checklist", text)]
    return write_report(out_dir, "Chapter 14 — Robust quant workflow (checklist)", sections)

CHAPTERS = {
    "chapter02": chapter02,
    "chapter03": chapter03,
    "chapter04": chapter04,
    "chapter05": chapter05,
    "chapter06": chapter06,
    "chapter07": chapter07,
    "chapter08": chapter08,
    "chapter09": chapter09,
    "chapter10": chapter10,
    "chapter11": chapter11,
    "chapter12": chapter12,
    "chapter13": chapter13,
    "chapter14": chapter14,
    "principia_ch04_inference": principia_ch04_inference,
}

def main():
    ap = argparse.ArgumentParser(prog="stats242xtb.run")
    ap.add_argument("chapter", choices=sorted(CHAPTERS.keys()))
    ap.add_argument("--out", required=True)

    ap.add_argument("--prices")
    ap.add_argument("--prices-multi")
    ap.add_argument("--returns")
    ap.add_argument("--trades")
    ap.add_argument("--quotes")
    ap.add_argument("--events")
    ap.add_argument("--lob")

    ap.add_argument("--lags", type=int, default=20)
    ap.add_argument("--returns-def", dest="returns_def", choices=["log","simple"], default="log")
    ap.add_argument("--vr-q", type=int, default=5)

    ap.add_argument("--J", type=int, default=60)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--top-frac", type=float, default=0.25)
    ap.add_argument("--overlap", action="store_true")

    ap.add_argument("--pair", default="A,B")
    ap.add_argument("--z-win", type=int, default=60)
    ap.add_argument("--z-entry", type=float, default=1.5)
    ap.add_argument("--adf-maxlag", type=int, default=1)

    ap.add_argument("--cov-type", default="HAC")
    ap.add_argument("--hac-lags", type=int, default=5)
    ap.add_argument("--hac-kernel", default="bartlett")
    ap.add_argument("--rolling-window", type=int, default=0)
    ap.add_argument("--rolling-step", type=int, default=5)

    ap.add_argument("--grid", default="1min")
    ap.add_argument("--sign-method", default="tick_rule")
    ap.add_argument("--bins", type=int, default=10)

    ap.add_argument("--total-shares", type=float, default=1_000_000)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--risk-aversion", type=float, default=0.15)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--impact", type=float, default=1e-6)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--schedule", default="exp")

    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--phi", type=float, default=0.4)
    ap.add_argument("--principia-sims", type=int, default=500)

    args = ap.parse_args()
    ensure_dir(args.out)
    ch = args.chapter
    fn = CHAPTERS[ch]

    kwargs = {"out_dir": args.out}
    if ch in ("chapter02","chapter03"):
        if not args.prices: raise SystemExit("--prices required")
        kwargs.update({"prices": args.prices, "lags": args.lags, "returns": args.returns_def})
        if ch == "chapter03":
            kwargs.update({"vr_q": args.vr_q})
    elif ch == "chapter04":
        if not args.returns: raise SystemExit("--returns required")
        kwargs.update({
            "returns": args.returns,
            "cov_type": args.cov_type,
            "hac_lags": args.hac_lags,
            "hac_kernel": args.hac_kernel,
            "rolling_window": args.rolling_window,
            "rolling_step": args.rolling_step,
        })
    elif ch in ("chapter05","chapter06","chapter07"):
        if not args.prices_multi: raise SystemExit("--prices-multi required")
        if ch == "chapter05":
            kwargs.update({"prices_multi": args.prices_multi, "J": args.J, "K": args.K, "top_frac": args.top_frac, "overlap": args.overlap})
        elif ch == "chapter06":
            kwargs.update({"prices_multi": args.prices_multi, "pair": args.pair, "z_win": args.z_win, "z_entry": args.z_entry, "adf_maxlag": args.adf_maxlag})
        else:
            kwargs.update({"prices_multi": args.prices_multi})
    elif ch == "chapter08":
        if not args.events: raise SystemExit("--events required")
        kwargs.update({"events": args.events})
    elif ch in ("chapter09","chapter11"):
        if not args.quotes or not args.trades: raise SystemExit("--quotes and --trades required")
        if ch == "chapter09":
            kwargs.update({"quotes": args.quotes, "trades": args.trades, "grid": args.grid})
        else:
            kwargs.update({"quotes": args.quotes, "trades": args.trades, "sign_method": args.sign_method})
    elif ch == "chapter10":
        if not args.trades: raise SystemExit("--trades required")
        kwargs.update({"trades": args.trades})
    elif ch == "chapter12":
        if not args.lob: raise SystemExit("--lob required")
        kwargs.update({"lob": args.lob, "bins": args.bins})
    elif ch == "chapter13":
        kwargs.update({"total_shares": args.total_shares, "N": args.N, "risk_aversion": args.risk_aversion, "sigma": args.sigma, "impact": args.impact, "sims": args.sims, "schedule": args.schedule})
    elif ch == "chapter14":
        pass
    elif ch == "principia_ch04_inference":
        kwargs.update({"n": args.n, "sims": args.principia_sims, "phi": args.phi, "cov_lags": args.hac_lags})

    md, html = fn(**kwargs)
    print(md); print(html)

if __name__ == "__main__":
    main()
