from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd

from .util import ensure_dir, read_csv_ts, relpath, safe_div
from .report import write_report
from . import finance as fin
from . import diagnostics as diag
from . import microstructure as ms
from . import models
from . import plots

def _md_imgs(out_dir: str, paths: list[str]) -> str:
    return "\n".join([f"![]({relpath(out_dir, p)})" for p in paths])

def chapter02(prices_path: str, out_dir: str, lags: int = 20) -> tuple[str, str]:
    df = read_csv_ts(prices_path)
    lr = fin.log_returns_from_close(df["close"])
    jb = diag.jarque_bera(lr)
    lb_r = diag.ljung_box(lr, lags=lags)
    lb_r2 = diag.ljung_box(lr**2, lags=lags)
    acf_r = diag.acf_simple(lr, max_lag=min(lags, 50))
    acf_r2 = diag.acf_simple(lr**2, max_lag=min(lags, 50))

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_series(lr, "Log returns (time series)", os.path.join(fig_dir, "returns_series.png")),
        plots.plot_hist(lr, "Log returns (histogram, density)", os.path.join(fig_dir, "returns_hist.png")),
        plots.plot_qq(lr, "QQ plot vs Normal (log returns)", os.path.join(fig_dir, "returns_qq.png")),
        plots.plot_acf_bar(acf_r, "ACF: returns", os.path.join(fig_dir, "acf_returns.png")),
        plots.plot_acf_bar(acf_r2, "ACF: squared returns (volatility clustering)", os.path.join(fig_dir, "acf_returns_sq.png")),
    ]

    sections = [
        ("What you ran", f"- prices: `{prices_path}`\n- n log-returns: {len(lr)}\n- lags: {lags}"),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Jarque–Bera normality check (log returns)", "```json\n" + json.dumps(jb, indent=2) + "\n```"),
        ("Ljung–Box on returns (linear autocorrelation?)", lb_r.to_string()),
        ("Ljung–Box on squared returns (volatility clustering)", lb_r2.to_string()),
        ("Interpretation prompts", "\n".join([
            "- Weak autocorr in returns + strong autocorr in squared returns = volatility clustering.",
            "- JB p-value near 0 is typical; focus on tails + stability across regimes.",
        ])),
    ]
    return write_report(out_dir, "Chapter 02 — Returns, tails, and diagnostics", sections)

def chapter03(prices_path: str, out_dir: str, lags: int = 20, vr_q: int = 5) -> tuple[str, str]:
    df = read_csv_ts(prices_path)
    lr = fin.log_returns_from_close(df["close"])

    acf_r = diag.acf_simple(lr, max_lag=min(lags, 50))
    acf_r2 = diag.acf_simple(lr**2, max_lag=min(lags, 50))
    vr = diag.variance_ratio(lr, q=vr_q)

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_acf_bar(acf_r, "ACF: returns (martingale sanity check)", os.path.join(fig_dir, "acf_returns.png")),
        plots.plot_acf_bar(acf_r2, "ACF: squared returns (structure in volatility)", os.path.join(fig_dir, "acf_returns_sq.png")),
    ]

    sections = [
        ("What you ran", f"- prices: `{prices_path}`\n- n log-returns: {len(lr)}\n- lags: {lags}\n- variance ratio q: {vr_q}"),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Variance ratio (random walk sanity check)", "```json\n" + json.dumps(vr, indent=2) + "\n```"),
        ("Interpretation prompts", "\n".join([
            "- VR ≈ 1 is consistent with a random walk in *returns* aggregation; VR < 1 suggests mean reversion; VR > 1 suggests momentum.",
            "- Most directional predictability is tiny; structure often lives in volatility/order flow/spreads.",
        ])),
    ]
    return write_report(out_dir, "Chapter 03 — Random walk vs martingale: what 'predictable' means", sections)

def chapter04(returns_path: str, out_dir: str) -> tuple[str, str]:
    df = read_csv_ts(returns_path)
    for col in ["strategy", "market"]:
        if col not in df.columns:
            raise ValueError("returns CSV must include 'strategy' and 'market' columns")
    rf = df["rf"] if "rf" in df.columns else 0.0
    y = pd.to_numeric(df["strategy"], errors="coerce") - pd.to_numeric(rf, errors="coerce")
    x = pd.to_numeric(df["market"], errors="coerce") - pd.to_numeric(rf, errors="coerce")
    xy = pd.DataFrame({"y": y, "x": x}).dropna()
    y = xy["y"]; x = xy["x"]

    import statsmodels.api as sm
    X = sm.add_constant(x.to_numpy())
    model = sm.OLS(y.to_numpy(), X).fit()

    alpha = float(model.params[0])
    beta = float(model.params[1])
    r2 = float(model.rsquared)
    dw = float(sm.stats.stattools.durbin_watson(model.resid))
    # White test (basic)
    try:
        white = sm.stats.diagnostic.het_white(model.resid, X)
        white_p = float(white[1])
    except Exception:
        white_p = float("nan")

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_scatter_fit(x, y, "CAPM regression: (strategy-rf) vs (market-rf)", os.path.join(fig_dir, "capm_scatter.png")),
        plots.plot_series(pd.Series(model.resid), "Residuals (time order)", os.path.join(fig_dir, "capm_resid_series.png")),
        plots.plot_hist(pd.Series(model.resid), "Residuals histogram", os.path.join(fig_dir, "capm_resid_hist.png"), bins=60),
    ]

    summary = {
        "alpha": alpha, "beta": beta, "r2": r2,
        "durbin_watson": dw,
        "white_p_value": white_p,
        "n": int(model.nobs),
    }

    sections = [
        ("What you ran", f"- returns: `{returns_path}`\n- n: {int(model.nobs)}"),
        ("Charts", _md_imgs(out_dir, figs)),
        ("CAPM fit summary", "```json\n" + json.dumps(summary, indent=2) + "\n```"),
        ("Interpretation prompts", "\n".join([
            "- If beta is large and alpha tiny, your 'alpha' is mostly market exposure in disguise.",
            "- White test p-value small suggests heteroskedasticity → use robust SEs in serious work.",
            "- Durbin–Watson far from ~2 suggests residual autocorrelation (time-series structure not captured by CAPM).",
        ])),
    ]
    return write_report(out_dir, "Chapter 04 — CAPM as a regression template", sections)

def chapter05(prices_multi_path: str, out_dir: str, J: int = 60, K: int = 20, top_frac: float = 0.25) -> tuple[str, str]:
    df = read_csv_ts(prices_multi_path)
    asset_cols = [c for c in df.columns if c != "ts"]
    if len(asset_cols) < 3:
        raise ValueError("prices_multi needs >= 3 asset columns")
    # regularize and compute daily returns
    R = fin.to_period_returns(df, "ts", asset_cols, rule="1D")
    if len(R) < J + K + 10:
        raise ValueError("Not enough history after resample for momentum demo.")
    # rolling momentum signal: past J return
    past = R.rolling(J).sum().shift(1)
    n_assets = len(asset_cols)
    n_top = max(1, int(np.floor(top_frac * n_assets)))
    pnl = []
    for t in range(J+1, len(R)-K):
        scores = past.iloc[t].dropna()
        if len(scores) < n_assets:
            continue
        winners = scores.sort_values(ascending=False).head(n_top).index
        losers = scores.sort_values(ascending=True).head(n_top).index
        # hold next K days: average winner return - average loser return
        hold = R.iloc[t:t+K]
        w_ret = hold[winners].mean(axis=1).sum()
        l_ret = hold[losers].mean(axis=1).sum()
        pnl.append(w_ret - l_ret)
    pnl = pd.Series(pnl, name="wml")

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_hist(pnl, "Winners-minus-losers PnL (per holding window)", os.path.join(fig_dir, "wml_hist.png"), bins=50),
        plots.plot_cum_pnl(pnl, "Cumulative WML (toy, overlapping windows)", os.path.join(fig_dir, "wml_cum.png")),
    ]

    summary = {
        "J_lookback_days": int(J),
        "K_hold_days": int(K),
        "top_frac": float(top_frac),
        "n_assets": int(n_assets),
        "n_samples": int(len(pnl)),
        "mean_pnl": float(np.mean(pnl)),
        "std_pnl": float(np.std(pnl, ddof=1)) if len(pnl) > 1 else float("nan"),
        "sharpe_like": float(np.mean(pnl) / np.std(pnl, ddof=1)) if len(pnl) > 2 else float("nan"),
    }

    sections = [
        ("What you ran", f"- prices_multi: `{prices_multi_path}`\n- assets: {asset_cols}\n- resample: 1D"),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Momentum (J/K) summary", "```json\n" + json.dumps(summary, indent=2) + "\n```"),
        ("Interpretation prompts", "\n".join([
            "- This is a toy cross-sectional momentum sort. Real results depend on universe, costs, and constraints.",
            "- Momentum at one horizon doesn't imply momentum at others.",
        ])),
    ]
    return write_report(out_dir, "Chapter 05 — Momentum as statistics (J/K sorts)", sections)

def chapter06(prices_multi_path: str, out_dir: str, pair: str = "A,B", z_win: int = 60, z_entry: float = 1.5) -> tuple[str, str]:
    df = read_csv_ts(prices_multi_path)
    a, b = [s.strip() for s in pair.split(",")]
    if a not in df.columns or b not in df.columns:
        raise ValueError(f"Pair columns not found. Need '{a}' and '{b}' in prices_multi.")
    x = df.set_index("ts").sort_index()
    # use daily resample
    px = x[[a, b]].resample("1D").last().dropna()
    logA = np.log(px[a]); logB = np.log(px[b])
    beta = models.hedge_ratio_ols(logA, logB)
    spread = logA - beta * logB
    spread.name = "spread"

    # stationarity test (ADF)
    from statsmodels.tsa.stattools import adfuller
    adf_stat, adf_p, *_ = adfuller(spread.dropna().to_numpy(), maxlag=1, regression="c", autolag=None)
    ar1 = models.fit_ar1(spread)

    z = fin.zscore(spread, win=z_win).dropna()

    # simple trading rule: enter when |z|>z_entry, exit when z crosses 0
    pos = 0
    pnl = []
    for i in range(1, len(z)):
        zi_prev = z.iloc[i-1]
        zi = z.iloc[i]
        # position changes
        if pos == 0:
            if zi > z_entry:
                pos = -1
            elif zi < -z_entry:
                pos = 1
        else:
            if (pos == 1 and zi >= 0) or (pos == -1 and zi <= 0):
                pos = 0
        # pnl in spread units (delta spread * position)
        ds = float(spread.loc[z.index[i]] - spread.loc[z.index[i-1]])
        pnl.append(pos * (-ds))  # if long spread, profit when spread falls? depends; we pick one convention
    pnl = pd.Series(pnl, index=z.index[1:], name="pnl")

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_series(spread.dropna(), f"Spread: log({a}) - beta*log({b}), beta={beta:.3f}", os.path.join(fig_dir, "spread.png")),
        plots.plot_series(z, f"Z-score (window={z_win})", os.path.join(fig_dir, "zscore.png")),
        plots.plot_cum_pnl(pnl, "Toy strategy cumulative PnL (spread units)", os.path.join(fig_dir, "pnl_cum.png")),
    ]

    summary = {
        "pair": [a, b],
        "beta": float(beta),
        "adf_stat": float(adf_stat),
        "adf_p_value": float(adf_p),
        "ar1": ar1,
        "z_window": int(z_win),
        "z_entry": float(z_entry),
        "n_days": int(len(px)),
        "n_trades_samples": int(len(pnl)),
        "mean_pnl": float(np.mean(pnl)),
        "sharpe_like": float(np.mean(pnl) / np.std(pnl, ddof=1)) if len(pnl) > 5 else float("nan"),
    }

    sections = [
        ("What you ran", f"- prices_multi: `{prices_multi_path}`\n- pair: {a},{b}\n- daily resample"),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Cointegration-ish diagnostics", "```json\n" + json.dumps(summary, indent=2) + "\n```"),
        ("Interpretation prompts", "\n".join([
            "- ADF p-value small suggests the spread is stationary *in this sample*; structural breaks can still kill it.",
            "- Half-life tells you whether mean reversion matches your horizon.",
            "- The toy PnL is not costed and uses a simplistic entry/exit; treat it as a lab, not a result.",
        ])),
    ]
    return write_report(out_dir, "Chapter 06 — Mean reversion & cointegration (pairs / OU intuition)", sections)

def chapter07(prices_multi_path: str, out_dir: str) -> tuple[str, str]:
    df = read_csv_ts(prices_multi_path)
    asset_cols = [c for c in df.columns if c != "ts"]
    R = fin.to_period_returns(df, "ts", asset_cols, rule="1D")
    w = models.long_only_min_var_weights(R)

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_bar(w["weights"], "Long-only minimum variance weights (toy)", os.path.join(fig_dir, "weights.png")),
    ]

    sections = [
        ("What you ran", f"- prices_multi: `{prices_multi_path}`\n- assets: {asset_cols}\n- daily returns"),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Min-variance (long-only) result", "```json\n" + json.dumps(w, indent=2) + "\n```"),
        ("Interpretation prompts", "\n".join([
            "- MV optimization is usually dominated by covariance estimation error.",
            "- Constraints (turnover, bounds, exposures) matter more than pretty algebra.",
        ])),
    ]
    return write_report(out_dir, "Chapter 07 — Portfolio construction (mean-variance under constraints)", sections)

def chapter08(events_path: str, out_dir: str) -> tuple[str, str]:
    df = read_csv_ts(events_path)
    ts = df["ts"].sort_values()
    dt = ts.diff().dt.total_seconds().dropna()
    dt = dt[dt > 0]
    # simple de-seasonalization: by hour-of-day
    hours = ts.iloc[1:].dt.hour
    by_hour = pd.DataFrame({"dt": dt.to_numpy(), "h": hours.to_numpy()}).groupby("h")["dt"].mean()
    dt_adj = dt / hours.map(by_hour).to_numpy()

    fit = models.acd11_fit(dt_adj.to_numpy())
    acf_e = diag.acf_simple(pd.Series(dt_adj.to_numpy()) / fit["mean_duration"], max_lag=30)

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_hist(pd.Series(dt), "Durations (raw seconds) histogram", os.path.join(fig_dir, "dur_hist_raw.png"), bins=80),
        plots.plot_hist(pd.Series(dt_adj), "Durations (seasonality-adjusted) histogram", os.path.join(fig_dir, "dur_hist_adj.png"), bins=80),
        plots.plot_acf_bar(acf_e, "ACF: normalized durations (quick look)", os.path.join(fig_dir, "acf_norm_dur.png")),
    ]

    sections = [
        ("What you ran", f"- events: `{events_path}`\n- n durations: {len(dt)}"),
        ("Charts", _md_imgs(out_dir, figs)),
        ("ACD(1,1) fit (exponential innovations)", "```json\n" + json.dumps(fit, indent=2) + "\n```"),
        ("Interpretation prompts", "\n".join([
            "- ACD is the 'GARCH of time': it models clustering of events.",
            "- alpha+beta close to 1 suggests persistent duration clustering.",
        ])),
    ]
    return write_report(out_dir, "Chapter 08 — Intraday durations and ACD", sections)

def chapter09(quotes_path: str, trades_path: str, out_dir: str) -> tuple[str, str]:
    q = read_csv_ts(quotes_path).set_index("ts").sort_index()
    t = read_csv_ts(trades_path).set_index("ts").sort_index()
    q["mid"] = (pd.to_numeric(q["bid"], errors="coerce") + pd.to_numeric(q["ask"], errors="coerce")) / 2.0
    # resample both to 1min grid using last
    q1 = q[["mid"]].resample("1min").last().dropna()
    t1 = t[["price"]].resample("1min").last().dropna()
    idx = q1.index.intersection(t1.index)
    q1 = q1.loc[idx]; t1 = t1.loc[idx]

    r_mid = fin.log_returns_from_close(q1["mid"])
    r_trd = fin.log_returns_from_close(t1["price"])
    acf_mid = diag.acf_simple(r_mid, max_lag=30)
    acf_trd = diag.acf_simple(r_trd, max_lag=30)

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_acf_bar(acf_mid, "ACF: mid returns", os.path.join(fig_dir, "acf_mid.png")),
        plots.plot_acf_bar(acf_trd, "ACF: trade returns (bid-ask bounce)", os.path.join(fig_dir, "acf_trade.png")),
        plots.plot_hist(r_mid, "Mid returns histogram", os.path.join(fig_dir, "mid_hist.png"), bins=60),
        plots.plot_hist(r_trd, "Trade returns histogram", os.path.join(fig_dir, "trade_hist.png"), bins=60),
    ]

    sections = [
        ("What you ran", f"- quotes: `{quotes_path}`\n- trades: `{trades_path}`\n- resample: 1min, aligned"),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Key idea", "\n".join([
            "Trade-to-trade returns often show negative serial dependence from bid–ask bounce.",
            "Midprice returns are usually closer to the efficient-price move (still noisy)."
        ])),
    ]
    return write_report(out_dir, "Chapter 09 — Microstructure noise: mid vs trade", sections)

def chapter10(trades_path: str, out_dir: str) -> tuple[str, str]:
    df = read_csv_ts(trades_path)
    roll = ms.roll_spread_from_trades(df["price"])
    fig_dir = os.path.join(out_dir, "figures")
    dp = pd.to_numeric(df["price"], errors="coerce").diff().dropna()
    figs = [
        plots.plot_series(pd.to_numeric(df["price"], errors="coerce"), "Trade price (time series)", os.path.join(fig_dir, "trade_price.png")),
        plots.plot_hist(dp, "Trade price changes ΔP histogram", os.path.join(fig_dir, "dp_hist.png"), bins=80),
    ]
    sections = [
        ("What you ran", f"- trades: `{trades_path}`\n- n trades: {len(df)}"),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Roll estimator", "```json\n" + json.dumps(roll, indent=2) + "\n```"),
    ]
    return write_report(out_dir, "Chapter 10 — Roll spread baseline", sections)

def chapter11(quotes_path: str, trades_path: str, out_dir: str) -> tuple[str, str]:
    q = read_csv_ts(quotes_path).set_index("ts").sort_index()
    t = read_csv_ts(trades_path).set_index("ts").sort_index()
    q["mid"] = (pd.to_numeric(q["bid"], errors="coerce") + pd.to_numeric(q["ask"], errors="coerce")) / 2.0
    tp = pd.to_numeric(t["price"], errors="coerce").dropna()
    sign = ms.tick_rule_sign(tp)
    t = t.loc[tp.index].copy()
    t["sign"] = sign

    # align each trade to latest mid quote
    mid_at_trade = q["mid"].reindex(t.index, method="ffill")
    mid_change = mid_at_trade.diff().shift(-1)  # next mid move after trade
    df = pd.DataFrame({"sign": t["sign"], "dmid_next": mid_change}).dropna()
    resp_buy = float(df.loc[df["sign"] > 0, "dmid_next"].mean())
    resp_sell = float(df.loc[df["sign"] < 0, "dmid_next"].mean())

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_hist(df["dmid_next"], "Next mid change after trade (distribution)", os.path.join(fig_dir, "dmid_hist.png"), bins=80),
    ]
    summary = {"E[dMid_next | Buy]": resp_buy, "E[dMid_next | Sell]": resp_sell, "n": int(len(df))}
    sections = [
        ("What you ran", f"- quotes: `{quotes_path}`\n- trades: `{trades_path}`"),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Price response to trade sign (toy)", "```json\n" + json.dumps(summary, indent=2) + "\n```"),
        ("Interpretation prompts", "\n".join([
            "- In informed-trading models, buys tend to move mid up (permanent-ish impact).",
            "- Separating bid–ask bounce from permanent response is the hard part.",
        ])),
    ]
    return write_report(out_dir, "Chapter 11 — Sequential trade: adverse selection intuition", sections)

def chapter12(lob_path: str, out_dir: str) -> tuple[str, str]:
    df = read_csv_ts(lob_path).set_index("ts").sort_index()
    mid = pd.to_numeric(df["mid"], errors="coerce")
    bid_sz = pd.to_numeric(df["bid_size"], errors="coerce")
    ask_sz = pd.to_numeric(df["ask_size"], errors="coerce")
    imb = (bid_sz - ask_sz) / (bid_sz + ask_sz)
    nxt = mid.diff().shift(-1)
    y = (nxt > 0).astype(int)  # next move up?
    tmp = pd.DataFrame({"imb": imb, "up": y}).dropna()
    # bin imbalance and compute empirical P(up)
    tmp["bin"] = pd.qcut(tmp["imb"], 10, duplicates="drop")
    p_up = tmp.groupby("bin")["up"].mean()

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_series(imb.dropna(), "Imbalance (best level)", os.path.join(fig_dir, "imb_series.png")),
        plots.plot_bar({str(k): float(v) for k, v in p_up.items()}, "Empirical P(next move up) by imbalance decile", os.path.join(fig_dir, "pup_by_bin.png")),
    ]
    sections = [
        ("What you ran", f"- lob snapshots: `{lob_path}`\n- n: {len(tmp)}"),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Empirical calibration", p_up.to_string()),
        ("Interpretation prompts", "\n".join([
            "- A queueing LOB model tries to compute these conditional probabilities from event intensities.",
            "- Even without a full model, imbalance is a strong baseline feature in many markets.",
        ])),
    ]
    return write_report(out_dir, "Chapter 12 — LOB as a queue: imbalance → conditional move (baseline)", sections)

def chapter13(out_dir: str, total_shares: float = 1_000_000, N: int = 20, risk_aversion: float = 0.15, sigma: float = 0.01, impact: float = 1e-6, sims: int = 2000) -> tuple[str, str]:
    # schedule (simple exponential weights)
    t = np.arange(N)
    w = np.exp(-risk_aversion * t)
    w = w / w.sum()
    schedule = total_shares * w

    rng = np.random.default_rng(0)
    shortfalls = []
    for _ in range(sims):
        # price path increments
        dP = sigma * rng.normal(size=N)
        # execution cost: slippage from impact (quadratic-ish proxy) + price risk
        # assume execute at each step with temporary impact proportional to rate
        rate = schedule
        tmp_impact = impact * (rate**2)
        # shortfall: sum(schedule * (price move before execute)) + impact term
        # toy: accumulated drift from dP
        P = np.cumsum(dP)
        sf = float(np.sum(schedule * P) + np.sum(tmp_impact))
        shortfalls.append(sf)
    sf = pd.Series(shortfalls, name="shortfall")

    fig_dir = os.path.join(out_dir, "figures")
    figs = [
        plots.plot_hist(sf, "Implementation shortfall (toy distribution)", os.path.join(fig_dir, "shortfall_hist.png"), bins=60),
        plots.plot_series(pd.Series(schedule), "AC-style schedule (shares per slice)", os.path.join(fig_dir, "schedule.png")),
    ]

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
        ("What you ran", "Toy execution simulation (not calibrated)."),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Summary", "```json\n" + json.dumps(summary, indent=2) + "\n```"),
        ("Interpretation prompts", "\n".join([
            "- Execution is a control problem: trade faster → less risk, more impact; trade slower → less impact, more risk.",
            "- In production: estimate sigma and impact from TCA and update because regimes change.",
        ])),
    ]
    return write_report(out_dir, "Chapter 13 — Execution: schedules & implementation shortfall (toy AC)", sections)

def chapter14(out_dir: str) -> tuple[str, str]:
    text = """<div class="note">
This chapter is intentionally checklist-shaped. Strategies die from *engineering reality* more often than from missing equations.
</div>

**Research hygiene**
- Define the information set (no lookahead, correct timestamps, trading calendar, corporate actions).
- Separate signal research from execution (spread, slippage, latency, rejects).
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
    sections = [("Checklist", text)]
    return write_report(out_dir, "Chapter 14 — A robust quant workflow (checklist)", sections)

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
    ap.add_argument("--vr-q", type=int, default=5)

    ap.add_argument("--J", type=int, default=60)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--top-frac", type=float, default=0.25)

    ap.add_argument("--pair", default="A,B")
    ap.add_argument("--z-win", type=int, default=60)
    ap.add_argument("--z-entry", type=float, default=1.5)

    ap.add_argument("--total-shares", type=float, default=1_000_000)
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--risk-aversion", type=float, default=0.15)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--impact", type=float, default=1e-6)
    ap.add_argument("--sims", type=int, default=2000)

    args = ap.parse_args()
    ensure_dir(args.out)

    ch = args.chapter
    fn = CHAPTERS[ch]

    if ch in ("chapter02", "chapter03"):
        if not args.prices:
            raise SystemExit("--prices required")
        if ch == "chapter02":
            md, html = fn(args.prices, args.out, lags=args.lags)
        else:
            md, html = fn(args.prices, args.out, lags=args.lags, vr_q=args.vr_q)

    elif ch == "chapter04":
        if not args.returns:
            raise SystemExit("--returns required")
        md, html = fn(args.returns, args.out)

    elif ch in ("chapter05", "chapter06", "chapter07"):
        if not args.prices_multi:
            raise SystemExit("--prices-multi required")
        if ch == "chapter05":
            md, html = fn(args.prices_multi, args.out, J=args.J, K=args.K, top_frac=args.top_frac)
        elif ch == "chapter06":
            md, html = fn(args.prices_multi, args.out, pair=args.pair, z_win=args.z_win, z_entry=args.z_entry)
        else:
            md, html = fn(args.prices_multi, args.out)

    elif ch == "chapter08":
        if not args.events:
            raise SystemExit("--events required")
        md, html = fn(args.events, args.out)

    elif ch in ("chapter09", "chapter11"):
        if not args.quotes or not args.trades:
            raise SystemExit("--quotes and --trades required")
        md, html = fn(args.quotes, args.trades, args.out)

    elif ch == "chapter10":
        if not args.trades:
            raise SystemExit("--trades required")
        md, html = fn(args.trades, args.out)

    elif ch == "chapter12":
        if not args.lob:
            raise SystemExit("--lob required")
        md, html = fn(args.lob, args.out)

    elif ch == "chapter13":
        md, html = fn(args.out, total_shares=args.total_shares, N=args.N, risk_aversion=args.risk_aversion, sigma=args.sigma, impact=args.impact, sims=args.sims)

    elif ch == "chapter14":
        md, html = fn(args.out)

    else:
        raise SystemExit("Unhandled chapter")

    print(md)
    print(html)

if __name__ == "__main__":
    main()
