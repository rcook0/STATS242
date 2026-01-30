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
from . import metrics as met
from . import plots_perf as pperf

from .regression import fit_ols, summarize, summary_md
from .diagnostics_bundle import run_residual_diagnostics, diagnostics_md
from .rolling import rolling_ols

from .principia.ch04_inference import principia_ch04_inference
from .provenance import write_run_meta
from .data_contracts import describe_contract


def _prov(out_dir: str, chapter: str, params: dict, inputs: dict) -> None:
    try:
        write_run_meta(out_dir=out_dir, chapter=chapter, params=params, inputs=inputs)
    except Exception:
        pass



def _md_imgs(out_dir: str, paths: list[str]) -> str:
    return "\n".join([f"![]({relpath(out_dir, p)})" for p in paths])

def chapter01(*, out_dir: str) -> tuple[str, str]:
    ensure_dir(out_dir)
    _prov(out_dir, "chapter01", params={}, inputs={})

    primer = """<div class="note">
This executable reader is organized around **data contracts** and **chapter runners**.
Core chapters should run quickly and produce deterministic outputs (given identical inputs).
</div>

**Build the book**
- `python -m stats242xtb.tools.make_toy_data --kind suite --out data`
- `python -m stats242xtb.run_all --book configs/book.yml --out outputs/book_build`

**Validate data**
- `python -m stats242xtb.tools.validate --kind trades --in data/trades_toy.csv --strict`
- `python -m stats242xtb.tools.validate --kind quotes --in raw.csv --canonicalize --map configs/colmap_quotes.yml`

**Per-chapter outputs**
- `report.md`, `report.html`
- `figures/*.png`
- `artifacts/run_meta.json` (provenance)
"""

    contracts = []
    for k in ["prices","prices_multi","returns_capm","trades","quotes","events","lob"]:
        contracts.append("```\n" + describe_contract(k) + "\n```")

    sections = [
        ("Primer", primer),
        ("Data contracts", "\n\n".join(contracts)),
        ("Available chapters", "\n".join([f"- {c}" for c in sorted(CHAPTERS.keys())])),
    ]
    return write_report(out_dir, "Chapter 01 — Primer, contracts, and reproducibility", sections)

def _parse_num_list(x, *, default: list[float]) -> list[float]:
    if x is None:
        return list(default)
    if isinstance(x, (list, tuple)):
        return [float(v) for v in x]
    s = str(x).strip()
    if not s:
        return list(default)
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]

def _parse_str_list(x, *, default: list[str]) -> list[str]:
    if x is None:
        return list(default)
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    s = str(x).strip()
    if not s:
        return list(default)
    return [p.strip() for p in s.split(",") if p.strip()]

def _write_json(out_dir: str, rel_path: str, obj) -> str:
    p = os.path.join(out_dir, rel_path)
    ensure_dir(os.path.dirname(p))
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return p



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

def chapter05(
    *,
    out_dir: str,
    prices_multi: str,
    J: int = 60,
    K: int = 20,
    top_frac: float = 0.25,
    quantiles: int = 10,
    overlap: bool = True,
    long_short: bool = True,
    cost_bps_list = "0,10,25,50",
) -> tuple[str, str]:
    """Cross-sectional momentum / reversal lab (research-grade).

    - Signal: past J-period cumulative return (no lookahead via shift(1)).
    - Portfolio: long top bucket; optionally short bottom bucket.
    - Holding: K periods with optional overlap (K-cohort averaging).
    - Outputs: equity + drawdown + turnover + cost sensitivity + JSON artifacts.
    """
    _prov(out_dir, "chapter05", params={
        "prices_multi": prices_multi, "J": J, "K": K, "top_frac": top_frac,
        "quantiles": quantiles, "overlap": overlap, "long_short": long_short,
        "cost_bps_list": cost_bps_list,
    }, inputs={"prices_multi": prices_multi})

    ensure_dir(out_dir)
    df = read_csv_ts(prices_multi)
    asset_cols = [c for c in df.columns if c != "ts"]
    if len(asset_cols) < 3:
        raise ValueError("prices_multi needs >= 3 asset columns")
    R = fin.to_period_returns(df, "ts", asset_cols, rule="1D").dropna()
    if len(R) < max(J + K + 50, 200):
        raise ValueError("Not enough history after resample for a robust momentum lab.")

    # signal uses info up to t-1, applied to t
    past = R.rolling(int(J)).sum().shift(1)

    n_assets = len(asset_cols)
    if top_frac is not None and top_frac > 0:
        n_bucket = max(1, int(np.floor(float(top_frac) * n_assets)))
    else:
        q = max(2, int(quantiles))
        n_bucket = max(1, int(np.floor(n_assets / q)))

    # cohort weights (for overlap) and total weights by date
    dates = list(R.index)
    cohorts: list[pd.Series] = []
    W = []

    for i, d in enumerate(dates):
        sig = past.loc[d]
        if sig.isna().any():
            W.append(pd.Series(np.zeros(n_assets), index=asset_cols))
            continue

        # rank by signal
        ranks = sig.rank(ascending=True, method="first")
        order = ranks.sort_values()
        losers = list(order.index[:n_bucket])
        winners = list(order.index[-n_bucket:])

        w = pd.Series(0.0, index=asset_cols)
        if long_short:
            w.loc[winners] = +1.0 / len(winners)
            w.loc[losers] = -1.0 / len(losers)
        else:
            w.loc[winners] = +1.0 / len(winners)

        if overlap:
            cohorts.append(w)
            if len(cohorts) > int(K):
                cohorts = cohorts[-int(K):]
            w_tot = sum(cohorts) / float(len(cohorts))
        else:
            # rebalance every K periods, hold constant
            if i == 0 or (i % int(K) == 0):
                cohorts = [w]
            w_tot = cohorts[0]

        W.append(w_tot)

    W = pd.DataFrame(W, index=R.index, columns=asset_cols)
    port_r = (W * R).sum(axis=1)
    port_r.name = "gross_log_return"

    turnover = met.turnover_from_weights(W)
    costs = _parse_num_list(cost_bps_list, default=[0.0, 10.0, 25.0, 50.0])

    gross_stats = met.perf_stats(port_r)
    gross_eq = met.equity_curve(port_r)
    gross_dd = met.drawdown(gross_eq)

    # cost sweep
    sweep = met.cost_sweep(port_r, turnover, costs, exposure_mult=1.0)
    # convenience: also compute net series for a reference cost (10bps if present, else first)
    ref_cost = 10.0 if 10.0 in costs else float(costs[0])
    net_r = met.apply_costs(port_r, turnover, ref_cost)
    net_eq = met.equity_curve(net_r)
    net_dd = met.drawdown(net_eq)

    # inference-lite: t-test of mean return (illustrative, not production)
    try:
        from scipy import stats as _st
        tt = _st.ttest_1samp(pd.to_numeric(port_r, errors="coerce").dropna().to_numpy(), 0.0, alternative="two-sided")
        ttest = {"t": float(tt.statistic), "pvalue": float(tt.pvalue)}
    except Exception:
        ttest = {"t": float("nan"), "pvalue": float("nan")}

    fig_dir = os.path.join(out_dir, "figures")
    figs = []
    figs.append(pperf.plot_equity_multi(
        {"gross": gross_eq, f"net_{ref_cost:g}bps": net_eq},
        f"Momentum equity (gross vs net {ref_cost:g}bps)",
        os.path.join(fig_dir, "equity.png"),
    ))
    figs.append(pperf.plot_drawdown(gross_dd, "Drawdown (gross)", os.path.join(fig_dir, "drawdown_gross.png")))
    figs.append(pperf.plot_drawdown(net_dd, f"Drawdown (net {ref_cost:g}bps)", os.path.join(fig_dir, "drawdown_net.png")))
    figs.append(pperf.plot_turnover(turnover, "Turnover (half-L1 weight change)", os.path.join(fig_dir, "turnover.png")))
    # cost curve: net Sharpe vs cost
    cost_to_sharpe = {float(k): float(v.get("sharpe", float("nan"))) for k, v in sweep.items()}
    figs.append(pperf.plot_cost_curve(cost_to_sharpe, "Cost sensitivity (net Sharpe)", os.path.join(fig_dir, "cost_curve.png"), ylabel="Sharpe"))

    summary = {
        "chapter": "chapter05",
        "inputs": {"prices_multi": prices_multi},
        "params": {"J": int(J), "K": int(K), "n_bucket": int(n_bucket), "overlap": bool(overlap), "long_short": bool(long_short)},
        "gross": {**gross_stats, "max_drawdown": float(gross_dd.min()), "avg_turnover": float(turnover.mean())},
        "ttest_mean_return": ttest,
        "cost_sweep": sweep,
        "ref_cost_bps": float(ref_cost),
    }

    _write_json(out_dir, "artifacts/summary.json", summary)

    mc = model_card("Chapter 05", {
        "Signal": f"Cross-sectional ranks of past J={J} day cumulative log return (shifted by 1 day).",
        "Portfolio": f"{'Long winners / short losers' if long_short else 'Long winners only'}; bucket size={n_bucket} assets.",
        "Holding": f"K={K} days, {'overlapping K-cohort average' if overlap else 'non-overlapping rebalances'}.",
        "Costs": f"Linear turnover cost sweep: {costs} bps.",
    }, notes=["Research-grade implementation"])

    # Markdown table for cost sweep
    rows = []
    for bps in [float(x) for x in costs]:
        st = sweep.get(str(float(bps)), {})
        rows.append((bps, st.get("mean"), st.get("vol"), st.get("sharpe"), st.get("n")))
    table = "| cost (bps) | mean | vol | sharpe | n |\n|---:|---:|---:|---:|---:|\n" + \
            "\n".join([f"| {bps:>8.1f} | {m:> .6f} | {v:> .6f} | {s:> .3f} | {int(n) if n==n else 0} |"
                       for (bps,m,v,s,n) in rows])

    sections = [
        ("Model card", mc),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Cost sweep (net stats)", table),
        ("Summary artifact", "Saved to `artifacts/summary.json`."),
    ]
    return write_report(out_dir, "Chapter 05 — Cross-sectional momentum (research grade)", sections)


def chapter06(
    *,
    out_dir: str,
    prices_multi: str,
    pair: str = "A,B",
    z_win: int = 60,
    z_entry: float = 1.5,
    z_exit: float = 0.5,
    adf_maxlag: int = 1,
    target_vol: float = 0.10,
    allow_flip: bool = True,
    cost_bps_list = "0,10,25,50",
) -> tuple[str, str]:
    """Pair mean-reversion / cointegration lab (research-grade).

    - Engle–Granger: coint test and residual ADF.
    - Spread: log(y) - beta log(x), beta via OLS.
    - Trading rule: z-score entry/exit with optional flips.
    - Outputs: spread/z/position plots; equity/drawdown; costs; trade stats; JSON artifacts.
    """
    _prov(out_dir, "chapter06", params={
        "prices_multi": prices_multi, "pair": pair, "z_win": z_win,
        "z_entry": z_entry, "z_exit": z_exit, "adf_maxlag": adf_maxlag,
        "target_vol": target_vol, "allow_flip": allow_flip, "cost_bps_list": cost_bps_list,
    }, inputs={"prices_multi": prices_multi})

    ensure_dir(out_dir)
    df = read_csv_ts(prices_multi)
    asset_cols = [c for c in df.columns if c != "ts"]
    a, b = [p.strip() for p in str(pair).split(",")]
    if a not in asset_cols or b not in asset_cols:
        raise ValueError(f"pair must be in columns; got {a},{b}; available={asset_cols}")

    P = df.set_index("ts").sort_index()[[a, b]].dropna()
    # daily sampling
    P = P.resample("1D").last().dropna()

    loga = np.log(pd.to_numeric(P[a], errors="coerce")).dropna()
    logb = np.log(pd.to_numeric(P[b], errors="coerce")).dropna()
    idx = loga.index.intersection(logb.index)
    loga = loga.loc[idx]; logb = logb.loc[idx]

    beta = models.hedge_ratio_ols(loga, logb)
    spread = (loga - beta * logb).rename("spread")

    # cointegration diagnostics (Engle–Granger)
    try:
        from statsmodels.tsa.stattools import coint, adfuller
        coint_t, coint_p, _ = coint(loga.to_numpy(), logb.to_numpy())
        adf = adfuller(spread.to_numpy(), maxlag=int(adf_maxlag), regression="c", autolag=None)
        adf_stat, adf_p = float(adf[0]), float(adf[1])
    except Exception:
        coint_t, coint_p, adf_stat, adf_p = float("nan"), float("nan"), float("nan"), float("nan")

    z = fin.zscore(spread, int(z_win)).rename("z")
    z_sig = z.shift(1)  # decision at t uses z_{t-1}

    # returns for pnl
    rA = loga.diff().rename("rA")
    rB = logb.diff().rename("rB")
    R = pd.concat([rA, rB, z_sig], axis=1).dropna()
    spread_ret = (R["rA"] - beta * R["rB"]).rename("spread_ret")

    # position state machine
    pos = []
    cur = 0.0
    for val in R["z"].to_numpy():
        zt = float(val)
        if cur == 0.0:
            if zt > float(z_entry):
                cur = -1.0
            elif zt < -float(z_entry):
                cur = +1.0
        else:
            if abs(zt) < float(z_exit):
                cur = 0.0
            elif allow_flip:
                if cur > 0 and zt > float(z_entry):
                    cur = -1.0
                elif cur < 0 and zt < -float(z_entry):
                    cur = +1.0
        pos.append(cur)
    pos = pd.Series(pos, index=R.index, name="pos")

    # volatility targeting (simple)
    strat_r = (pos * spread_ret).rename("gross_log_return")
    if target_vol and target_vol > 0:
        # scale by rolling vol estimate
        vol = strat_r.rolling(60).std(ddof=1) * np.sqrt(met.TRADING_DAYS)
        scale = (float(target_vol) / vol).clip(0.0, 5.0).fillna(0.0)
        strat_r = (scale * strat_r).rename("gross_log_return")
        exposure_mult = float((1.0 + abs(beta)))  # legs
    else:
        exposure_mult = float((1.0 + abs(beta)))

    # costs: applied on position changes, scaled by gross exposure
    turnover = (pos.diff().abs().fillna(0.0)).rename("turnover")
    costs = _parse_num_list(cost_bps_list, default=[0.0, 10.0, 25.0, 50.0])
    sweep = met.cost_sweep(strat_r, turnover, costs, exposure_mult=exposure_mult)

    ref_cost = 10.0 if 10.0 in costs else float(costs[0])
    net_r = met.apply_costs(strat_r, turnover, ref_cost, exposure_mult=exposure_mult)

    gross_eq = met.equity_curve(strat_r)
    net_eq = met.equity_curve(net_r)
    gross_dd = met.drawdown(gross_eq)
    net_dd = met.drawdown(net_eq)

    trade_stats = met.trades_from_position(pos, strat_r)

    # OU fit via AR(1) mapping (spread is level series)
    try:
        ar1 = models.fit_ar1(spread.dropna())
        phi = float(ar1["phi"])
        kappa = float(-np.log(phi)) if 0 < phi < 1 else float("nan")
        mu = float(ar1["mu"])
        ou = {"phi": phi, "kappa": kappa, "mu": mu, "half_life_steps": float(ar1.get("half_life_steps", float("nan")))}
    except Exception:
        ou = {"phi": float("nan"), "kappa": float("nan"), "mu": float("nan"), "half_life_steps": float("nan")}

    fig_dir = os.path.join(out_dir, "figures")
    figs = []
    figs.append(plots.plot_series(spread, f"Spread: log({a}) - beta log({b}), beta={beta:.3f}", os.path.join(fig_dir, "spread.png")))
    figs.append(plots.plot_series(z.dropna(), f"Z-score (win={z_win}), entry={z_entry}, exit={z_exit}", os.path.join(fig_dir, "zscore.png")))
    figs.append(plots.plot_series(pos, "Position (+1 long spread, -1 short spread)", os.path.join(fig_dir, "position.png")))
    figs.append(pperf.plot_equity_multi({"gross": gross_eq, f"net_{ref_cost:g}bps": net_eq},
                                        f"Pair strategy equity (gross vs net {ref_cost:g}bps)",
                                        os.path.join(fig_dir, "equity.png")))
    figs.append(pperf.plot_turnover(turnover, "Turnover (|Δpos|)", os.path.join(fig_dir, "turnover.png")))
    cost_to_sharpe = {float(k): float(v.get("sharpe", float("nan"))) for k, v in sweep.items()}
    figs.append(pperf.plot_cost_curve(cost_to_sharpe, "Cost sensitivity (net Sharpe)", os.path.join(fig_dir, "cost_curve.png"), ylabel="Sharpe"))

    summary = {
        "chapter": "chapter06",
        "inputs": {"prices_multi": prices_multi},
        "params": {"pair": pair, "z_win": int(z_win), "z_entry": float(z_entry), "z_exit": float(z_exit), "target_vol": float(target_vol)},
        "diagnostics": {"beta": float(beta), "coint_t": float(coint_t), "coint_p": float(coint_p), "adf_stat": float(adf_stat), "adf_p": float(adf_p), "ou": ou},
        "gross": {**met.perf_stats(strat_r), "max_drawdown": float(gross_dd.min())},
        "trade_stats": trade_stats,
        "cost_sweep": sweep,
        "ref_cost_bps": float(ref_cost),
    }
    _write_json(out_dir, "artifacts/summary.json", summary)

    mc = model_card("Chapter 06", {
        "Spread": f"log({a}) - beta log({b}), beta={beta:.3f} (OLS).",
        "Cointegration": f"Engle–Granger coint p={coint_p:.4g}; ADF on residual p={adf_p:.4g}.",
        "Signal": f"z-score of spread (win={z_win}) shifted by 1 period; enter at ±{z_entry}, exit at |z|<{z_exit}.",
        "Sizing": f"Vol targeting to {target_vol:.0%} annual (cap 5×) on spread returns.",
        "Costs": f"Linear cost sweep: {costs} bps scaled by gross exposure (1+|beta|={1+abs(beta):.3f}).",
    }, notes=["Research-grade implementation"])

    rows = []
    for bps in [float(x) for x in costs]:
        st = sweep.get(str(float(bps)), {})
        rows.append((bps, st.get("mean"), st.get("vol"), st.get("sharpe"), st.get("n")))
    table = "| cost (bps) | mean | vol | sharpe | n |\n|---:|---:|---:|---:|---:|\n" + \
            "\n".join([f"| {bps:>8.1f} | {m:> .6f} | {v:> .6f} | {s:> .3f} | {int(n) if n==n else 0} |"
                       for (bps,m,v,s,n) in rows])

    extra = "```json\n" + json.dumps({"beta": beta, "cointegration_p": coint_p, "adf_p": adf_p, "ou": ou, "trade_stats": trade_stats}, indent=2) + "\n```"

    sections = [
        ("Model card", mc),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Cost sweep (net stats)", table),
        ("Diagnostics", extra),
        ("Summary artifact", "Saved to `artifacts/summary.json`."),
    ]
    return write_report(out_dir, "Chapter 06 — Pair mean reversion / cointegration (research grade)", sections)


def chapter07(
    *,
    out_dir: str,
    prices_multi: str,
    methods = "equal,inv_vol,min_var,risk_parity,mean_var",
    rebalance_days: int = 21,
    lookback_days: int = 252,
    shrink: float = 0.2,
    risk_aversion: float = 10.0,
    cost_bps_list = "0,10,25,50",
) -> tuple[str, str]:
    """Portfolio construction + rebalancing lab (research-grade).

    - Universe: multi-asset daily returns.
    - Methods: equal, inverse-vol, long-only min-var, risk-parity, mean-variance (long-only).
    - Backtest: rolling lookback, periodic rebalances, turnover + cost sweep.
    """
    _prov(out_dir, "chapter07", params={
        "prices_multi": prices_multi, "methods": methods, "rebalance_days": rebalance_days,
        "lookback_days": lookback_days, "shrink": shrink, "risk_aversion": risk_aversion,
        "cost_bps_list": cost_bps_list,
    }, inputs={"prices_multi": prices_multi})

    ensure_dir(out_dir)
    df = read_csv_ts(prices_multi)
    asset_cols = [c for c in df.columns if c != "ts"]
    if len(asset_cols) < 3:
        raise ValueError("prices_multi needs >= 3 asset columns")
    R = fin.to_period_returns(df, "ts", asset_cols, rule="1D").dropna()
    if len(R) < max(int(lookback_days) + 100, 400):
        raise ValueError("Not enough history for rolling portfolio construction lab.")

    methods_list = _parse_str_list(methods, default=["equal","inv_vol","min_var","risk_parity","mean_var"])
    costs = _parse_num_list(cost_bps_list, default=[0.0, 10.0, 25.0, 50.0])
    ref_cost = 10.0 if 10.0 in costs else float(costs[0])

    def w_fn(name: str, Rw: pd.DataFrame) -> pd.Series:
        name = name.lower()
        if name == "equal":
            return met.weight_equal(Rw.columns)
        if name == "inv_vol":
            return met.weight_inv_vol(Rw)
        if name == "min_var":
            return met.weight_min_var_long_only(Rw, shrink=float(shrink))
        if name == "risk_parity":
            return met.weight_risk_parity_long_only(Rw, shrink=float(shrink))
        if name == "mean_var":
            return met.weight_mean_var_long_only(Rw, risk_aversion=float(risk_aversion), shrink=float(shrink))
        raise ValueError(f"Unknown method: {name}")

    # backtest loop
    W_all = {}
    R_all = {}
    T_all = {}
    sweeps = {}
    gross_stats = {}
    net_stats = {}

    idx = R.index
    for name in methods_list:
        W = []
        cur_w = pd.Series(0.0, index=asset_cols)
        for i, d in enumerate(idx):
            if i < int(lookback_days):
                W.append(cur_w)
                continue
            if (i - int(lookback_days)) % int(rebalance_days) == 0:
                window = R.iloc[i - int(lookback_days):i]
                cur_w = w_fn(name, window).reindex(asset_cols).fillna(0.0)
            W.append(cur_w)
        W = pd.DataFrame(W, index=idx, columns=asset_cols)
        port_r = (W * R).sum(axis=1).rename("gross_log_return")
        turnover = met.turnover_from_weights(W)

        sweep = met.cost_sweep(port_r, turnover, costs, exposure_mult=1.0)
        net_r = met.apply_costs(port_r, turnover, ref_cost)

        W_all[name] = W
        R_all[name] = port_r
        T_all[name] = turnover
        sweeps[name] = sweep
        gross_stats[name] = {**met.perf_stats(port_r), "max_drawdown": met.max_drawdown(met.equity_curve(port_r)), "avg_turnover": float(turnover.mean())}
        net_stats[name] = {**met.perf_stats(net_r), "max_drawdown": met.max_drawdown(met.equity_curve(net_r)), "avg_turnover": float(turnover.mean())}

    # Plots
    fig_dir = os.path.join(out_dir, "figures")
    figs = []
    eq_gross = {name: met.equity_curve(R_all[name]) for name in methods_list}
    eq_net = {name: met.equity_curve(met.apply_costs(R_all[name], T_all[name], ref_cost)) for name in methods_list}
    figs.append(pperf.plot_equity_multi(eq_gross, "Equity curves (gross) by method", os.path.join(fig_dir, "equity_gross.png")))
    figs.append(pperf.plot_equity_multi(eq_net, f"Equity curves (net {ref_cost:g}bps) by method", os.path.join(fig_dir, "equity_net.png")))

    # turnover for one representative method (median by avg turnover)
    med_name = sorted(methods_list, key=lambda n: float(np.nan_to_num(gross_stats[n].get("avg_turnover", 0.0))))[len(methods_list)//2]
    figs.append(pperf.plot_turnover(T_all[med_name], f"Turnover series ({med_name})", os.path.join(fig_dir, "turnover.png")))
    figs.append(pperf.plot_weights_heatmap(W_all[med_name].fillna(0.0), f"Weights heatmap ({med_name})", os.path.join(fig_dir, "weights_heatmap.png")))

    # cost curve for Sharpe by method (at each bps)
    # store just net sharpe for plotting at each cost per method via separate figure
    for name in methods_list[:3]:  # keep the report compact
        c2s = {float(k): float(v.get("sharpe", float("nan"))) for k, v in sweeps[name].items()}
        figs.append(pperf.plot_cost_curve(c2s, f"Cost sensitivity (net Sharpe) — {name}", os.path.join(fig_dir, f"cost_curve_{name}.png"), ylabel="Sharpe"))

    # Artifacts
    summary = {
        "chapter": "chapter07",
        "inputs": {"prices_multi": prices_multi},
        "params": {"methods": methods_list, "rebalance_days": int(rebalance_days), "lookback_days": int(lookback_days), "shrink": float(shrink), "risk_aversion": float(risk_aversion)},
        "ref_cost_bps": float(ref_cost),
        "gross_stats": gross_stats,
        "net_stats_ref_cost": net_stats,
        "cost_sweeps": sweeps,
        "representative_method": med_name,
    }
    _write_json(out_dir, "artifacts/summary.json", summary)

    # Markdown table
    hdr = "| method | sharpe (gross) | mdd (gross) | avg turnover | sharpe (net) | mdd (net) |\n|---|---:|---:|---:|---:|---:|\n"
    rows = []
    for name in methods_list:
        gs = gross_stats[name]
        ns = net_stats[name]
        rows.append(
            f"| {name} | {gs.get('sharpe', float('nan')): .3f} | {gs.get('max_drawdown', float('nan')): .3f} | {gs.get('avg_turnover', float('nan')): .4f} | "
            f"{ns.get('sharpe', float('nan')): .3f} | {ns.get('max_drawdown', float('nan')): .3f} |"
        )
    table = hdr + "\n".join(rows)

    mc = model_card("Chapter 07", {
        "Rebalancing": f"Every {rebalance_days} trading days with lookback={lookback_days} days.",
        "Universe": f"{len(asset_cols)} assets from prices_multi -> daily log returns.",
        "Methods": ", ".join(methods_list),
        "Costs": f"Linear turnover cost sweep: {costs} bps; report includes net at {ref_cost:g}bps.",
        "Shrinkage": f"Diagonal shrinkage λ={shrink} for covariance-based methods.",
    }, notes=["Research-grade implementation"])

    sections = [
        ("Model card", mc),
        ("Charts", _md_imgs(out_dir, figs)),
        ("Method comparison", table),
        ("Summary artifact", "Saved to `artifacts/summary.json`."),
    ]
    return write_report(out_dir, "Chapter 07 — Portfolio construction & rebalancing (research grade)", sections)


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

    ap.add_argument("--quantiles", type=int, default=10)
    ap.add_argument("--long-only", action="store_true")
    ap.add_argument("--cost-bps-list", default="0,10,25,50")


    ap.add_argument("--pair", default="A,B")
    ap.add_argument("--z-win", type=int, default=60)
    ap.add_argument("--z-entry", type=float, default=1.5)
    ap.add_argument("--adf-maxlag", type=int, default=1)

    ap.add_argument("--z-exit", type=float, default=0.5)
    ap.add_argument("--target-vol", type=float, default=0.10)
    ap.add_argument("--no-flip", action="store_true")


    ap.add_argument("--cov-type", default="HAC")
    ap.add_argument("--hac-lags", type=int, default=5)
    ap.add_argument("--hac-kernel", default="bartlett")
    ap.add_argument("--rolling-window", type=int, default=0)
    ap.add_argument("--rolling-step", type=int, default=5)

    ap.add_argument("--grid", default="1min")
    ap.add_argument("--sign-method", default="tick_rule")

    ap.add_argument("--methods", default="equal,inv_vol,min_var,risk_parity,mean_var")
    ap.add_argument("--rebalance-days", type=int, default=21)
    ap.add_argument("--lookback-days", type=int, default=252)
    ap.add_argument("--shrink", type=float, default=0.2)
    ap.add_argument("--risk-aversion", type=float, default=10.0)

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
            kwargs.update({
                "prices_multi": args.prices_multi,
                "J": args.J,
                "K": args.K,
                "top_frac": args.top_frac,
                "quantiles": args.quantiles,
                "overlap": args.overlap,
                "long_short": (not args.long_only),
                "cost_bps_list": args.cost_bps_list,
            })
        elif ch == "chapter06":
            kwargs.update({
                "prices_multi": args.prices_multi,
                "pair": args.pair,
                "z_win": args.z_win,
                "z_entry": args.z_entry,
                "z_exit": args.z_exit,
                "adf_maxlag": args.adf_maxlag,
                "target_vol": args.target_vol,
                "allow_flip": (not args.no_flip),
                "cost_bps_list": args.cost_bps_list,
            })
        else:
            kwargs.update({
                "prices_multi": args.prices_multi,
                "methods": args.methods,
                "rebalance_days": args.rebalance_days,
                "lookback_days": args.lookback_days,
                "shrink": args.shrink,
                "risk_aversion": args.risk_aversion,
                "cost_bps_list": args.cost_bps_list,
            })
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
