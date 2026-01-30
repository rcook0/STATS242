from __future__ import annotations
import argparse
import json
import os

from .util import ensure_dir, read_csv_ts, relpath
from . import diagnostics as diag
from . import microstructure as ms
from . import plots
from .report import write_report

def chapter02(prices_path: str, out_dir: str, lags: int = 20) -> tuple[str, str]:
    df = read_csv_ts(prices_path)
    if "close" not in df.columns:
        raise ValueError("prices CSV must include 'close'")

    lr = diag.log_returns(df["close"])
    jb = diag.jarque_bera(lr)
    lb_r = diag.ljung_box(lr, lags=lags)
    lb_r2 = diag.ljung_box(lr**2, lags=lags)
    acf_r = diag.acf_simple(lr, max_lag=min(lags, 50))
    acf_r2 = diag.acf_simple(lr**2, max_lag=min(lags, 50))

    fig_dir = os.path.join(out_dir, "figures")
    f1 = plots.plot_series(lr, "Log returns (time series)", os.path.join(fig_dir, "returns_series.png"))
    f2 = plots.plot_hist(lr, "Log returns (histogram, density)", os.path.join(fig_dir, "returns_hist.png"))
    f3 = plots.plot_qq(lr, "QQ plot vs Normal (log returns)", os.path.join(fig_dir, "returns_qq.png"))
    f4 = plots.plot_acf_bar(acf_r, "ACF: returns", os.path.join(fig_dir, "acf_returns.png"))
    f5 = plots.plot_acf_bar(acf_r2, "ACF: squared returns (volatility clustering)", os.path.join(fig_dir, "acf_returns_sq.png"))

    md_figs = "\n".join([
        f"![]({relpath(out_dir, f1)})",
        f"![]({relpath(out_dir, f2)})",
        f"![]({relpath(out_dir, f3)})",
        f"![]({relpath(out_dir, f4)})",
        f"![]({relpath(out_dir, f5)})",
    ])

    sections = []
    sections.append(("What you ran", f"- prices: `{prices_path}`\n- n log-returns: {len(lr)}\n- lags: {lags}"))
    sections.append(("Charts", md_figs))
    sections.append(("Jarque–Bera normality check (log returns)", "```json\n" + json.dumps(jb, indent=2) + "\n```"))
    sections.append(("Ljung–Box on returns (linear autocorrelation?)", lb_r.to_string()))
    sections.append(("Ljung–Box on squared returns (volatility clustering)", lb_r2.to_string()))
    sections.append(("Interpretation prompts", "\n".join([
        "- Weak autocorr in returns + strong autocorr in squared returns = volatility clustering.",
        "- JB p-value near 0 is typical; focus on tail risk + stability across regimes.",
        "- If QQ plot bends in tails, Gaussian risk is lying to you."
    ])))

    return write_report(out_dir, "Chapter 02 — Returns, tails, and diagnostics", sections)

def chapter10(trades_path: str, out_dir: str) -> tuple[str, str]:
    df = read_csv_ts(trades_path)
    if "price" not in df.columns:
        raise ValueError("trades CSV must include 'price'")

    roll = ms.roll_spread_from_trades(df["price"])
    fig_dir = os.path.join(out_dir, "figures")
    f1 = plots.plot_series(df["price"], "Trade price (time series)", os.path.join(fig_dir, "trade_price_series.png"))
    dp = df["price"].diff().dropna()
    f2 = plots.plot_hist(dp, "Trade price changes ΔP (histogram)", os.path.join(fig_dir, "dp_hist.png"), bins=80)

    md_figs = "\n".join([
        f"![]({relpath(out_dir, f1)})",
        f"![]({relpath(out_dir, f2)})",
    ])

    sections = []
    sections.append(("What you ran", f"- trades: `{trades_path}`\n- n trades: {len(df)}"))
    sections.append(("Charts", md_figs))
    sections.append(("Roll estimator", "```json\n" + json.dumps(roll, indent=2) + "\n```"))
    sections.append(("Interpretation prompts", "\n".join([
        "- If lag-1 autocovariance isn't negative, Roll assumptions are violated (very common).",
        "- Compare `roll_spread` to quoted spread if you have quotes.",
        "- Use Roll as a diagnostic baseline, not a production estimator."
    ])))

    return write_report(out_dir, "Chapter 10 — Roll spread estimator", sections)

CHAPTERS = {"chapter02": chapter02, "chapter10": chapter10}

def main():
    ap = argparse.ArgumentParser(prog="stats242xtb.run")
    ap.add_argument("chapter", choices=sorted(CHAPTERS.keys()))
    ap.add_argument("--out", required=True)
    ap.add_argument("--prices")
    ap.add_argument("--trades")
    ap.add_argument("--lags", type=int, default=20)
    args = ap.parse_args()

    ensure_dir(args.out)

    if args.chapter == "chapter02":
        if not args.prices:
            raise SystemExit("--prices is required for chapter02")
        md_path, html_path = chapter02(args.prices, args.out, lags=args.lags)
    else:
        if not args.trades:
            raise SystemExit("--trades is required for chapter10")
        md_path, html_path = chapter10(args.trades, args.out)

    print(md_path)
    print(html_path)

if __name__ == "__main__":
    main()
