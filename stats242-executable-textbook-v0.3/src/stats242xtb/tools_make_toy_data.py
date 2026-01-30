from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def make_prices(out_file: str, n: int = 2000, start: str = "2020-01-01T00:00:00Z", freq: str = "1min") -> str:
    ts0 = pd.to_datetime(start, utc=True)
    ts = pd.date_range(ts0, periods=n, freq=freq)
    rng = np.random.default_rng(42)
    vol = 0.001
    r = []
    for _ in range(n):
        vol = 0.99 * vol + 0.01 * (0.0005 + 0.002 * abs(rng.normal()))
        r.append(vol * rng.normal())
    p = 100 * np.exp(np.cumsum(r))
    pd.DataFrame({"ts": ts, "close": p}).to_csv(out_file, index=False)
    return out_file

def make_prices_multi(out_file: str, n: int = 800, start: str = "2019-01-01T00:00:00Z", freq: str = "1D", assets=("A","B","C","D")) -> str:
    ts0 = pd.to_datetime(start, utc=True)
    ts = pd.date_range(ts0, periods=n, freq=freq)
    rng = np.random.default_rng(123)
    base = 100 * np.exp(np.cumsum(0.001 * rng.normal(size=n)))
    data = {"ts": ts}
    for i, a in enumerate(assets):
        noise = np.cumsum(0.0025 * rng.normal(size=n))
        drift = 0.0002 * (i - 1.5) * np.arange(n) / n
        data[a] = base * np.exp(noise + drift)
    pd.DataFrame(data).to_csv(out_file, index=False)
    return out_file

def make_returns_capm(out_file: str, n: int = 800, start: str = "2019-01-01T00:00:00Z", freq: str = "1D") -> str:
    ts0 = pd.to_datetime(start, utc=True)
    ts = pd.date_range(ts0, periods=n, freq=freq)
    rng = np.random.default_rng(9)
    mkt = 0.0002 + 0.01 * rng.normal(size=n)
    rf = np.full(n, 0.00002)
    beta = 1.1
    alpha = 0.00005
    strat = alpha + beta * (mkt - rf) + 0.012 * rng.normal(size=n)
    pd.DataFrame({"ts": ts, "strategy": strat, "market": mkt, "rf": rf}).to_csv(out_file, index=False)
    return out_file

def make_trades(out_file: str, n: int = 4000, start: str = "2020-01-01T00:00:00Z") -> str:
    ts0 = pd.to_datetime(start, utc=True)
    rng = np.random.default_rng(7)
    ts = [ts0 + pd.Timedelta(seconds=int(i * 2 + rng.integers(0, 2))) for i in range(n)]
    mid = 100 * np.exp(np.cumsum(0.0002 * rng.normal(size=n)))
    spread = 0.02 + 0.03 * np.abs(rng.normal(size=n))
    side = rng.choice([-1, 1], size=n)
    price = mid + side * spread / 2
    size = rng.integers(1, 10, size=n)
    pd.DataFrame({"ts": ts, "price": price, "size": size}).to_csv(out_file, index=False)
    return out_file

def make_quotes(out_file: str, n: int = 4000, start: str = "2020-01-01T00:00:00Z") -> str:
    ts0 = pd.to_datetime(start, utc=True)
    rng = np.random.default_rng(8)
    ts = [ts0 + pd.Timedelta(seconds=int(i * 2 + rng.integers(0, 2))) for i in range(n)]
    mid = 100 * np.exp(np.cumsum(0.00015 * rng.normal(size=n)))
    spr = 0.02 + 0.02 * np.abs(rng.normal(size=n))
    bid = mid - spr/2
    ask = mid + spr/2
    bid_size = rng.integers(50, 300, size=n)
    ask_size = rng.integers(50, 300, size=n)
    pd.DataFrame({"ts": ts, "bid": bid, "ask": ask, "bid_size": bid_size, "ask_size": ask_size}).to_csv(out_file, index=False)
    return out_file

def make_events(out_file: str, n: int = 5000, start: str = "2020-01-01T00:00:00Z") -> str:
    ts0 = pd.to_datetime(start, utc=True)
    rng = np.random.default_rng(5)
    # ACD-ish irregular times with clustering: durations drawn from mixture
    durations = rng.exponential(scale=1.2, size=n) * (1 + 2*(rng.random(size=n) < 0.1))
    t = np.cumsum(durations)
    ts = [ts0 + pd.Timedelta(seconds=float(s)) for s in t]
    pd.DataFrame({"ts": ts}).to_csv(out_file, index=False)
    return out_file

def make_lob_snapshots(out_file: str, n: int = 4000, start: str = "2020-01-01T00:00:00Z") -> str:
    ts0 = pd.to_datetime(start, utc=True)
    rng = np.random.default_rng(33)
    ts = [ts0 + pd.Timedelta(seconds=int(i * 2)) for i in range(n)]
    mid = 100 * np.exp(np.cumsum(0.0001 * rng.normal(size=n)))
    bid_size = rng.integers(50, 400, size=n).astype(float)
    ask_size = rng.integers(50, 400, size=n).astype(float)
    # induce mild correlation: if bid_size > ask_size, make next mid move more likely up
    for i in range(n-1):
        imb = (bid_size[i] - ask_size[i]) / (bid_size[i] + ask_size[i])
        mid[i+1] *= np.exp(0.00008 * np.sign(imb) + 0.00005 * rng.normal())
    pd.DataFrame({"ts": ts, "mid": mid, "bid_size": bid_size, "ask_size": ask_size}).to_csv(out_file, index=False)
    return out_file

def make_suite(out_dir: str) -> str:
    _ensure_dir(out_dir)
    make_prices(os.path.join(out_dir, "prices_toy.csv"))
    make_prices_multi(os.path.join(out_dir, "prices_multi_toy.csv"))
    make_returns_capm(os.path.join(out_dir, "returns_capm_toy.csv"))
    make_trades(os.path.join(out_dir, "trades_toy.csv"))
    make_quotes(os.path.join(out_dir, "quotes_toy.csv"))
    make_events(os.path.join(out_dir, "events_toy.csv"))
    make_lob_snapshots(os.path.join(out_dir, "lob_toy.csv"))
    return out_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["prices", "prices_multi", "returns_capm", "trades", "quotes", "events", "lob", "suite"], default="prices")
    ap.add_argument("--out", required=True, help="Output path: file.csv (for single) or directory (for suite)")
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    if args.kind == "suite":
        make_suite(args.out)
        print(args.out)
        return

    if args.kind == "prices":
        make_prices(args.out, n=args.n)
    elif args.kind == "prices_multi":
        make_prices_multi(args.out, n=max(300, args.n))
    elif args.kind == "returns_capm":
        make_returns_capm(args.out, n=max(300, args.n))
    elif args.kind == "trades":
        make_trades(args.out, n=max(4000, args.n))
    elif args.kind == "quotes":
        make_quotes(args.out, n=max(4000, args.n))
    elif args.kind == "events":
        make_events(args.out, n=max(5000, args.n))
    elif args.kind == "lob":
        make_lob_snapshots(args.out, n=max(4000, args.n))
    print(args.out)

if __name__ == "__main__":
    main()
