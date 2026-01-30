from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

def make_toy_prices(n: int, out: str, start: str = "2020-01-01T00:00:00Z") -> str:
    ts0 = pd.to_datetime(start, utc=True)
    ts = [ts0 + pd.Timedelta(minutes=i) for i in range(n)]
    rng = np.random.default_rng(42)
    vol = 0.001
    r = []
    for _ in range(n):
        vol = 0.99 * vol + 0.01 * (0.0005 + 0.002 * abs(rng.normal()))
        r.append(vol * rng.normal())
    p = 100 * np.exp(np.cumsum(r))
    pd.DataFrame({"ts": ts, "close": p}).to_csv(out, index=False)
    return out

def make_toy_trades(n: int, out: str, start: str = "2020-01-01T00:00:00Z") -> str:
    ts0 = pd.to_datetime(start, utc=True)
    rng = np.random.default_rng(7)
    ts = [ts0 + pd.Timedelta(seconds=int(i * 2 + rng.integers(0, 2))) for i in range(n)]
    mid = 100 * np.exp(np.cumsum(0.0002 * rng.normal(size=n)))
    spread = 0.02 + 0.03 * np.abs(rng.normal(size=n))
    side = rng.choice([-1, 1], size=n)
    price = mid + side * spread / 2
    size = rng.integers(1, 10, size=n)
    pd.DataFrame({"ts": ts, "price": price, "size": size}).to_csv(out, index=False)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["prices", "trades"], default="prices")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()
    if args.kind == "prices":
        make_toy_prices(args.n, args.out)
    else:
        make_toy_trades(max(args.n, 4000), args.out)
    print(args.out)

if __name__ == "__main__":
    main()
