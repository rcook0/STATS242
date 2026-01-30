from __future__ import annotations
import argparse
import json
import os
import yaml
import pandas as pd

from .data_contracts import validate as validate_df, describe_contract, CONTRACTS
from .adapters import (
    canonicalize_prices,
    canonicalize_quotes,
    canonicalize_trades,
    canonicalize_events,
    canonicalize_lob,
)

ADAPTERS = {
    "prices": canonicalize_prices,
    "quotes": canonicalize_quotes,
    "trades": canonicalize_trades,
    "events": canonicalize_events,
    "lob": canonicalize_lob,
}

def main():
    ap = argparse.ArgumentParser(prog="stats242xtb.tools.validate")
    ap.add_argument("--kind", choices=sorted(CONTRACTS.keys()), required=True)
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV path")
    ap.add_argument("--schema", default="auto")
    ap.add_argument("--map", dest="map_path", default=None, help="Optional YAML mapping for adapters")
    ap.add_argument("--canonicalize", action="store_true", help="Run adapter canonicalization (where supported) before validation")
    ap.add_argument("--strict", action="store_true", help="Fail on key contract violations")
    ap.add_argument("--out", default=None, help="Write JSON report to this path")
    ap.add_argument("--describe", action="store_true", help="Print the data contract description and exit")
    ap.add_argument("--head", type=int, default=5, help="Print head(n) after validation")
    args = ap.parse_args()

    if args.describe:
        print(describe_contract(args.kind))
        return

    df = pd.read_csv(args.inp)

    mapping = None
    if args.map_path:
        with open(args.map_path, "r", encoding="utf-8") as f:
            mapping = yaml.safe_load(f) or {}

    if args.canonicalize and args.kind in ADAPTERS:
        df = ADAPTERS[args.kind](df, schema=args.schema, mapping=mapping)

    report = validate_df(args.kind, df, strict=args.strict)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(args.out)
    else:
        print(json.dumps(report, indent=2))

    if args.head and args.head > 0:
        print("\n--- head() ---")
        print(df.head(args.head).to_string(index=False))

if __name__ == "__main__":
    main()
