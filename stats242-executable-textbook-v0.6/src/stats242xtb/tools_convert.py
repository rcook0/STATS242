from __future__ import annotations
import argparse
import json
import os
import yaml
import pandas as pd

from ..adapters import (
    canonicalize_prices,
    canonicalize_quotes,
    canonicalize_trades,
    canonicalize_events,
    canonicalize_lob,
)

KIND_FN = {
    "prices": canonicalize_prices,
    "quotes": canonicalize_quotes,
    "trades": canonicalize_trades,
    "events": canonicalize_events,
    "lob": canonicalize_lob,
}

def main():
    ap = argparse.ArgumentParser(prog="stats242xtb.tools.convert")
    ap.add_argument("--kind", choices=sorted(KIND_FN.keys()), required=True)
    ap.add_argument("--schema", default="auto", help="Schema hint: auto|taq|crypto|... (best-effort)")
    ap.add_argument("--in", dest="inp", required=True, help="Input CSV path")
    ap.add_argument("--out", required=True, help="Output canonical CSV path")
    ap.add_argument("--map", dest="map_path", default=None, help="YAML mapping file (optional)")
    ap.add_argument("--preview", action="store_true", help="Print inferred columns and head()")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    mapping = None
    if args.map_path:
        with open(args.map_path, "r", encoding="utf-8") as f:
            mapping = yaml.safe_load(f) or {}

    fn = KIND_FN[args.kind]
    out = fn(df, schema=args.schema, mapping=mapping)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    if args.preview:
        print("Output columns:", list(out.columns))
        print(out.head().to_string(index=False))
    print(args.out)

if __name__ == "__main__":
    main()
