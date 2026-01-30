from __future__ import annotations
import argparse
import os
from .book import build as build_book
from .util import ensure_dir, utc_now_stamp

def main():
    ap = argparse.ArgumentParser(prog="stats242xtb.run_all")
    ap.add_argument("--book", required=True)
    ap.add_argument("--out", required=True, help="Root output folder")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    ensure_dir(args.out)
    stamp = utc_now_stamp()
    run_dir = os.path.join(args.out, stamp)
    build_book(args.book, run_dir, data_root=args.data_root)

    latest = os.path.join(args.out, "latest")
    ensure_dir(latest)
    with open(os.path.join(latest, "README.txt"), "w", encoding="utf-8") as f:
        f.write(f"Latest build is in: {run_dir}\n")
    with open(os.path.join(latest, "index.html"), "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><meta http-equiv='refresh' content='0; url=../{stamp}/index.html'>")
    print(run_dir)

if __name__ == "__main__":
    main()
