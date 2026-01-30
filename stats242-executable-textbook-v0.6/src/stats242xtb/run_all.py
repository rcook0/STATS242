from __future__ import annotations
import argparse
import os
from .book import build as build_book
from .util import ensure_dir, utc_now_stamp

def main():
    ap = argparse.ArgumentParser(prog="stats242xtb.run_all")
    ap.add_argument("--book", required=True)
    ap.add_argument("--out", required=True, help="Root output folder")
    ap.add_argument("--stamp", action="store_true", help="Write into a timestamped subfolder and update <out>/latest symlink-like folder")
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    ensure_dir(args.out)
    if args.stamp or True:
        # default: stamp always; creates reproducible snapshots
        stamp = utc_now_stamp()
        run_dir = os.path.join(args.out, stamp)
    else:
        run_dir = args.out

    build_book(args.book, run_dir, data_root=args.data_root)

    # update latest pointer (directory copy of index linking)
    latest = os.path.join(args.out, "latest")
    try:
        # replace latest folder with small redirect index
        ensure_dir(latest)
        with open(os.path.join(latest, "README.txt"), "w", encoding="utf-8") as f:
            f.write(f"Latest build is in: {run_dir}\n")
        # create a tiny html that redirects
        with open(os.path.join(latest, "index.html"), "w", encoding="utf-8") as f:
            f.write(f"""<!doctype html><meta http-equiv="refresh" content="0; url=../{stamp}/index.html">""")
    except Exception:
        pass

    print(run_dir)

if __name__ == "__main__":
    main()
