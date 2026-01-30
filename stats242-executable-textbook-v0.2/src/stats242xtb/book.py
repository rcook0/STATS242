from __future__ import annotations
import argparse
import os
import json
from datetime import datetime
import yaml

from .util import ensure_dir, relpath
from .run import CHAPTERS

def _resolve_path(p: str | None, base: str) -> str | None:
    if p is None:
        return None
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base, p))

def build(book_path: str, out_dir: str, data_root: str | None = None) -> tuple[str, str]:
    book_dir = ensure_dir(out_dir)
    base = os.path.dirname(os.path.abspath(book_path))
    if data_root is None:
        data_root = base

    with open(book_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    chapters = cfg.get("chapters", [])
    if not chapters:
        raise ValueError("book.yml must contain a top-level 'chapters:' list")

    results = []
    for item in chapters:
        key = item["chapter"]
        name = item.get("name", key)
        sub = item.get("out", key)
        args = item.get("args", {}) or {}
        chapter_out = ensure_dir(os.path.join(book_dir, sub))

        if key not in CHAPTERS:
            raise ValueError(f"Unknown chapter '{key}'. Available: {sorted(CHAPTERS.keys())}")

        if "prices" in args:
            args["prices"] = _resolve_path(args["prices"], data_root)
        if "trades" in args:
            args["trades"] = _resolve_path(args["trades"], data_root)

        fn = CHAPTERS[key]
        if key == "chapter02":
            md_path, html_path = fn(args["prices"], chapter_out, lags=int(args.get("lags", 20)))
        elif key == "chapter10":
            md_path, html_path = fn(args["trades"], chapter_out)
        else:
            md_path, html_path = fn(**args, out_dir=chapter_out)

        results.append({
            "chapter": key,
            "name": name,
            "out": sub,
            "report_md": relpath(book_dir, md_path),
            "report_html": relpath(book_dir, html_path),
        })

    index_md_lines = ["# STATS 242 Executable Textbook — Book Build", ""]
    for r in results:
        index_md_lines.append(f"- **{r['name']}** — [HTML]({r['report_html']}) | [MD]({r['report_md']})")
    index_md = "\n".join(index_md_lines) + "\n"
    index_md_path = os.path.join(book_dir, "index.md")
    with open(index_md_path, "w", encoding="utf-8") as f:
        f.write(index_md)

    html_items = "\n".join([f"<li><b>{r['name']}</b> — <a href='{r['report_html']}'>HTML</a> | <a href='{r['report_md']}'>MD</a></li>" for r in results])
    index_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>STATS 242 Book Build</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 32px; max-width: 980px; }}
    li {{ margin: 8px 0; }}
    a {{ text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .meta {{ color: #666; }}
  </style>
</head>
<body>
  <h1>STATS 242 Executable Textbook — Book Build</h1>
  <p class="meta">Built {datetime.utcnow().isoformat()}Z from <code>{os.path.basename(book_path)}</code>.</p>
  <ul>
    {html_items}
  </ul>
</body>
</html>
"""
    index_html_path = os.path.join(book_dir, "index.html")
    with open(index_html_path, "w", encoding="utf-8") as f:
        f.write(index_html)

    manifest_path = os.path.join(book_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"book": os.path.basename(book_path), "results": results}, f, indent=2)

    return index_md_path, index_html_path

def main():
    ap = argparse.ArgumentParser(prog="stats242xtb.book")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--book", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--data-root", default=None)
    args = ap.parse_args()

    if args.cmd == "build":
        md, html = build(args.book, args.out, data_root=args.data_root)
        print(md)
        print(html)

if __name__ == "__main__":
    main()
