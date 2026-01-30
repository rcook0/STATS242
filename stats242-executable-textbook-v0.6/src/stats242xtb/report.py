from __future__ import annotations
import os
from typing import List, Tuple
from .util import ensure_dir

def write_report_md(out_dir: str, title: str, sections: List[Tuple[str, str]]) -> str:
    ensure_dir(out_dir)
    lines = [f"# {title}", ""]
    for h, body in sections:
        lines += [f"## {h}", "", body.rstrip(), ""]
    path = os.path.join(out_dir, "report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

def write_report_html(out_dir: str, title: str, md_text: str, css: str = "") -> str:
    ensure_dir(out_dir)
    try:
        import markdown as md
        body = md.markdown(md_text, extensions=["fenced_code", "tables", "toc"], output_format="html5")
    except Exception:
        import html
        body = "<pre>" + html.escape(md_text) + "</pre>"

    default_css = """
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.55; margin: 32px; max-width: 980px; }
    h1,h2,h3 { line-height: 1.2; }
    code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    pre { padding: 12px; background: #f6f8fa; overflow-x: auto; border-radius: 10px; }
    table { border-collapse: collapse; }
    table, th, td { border: 1px solid #ddd; }
    th, td { padding: 6px 8px; vertical-align: top; }
    img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 10px; }
    .note { background: #fff8c5; padding: 10px 12px; border-radius: 10px; border: 1px solid #f1e09a; }
    """
    css_all = default_css + "\n" + (css or "")
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>{title}</title>
  <style>{css_all}</style>
</head>
<body>
{body}
</body>
</html>
"""
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_text)
    return path

def write_report(out_dir: str, title: str, sections: List[Tuple[str, str]]) -> tuple[str, str]:
    md_path = write_report_md(out_dir, title, sections)
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    html_path = write_report_html(out_dir, title, md_text)
    return md_path, html_path
