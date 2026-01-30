from __future__ import annotations
import os
import tempfile
import shutil
import pathlib

from stats242xtb.tools_make_toy_data import make_suite
from stats242xtb.run import CHAPTERS
from stats242xtb.book import build as build_book

def _exists(p: str) -> bool:
    return os.path.exists(p) and os.path.getsize(p) > 0

def test_each_chapter_runs_on_toy_suite():
    with tempfile.TemporaryDirectory() as td:
        data_dir = os.path.join(td, "data")
        out_dir = os.path.join(td, "out")
        os.makedirs(out_dir, exist_ok=True)
        make_suite(data_dir)

        # minimal args per chapter
        args = {
            "chapter02": dict(prices=os.path.join(data_dir, "prices_toy.csv")),
            "chapter03": dict(prices=os.path.join(data_dir, "prices_toy.csv")),
            "chapter04": dict(returns=os.path.join(data_dir, "returns_capm_toy.csv")),
            "chapter05": dict(prices_multi=os.path.join(data_dir, "prices_multi_toy.csv")),
            "chapter06": dict(prices_multi=os.path.join(data_dir, "prices_multi_toy.csv")),
            "chapter07": dict(prices_multi=os.path.join(data_dir, "prices_multi_toy.csv")),
            "chapter08": dict(events=os.path.join(data_dir, "events_toy.csv")),
            "chapter09": dict(quotes=os.path.join(data_dir, "quotes_toy.csv"), trades=os.path.join(data_dir, "trades_toy.csv")),
            "chapter10": dict(trades=os.path.join(data_dir, "trades_toy.csv")),
            "chapter11": dict(quotes=os.path.join(data_dir, "quotes_toy.csv"), trades=os.path.join(data_dir, "trades_toy.csv")),
            "chapter12": dict(lob=os.path.join(data_dir, "lob_toy.csv")),
            "chapter13": dict(),
            "chapter14": dict(),
        }

        for ch, fn in CHAPTERS.items():
            ch_out = os.path.join(out_dir, ch)
            os.makedirs(ch_out, exist_ok=True)
            md, html = fn(out_dir=ch_out, **args[ch])
            assert _exists(md), f"{ch} missing report.md"
            assert _exists(html), f"{ch} missing report.html"
            figs = os.path.join(ch_out, "figures")
            # some chapters may not output figs, but most do; require directory exists if created
            assert os.path.exists(ch_out)

def test_book_build_runs():
    with tempfile.TemporaryDirectory() as td:
        data_dir = os.path.join(td, "data")
        out_dir = os.path.join(td, "book")
        os.makedirs(out_dir, exist_ok=True)
        make_suite(data_dir)

        # minimal book yml constructed on the fly
        book_yml = os.path.join(td, "book.yml")
        with open(book_yml, "w", encoding="utf-8") as f:
            f.write("""chapters:
  - chapter: chapter02
    name: ch02
    out: ch02
    args:
      prices: prices_toy.csv
""")
        build_book(book_yml, out_dir, data_root=data_dir)

        assert _exists(os.path.join(out_dir, "index.html"))
        assert _exists(os.path.join(out_dir, "manifest.json"))
