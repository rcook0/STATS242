# STATS 242 Executable Textbook (v0.7)

This repository turns chapters into runnable programs that produce:
- `report.md`
- `report.html` (with embedded charts)
- `figures/*.png`
- `artifacts/*.json` (structured outputs when applicable)

v0.7 adds two lanes:

## Lane 1 — Core
Fast, reproducible, “instrument-grade” implementations that you can run by default.

## Lane 2 — Principia
Deep, derivation-heavy and simulation-heavy companions. They run via the same book builder but are registered in a separate YAML registry.

---

## Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

---

## Build the Core book (fast)

```bash
python -m stats242xtb.tools.make_toy_data --kind suite --out data
python -m stats242xtb.run_all --book configs/book.yml --out outputs/book_build
```

Open: `outputs/book_build/latest/index.html`

---

## Build Principia (deeper)

```bash
python -m stats242xtb.run_all --book configs/principia.yml --out outputs/principia_build
```

Open: `outputs/principia_build/latest/index.html`

---

## Run one chapter

```bash
python -m stats242xtb.run chapter04 --returns data/returns_capm_toy.csv --out outputs/ch04 \
  --cov-type HAC --hac-lags 5 --hac-kernel bartlett \
  --rolling-window 120 --rolling-step 5
```

---

## What v0.7 focuses on (Core)
- A regression layer with **covariance choices**: OLS, HC0–HC3, HAC (Newey–West).
- A reusable **diagnostics bundle** that standardizes residual checks and plots.
- Optional **rolling regression** stability plots for Chapter 04.

Generated: 2026-01-12
