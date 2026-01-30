# STATS 242 Executable Textbook (v0.2)

This repo turns the **STATS 242 / MSE 242** reader into something you can **run**.

You get:
- **chapter runners** that load data, compute diagnostics, and emit:
  - `report.md` (diffable)
  - `report.html` (shareable)
  - `figures/*.png` (charts)
- a **book registry** that can auto-run chapters into a **book build folder** with an index.

## Quickstart

### 1) Create a venv + install deps
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Generate toy data (so everything runs immediately)
```bash
python -m stats242xtb.tools.make_toy_data --kind prices --out data/prices_toy.csv
python -m stats242xtb.tools.make_toy_data --kind trades --out data/trades_toy.csv
```

### 3) Run a chapter report
```bash
python -m stats242xtb.run chapter02 --prices data/prices_toy.csv --out outputs/ch02
python -m stats242xtb.run chapter10 --trades data/trades_toy.csv --out outputs/ch10
```

### 4) Build the book (auto-run registry)
```bash
python -m stats242xtb.book build --book configs/book.yml --out outputs/book_build
```

Open:
- `outputs/ch02/report.html`
- `outputs/book_build/index.html`

---

Generated: 2026-01-10
