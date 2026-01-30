# STATS 242 Executable Textbook (v0.6)

This is a runnable course reader: chapters become scripts that produce **reports + charts**,
and a book registry can build an **indexed output folder**.

This version bundles:
- **v0.4**: explicit model choices (“model cards”) per chapter
- **v0.5**: data adapters + conversion tool (schema + column mapping) — TAQ supported as a best-effort template
- **v0.6**: per-chapter unit tests + a `run-all` command

## Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Fast start (runs the whole book on toy data)
```bash
python -m stats242xtb.tools.make_toy_data --kind suite --out data
python -m stats242xtb.run_all --book configs/book.yml --out outputs/book_build
```

Open: `outputs/book_build/latest/index.html`

## Run one chapter
```bash
python -m stats242xtb.run chapter06 --prices-multi data/prices_multi_toy.csv --pair A,B --out outputs/ch06
```

## Convert raw data → canonical format (adapters)
Convert “TAQ-like” trades to canonical `trades.csv`:
```bash
python -m stats242xtb.tools.convert --kind trades --schema taq --in raw_trades.csv --out data/trades.csv
```

If your columns differ, supply a mapping YAML:
```bash
python -m stats242xtb.tools.convert --kind quotes --schema taq --in raw_quotes.csv --out data/quotes.csv --map configs/colmap_quotes.yml
```

## Tests (v0.6)
```bash
pytest -q
```

---

Generated: 2026-01-10
