# STATS 242 Executable Textbook (v0.3 — chapter runners filled out)

Goal: make the STATS 242 / MSE 242 ideas **runnable**.

Every chapter produces:
- `report.md` (diffable)
- `report.html` (shareable)
- `figures/*.png` (charts)

## Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Generate a full toy dataset suite (so the whole book runs immediately)
```bash
python -m stats242xtb.tools.make_toy_data --kind suite --out data
```

## Run one chapter
```bash
python -m stats242xtb.run chapter02 --prices data/prices_toy.csv --out outputs/ch02
python -m stats242xtb.run chapter06 --prices-multi data/prices_multi_toy.csv --pair A,B --out outputs/ch06
python -m stats242xtb.run chapter08 --events data/events_toy.csv --out outputs/ch08
```

## Build the book (auto-run registry)
```bash
python -m stats242xtb.book build --book configs/book.yml --out outputs/book_build
```
Open `outputs/book_build/index.html`.

## Implemented chapters (runners)
- `chapter02`: returns, tails, JB, volatility clustering (prices)
- `chapter03`: random walk / martingale sanity checks + variance ratio (prices)
- `chapter04`: CAPM regression template + residual diagnostics (returns CSV)
- `chapter05`: cross-sectional momentum J/K (multi-asset prices)
- `chapter06`: pairs + spread, ADF, AR(1)/OU half-life, simple z-score strategy (multi-asset prices)
- `chapter07`: mean-variance (long-only min-var) + weights + frontier sketch (multi-asset prices)
- `chapter08`: durations + ACD(1,1) fit (events timestamps)
- `chapter09`: microstructure noise: mid vs trade (quotes + trades)
- `chapter10`: Roll spread baseline (trades)
- `chapter11`: trade-sign response (quotes + trades)
- `chapter12`: LOB-ish imbalance → next move empirical calibration (lob snapshots + mid)
- `chapter13`: execution: AC schedule + implementation shortfall simulation (params)
- `chapter14`: robust quant workflow checklist (no data)

## Data formats

### prices CSV
`ts` (ISO), `close`

### prices_multi CSV
`ts` (ISO) and one column per asset: e.g. `A,B,C,...` of closes

### returns CSV (CAPM)
`ts` (ISO), `strategy`, `market`, optional `rf`

### trades CSV
`ts` (ISO), `price`, optional `size`, optional `side` ("B"/"S")

### quotes CSV
`ts` (ISO), `bid`, `ask`, optional `bid_size`, `ask_size`

### events CSV
`ts` (ISO) timestamps of events (trades/quote updates)

### LOB snapshots CSV (toy abstraction)
`ts` (ISO), `mid`, `bid_size`, `ask_size` (best level depth)

---

Generated: 2026-01-10
