Put your datasets here.

Recommended:
- prices_*.csv   (OHLCV-ish)
- trades_*.csv   (tick trades)
- quotes_*.csv   (best bid/ask)

Toy data:
python -m stats242xtb.tools.make_toy_data --kind prices --out data/prices_toy.csv
python -m stats242xtb.tools.make_toy_data --kind trades --out data/trades_toy.csv
