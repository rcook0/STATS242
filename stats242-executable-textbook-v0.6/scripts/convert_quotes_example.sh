#!/usr/bin/env bash
python -m stats242xtb.tools.convert --kind quotes --schema taq --in raw_quotes.csv --out data/quotes.csv --map configs/colmap_quotes.yml --preview
