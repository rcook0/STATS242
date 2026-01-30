#!/usr/bin/env bash
set -e
python -m stats242xtb.tools.validate --kind prices --in data/prices_toy.csv --strict
python -m stats242xtb.tools.validate --kind prices_multi --in data/prices_multi_toy.csv
python -m stats242xtb.tools.validate --kind returns_capm --in data/returns_capm_toy.csv
python -m stats242xtb.tools.validate --kind trades --in data/trades_toy.csv
python -m stats242xtb.tools.validate --kind quotes --in data/quotes_toy.csv
python -m stats242xtb.tools.validate --kind events --in data/events_toy.csv
python -m stats242xtb.tools.validate --kind lob --in data/lob_toy.csv
