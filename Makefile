PYTHON ?= python3

.PHONY: install data notebook run test clean

install:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/fetch_data.py

notebook:
	jupyter notebook notebooks/CustomerAtlas_5D_KMeans.ipynb

run:
	$(PYTHON) scripts/run_analysis.py

test:
	pytest -q

clean:
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__
	rm -f data/online_retail.csv
