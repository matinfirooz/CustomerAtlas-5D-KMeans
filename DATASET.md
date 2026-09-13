# Dataset

This project uses the **UCI Online Retail** dataset.

- UCI dataset ID: `352`
- Instances: `541,909`
- Dataset type: multivariate, sequential, time-series
- Associated tasks: classification, clustering
- Source organization: a UK-based non-store online retailer
- Time period: 1 Dec 2010 to 9 Dec 2011
- DOI: `10.24432/C5BW33`
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Citation

Chen, D. (2015). *Online Retail* [Dataset].
UCI Machine Learning Repository.
https://doi.org/10.24432/C5BW33

The dataset itself is not committed into this repository because the source file
is large. Run:

```bash
python scripts/fetch_data.py
```

or execute the notebook; both cache a local copy under:

```text
data/online_retail.csv
```

Please preserve the UCI attribution when redistributing derived dataset artifacts.
