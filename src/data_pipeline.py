from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


FIVE_D_FEATURES = [
    "RecencyDays",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "ProductDiversity",
]


def fetch_uci_online_retail(cache_path: str | Path = "data/online_retail.csv") -> pd.DataFrame:
    """
    Fetch UCI Online Retail (dataset id=352) via ucimlrepo and cache it locally.

    Source:
      Chen, D. (2015). Online Retail [Dataset].
      UCI Machine Learning Repository.
      DOI: 10.24432/C5BW33
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["InvoiceDate"])
        return df

    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise RuntimeError(
            "Install ucimlrepo first: pip install ucimlrepo"
        ) from exc

    dataset = fetch_ucirepo(id=352)
    df = dataset.data.features.copy()

    # Depending on ucimlrepo version, ID-like fields may be outside features.
    # Recover the original dataset frame when available.
    original = getattr(dataset.data, "original", None)
    if original is not None and len(original):
        df = original.copy()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if "InvoiceDate" in df.columns:
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df.to_csv(cache_path, index=False)
    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "InvoiceNo", "StockCode", "Quantity", "InvoiceDate",
        "UnitPrice", "CustomerID"
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}. "
            "If ucimlrepo returned only feature columns, update ucimlrepo or "
            "use the cached original UCI file."
        )

    x = df.copy()
    x["InvoiceNo"] = x["InvoiceNo"].astype(str)
    x["StockCode"] = x["StockCode"].astype(str)
    x["InvoiceDate"] = pd.to_datetime(x["InvoiceDate"], errors="coerce")
    x["CustomerID"] = pd.to_numeric(x["CustomerID"], errors="coerce")
    x["Quantity"] = pd.to_numeric(x["Quantity"], errors="coerce")
    x["UnitPrice"] = pd.to_numeric(x["UnitPrice"], errors="coerce")

    x = x.dropna(subset=[
        "CustomerID", "InvoiceDate", "Quantity", "UnitPrice", "InvoiceNo", "StockCode"
    ])

    # Remove cancellations / returns and non-positive sales.
    x = x[~x["InvoiceNo"].str.upper().str.startswith("C")]
    x = x[(x["Quantity"] > 0) & (x["UnitPrice"] > 0)]

    x["CustomerID"] = x["CustomerID"].astype("int64")
    x["Revenue"] = x["Quantity"] * x["UnitPrice"]
    return x


def engineer_customer_5d(transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw transactions into exactly five customer dimensions:

    1. RecencyDays      - days since the customer's last purchase
    2. Frequency        - number of unique invoices/orders
    3. Monetary         - total positive revenue
    4. AvgOrderValue    - revenue per unique invoice
    5. ProductDiversity - unique products purchased
    """
    if transactions.empty:
        raise ValueError("No transactions remain after cleaning.")

    snapshot_date = transactions["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)

    agg = transactions.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("Revenue", "sum"),
        ProductDiversity=("StockCode", "nunique"),
        Country=("Country", lambda s: s.mode().iloc[0] if "Country" in transactions.columns and len(s.mode()) else "Unknown"),
    )

    agg["RecencyDays"] = (snapshot_date - agg["LastPurchase"].dt.normalize()).dt.days
    agg["AvgOrderValue"] = agg["Monetary"] / agg["Frequency"].clip(lower=1)

    ordered = agg[
        ["RecencyDays", "Frequency", "Monetary", "AvgOrderValue", "ProductDiversity", "Country"]
    ].copy()

    return ordered.sort_index()


def winsorize_upper(df: pd.DataFrame, columns: list[str], q: float = 0.995) -> tuple[pd.DataFrame, pd.Series]:
    out = df.copy()
    caps = out[columns].quantile(q)
    for col in columns:
        out[col] = out[col].clip(upper=float(caps[col]))
    return out, caps


def prepare_kmeans_matrix(
    customer_df: pd.DataFrame,
    quantile_cap: float = 0.995,
):
    """
    K-Means uses Euclidean geometry, so:
      1) cap extreme upper tails,
      2) log1p-transform positive skewed features,
      3) standardize all five dimensions.
    """
    from sklearn.preprocessing import StandardScaler

    features = customer_df[FIVE_D_FEATURES].astype(float).copy()
    capped, caps = winsorize_upper(features, FIVE_D_FEATURES, quantile_cap)

    logged = np.log1p(capped)
    scaler = StandardScaler()
    X = scaler.fit_transform(logged)

    return X, logged, scaler, caps
