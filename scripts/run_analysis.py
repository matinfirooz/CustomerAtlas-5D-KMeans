#!/usr/bin/env python3
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.cluster import KMeans

from src.data_pipeline import (
    FIVE_D_FEATURES,
    fetch_uci_online_retail,
    clean_transactions,
    engineer_customer_5d,
    prepare_kmeans_matrix,
)
from src.evaluation import evaluate_k_range, choose_k


def main():
    out = ROOT / "outputs"
    out.mkdir(exist_ok=True)

    raw = fetch_uci_online_retail(ROOT / "data" / "online_retail.csv")
    clean = clean_transactions(raw)
    customers = engineer_customer_5d(clean)

    X, logged, scaler, caps = prepare_kmeans_matrix(customers)

    metrics = evaluate_k_range(X)
    best_k, ranked = choose_k(metrics)

    model = KMeans(n_clusters=best_k, n_init=30, random_state=42)
    labels = model.fit_predict(X)

    assigned = customers.copy()
    assigned["Cluster"] = labels
    assigned.to_csv(out / "customer_clusters.csv")

    profiles = assigned.groupby("Cluster")[FIVE_D_FEATURES].median()
    profiles.to_csv(out / "cluster_profiles_median.csv")

    metrics.to_csv(out / "k_selection_metrics.csv", index=False)
    ranked.to_csv(out / "k_selection_ranked.csv", index=False)

    summary = {
        "raw_transactions": int(len(raw)),
        "clean_transactions": int(len(clean)),
        "customers": int(len(customers)),
        "dimensions": FIVE_D_FEATURES,
        "best_k": int(best_k),
        "inertia": float(model.inertia_),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print("\nCluster medians:")
    print(profiles.round(2))


if __name__ == "__main__":
    main()
