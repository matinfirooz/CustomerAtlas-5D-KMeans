from __future__ import annotations

from itertools import combinations
import numpy as np
import pandas as pd


def clustering_stability(X, k: int, seeds=range(8)) -> float:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    labels = []
    for seed in seeds:
        labels.append(
            KMeans(n_clusters=k, n_init=10, random_state=int(seed)).fit_predict(X)
        )

    scores = [
        adjusted_rand_score(labels[i], labels[j])
        for i, j in combinations(range(len(labels)), 2)
    ]
    return float(np.mean(scores)) if scores else 1.0


def evaluate_k_range(X, k_values=range(2, 9), silhouette_sample=3000, random_state=42):
    from sklearn.cluster import KMeans
    from sklearn.metrics import (
        silhouette_score,
        calinski_harabasz_score,
        davies_bouldin_score,
    )

    rows = []
    rng = np.random.default_rng(random_state)

    if len(X) > silhouette_sample:
        sample_idx = rng.choice(len(X), size=silhouette_sample, replace=False)
        X_sil = X[sample_idx]
    else:
        sample_idx = None
        X_sil = X

    for k in k_values:
        model = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        labels = model.fit_predict(X)

        sil_labels = labels[sample_idx] if sample_idx is not None else labels

        rows.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette_score(X_sil, sil_labels)),
            "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
            "davies_bouldin": float(davies_bouldin_score(X, labels)),
            "stability_ari": clustering_stability(X, k),
        })

    return pd.DataFrame(rows)


def choose_k(metrics: pd.DataFrame) -> tuple[int, pd.DataFrame]:
    """
    Rank aggregation rather than blindly trusting a single metric.

    Higher is better:
      silhouette, Calinski-Harabasz, stability

    Lower is better:
      Davies-Bouldin

    Inertia is intentionally visualized but not directly ranked because it
    monotonically decreases with k.
    """
    ranked = metrics.copy()
    ranked["rank_sil"] = ranked["silhouette"].rank(ascending=False, method="min")
    ranked["rank_ch"] = ranked["calinski_harabasz"].rank(ascending=False, method="min")
    ranked["rank_db"] = ranked["davies_bouldin"].rank(ascending=True, method="min")
    ranked["rank_stability"] = ranked["stability_ari"].rank(ascending=False, method="min")

    ranked["rank_score"] = (
        ranked["rank_sil"] +
        ranked["rank_ch"] +
        ranked["rank_db"] +
        ranked["rank_stability"]
    )

    ranked = ranked.sort_values(["rank_score", "k"]).reset_index(drop=True)
    return int(ranked.iloc[0]["k"]), ranked


def cluster_feature_importance(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Between-cluster variance / total variance for each standardized feature.
    This is descriptive, not causal.
    """
    global_mean = X.mean(axis=0)
    total = np.sum((X - global_mean) ** 2, axis=0)

    between = np.zeros(X.shape[1], dtype=float)
    for cluster in np.unique(labels):
        Xc = X[labels == cluster]
        if len(Xc) == 0:
            continue
        between += len(Xc) * (Xc.mean(axis=0) - global_mean) ** 2

    return np.divide(between, total, out=np.zeros_like(between), where=total > 0)
