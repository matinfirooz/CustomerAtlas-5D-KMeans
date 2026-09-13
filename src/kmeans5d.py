from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class KMeansRun:
    centroids: np.ndarray
    labels: np.ndarray
    inertia: float
    history: list[float]
    n_iter: int


class KMeansScratch:
    """
    Vectorized K-Means implementation with:
      - K-Means++ initialization
      - multiple random restarts
      - empty-cluster recovery
      - inertia history
      - deterministic random_state

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    n_init : int
        Number of independent K-Means++ restarts.
    max_iter : int
        Maximum Lloyd iterations per restart.
    tol : float
        Stop when maximum centroid displacement is below tol.
    random_state : int
        Reproducibility seed.
    """

    def __init__(
        self,
        n_clusters: int = 4,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        self.n_clusters = int(n_clusters)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = int(random_state)

        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.history_: list[float] = []
        self.n_iter_: int = 0

    @staticmethod
    def _sq_distances(X: np.ndarray, C: np.ndarray) -> np.ndarray:
        # ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x.c
        x2 = np.sum(X * X, axis=1, keepdims=True)
        c2 = np.sum(C * C, axis=1)[None, :]
        d2 = x2 + c2 - 2.0 * X @ C.T
        return np.maximum(d2, 0.0)

    def _kmeans_plus_plus(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n_samples, n_features = X.shape
        C = np.empty((self.n_clusters, n_features), dtype=np.float64)

        first = rng.integers(0, n_samples)
        C[0] = X[first]

        closest_d2 = self._sq_distances(X, C[:1]).ravel()

        for c in range(1, self.n_clusters):
            total = closest_d2.sum()
            if total <= 1e-15:
                idx = rng.integers(0, n_samples)
            else:
                probs = closest_d2 / total
                idx = rng.choice(n_samples, p=probs)

            C[c] = X[idx]
            new_d2 = self._sq_distances(X, C[c:c+1]).ravel()
            closest_d2 = np.minimum(closest_d2, new_d2)

        return C

    def _recover_empty_clusters(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
        d2: np.ndarray,
    ) -> np.ndarray:
        new_centroids = centroids.copy()
        min_d2 = d2[np.arange(len(X)), labels].copy()

        for k in range(self.n_clusters):
            members = X[labels == k]
            if len(members) > 0:
                new_centroids[k] = members.mean(axis=0)
            else:
                # Re-seed an empty cluster using the currently worst represented point.
                farthest = int(np.argmax(min_d2))
                new_centroids[k] = X[farthest]
                min_d2[farthest] = -np.inf

        return new_centroids

    def _single_run(self, X: np.ndarray, seed: int) -> KMeansRun:
        rng = np.random.default_rng(seed)
        centroids = self._kmeans_plus_plus(X, rng)
        history: list[float] = []

        for iteration in range(1, self.max_iter + 1):
            d2 = self._sq_distances(X, centroids)
            labels = d2.argmin(axis=1)
            inertia = float(d2[np.arange(len(X)), labels].sum())
            history.append(inertia)

            new_centroids = self._recover_empty_clusters(X, labels, centroids, d2)
            shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
            centroids = new_centroids

            if shift <= self.tol:
                break

        d2 = self._sq_distances(X, centroids)
        labels = d2.argmin(axis=1)
        inertia = float(d2[np.arange(len(X)), labels].sum())
        history.append(inertia)

        return KMeansRun(
            centroids=centroids,
            labels=labels,
            inertia=inertia,
            history=history,
            n_iter=iteration,
        )

    def fit(self, X: np.ndarray) -> "KMeansScratch":
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if len(X) < self.n_clusters:
            raise ValueError("n_samples must be >= n_clusters")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinity")

        master = np.random.default_rng(self.random_state)
        seeds = master.integers(0, 2**31 - 1, size=self.n_init)

        best: KMeansRun | None = None
        for seed in seeds:
            run = self._single_run(X, int(seed))
            if best is None or run.inertia < best.inertia:
                best = run

        assert best is not None
        self.cluster_centers_ = best.centroids
        self.labels_ = best.labels
        self.inertia_ = best.inertia
        self.history_ = best.history
        self.n_iter_ = best.n_iter
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("fit must be called first")
        X = np.asarray(X, dtype=np.float64)
        return self._sq_distances(X, self.cluster_centers_).argmin(axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.cluster_centers_ is None:
            raise RuntimeError("fit must be called first")
        X = np.asarray(X, dtype=np.float64)
        return np.sqrt(self._sq_distances(X, self.cluster_centers_))
