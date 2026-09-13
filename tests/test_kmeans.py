import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.kmeans5d import KMeansScratch


def test_kmeans_scratch_finds_separated_5d_clusters():
    rng = np.random.default_rng(7)
    centers = np.array([
        [-5, -5, -5, -5, -5],
        [ 0,  0,  0,  0,  0],
        [ 5,  5,  5,  5,  5],
    ], dtype=float)

    X = np.vstack([
        rng.normal(loc=c, scale=0.35, size=(80, 5))
        for c in centers
    ])
    y = np.repeat(np.arange(3), 80)

    model = KMeansScratch(
        n_clusters=3,
        n_init=6,
        max_iter=100,
        random_state=12,
    )
    labels = model.fit_predict(X)

    assert adjusted_rand_score(y, labels) > 0.98
    assert model.inertia_ > 0
    assert model.cluster_centers_.shape == (3, 5)


def test_predict_shape():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 5))
    model = KMeansScratch(n_clusters=4, n_init=3).fit(X)
    pred = model.predict(X[:11])
    assert pred.shape == (11,)
