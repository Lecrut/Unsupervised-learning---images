from typing import Tuple, Optional, Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def run_minibatch_kmeans(
    X: np.ndarray,
    n_clusters: int = 10,
    random_state: int = 42,
    batch_size: int = 256,
    n_init: int = 10,
    compute_silhouette: bool = True,
    standardize: bool = True,
) -> Tuple[np.ndarray, Optional[float], MiniBatchKMeans]:
    """
    Klasteryzacja MiniBatch KMeans (Sculley, 2010) na macierzy cech X.
    Opcjonalnie standaryzuje cechy przed klasteryzacją.
    """
    X_use = StandardScaler().fit_transform(X) if standardize else X
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        batch_size=batch_size,
        n_init=n_init,
    )
    labels = model.fit_predict(X_use)
    sil = None
    if compute_silhouette and len(np.unique(labels)) > 1:
        sil = silhouette_score(X_use, labels)
    return labels, sil, model


def sweep_minibatch_kmeans(
    X: np.ndarray,
    cluster_grid: Sequence[int] = (5, 8, 10, 12, 16),
    random_state: int = 42,
    batch_size: int = 256,
    n_init: int = 10,
    standardize: bool = True,
) -> Tuple[np.ndarray, int, Optional[float], MiniBatchKMeans]:
    """
    Przegląd kilku wartości k, wybór najlepszego wg silhouette.
    Zwraca etykiety, wybrane k, silhouette, model.
    """
    best = {"sil": -1.0, "labels": None, "k": None, "model": None}
    for k in cluster_grid:
        labels, sil, model = run_minibatch_kmeans(
            X,
            n_clusters=k,
            random_state=random_state,
            batch_size=batch_size,
            n_init=n_init,
            compute_silhouette=True,
            standardize=standardize,
        )
        score = -1.0 if sil is None else sil
        if score > best["sil"]:
            best = {"sil": score, "labels": labels, "k": k, "model": model}
    return best["labels"], best["k"], (None if best["sil"] < 0 else best["sil"]), best["model"]
