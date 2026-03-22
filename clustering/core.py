# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from .config import CLUSTERING_CONFIG, MIN_CLUSTER_SIZE, DEFAULT_N_CLUSTERS


class ClusteringResult:
    def __init__(self, labels, n_clusters, algorithm, metrics, centers=None):
        self.labels = labels
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.metrics = metrics
        self.centers = centers

    def get_cluster_indices(self, cid):
        return np.where(self.labels == cid)[0]

    def get_cluster_sizes(self):
        sizes = {}
        for label in self.labels:
            if label == -1:
                continue
            sizes[int(label)] = sizes.get(int(label), 0) + 1
        return sizes

    def to_dict(self):
        return {
            "n_clusters": self.n_clusters,
            "algorithm": self.algorithm,
            "metrics": self.metrics,
            "cluster_sizes": self.get_cluster_sizes(),
        }


def silhouette(X, y):
    from sklearn.metrics import silhouette_score
    mask = y != -1
    if mask.sum() < 2:
        return -1.0
    return float(silhouette_score(X[mask], y[mask]))


def calinski(X, y):
    from sklearn.metrics import calinski_harabasz_score
    mask = y != -1
    if mask.sum() < 2:
        return 0.0
    return float(calinski_harabasz_score(X[mask], y[mask]))


def davies_bouldin(X, y):
    from sklearn.metrics import davies_bouldin_score
    mask = y != -1
    if mask.sum() < 2:
        return float("inf")
    return float(davies_bouldin_score(X[mask], y[mask]))


def evaluate_clustering(features, labels):
    m = {}
    unique = set(labels)
    unique.discard(-1)
    if len(unique) < 2:
        return {"silhouette": -1.0, "calinski_harabasz": 0.0, "davies_bouldin": float("inf")}
    try:
        m["silhouette"] = silhouette(features, labels)
        m["calinski_harabasz"] = calinski(features, labels)
        m["davies_bouldin"] = davies_bouldin(features, labels)
    except Exception:
        m["silhouette"] = -1.0
        m["calinski_harabasz"] = 0.0
        m["davies_bouldin"] = float("inf")
    return m


class FeatureClusterer:
    def __init__(self, algorithm="kmeans", n_clusters=DEFAULT_N_CLUSTERS, random_state=42):
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._model = None

    def _kmeans(self):
        from sklearn.cluster import KMeans
        cfg = CLUSTERING_CONFIG.get("kmeans", {})
        return KMeans(
            n_clusters=self.n_clusters,
            n_init=cfg.get("n_init", 10),
            max_iter=cfg.get("max_iter", 300),
            random_state=self.random_state,
        )

    def _hdbscan(self):
        try:
            import hdbscan
            cfg = CLUSTERING_CONFIG.get("hdbscan", {})
            return hdbscan.HDBSCAN(
                min_cluster_size=cfg.get("min_cluster_size", 5),
                min_samples=cfg.get("min_samples", 3),
                metric=cfg.get("metric", "euclidean"),
            )
        except ImportError:
            return self._kmeans()

    def _hierarchical(self):
        from sklearn.cluster import AgglomerativeClustering
        cfg = CLUSTERING_CONFIG.get("hierarchical", {})
        return AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=cfg.get("linkage", "ward"),
        )

    def fit(self, features):
        if self.algorithm == "kmeans":
            model = self._kmeans()
        elif self.algorithm == "hdbscan":
            model = self._hdbscan()
        elif self.algorithm == "hierarchical":
            model = self._hierarchical()
        else:
            model = self._kmeans()
        self._model = model
        labels = model.fit_predict(features)
        unique = set(labels)
        unique.discard(-1)
        n_c = len(unique)
        centers = model.cluster_centers_ if hasattr(model, "cluster_centers_") else None
        return ClusteringResult(labels, n_c, self.algorithm, evaluate_clustering(features, labels), centers)

    def predict(self, features):
        if self._model is None:
            raise RuntimeError("not fitted")
        if hasattr(self._model, "predict"):
            return self._model.predict(features)
        return np.full(len(features), -1, dtype=int)


class MultiViewClustering:
    def __init__(self, algorithms=None, n_clusters=DEFAULT_N_CLUSTERS, random_state=42):
        if algorithms is None:
            algorithms = ["kmeans", "hierarchical"]
        self.algorithms = algorithms
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.results = {}

    def fit_all(self, features):
        for algo in self.algorithms:
            cl = FeatureClusterer(algo, self.n_clusters, self.random_state)
            self.results[algo] = cl.fit(features)
        return self.results

    def get_best(self, metric="silhouette"):
        best_algo = None
        best_score = float("-inf")
        for algo, res in self.results.items():
            score = res.metrics.get(metric, float("-inf"))
            if metric == "davies_bouldin":
                score = -score
            if score > best_score:
                best_score = score
                best_algo = algo
        if best_algo is None:
            best_algo = list(self.results.keys())[0]
        return best_algo, self.results[best_algo]

    def consensus_labels(self):
        if not self.results:
            raise RuntimeError("no results")
        from sklearn.cluster import KMeans
        all_labels = [r.labels for r in self.results.values()]
        label_matrix = np.column_stack(all_labels).astype(np.float32)
        km = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        return km.fit_predict(label_matrix)


def cluster_triplets(fused_features, pairs, n_clusters=DEFAULT_N_CLUSTERS,
                    algorithm="kmeans", random_state=42):
    cl = FeatureClusterer(algorithm, n_clusters, random_state)
    res = cl.fit(fused_features)
    rows = [{"drug_id": did, "target_id": tid, "cluster_id": int(res.labels[i])}
            for i, (did, tid) in enumerate(pairs)]
    return res, pd.DataFrame(rows)


def cluster_targets(target_features, target_ids, n_clusters=DEFAULT_N_CLUSTERS,
                    algorithm="kmeans", random_state=42):
    cl = FeatureClusterer(algorithm, n_clusters, random_state)
    res = cl.fit(target_features)
    rows = [{"target_id": tid, "cluster_id": int(res.labels[i])} for i, tid in enumerate(target_ids)]
    return res, pd.DataFrame(rows)


def find_optimal_k(features, k_range=None, metric="silhouette"):
    if k_range is None:
        k_range = list(range(2, min(21, len(features) // 2 + 1)))
    best_k = k_range[0]
    best_score = float("-inf")
    for k in k_range:
        cl = FeatureClusterer("kmeans", k)
        score = cl.fit(features).metrics.get(metric, float("-inf"))
        if metric == "davies_bouldin":
            score = -score
        if score > best_score:
            best_score = score
            best_k = k
    return best_k
