# -*- coding: utf-8 -*-
import os

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(_BASE_DIR, "dataset")

CLUSTER_OUTPUT_DIR = os.path.join(_BASE_DIR, "clustering_output")
os.makedirs(CLUSTER_OUTPUT_DIR, exist_ok=True)

CLUSTERING_CONFIG = {
    "kmeans": {"n_clusters": 10, "n_init": 10, "max_iter": 300, "random_state": 42},
    "hdbscan": {"min_cluster_size": 5, "min_samples": 3, "metric": "euclidean"},
    "hierarchical": {"n_clusters": 10, "linkage": "ward", "metric": "euclidean"},
}

TRIPLET_CLUSTERING_CONFIG = {"algorithm": "kmeans", "n_clusters": 10, "random_state": 42, "feature_source": "fused"}
TARGET_CLUSTERING_CONFIG = {"algorithm": "kmeans", "n_clusters": 10, "random_state": 42, "feature_source": "target"}

EVALUATION_METRICS = ["silhouette", "calinski_harabasz", "davies_bouldin"]
MIN_CLUSTER_SIZE = 3
MAX_CLUSTERS = 50
DEFAULT_N_CLUSTERS = 10
