"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import normalize
from collections import defaultdict
import pickle
from config import SUBSPACE_CONFIG

class SubspaceClusterer:
    """
    
    def __init__(self, n_clusters: int = None, method: str = 'kmeans'):

        if n_clusters is not None:
            self.n_clusters = n_clusters
        

        features = normalize(features)
        

        if self.method == 'kmeans':
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
        elif self.method == 'hierarchical':
            self.clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                linkage='ward'
            )
        elif self.method == 'dbscan':
            self.clusterer = DBSCAN(
                eps=0.5,
                min_samples=5
            )
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        

        self.labels = self.clusterer.fit_predict(features)
        

        if self.method == 'kmeans':
            self.cluster_centers = self.clusterer.cluster_centers_
        else:
            self._compute_cluster_centers(features)
        

        self._compute_cluster_info(features)
        
        return self.labels
    
    def _compute_cluster_centers(self, features: np.ndarray):

        unique_labels = np.unique(self.labels)
        
        for label in unique_labels:
            mask = self.labels == label
            cluster_features = features[mask]
            
            self.cluster_info[label] = {
                'size': mask.sum(),
                'center': self.cluster_centers[label] if label < len(self.cluster_centers) else None,
                'variance': cluster_features.var(axis=0).mean() if len(cluster_features) > 1 else 0,
                'indices': np.where(mask)[0].tolist()
            }
    
    def predict(self, features: np.ndarray) -> np.ndarray:

        return np.where(self.labels == cluster_id)[0].tolist()
    
    def evaluate_clustering(self, features: np.ndarray) -> Dict[str, float]:

        features = normalize(features)
        scores = {}
        
        for k in range(k_range[0], k_range[1] + 1):
            clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = clusterer.fit_predict(features)
            
            if len(np.unique(labels)) > 1:
                score = silhouette_score(features, labels)
                scores[k] = score
                print(f"K={k}, Silhouette Score={score:.4f}")
        
        return scores

class TripletSubspaceManager:

        self.n_clusters = n_clusters
        self.clusterer = SubspaceClusterer(n_clusters=n_clusters)
        self.subspaces = defaultdict(list)
        self.triplet_to_subspace = {}
        self.subspace_entities = defaultdict(set)
        self.subspace_relations = defaultdict(set)
        
    def cluster_triplets(self, triplets: List[Dict], features: np.ndarray) -> Dict[int, List[Dict]]:

        return self.subspaces.get(subspace_id, [])
    
    def get_entity_subspaces(self, entity_id: int) -> List[int]:

        return self.subspace_entities.get(subspace_id, set())
    
    def get_subspace_relations(self, subspace_id: int) -> set:
 
        stats = {
            'n_subspaces': len(self.subspaces),
            'subspace_sizes': {},
            'entity_counts': {},
            'relation_counts': {},
            'overlap_matrix': None
        }
        
        for subspace_id in self.subspaces:
            stats['subspace_sizes'][subspace_id] = len(self.subspaces[subspace_id])
            stats['entity_counts'][subspace_id] = len(self.subspace_entities[subspace_id])
            stats['relation_counts'][subspace_id] = len(self.subspace_relations[subspace_id])
        

        n = len(self.subspaces)
        overlap_matrix = np.zeros((n, n))
        subspace_ids = list(self.subspaces.keys())
        
        for i, id1 in enumerate(subspace_ids):
            for j, id2 in enumerate(subspace_ids):
                if i != j:
                    entities1 = self.subspace_entities[id1]
                    entities2 = self.subspace_entities[id2]
                    overlap = len(entities1 & entities2)
                    overlap_matrix[i, j] = overlap
        
        stats['overlap_matrix'] = overlap_matrix
        
        return stats
    
    def save(self, filepath: str):

        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.n_clusters = data['n_clusters']
        self.subspaces = defaultdict(list, data['subspaces'])
        self.triplet_to_subspace = data['triplet_to_subspace']
        self.subspace_entities = defaultdict(set, {k: set(v) for k, v in data['subspace_entities'].items()})
        self.subspace_relations = defaultdict(set, {k: set(v) for k, v in data['subspace_relations'].items()})
        self.clusterer.labels = data['clusterer_labels']
        self.clusterer.cluster_centers = data['cluster_centers']
        print(f"子空间信息已从 {filepath} 加载")

class SimilarityCalculator:


        features1_norm = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
        features2_norm = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)
        
        return np.dot(features1_norm, features2_norm.T)
    
    @staticmethod
    def euclidean_distance(features1: np.ndarray, features2: np.ndarray) -> np.ndarray:

        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def compute_triplet_similarity(triplet1: Dict, triplet2: Dict, 
                                   entity_features: np.ndarray,
                                   relation_features: Optional[np.ndarray] = None) -> float: