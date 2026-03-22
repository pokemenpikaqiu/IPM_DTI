# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict


class StatisticalKnowledgeBase:
    def __init__(self, analysis_output_dir: str, dataset_name: str):
        self.dataset_name = dataset_name
        self.base_dir = analysis_output_dir
        self.drug_type_map: Dict[str, int] = {}
        self.target_type_map: Dict[str, int] = {}
        self.interaction_matrix: Dict[Tuple[int, int], Dict] = {}
        self.type_to_clusters: Dict[Tuple[str, int], Set[int]] = defaultdict(set)
        self._load_data()

    def _load_data(self):
        mapping_path = os.path.join(self.base_dir, "bio_type_mapping.csv")
        if os.path.exists(mapping_path):
            df = pd.read_csv(mapping_path)
            for _, row in df.iterrows():
                eid = str(row["entity_id"])
                bio_type = int(row["bio_type"])
                if row["entity_type"] == "drug":
                    self.drug_type_map[eid] = bio_type
                else:
                    self.target_type_map[eid] = bio_type

        patterns_path = os.path.join(self.base_dir, "interaction_patterns.json")
        if os.path.exists(patterns_path):
            with open(patterns_path) as f:
                data = json.load(f)
            for key, stats in data.get("interaction_matrix", {}).items():
                d_type, t_type = map(int, key.split("_"))
                self.interaction_matrix[(d_type, t_type)] = stats

    def build_cluster_type_index(self, cluster_dirs: List[str]):
        for ctype_idx, cluster_dir in enumerate(cluster_dirs):
            triples_path = os.path.join(cluster_dir, "expanded_triples.txt")
            if not os.path.exists(triples_path):
                continue
            with open(triples_path) as f:
                for line in f:
                    h, r, t = line.strip().split("\t")
                    for entity in [h, t]:
                        d_type = self.get_drug_type(entity)
                        if d_type is not None:
                            self.type_to_clusters[("drug", d_type)].add(ctype_idx)
                        t_type = self.get_target_type(entity)
                        if t_type is not None:
                            self.type_to_clusters[("target", t_type)].add(ctype_idx)

    def get_drug_type(self, drug_id: str) -> Optional[int]:
        return self.drug_type_map.get(drug_id)

    def get_target_type(self, target_id: str) -> Optional[int]:
        return self.target_type_map.get(target_id)

    def predict_target_types(self, drug_id: str, top_k: int = 3) -> List[Tuple[int, float]]:
        drug_type = self.get_drug_type(drug_id)
        if drug_type is None:
            return []
        candidates = []
        for t_type in range(5):
            key = (drug_type, t_type)
            if key in self.interaction_matrix:
                prob = self.interaction_matrix[key]["interaction_rate"]
                count = self.interaction_matrix[key]["count"]
                candidates.append((t_type, prob, count))
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [(t_type, prob) for t_type, prob, _ in candidates[:top_k]]

    def predict_drug_types(self, target_id: str, top_k: int = 3) -> List[Tuple[int, float]]:
        target_type = self.get_target_type(target_id)
        if target_type is None:
            return []
        candidates = []
        for d_type in range(5):
            key = (d_type, target_type)
            if key in self.interaction_matrix:
                prob = self.interaction_matrix[key]["interaction_rate"]
                count = self.interaction_matrix[key]["count"]
                candidates.append((d_type, prob, count))
        candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [(d_type, prob) for d_type, prob, _ in candidates[:top_k]]

    def get_clusters_by_target_type(self, target_type: int) -> Set[int]:
        return self.type_to_clusters.get(("target", target_type), set())

    def get_clusters_by_drug_type(self, drug_type: int) -> Set[int]:
        return self.type_to_clusters.get(("drug", drug_type), set())


class ClusterJumpingStrategy:
    def __init__(self, kb: StatisticalKnowledgeBase):
        self.kb = kb

    def jump_for_drug(self, drug_id: str) -> Tuple[Set[int], List[Tuple[int, float]], str]:
        drug_type = self.kb.get_drug_type(drug_id)
        if drug_type is None:
            return set(), [], "unknown_drug_type"
        target_types = self.kb.predict_target_types(drug_id, top_k=3)
        relevant = set()
        for t_type, prob in target_types:
            relevant.update(self.kb.get_clusters_by_target_type(t_type))
        return relevant, target_types, "type_based_jump"

    def jump_for_target(self, target_id: str) -> Tuple[Set[int], List[Tuple[int, float]], str]:
        target_type = self.kb.get_target_type(target_id)
        if target_type is None:
            return set(), [], "unknown_target_type"
        drug_types = self.kb.predict_drug_types(target_id, top_k=3)
        relevant = set()
        for d_type, prob in drug_types:
            relevant.update(self.kb.get_clusters_by_drug_type(d_type))
        return relevant, drug_types, "type_based_jump"

    def find_fallback_clusters(self, entity_id: str, entity_type: str,
                              max_clusters: int = 3) -> Set[int]:
        return set()


class SimpleRandomWalkPredictor:
    def __init__(self, environment):
        self.env = environment

    def predict(self, start_entity: str, n_walks: int = 50, max_steps: int = 30,
                stop_prob: float = 0.5) -> List[Dict]:
        from collections import defaultdict
        predictions = defaultdict(lambda: {"count": 0, "paths": []})

        for _ in range(n_walks):
            current = start_entity
            path = [current]

            for _ in range(max_steps):
                neighbors = self.env.get_neighbors(current)
                if not neighbors:
                    break
                next_entity, relation = random.choice(neighbors)
                path.append(next_entity)

                start_type = self.env.get_entity_type(start_entity)
                next_type = self.env.get_entity_type(next_entity)

                if start_type != next_type and next_type in ("drug", "target"):
                    if start_type == "drug":
                        pair = (start_entity, next_entity)
                    else:
                        pair = (next_entity, start_entity)
                    predictions[pair]["count"] += 1
                    predictions[pair]["paths"].append(path.copy())

                if random.random() > stop_prob:
                    break
                current = next_entity

        results = []
        for (drug, target), data in predictions.items():
            prob = data["count"] / n_walks
            results.append({
                "drug_id": drug,
                "target_id": target,
                "walk_count": data["count"],
                "probability": float(prob),
            })
        results.sort(key=lambda x: x["probability"], reverse=True)
        return results
