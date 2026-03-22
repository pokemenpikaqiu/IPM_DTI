# -*- coding: utf-8 -*-
import os
import numpy as np
from typing import List, Dict, Optional

from ..core.environment import RLEnvironment
from .jumping import StatisticalKnowledgeBase, ClusterJumpingStrategy, SimpleRandomWalkPredictor


class SmartPredictor:
    def __init__(self, kb: StatisticalKnowledgeBase, cluster_kgs_dir: str,
                 drug_features: Dict[str, np.ndarray], target_features: Dict[str, np.ndarray]):
        self.kb = kb
        self.cluster_kgs_dir = cluster_kgs_dir
        self.drug_features = drug_features
        self.target_features = target_features
        self.jumping_strategy = ClusterJumpingStrategy(kb)
        self._cache: Dict[int, RLEnvironment] = {}

    def _load_cluster_env(self, cid: int) -> Optional[RLEnvironment]:
        if cid in self._cache:
            return self._cache[cid]
        triples_path = os.path.join(self.cluster_kgs_dir, f"cluster_{cid}", "expanded_triples.txt")
        if not os.path.exists(triples_path):
            return None
        env = RLEnvironment(
            triples_path=triples_path,
            drug_features=self.drug_features,
            target_features=self.target_features,
            max_steps=50,
        )
        self._cache[cid] = env
        return env

    def predict_for_drug(self, drug_id: str, n_walks_per_cluster: int = 50,
                        top_k: int = 100) -> Dict:
        relevant_clusters, target_types, strategy = self.jumping_strategy.jump_for_drug(drug_id)
        drug_type = self.kb.get_drug_type(drug_id)
        searched = []
        all_preds = []

        for cid in sorted(relevant_clusters):
            env = self._load_cluster_env(cid)
            if env is None or drug_id not in env.entities:
                continue
            searched.append(cid)
            predictor = SimpleRandomWalkPredictor(env)
            preds = predictor.predict(drug_id, n_walks=n_walks_per_cluster)
            for p in preds:
                p["cluster_id"] = cid
                p["strategy"] = strategy
            all_preds.extend(preds)

        if not all_preds:
            for cid in range(16):
                env = self._load_cluster_env(cid)
                if env is None or drug_id not in env.entities:
                    continue
                searched.append(cid)
                predictor = SimpleRandomWalkPredictor(env)
                preds = predictor.predict(drug_id, n_walks=n_walks_per_cluster)
                for p in preds:
                    p["cluster_id"] = cid
                    p["strategy"] = "fallback_random"
                all_preds.extend(preds)

        if len(searched) >= 3:
            pass

        unique_preds = self._dedupe(all_preds)
        unique_preds.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "drug_id": drug_id,
            "drug_type": drug_type,
            "predicted_target_types": target_types,
            "searched_clusters": searched,
            "n_searched": len(searched),
            "total_clusters": 16,
            "coverage": (1 - len(searched) / 16) if searched else 0,
            "predictions": unique_preds[:top_k],
        }

    def predict_for_target(self, target_id: str, n_walks_per_cluster: int = 50,
                          top_k: int = 100) -> Dict:
        relevant_clusters, drug_types, strategy = self.jumping_strategy.jump_for_target(target_id)
        target_type = self.kb.get_target_type(target_id)
        searched = []
        all_preds = []

        for cid in sorted(relevant_clusters)[:5]:
            env = self._load_cluster_env(cid)
            if env is None or target_id not in env.entities:
                continue
            searched.append(cid)
            predictor = SimpleRandomWalkPredictor(env)
            preds = predictor.predict(target_id, n_walks=n_walks_per_cluster)
            for p in preds:
                p["cluster_id"] = cid
                p["strategy"] = strategy
            all_preds.extend(preds)

        unique_preds = self._dedupe(all_preds, key_field="target_id")
        unique_preds.sort(key=lambda x: x["probability"], reverse=True)

        return {
            "target_id": target_id,
            "target_type": target_type,
            "predicted_drug_types": drug_types,
            "searched_clusters": searched,
            "n_searched": len(searched),
            "total_clusters": 16,
            "coverage": (1 - len(searched) / 16) if searched else 0,
            "predictions": unique_preds[:top_k],
        }

    def _dedupe(self, predictions: List[Dict], key_field: str = "target_id") -> List[Dict]:
        seen = set()
        unique = []
        for p in predictions:
            if "drug_id" in p and "target_id" in p:
                key = (p["drug_id"], p["target_id"]) if key_field == "target_id" else (p["drug_id"], p["target_id"])
            else:
                key = p.get(key_field, str(p))
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique
