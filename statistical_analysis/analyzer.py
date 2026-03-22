# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_interactions(dataset_name: str, data_root: str) -> pd.DataFrame:
    paths = [
        os.path.join(data_root, f"{dataset_name}_dataset.csv"),
        os.path.join(data_root, dataset_name, f"{dataset_name}_dataset.csv"),
        os.path.join(os.path.dirname(data_root.rstrip(os.sep)), "dataset.csv"),
    ]
    csv_path = None
    for p in paths:
        if os.path.exists(p):
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError(f"Dataset not found: {dataset_name}")
    df = pd.read_csv(csv_path)
    label_col = "label" if "label" in df.columns else df.columns[-1]
    return df[["drug_id", "target_id", label_col]].rename(columns={label_col: "label"})


def load_features(dataset_name: str, data_root: str) -> Tuple[Dict, Dict]:
    root = os.path.join(data_root, dataset_name)
    drug_emb = np.load(os.path.join(root, f"{dataset_name}_drug_features.npy"))
    drug_ids = np.load(os.path.join(root, f"{dataset_name}_drug_ids.npy"), allow_pickle=True)
    drug_feats = {str(d): drug_emb[i] for i, d in enumerate(drug_ids)}
    target_emb = np.load(os.path.join(root, f"{dataset_name}_target_features.npy"))
    target_ids = np.load(os.path.join(root, f"{dataset_name}_target_ids.npy"), allow_pickle=True)
    target_feats = {str(t): target_emb[i] for i, t in enumerate(target_ids)}
    return drug_feats, target_feats


class BioTypeClassifier:
    def __init__(self, n_drug_types=5, n_target_types=5, seed=42):
        self.n_drug = n_drug_types
        self.n_target = n_target_types
        self.seed = seed
        self.d_clf = None
        self.t_clf = None
        self.d_map = {}
        self.t_map = {}

    def fit(self, drug_feats, target_feats):
        d_ids = list(drug_feats.keys())
        d_mat = np.stack([drug_feats[d] for d in d_ids])
        self.d_clf = KMeans(n_clusters=self.n_drug, random_state=self.seed, n_init=10)
        d_labels = self.d_clf.fit_predict(d_mat)
        self.d_map = {d_ids[i]: int(d_labels[i]) for i in range(len(d_ids))}

        t_ids = list(target_feats.keys())
        t_mat = np.stack([target_feats[t] for t in t_ids])
        self.t_clf = KMeans(n_clusters=self.n_target, random_state=self.seed, n_init=10)
        t_labels = self.t_clf.fit_predict(t_mat)
        self.t_map = {t_ids[i]: int(t_labels[i]) for i in range(len(t_ids))}
        return self

    def get_drug_type(self, drug_id: str) -> Optional[int]:
        return self.d_map.get(drug_id)

    def get_target_type(self, target_id: str) -> Optional[int]:
        return self.t_map.get(target_id)

    def stats(self) -> Dict:
        return {
            "drug_type_dist": dict(Counter(self.d_map.values())),
            "target_type_dist": dict(Counter(self.t_map.values())),
        }


class InteractionPatternAnalyzer:
    def __init__(self, clf: BioTypeClassifier):
        self.clf = clf
        self.matrix = None
        self.global_stats = {}

    def analyze(self, df: pd.DataFrame) -> Dict:
        df = df.copy()
        df["d_type"] = df["drug_id"].map(lambda x: self.clf.get_drug_type(str(x)))
        df["t_type"] = df["target_id"].map(lambda x: self.clf.get_target_type(str(x)))
        df = df.dropna(subset=["d_type", "t_type"])
        df["d_type"] = df["d_type"].astype(int)
        df["t_type"] = df["t_type"].astype(int)

        type_mat = defaultdict(lambda: {"count": 0, "pos": 0, "neg": 0})
        for _, row in df.iterrows():
            dt, tt = int(row["d_type"]), int(row["t_type"])
            label = int(row["label"])
            key = (dt, tt)
            type_mat[key]["count"] += 1
            if label == 1:
                type_mat[key]["pos"] += 1
            else:
                type_mat[key]["neg"] += 1

        for key in type_mat:
            s = type_mat[key]
            s["pos_rate"] = s["pos"] / max(s["count"], 1)
            s["neg_rate"] = s["neg"] / max(s["count"], 1)

        self.matrix = dict(type_mat)

        d_prefs = defaultdict(lambda: defaultdict(int))
        t_prefs = defaultdict(lambda: defaultdict(int))
        for (dt, tt), stats in type_mat.items():
            d_prefs[dt][tt] = stats["count"]
            t_prefs[tt][dt] = stats["count"]

        conserved = []
        for (dt, tt), stats in type_mat.items():
            if stats["count"] >= 10 and stats["pos_rate"] > 0.7:
                conserved.append({
                    "drug_type": dt, "target_type": tt,
                    "count": stats["count"], "pos_rate": stats["pos_rate"],
                })
        conserved.sort(key=lambda x: x["count"], reverse=True)

        mat_str_keys = {f"{k[0]}_{k[1]}": v for k, v in self.matrix.items()}

        return {
            "interaction_matrix": mat_str_keys,
            "drug_preferences": {str(k): dict(v) for k, v in d_prefs.items()},
            "target_preferences": {str(k): dict(v) for k, v in t_prefs.items()},
            "conserved_patterns": conserved,
            "n_interactions": len(df),
        }


class CrossDatasetComparator:
    def __init__(self):
        self.results = {}

    def add(self, name: str, analyzer: InteractionPatternAnalyzer):
        self.results[name] = analyzer

    def compare(self) -> Dict:
        if len(self.results) < 2:
            return {"n_datasets": len(self.results)}

        all_patterns = {}
        for name, ana in self.results.items():
            patterns = set()
            for key in ana.matrix.keys():
                s = ana.matrix[key]
                if s["count"] >= 5 and s["pos_rate"] > 0.5:
                    patterns.add(key)
            all_patterns[name] = patterns

        common = set.intersection(*all_patterns.values())
        similarities = {}
        datasets = list(self.results.keys())
        for i in range(len(datasets)):
            for j in range(i + 1, len(datasets)):
                d1, d2 = datasets[i], datasets[j]
                s1, s2 = all_patterns[d1], all_patterns[d2]
                inter = len(s1 & s2)
                union = len(s1 | s2)
                jaccard = inter / union if union > 0 else 0
                similarities[f"{d1}_vs_{d2}"] = {
                    "jaccard": jaccard, "intersection": inter, "union": union,
                }

        return {
            "n_datasets": len(self.results),
            "common_patterns": [f"{k[0]}_{k[1]}" for k in common],
            "n_common": len(common),
            "similarities": similarities,
        }


def heatmap_data(matrix, n_drug, n_target):
    hm = np.zeros((n_drug, n_target))
    cnts = np.zeros((n_drug, n_target))
    for (dt, tt), stats in matrix.items():
        hm[dt, tt] = stats["pos_rate"]
        cnts[dt, tt] = stats["count"]
    return hm, cnts


def save_analysis(dataset_name, clf, analyzer, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    type_stats = clf.stats()
    with open(os.path.join(out_dir, "bio_type_stats.json"), "w") as f:
        json.dump(type_stats, f, indent=2)
    with open(os.path.join(out_dir, "interaction_patterns.json"), "w") as f:
        json.dump(analyzer.analyze.__self__.matrix, f, indent=2)
    hm, cnts = heatmap_data(analyzer.matrix, clf.n_drug, clf.n_target)
    np.save(os.path.join(out_dir, "heatmap_rates.npy"), hm)
    np.save(os.path.join(out_dir, "heatmap_counts.npy"), cnts)
    df = pd.DataFrame(
        [{"entity_id": eid, "entity_type": "drug", "bio_type": t} for eid, t in clf.d_map.items()] +
        [{"entity_id": eid, "entity_type": "target", "bio_type": t} for eid, t in clf.t_map.items()]
    )
    df.to_csv(os.path.join(out_dir, "bio_type_mapping.csv"), index=False)


def analyze_single(dataset_name, data_root, n_drug=5, n_target=5, out_dir=None):
    interactions_df = load_interactions(dataset_name, data_root)
    drug_feats, target_feats = load_features(dataset_name, data_root)

    clf = BioTypeClassifier(n_drug_types=n_drug, n_target_types=n_target)
    clf.fit(drug_feats, target_feats)
    type_stats = clf.stats()

    analyzer = InteractionPatternAnalyzer(clf)
    analysis = analyzer.analyze(interactions_df)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "bio_type_stats.json"), "w") as f:
            json.dump(type_stats, f, indent=2)
        with open(os.path.join(out_dir, "interaction_patterns.json"), "w") as f:
            json.dump(analysis, f, indent=2)
        hm, cnts = heatmap_data(analyzer.matrix, n_drug, n_target)
        np.save(os.path.join(out_dir, "heatmap_rates.npy"), hm)
        np.save(os.path.join(out_dir, "heatmap_counts.npy"), cnts)
        df = pd.DataFrame(
            [{"entity_id": eid, "entity_type": "drug", "bio_type": t} for eid, t in clf.d_map.items()] +
            [{"entity_id": eid, "entity_type": "target", "bio_type": t} for eid, t in clf.t_map.items()]
        )
        df.to_csv(os.path.join(out_dir, "bio_type_mapping.csv"), index=False)

    return clf, analyzer


def run_analysis(datasets: List[str], data_root: str, n_drug=5, n_target=5, out_dir=None):
    if out_dir is None:
        out_dir = os.path.join(PROJECT_ROOT, "statistical_analysis_output")
    os.makedirs(out_dir, exist_ok=True)

    comparator = CrossDatasetComparator()
    all_clf = {}

    for dataset in datasets:
        clf, analyzer = analyze_single(dataset, data_root, n_drug, n_target, out_dir)
        all_clf[dataset] = clf
        comparator.add(dataset, analyzer)

    if len(datasets) > 1:
        comparison = comparator.compare()
        with open(os.path.join(out_dir, "cross_dataset_comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)

    summary = {
        "datasets": datasets,
        "n_drug_types": n_drug,
        "n_target_types": n_target,
        "classifiers": {d: {"drug_types": all_clf[d].n_drug, "target_types": all_clf[d].n_target} for d in datasets},
    }
    with open(os.path.join(out_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return comparator, all_clf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["lou", "yamanishi", "zheng"],
                       choices=["lou", "yamanishi", "zheng"])
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--n_drug_types", type=int, default=5)
    parser.add_argument("--n_target_types", type=int, default=5)
    args = parser.parse_args()

    data_root = args.data_root or os.path.join(PROJECT_ROOT, "dataset")
    run_analysis(args.datasets, data_root, args.n_drug_types, args.n_target_types, args.output_dir)
