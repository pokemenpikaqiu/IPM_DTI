# -*- coding: utf-8 -*-
import os
import json
import argparse
import numpy as np
import pandas as pd

from .config import CLUSTER_OUTPUT_DIR, DEFAULT_N_CLUSTERS
from .core import FeatureClusterer, cluster_triplets, cluster_targets, find_optimal_k


def load_features(dataset_name, feature_type="fused", fusion="concat"):
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if feature_type == "fused":
        fusion_dir = os.path.join(base, "fusion_output", dataset_name)
        fused_path = os.path.join(fusion_dir, f"{dataset_name}_{fusion}_fused.npy")
        pairs_path = os.path.join(fusion_dir, f"{dataset_name}_{fusion}_pairs.npy")
        if not os.path.exists(fused_path):
            data_dir = os.path.join(base, "dataset", dataset_name)
            fused_path = os.path.join(data_dir, f"{dataset_name}_{fusion}_fused.npy")
            pairs_path = os.path.join(data_dir, f"{dataset_name}_{fusion}_pairs.npy")
        feats = np.load(fused_path)
        pairs = [(str(p[0]), str(p[1])) for p in np.load(pairs_path, allow_pickle=True)]
        return feats, [p[0] for p in pairs], [p[1] for p in pairs]
    elif feature_type == "drug":
        data_dir = os.path.join(base, "dataset", dataset_name)
        feats = np.load(os.path.join(data_dir, f"{dataset_name}_drug_features.npy"))
        ids = np.load(os.path.join(data_dir, f"{dataset_name}_drug_ids.npy"), allow_pickle=True).tolist()
        return feats, ids, None
    elif feature_type == "target":
        data_dir = os.path.join(base, "dataset", dataset_name)
        feats = np.load(os.path.join(data_dir, f"{dataset_name}_target_features.npy"))
        ids = np.load(os.path.join(data_dir, f"{dataset_name}_target_ids.npy"), allow_pickle=True).tolist()
        return feats, ids, None
    raise ValueError(f"unknown feature_type: {feature_type}")


def save_results(res, df, output_dir, dataset_name, suffix="triplets"):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"{suffix}_clustered.csv"), index=False)
    np.save(os.path.join(output_dir, f"{suffix}_labels.npy"), res.labels)
    if res.centers is not None:
        np.save(os.path.join(output_dir, f"{suffix}_centers.npy"), res.centers)
    summary = res.to_dict()
    summary["dataset"] = dataset_name
    summary["suffix"] = suffix
    with open(os.path.join(output_dir, f"{suffix}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


def build_profiles(triplet_df, target_df=None):
    profs = {}
    for cid, grp in triplet_df.groupby("cluster_id"):
        cid = int(cid)
        profs[cid] = {
            "cluster_id": cid,
            "n_interactions": len(grp),
            "drug_ids": grp["drug_id"].unique().tolist(),
            "target_ids": grp["target_id"].unique().tolist(),
            "n_drugs": grp["drug_id"].nunique(),
            "n_targets": grp["target_id"].nunique(),
        }
    if target_df is not None and "cluster_id" in target_df.columns:
        for cid, grp in target_df.groupby("cluster_id"):
            cid = int(cid)
            if cid not in profs:
                profs[cid] = {"cluster_id": cid}
            profs[cid]["target_cluster_members"] = grp["target_id"].tolist()
    return profs


def run_clustering_pipeline(dataset_name, algorithm="kmeans", n_clusters=DEFAULT_N_CLUSTERS,
                            output_dir=None, fusion="concat", auto_k=False, random_state=42):
    if output_dir is None:
        output_dir = os.path.join(CLUSTER_OUTPUT_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    try:
        feats, d_ids, t_ids = load_features(dataset_name, "fused", fusion)
        pairs = list(zip(d_ids, t_ids if t_ids else []))
    except FileNotFoundError:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base, "dataset", dataset_name)
        d_path = os.path.join(data_dir, f"{dataset_name}_drug_features.npy")
        t_path = os.path.join(data_dir, f"{dataset_name}_target_features.npy")
        if os.path.exists(d_path) and os.path.exists(t_path):
            d_feats = np.load(d_path)
            t_feats = np.load(t_path)
            feats = np.concatenate([d_feats, t_feats], axis=1) if len(d_feats) == len(t_feats) else d_feats
            d_ids = np.load(os.path.join(data_dir, f"{dataset_name}_drug_ids.npy"), allow_pickle=True).tolist()
            t_ids = np.load(os.path.join(data_dir, f"{dataset_name}_target_ids.npy"), allow_pickle=True).tolist()
            pairs = list(zip(d_ids, t_ids))
        else:
            raise FileNotFoundError(f"no features for {dataset_name}")

    if auto_k:
        n_clusters = find_optimal_k(feats)

    res, df = cluster_triplets(feats, pairs, n_clusters, algorithm, random_state)
    save_results(res, df, output_dir, dataset_name, "triplets")

    target_res, target_df = None, None
    try:
        t_feats, t_ids, _ = load_features(dataset_name, "target")
        target_res, target_df = cluster_targets(t_feats, t_ids, n_clusters, algorithm, random_state)
        save_results(target_res, target_df, output_dir, dataset_name, "targets")
    except FileNotFoundError:
        pass

    profs = build_profiles(df, target_df)
    with open(os.path.join(output_dir, "cluster_profiles.json"), "w") as f:
        json.dump(profs, f, indent=2)

    df.to_csv(os.path.join(output_dir, "interactions_clustered.csv"), index=False)

    summary = {
        "dataset": dataset_name,
        "algorithm": algorithm,
        "n_clusters": res.n_clusters,
        "n_triplets": len(df),
        "triplet_metrics": res.metrics,
        "output_dir": output_dir,
    }
    if target_res:
        summary["target_metrics"] = target_res.metrics
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_clustering_output(dataset_name, output_dir=None, suffix="triplets"):
    if output_dir is None:
        output_dir = os.path.join(CLUSTER_OUTPUT_DIR, dataset_name)
    csv_path = os.path.join(output_dir, f"{suffix}_clustered.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    return pd.read_csv(csv_path), np.load(os.path.join(output_dir, f"{suffix}_labels.npy"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--algorithm", type=str, default="kmeans",
                        choices=["kmeans", "hdbscan", "hierarchical"])
    parser.add_argument("--n_clusters", type=int, default=DEFAULT_N_CLUSTERS)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--fusion_mode", type=str, default="concat")
    parser.add_argument("--auto_k", action="store_true")
    args = parser.parse_args()
    run_clustering_pipeline(
        args.dataset, args.algorithm, args.n_clusters,
        args.output_dir, args.fusion_mode, args.auto_k,
    )
