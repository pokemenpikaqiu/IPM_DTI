# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_kge_preds(kge_dir: str, dataset: str) -> Dict[str, List[Dict]]:
    results = {}
    base = os.path.join(kge_dir, dataset)
    if not os.path.exists(base):
        return results
    for cluster_dir in os.listdir(base):
        if not cluster_dir.startswith("cluster_"):
            continue
        cid = cluster_dir.split("_")[1]
        pred_path = os.path.join(base, cluster_dir, "predictions.json")
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                results[cid] = json.load(f)
    return results


def load_rl_preds(rl_dir: str, dataset: str) -> Dict[str, List[Dict]]:
    results = {}
    base = os.path.join(rl_dir, dataset)
    if not os.path.exists(base):
        return results
    for cluster_dir in os.listdir(base):
        if not cluster_dir.startswith("cluster_"):
            continue
        cid = cluster_dir.split("_")[1]
        pred_path = os.path.join(base, cluster_dir, "predictions.json")
        if os.path.exists(pred_path):
            with open(pred_path) as f:
                data = json.load(f)
                results[cid] = data.get("predictions", [])
    return results


class ResultFusion:
    def __init__(self, alpha=0.4, beta=0.4, gamma=0.2, decay=0.95):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.decay = decay
        self.kge_preds = {}
        self.rl_preds = {}

    def load(self, kge_results, rl_results):
        self.kge_preds = kge_results
        self.rl_preds = rl_results

    def _norm_kge(self, rank, total):
        return self.decay ** rank

    def _aggregate(self):
        pair_stats = defaultdict(lambda: {"ks": [], "rp": [], "clusters": set()})

        for cid, preds in self.kge_preds.items():
            for p in preds:
                pair = (p["drug_id"], p["target_id"])
                rank = preds.index(p)
                ks = self._norm_kge(rank, len(preds))
                pair_stats[pair]["ks"].append((cid, ks))
                pair_stats[pair]["clusters"].add(cid)

        for cid, preds in self.rl_preds.items():
            for p in preds:
                pair = (p["drug_id"], p["target_id"])
                rp = p["probability"]
                pair_stats[pair]["rp"].append((cid, rp))
                pair_stats[pair]["clusters"].add(cid)

        for pair, stats in pair_stats.items():
            stats["kge_score"] = min([s for _, s in stats["ks"]], default=0)
            stats["rl_prob"] = max([p for _, p in stats["rp"]], default=0)
            stats["n_clusters"] = len(stats["clusters"])

        return pair_stats

    def _fuse(self, pair_stats):
        results = []
        for (did, tid), stats in pair_stats.items():
            ks = stats["kge_score"]
            rp = stats["rl_prob"]
            n_clusters = stats["n_clusters"]

            kge_freq = len(stats["ks"])
            rl_freq = len(stats["rp"])
            total_freq = kge_freq + rl_freq

            fusion_score = (
                self.alpha * ks +
                self.beta * rp +
                self.gamma * min(n_clusters / 10, 1.0)
            )

            results.append({
                "drug_id": did,
                "target_id": tid,
                "fusion_score": float(fusion_score),
                "kge_score": float(ks),
                "rl_prob": float(rp),
                "n_clusters": n_clusters,
                "kge_freq": kge_freq,
                "rl_freq": rl_freq,
                "total_freq": total_freq,
                "clusters": sorted(list(stats["clusters"])),
            })

        results.sort(key=lambda x: x["fusion_score"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        return results

    def fuse(self, top_k=1000):
        pair_stats = self._aggregate()
        return self._fuse(pair_stats)[:top_k]


def eval_preds(predictions: List[Dict], gt_path: str, top_ks=[100, 200, 500]) -> Dict:
    true_pairs = set()
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            for line in f:
                h, _, t = line.strip().split()
                true_pairs.add((h, t))
    if not true_pairs:
        return {"hit_rate": {}}

    metrics = {}
    for k in top_ks:
        hits = sum(1 for p in predictions[:k] if (p["drug_id"], p["target_id"]) in true_pairs)
        metrics[f"hit_rate@{k}"] = hits / min(k, len(true_pairs))
    return metrics


def save_fusion_results(predictions, metrics, output_path, params):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output = {
        "params": params,
        "metrics": metrics,
        "n_predictions": len(predictions),
        "predictions": predictions,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    csv_path = output_path.replace(".json", ".csv")
    pd.DataFrame(predictions).to_csv(csv_path, index=False)


def run_fusion_pipeline(dataset, kge_dir, rl_dir, kg_dir, output_dir,
                      alpha=0.4, beta=0.4, gamma=0.2, top_k=1000):
    kge_results = load_kge_preds(kge_dir, dataset)
    rl_results = load_rl_preds(rl_dir, dataset)

    if not kge_results and not rl_results:
        print("no predictions found")
        return []

    fusion = ResultFusion(alpha=alpha, beta=beta, gamma=gamma)
    fusion.load(kge_results, rl_results)
    predictions = fusion.fuse(top_k=top_k)

    gt_path = os.path.join(kg_dir, dataset, "global", "global_triples.txt")
    metrics = eval_preds(predictions, gt_path)

    out_path = os.path.join(output_dir, dataset, "fusion_results.json")
    save_fusion_results(predictions, metrics, out_path,
                       {"alpha": alpha, "beta": beta, "gamma": gamma, "top_k": top_k})

    summary_path = os.path.join(output_dir, dataset, "fusion_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "dataset": dataset,
            "weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
            "n_kge_clusters": len(kge_results),
            "n_rl_clusters": len(rl_results),
            "n_predictions": len(predictions),
            "metrics": metrics,
            "top_10": predictions[:10],
        }, f, indent=2)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--kge_dir", type=str, default=None)
    parser.add_argument("--rl_dir", type=str, default=None)
    parser.add_argument("--kg_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--top_k", type=int, default=1000)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(__file__))
    kge_dir = args.kge_dir or os.path.join(base, "kg_embedding_output")
    rl_dir = args.rl_dir or os.path.join(base, "rl_prediction_output")
    kg_dir = args.kg_dir or os.path.join(base, "knowledge_graph_output")
    out_dir = args.output_dir or os.path.join(base, "result_fusion_output")

    run_fusion_pipeline(args.dataset, kge_dir, rl_dir, kg_dir, out_dir,
                      args.alpha, args.beta, args.gamma, args.top_k)
