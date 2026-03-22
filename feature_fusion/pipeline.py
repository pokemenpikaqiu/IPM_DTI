# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import json

from .config import DATA_ROOT, DATASET_FILES, FUSED_EMBED_DIM, OUTPUT_DIR
from .dataset import DTIDataset
from .extract_all import extract_all_features
from .fusion_network import FusionMLP


def run_fusion_pipeline(dataset_name, fusion="concat", output_dir=None,
                       drug_model="fingerprint", target_model="fegs",
                       use_api=True, force=False):
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    feat_dir = os.path.join(DATA_ROOT, dataset_name)
    alignment = extract_all_features(
        dataset_name, feat_dir,
        drug_model=drug_model, target_model=target_model,
        use_api=use_api, force=force,
    )

    dataset = DTIDataset(dataset_name)
    pos_pairs = dataset.get_positive_pairs()

    net = FusionMLP(
        drug_dim=alignment.drug_features.shape[1],
        target_dim=alignment.target_features.shape[1],
        output_dim=FUSED_EMBED_DIM,
        fusion=fusion,
    )

    fused, valid = net.fuse_batch(
        alignment.drug_features, alignment.target_features,
        alignment.drug_ids, alignment.target_ids,
        pos_pairs,
    )

    fused_path = os.path.join(output_dir, f"{dataset_name}_{fusion}_fused.npy")
    pairs_path = os.path.join(output_dir, f"{dataset_name}_{fusion}_pairs.npy")
    np.save(fused_path, fused)
    np.save(pairs_path, np.array(valid))

    summary = {
        "dataset": dataset_name,
        "fusion_mode": fusion,
        "n_drugs": len(alignment.drug_ids),
        "n_targets": len(alignment.target_ids),
        "n_pos_pairs": len(pos_pairs),
        "n_fused": len(valid),
        "fused_dim": fused.shape[1] if len(fused) > 0 else 0,
        "fused_path": fused_path,
        "pairs_path": pairs_path,
    }
    with open(os.path.join(output_dir, f"{dataset_name}_{fusion}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def load_fused_output(dataset_name, fusion="concat", output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, dataset_name)
    fused_path = os.path.join(output_dir, f"{dataset_name}_{fusion}_fused.npy")
    pairs_path = os.path.join(output_dir, f"{dataset_name}_{fusion}_pairs.npy")
    if not os.path.exists(fused_path):
        raise FileNotFoundError(f"Fused features not found: {fused_path}")
    fused = np.load(fused_path)
    pairs = [(str(p[0]), str(p[1])) for p in np.load(pairs_path, allow_pickle=True)]
    return fused, pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_FILES.keys()))
    parser.add_argument("--fusion_mode", type=str, default="concat",
                        choices=["concat", "hadamard", "cross_attention"])
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--drug_model", type=str, default="fingerprint")
    parser.add_argument("--target_model", type=str, default="fegs")
    parser.add_argument("--no_api", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_fusion_pipeline(
        args.dataset, fusion=args.fusion_mode, output_dir=args.output_dir,
        drug_model=args.drug_model, target_model=args.target_model,
        use_api=not args.no_api, force=args.force,
    )
