# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd

from .config import DATA_ROOT, DATASET_FILES, DRUG_EMBED_DIM, TARGET_EMBED_DIM
from .extract_drug import extract_drug_features
from .extract_target import extract_target_features
from .dataset import FeatureAlignment


def is_drugbank(drug_id):
    return isinstance(drug_id, str) and drug_id.startswith("DB") and len(drug_id) >= 7


def is_uniprot(target_id):
    if not isinstance(target_id, str):
        return False
    return target_id.startswith(("P", "Q", "O"))


def load_csv(dataset_name):
    csv_path = DATASET_FILES.get(dataset_name)
    if csv_path is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df, df["drug_id"].unique().tolist(), df["target_id"].unique().tolist()


def extract_all_features(
    dataset_name,
    output_dir=None,
    drug_model="fingerprint",
    target_model="fegs",
    use_api=True,
    force=False,
):
    if output_dir is None:
        output_dir = os.path.join(DATA_ROOT, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    df, drug_ids, tids = load_csv(dataset_name)

    dfs, valid_drugs = extract_drug_features(
        drug_ids, output_dir, dataset_name,
        model_type=drug_model, force_recompute=force,
    )

    tfs, valid_tids = extract_target_features(
        tids, output_dir, dataset_name,
        model_type=target_model, use_api=use_api, force_recompute=force,
    )

    alignment = FeatureAlignment(dfs, tfs, valid_drugs, valid_tids)
    alignment.normalize(method="l2")
    alignment.save(output_dir, dataset_name)

    summary_path = os.path.join(output_dir, f"{dataset_name}_feature_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"dataset: {dataset_name}\n")
        f.write(f"n_drugs: {len(valid_drugs)}\n")
        f.write(f"n_targets: {len(valid_tids)}\n")
        f.write(f"drug_dim: {dfs.shape[1]}\n")
        f.write(f"target_dim: {tfs.shape[1]}\n")
    return alignment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_FILES.keys()))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--drug_model", type=str, default="fingerprint")
    parser.add_argument("--target_model", type=str, default="fegs")
    parser.add_argument("--no_api", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    extract_all_features(
        args.dataset, args.output_dir,
        drug_model=args.drug_model, target_model=args.target_model,
        use_api=not args.no_api, force=args.force,
    )
