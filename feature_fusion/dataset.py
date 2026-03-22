# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

from .config import DATA_ROOT, DATASET_FILES, DRUG_EMBED_FILE, TARGET_EMBED_FILE
from .config import FUSED_FEATURES_FILE, DRUG_EMBED_DIM, TARGET_EMBED_DIM, ALIGNMENT_CONFIG


def load_raw_dataset(dataset_name):
    csv_path = DATASET_FILES.get(dataset_name)
    if csv_path is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_fused_data(dataset_name):
    fused_path = FUSED_FEATURES_FILE.format(dataset=dataset_name)
    if not os.path.exists(fused_path):
        raise FileNotFoundError(f"Fused features not found: {fused_path}")
    try:
        import h5py
        with h5py.File(fused_path, "r") as f:
            drug_features = f["drug_features"][:]
            target_features = f["target_features"][:]
            drug_ids = [x.decode() if isinstance(x, bytes) else x for x in f["drug_ids"][:]]
            target_ids = [x.decode() if isinstance(x, bytes) else x for x in f["target_ids"][:]]
    except ImportError:
        drug_features = np.load(fused_path.replace(".h5", "_drug.npy"))
        target_features = np.load(fused_path.replace(".h5", "_target.npy"))
        drug_ids = []
        target_ids = []
    df = load_raw_dataset(dataset_name)
    return drug_features, target_features, df


class DTIDataset:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.df = load_raw_dataset(dataset_name)
        self.drug_ids = self.df["drug_id"].unique().tolist()
        self.target_ids = self.df["target_id"].unique().tolist()
        self.drug_to_idx = {d: i for i, d in enumerate(self.drug_ids)}
        self.target_to_idx = {t: i for i, t in enumerate(self.target_ids)}

    def get_positive_pairs(self):
        pos = self.df[self.df["label"] == 1]
        return list(zip(pos["drug_id"].tolist(), pos["target_id"].tolist()))

    def get_all_pairs(self):
        return list(zip(
            self.df["drug_id"].tolist(),
            self.df["target_id"].tolist(),
            self.df["label"].tolist(),
        ))

    def get_drug_count(self):
        return len(self.drug_ids)

    def get_target_count(self):
        return len(self.target_ids)

    def get_interaction_matrix(self):
        n_drugs = len(self.drug_ids)
        n_targets = len(self.target_ids)
        mat = np.zeros((n_drugs, n_targets), dtype=np.float32)
        for _, row in self.df.iterrows():
            d_idx = self.drug_to_idx.get(row["drug_id"])
            t_idx = self.target_to_idx.get(row["target_id"])
            if d_idx is not None and t_idx is not None:
                mat[d_idx, t_idx] = float(row["label"])
        return mat


class FeatureAlignment:
    def __init__(self, drug_features, target_features, drug_ids, target_ids):
        self.drug_features = drug_features
        self.target_features = target_features
        self.drug_ids = drug_ids
        self.target_ids = target_ids
        self.drug_to_idx = {d: i for i, d in enumerate(drug_ids)}
        self.target_to_idx = {t: i for i, t in enumerate(target_ids)}

    def normalize(self, method="l2"):
        if method == "l2":
            dnorms = np.linalg.norm(self.drug_features, axis=1, keepdims=True)
            dnorms = np.where(dnorms == 0, 1.0, dnorms)
            self.drug_features = self.drug_features / dnorms
            tnorms = np.linalg.norm(self.target_features, axis=1, keepdims=True)
            tnorms = np.where(tnorms == 0, 1.0, tnorms)
            self.target_features = self.target_features / tnorms
        elif method == "minmax":
            d_min = self.drug_features.min(axis=0)
            d_max = self.drug_features.max(axis=0)
            d_range = np.where(d_max - d_min == 0, 1.0, d_max - d_min)
            self.drug_features = (self.drug_features - d_min) / d_range
            t_min = self.target_features.min(axis=0)
            t_max = self.target_features.max(axis=0)
            t_range = np.where(t_max - t_min == 0, 1.0, t_max - t_min)
            self.target_features = (self.target_features - t_min) / t_range
        return self

    def get_drug_feature(self, drug_id):
        idx = self.drug_to_idx.get(drug_id)
        return None if idx is None else self.drug_features[idx]

    def get_target_feature(self, target_id):
        idx = self.target_to_idx.get(target_id)
        return None if idx is None else self.target_features[idx]

    def get_pair_features(self, drug_id, target_id):
        d = self.get_drug_feature(drug_id)
        t = self.get_target_feature(target_id)
        if d is None or t is None:
            return None
        return np.concatenate([d, t])

    def save(self, output_dir, dataset_name):
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"{dataset_name}_drug_features.npy"), self.drug_features)
        np.save(os.path.join(output_dir, f"{dataset_name}_target_features.npy"), self.target_features)
        np.save(os.path.join(output_dir, f"{dataset_name}_drug_ids.npy"), np.array(self.drug_ids))
        np.save(os.path.join(output_dir, f"{dataset_name}_target_ids.npy"), np.array(self.target_ids))

    @classmethod
    def load(cls, output_dir, dataset_name):
        drug_features = np.load(os.path.join(output_dir, f"{dataset_name}_drug_features.npy"))
        target_features = np.load(os.path.join(output_dir, f"{dataset_name}_target_features.npy"))
        drug_ids = np.load(os.path.join(output_dir, f"{dataset_name}_drug_ids.npy"), allow_pickle=True).tolist()
        target_ids = np.load(os.path.join(output_dir, f"{dataset_name}_target_ids.npy"), allow_pickle=True).tolist()
        return cls(drug_features, target_features, drug_ids, target_ids)
