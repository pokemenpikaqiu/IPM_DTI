# -*- coding: utf-8 -*-
import os
import hashlib
import numpy as np

from .config import DRUG_EMBED_DIM, DRUG_FEATURE_CONFIG


class DrugFeatureExtractor:
    def __init__(self, model_type="fingerprint", embed_dim=DRUG_EMBED_DIM):
        self.model_type = model_type
        self.embed_dim = embed_dim
        self._cache = {}

    def _hash_vec(self, drug_id):
        seed = int(hashlib.md5(drug_id.encode()).hexdigest(), 16) % (2 ** 32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.embed_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def _morgan_fp(self, drug_id):
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            mol = Chem.MolFromSmiles(drug_id)
            if mol is None:
                return self._hash_vec(drug_id)
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol,
                radius=DRUG_FEATURE_CONFIG["radius"],
                nBits=DRUG_FEATURE_CONFIG["n_bits"],
            )
            arr = np.zeros(DRUG_FEATURE_CONFIG["n_bits"], dtype=np.float32)
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, arr)
            if len(arr) < self.embed_dim:
                arr = np.pad(arr, (0, self.embed_dim - len(arr)))
            elif len(arr) > self.embed_dim:
                arr = arr[:self.embed_dim]
            return arr
        except ImportError:
            return self._hash_vec(drug_id)

    def extract_one(self, drug_id):
        if drug_id in self._cache:
            return self._cache[drug_id]
        if self.model_type == "fingerprint":
            feat = self._morgan_fp(drug_id)
        elif self.model_type == "graphdta":
            feat = self._hash_vec(drug_id)
        else:
            feat = self._hash_vec(drug_id)
        self._cache[drug_id] = feat
        return feat

    def extract_batch(self, drug_ids, show_progress=True):
        feats = []
        valid = []
        it = drug_ids
        if show_progress:
            try:
                from tqdm import tqdm
                it = tqdm(drug_ids, desc="Extracting drug features")
            except ImportError:
                pass
        for did in it:
            feat = self.extract_one(did)
            feats.append(feat)
            valid.append(did)
        return np.array(feats, dtype=np.float32), valid

    def save_features(self, features, drug_ids, output_dir, dataset_name):
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"{dataset_name}_drug_features.npy"), features)
        np.save(os.path.join(output_dir, f"{dataset_name}_drug_ids.npy"), np.array(drug_ids))

    def load_features(self, output_dir, dataset_name):
        feat_path = os.path.join(output_dir, f"{dataset_name}_drug_features.npy")
        ids_path = os.path.join(output_dir, f"{dataset_name}_drug_ids.npy")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Drug features not found: {feat_path}")
        return np.load(feat_path), np.load(ids_path, allow_pickle=True).tolist()


def extract_drug_features(drug_ids, output_dir, dataset_name,
                          model_type="fingerprint", embed_dim=DRUG_EMBED_DIM,
                          force_recompute=False):
    extractor = DrugFeatureExtractor(model_type=model_type, embed_dim=embed_dim)
    feat_path = os.path.join(output_dir, f"{dataset_name}_drug_features.npy")
    if os.path.exists(feat_path) and not force_recompute:
        return extractor.load_features(output_dir, dataset_name)
    features, valid_ids = extractor.extract_batch(drug_ids)
    extractor.save_features(features, valid_ids, output_dir, dataset_name)
    return features, valid_ids
