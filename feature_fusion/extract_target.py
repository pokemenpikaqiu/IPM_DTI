# -*- coding: utf-8 -*-
import os
import hashlib
import numpy as np
import time

from .config import TARGET_EMBED_DIM, TARGET_FEATURE_CONFIG

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def _kmer_comp(seq, k=3):
    aa_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    n = len(AMINO_ACIDS)
    n_kmers = n ** k
    counts = np.zeros(n_kmers, dtype=np.float32)
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        idx = 0
        ok = True
        for ch in kmer:
            if ch not in aa_idx:
                ok = False
                break
            idx = idx * n + aa_idx[ch]
        if ok:
            counts[idx] += 1
    total = counts.sum()
    if total > 0:
        counts = counts / total
    return counts


def _fegs(seq, dim=TARGET_EMBED_DIM):
    k3 = _kmer_comp(seq, k=3)
    if len(k3) >= dim:
        return k3[:dim].astype(np.float32)
    return np.pad(k3, (0, dim - len(k3))).astype(np.float32)


def _hash_vec(target_id, dim=TARGET_EMBED_DIM):
    seed = int(hashlib.md5(target_id.encode()).hexdigest(), 16) % (2 ** 32)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def _fetch_uniprot(uniprot_id, retries=3):
    import urllib.request, urllib.error
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                content = resp.read().decode("utf-8")
            lines = content.strip().split("\n")
            return "".join(lines[1:])
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


class TargetFeatureExtractor:
    def __init__(self, model_type="fegs", embed_dim=TARGET_EMBED_DIM, use_api=True):
        self.model_type = model_type
        self.embed_dim = embed_dim
        self.use_api = use_api
        self._seq_cache = {}
        self._feat_cache = {}

    def _get_seq(self, target_id):
        if target_id in self._seq_cache:
            return self._seq_cache[target_id]
        if self.use_api and target_id.startswith(("P", "Q", "O")):
            seq = _fetch_uniprot(target_id)
            if seq:
                self._seq_cache[target_id] = seq
                return seq
        return None

    def extract_one(self, target_id):
        if target_id in self._feat_cache:
            return self._feat_cache[target_id]
        seq = self._get_seq(target_id)
        if seq:
            feat = _fegs(seq, self.embed_dim)
        else:
            feat = _hash_vec(target_id, self.embed_dim)
        self._feat_cache[target_id] = feat
        return feat

    def extract_batch(self, target_ids, show_progress=True):
        feats = []
        valid = []
        it = target_ids
        if show_progress:
            try:
                from tqdm import tqdm
                it = tqdm(target_ids, desc="Extracting target features")
            except ImportError:
                pass
        for tid in it:
            feat = self.extract_one(tid)
            feats.append(feat)
            valid.append(tid)
        return np.array(feats, dtype=np.float32), valid

    def save_features(self, features, target_ids, output_dir, dataset_name):
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, f"{dataset_name}_target_features.npy"), features)
        np.save(os.path.join(output_dir, f"{dataset_name}_target_ids.npy"), np.array(target_ids))

    def load_features(self, output_dir, dataset_name):
        feat_path = os.path.join(output_dir, f"{dataset_name}_target_features.npy")
        ids_path = os.path.join(output_dir, f"{dataset_name}_target_ids.npy")
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Target features not found: {feat_path}")
        return np.load(feat_path), np.load(ids_path, allow_pickle=True).tolist()

    def save_sequences(self, output_dir, dataset_name):
        os.makedirs(output_dir, exist_ok=True)
        seq_path = os.path.join(output_dir, f"{dataset_name}_sequences.txt")
        with open(seq_path, "w") as f:
            for tid, seq in self._seq_cache.items():
                f.write(f"{tid}\t{seq}\n")

    def load_sequences(self, output_dir, dataset_name):
        seq_path = os.path.join(output_dir, f"{dataset_name}_sequences.txt")
        if not os.path.exists(seq_path):
            return
        with open(seq_path, "r") as f:
            for line in f:
                line = line.strip()
                if "\t" in line:
                    tid, seq = line.split("\t", 1)
                    self._seq_cache[tid] = seq


def extract_target_features(target_ids, output_dir, dataset_name,
                           model_type="fegs", embed_dim=TARGET_EMBED_DIM,
                           use_api=True, force_recompute=False):
    extractor = TargetFeatureExtractor(model_type=model_type, embed_dim=embed_dim, use_api=use_api)
    feat_path = os.path.join(output_dir, f"{dataset_name}_target_features.npy")
    if os.path.exists(feat_path) and not force_recompute:
        return extractor.load_features(output_dir, dataset_name)
    extractor.load_sequences(output_dir, dataset_name)
    features, valid_ids = extractor.extract_batch(target_ids)
    extractor.save_features(features, valid_ids, output_dir, dataset_name)
    extractor.save_sequences(output_dir, dataset_name)
    return features, valid_ids
