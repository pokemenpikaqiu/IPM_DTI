# -*- coding: utf-8 -*-
import os

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")

DATASET_FILES = {
    "lou": os.path.join(DATA_ROOT, "lou_dataset.csv"),
    "yamanishi": os.path.join(DATA_ROOT, "yamanishi_dataset.csv"),
    "zheng": os.path.join(DATA_ROOT, "zheng_dataset.csv"),
}

DRUG_EMBED_FILE = os.path.join(DATA_ROOT, "{dataset}", "drug_features.h5")
TARGET_EMBED_FILE = os.path.join(DATA_ROOT, "{dataset}", "target_features.h5")
FUSED_FEATURES_FILE = os.path.join(DATA_ROOT, "{dataset}", "fused_features.h5")

DRUG_EMBED_DIM = 1024
TARGET_EMBED_DIM = 512
FUSED_EMBED_DIM = 1536

DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

FUSION_CONFIG = {
    "concat": {
        "input_dim": DRUG_EMBED_DIM + TARGET_EMBED_DIM,
        "output_dim": FUSED_EMBED_DIM,
    },
    "hadamard": {
        "input_dim": DRUG_EMBED_DIM + TARGET_EMBED_DIM,
        "output_dim": FUSED_EMBED_DIM,
    },
    "cross_attention": {
        "input_dim": DRUG_EMBED_DIM + TARGET_EMBED_DIM,
        "output_dim": FUSED_EMBED_DIM,
        "n_heads": 8,
    },
}

ALIGNMENT_CONFIG = {
    "max_drugs": 1000,
    "max_targets": 500,
    "normalization": "l2",
}

DRUG_FEATURE_CONFIG = {
    "model_type": "fingerprint",
    "embed_dim": DRUG_EMBED_DIM,
    "fingerprint_type": "morgan",
    "radius": 2,
    "n_bits": 2048,
}

TARGET_FEATURE_CONFIG = {
    "model_type": "fegs",
    "embed_dim": TARGET_EMBED_DIM,
    "k_mer": 3,
    "svm_dim": 1965,
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "fusion_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
