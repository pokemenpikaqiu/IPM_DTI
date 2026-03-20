#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Drug-Target Interaction Prediction System
Configuration Module
"""

import os

# ==================== Data Configuration ====================
DATA_CONFIG = {
    'data_dir': './data',
    'raw_data_file': 'zheng_dataset.csv',
    'processed_data_file': 'processed_data.pkl',
}

DATA_PATH = './data/zheng_dataset.csv'
OUTPUT_DIR = './output'

# ==================== Knowledge Graph Configuration ====================
KG_CONFIG = {
    'walk_length': 10,
    'num_walks': 5,
    'window_size': 5,
    'embedding_dim': 128,
}

# ==================== Feature Configuration ====================
FEATURE_CONFIG = {
    'drug_feature_dim': 256,
    'target_feature_dim': 256,
    'use_morgan_fingerprint': True,
    'morgan_radius': 2,
    'morgan_bits': 2048,
    'use_sequence_features': True,
    'sequence_embedding_dim': 128,
}

DRUG_FEATURE_DIM = 256
TARGET_FEATURE_DIM = 256
HIDDEN_DIM = 128
FUSION_DIM = 64

# ==================== Fusion Network Configuration ====================
FUSION_CONFIG = {
    'drug_input_dim': 256,
    'target_input_dim': 256,
    'hidden_dim': 128,
    'output_dim': 64,
    'num_heads': 4,
    'dropout': 0.1,
    'use_attention': True,
}

# ==================== Subspace Configuration ====================
SUBSPACE_CONFIG = {
    'n_clusters': 5,
    'similarity_threshold': 0.7,
    'min_cluster_size': 10,
    'max_iter': 100,
    'random_state': 42,
}

NUM_SUBSPACES = 5
SIMILARITY_THRESHOLD = 0.7
RANDOM_WALK_STEPS = 10
RESTART_PROBABILITY = 0.15

# ==================== Model Configuration ====================
MODEL_CONFIG = {
    'hidden_dim': 128,
    'num_layers': 3,
    'dropout': 0.1,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'batch_size': 64,
    'epochs': 100,
}

# ==================== Training Configuration ====================
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# ==================== Reinforcement Learning Configuration ====================
RL_CONFIG = {
    'state_dim': 128,
    'action_dim': 64,
    'hidden_dim': 128,
    'learning_rate': 0.0005,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'buffer_size': 10000,
    'batch_size': 64,
    'target_update': 10,
    'episodes': 500,
    'max_steps': 50,
}

RL_HIDDEN_DIM = 128
RL_LEARNING_RATE = 0.0005
RL_GAMMA = 0.99
RL_EPISODES = 500
RL_MAX_STEPS = 50

# ==================== Knowledge Graph Embedding Configuration ====================
EMBEDDING_CONFIG = {
    'model_type': 'TransE',
    'embedding_dim': 128,
    'margin': 1.0,
    'norm': 2,
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 100,
    'negative_samples': 5,
}

EMBEDDING_DIM = 128
KG_EMBEDDING_MODEL = 'TransE'
KG_MARGIN = 1.0
KG_NORM = 2

# ==================== Result Fusion Configuration ====================
RL_WEIGHT = 0.5
KG_WEIGHT = 0.5
RANK_METHOD = 'weighted'

# ==================== Logging Configuration ====================
LOG_LEVEL = 'INFO'
LOG_FILE = 'dti_prediction.log'

# ==================== Device Configuration ====================
USE_GPU = True
GPU_ID = 0


def get_device():
    """Get computation device (GPU/CPU)"""
    import torch
    if USE_GPU and torch.cuda.is_available():
        return torch.device(f'cuda:{GPU_ID}')
    return torch.device('cpu')


def update_config(**kwargs):
    """Update configuration parameters"""
    for key, value in kwargs.items():
        if key in globals():
            globals()[key] = value
        else:
            raise ValueError(f"Unknown config parameter: {key}")


def print_config():
    """Print current configuration"""
    print("=" * 50)
    print("Current Configuration:")
    print("=" * 50)
    print(f"Data Path: {DATA_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Embedding Dim: {EMBEDDING_DIM}")
    print(f"KG Model: {KG_EMBEDDING_MODEL}")
    print("=" * 50)


if __name__ == '__main__':
    print_config()
