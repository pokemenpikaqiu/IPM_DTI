# -*- coding: utf-8 -*-
import os
import numpy as np

from .config import DRUG_EMBED_DIM, TARGET_EMBED_DIM, FUSED_EMBED_DIM, FUSION_CONFIG


class LinearLayer:
    def __init__(self, in_d, out_d, seed=42):
        rng = np.random.RandomState(seed)
        scale = np.sqrt(2.0 / in_d)
        self.W = (rng.randn(in_d, out_d) * scale).astype(np.float32)
        self.b = np.zeros(out_d, dtype=np.float32)

    def forward(self, x):
        return x @ self.W + self.b


class BatchNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim, dtype=np.float32)
        self.beta = np.zeros(dim, dtype=np.float32)
        self.eps = eps

    def forward(self, x):
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        return self.gamma * ((x - mean) / np.sqrt(var + self.eps)) + self.beta


def relu(x):
    return np.maximum(0, x)


def dropout(x, rate=0.1, train=False):
    if not train or rate == 0:
        return x
    mask = (np.random.rand(*x.shape) > rate).astype(np.float32)
    return x * mask / (1.0 - rate)


class FusionMLP:
    def __init__(self, drug_dim=DRUG_EMBED_DIM, target_dim=TARGET_EMBED_DIM,
                 output_dim=FUSED_EMBED_DIM, fusion="concat",
                 hidden_dims=None, dropout_rate=0.1):
        self.drug_dim = drug_dim
        self.target_dim = target_dim
        self.output_dim = output_dim
        self.fusion = fusion
        self.dropout_rate = dropout_rate
        if hidden_dims is None:
            hidden_dims = [1024, 512]
        self.hidden_dims = hidden_dims

        if fusion == "concat" or fusion == "cross_attention":
            in_d = drug_dim + target_dim
        elif fusion == "hadamard":
            in_d = min(drug_dim, target_dim)
        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        self.layers = []
        self.bns = []
        prev = in_d
        for i, h in enumerate(hidden_dims):
            self.layers.append(LinearLayer(prev, h, seed=i))
            self.bns.append(BatchNorm(h))
            prev = h
        self.out_layer = LinearLayer(prev, output_dim, seed=len(hidden_dims))

        if fusion == "cross_attention":
            n_heads = FUSION_CONFIG["cross_attention"].get("n_heads", 8)
            self.n_heads = n_heads
            rng = np.random.RandomState(99)
            scale = np.sqrt(2.0 / drug_dim)
            self.W_q = (rng.randn(drug_dim, drug_dim) * scale).astype(np.float32)
            self.W_k = (rng.randn(target_dim, drug_dim) * scale).astype(np.float32)
            self.W_v = (rng.randn(target_dim, drug_dim) * scale).astype(np.float32)

    def _concat_fuse(self, d, t):
        if d.ndim == 1:
            return np.concatenate([d, t])
        return np.concatenate([d, t], axis=-1)

    def _hadamard_fuse(self, d, t):
        mn = min(self.drug_dim, self.target_dim)
        return d[..., :mn] * t[..., :mn]

    def _cross_attn_fuse(self, d, t):
        squeeze = False
        if d.ndim == 1:
            d = d[np.newaxis, :]
            t = t[np.newaxis, :]
            squeeze = True
        Q = d @ self.W_q
        K = t @ self.W_k
        V = t @ self.W_v
        scale = np.sqrt(float(self.drug_dim))
        scores = (Q @ K.T) / scale
        attn = np.exp(scores - scores.max(axis=-1, keepdims=True))
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-9)
        out = np.concatenate([d, attn @ V], axis=-1)
        if squeeze:
            out = out[0]
        return out

    def _mlp(self, x, train=False):
        for layer, bn in zip(self.layers, self.bns):
            x = layer.forward(x)
            if x.ndim == 2:
                x = bn.forward(x)
            x = relu(x)
            x = dropout(x, self.dropout_rate, train)
        return self.out_layer.forward(x)

    def forward(self, d, t, train=False):
        if self.fusion == "concat":
            fused = self._concat_fuse(d, t)
        elif self.fusion == "hadamard":
            fused = self._hadamard_fuse(d, t)
        else:
            fused = self._cross_attn_fuse(d, t)
        return self._mlp(fused, train)

    def fuse_batch(self, drug_feats, target_feats, drug_ids, target_ids, pairs):
        d2i = {d: i for i, d in enumerate(drug_ids)}
        t2i = {t: i for i, t in enumerate(target_ids)}
        results = []
        valid_pairs = []
        for did, tid in pairs:
            di = d2i.get(did)
            ti = t2i.get(tid)
            if di is None or ti is None:
                continue
            results.append(self.forward(drug_feats[di], target_feats[ti]))
            valid_pairs.append((did, tid))
        if not results:
            return np.zeros((0, self.output_dim), dtype=np.float32), []
        return np.array(results, dtype=np.float32), valid_pairs

    def save_weights(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        w = {}
        for i, (layer, bn) in enumerate(zip(self.layers, self.bns)):
            w[f"layer_{i}_W"] = layer.W
            w[f"layer_{i}_b"] = layer.b
            w[f"bn_{i}_gamma"] = bn.gamma
            w[f"bn_{i}_beta"] = bn.beta
        w["output_W"] = self.out_layer.W
        w["output_b"] = self.out_layer.b
        np.savez(path, **w)

    def load_weights(self, path):
        data = np.load(path)
        for i, (layer, bn) in enumerate(zip(self.layers, self.bns)):
            layer.W = data[f"layer_{i}_W"]
            layer.b = data[f"layer_{i}_b"]
            bn.gamma = data[f"bn_{i}_gamma"]
            bn.beta = data[f"bn_{i}_beta"]
        self.out_layer.W = data["output_W"]
        self.out_layer.b = data["output_b"]
