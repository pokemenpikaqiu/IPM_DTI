# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RLEnvironment:
    def __init__(self, drug_feats, target_feats, drug_ids, target_ids,
                 adj=None, embed_dim=64, max_steps=50):
        self.drug_feats = drug_feats
        self.target_feats = target_feats
        self.drug_ids = drug_ids
        self.target_ids = target_ids
        self.n_drugs = len(drug_ids)
        self.n_targets = len(target_ids)
        self.embed_dim = embed_dim
        self.max_steps = max_steps
        self.d2i = {d: i for i, d in enumerate(drug_ids)}
        self.t2i = {t: i for i, t in enumerate(target_ids)}
        self.adj = adj if adj is not None else np.zeros((self.n_drugs, self.n_targets), dtype=np.float32)
        self.state_dim = embed_dim
        self.action_dim = self.n_targets
        self._drug_idx = 0
        self._step = 0
        self._visited = set()
        self._init_emb()

    def _init_emb(self):
        rng = np.random.RandomState(42)
        scale = np.sqrt(2.0 / self.embed_dim)
        self.drug_emb = (rng.randn(self.n_drugs, self.embed_dim) * scale).astype(np.float32)
        self.target_emb = (rng.randn(self.n_targets, self.embed_dim) * scale).astype(np.float32)

    def reset(self, drug_id=None):
        self._drug_idx = self.d2i.get(drug_id, np.random.randint(0, self.n_drugs))
        self._step = 0
        self._visited = set()
        return self.drug_emb[self._drug_idx].copy()

    def step(self, action):
        if action < 0 or action >= self.n_targets:
            return self.drug_emb[self._drug_idx].copy(), -1.0, True, {"error": "invalid_action"}
        self._step += 1
        self._visited.add(action)
        reward = 1.0 if self.adj[self._drug_idx, action] > 0 else -0.1
        if action in self._visited:
            reward -= 0.5
        done = self._step >= self.max_steps
        return self.drug_emb[self._drug_idx].copy(), reward, done, {"drug_idx": self._drug_idx, "target_idx": action}

    def valid_actions(self):
        return list(range(self.n_targets))

    def set_interactions(self, interactions):
        self.adj.fill(0)
        for did, tid, label in interactions:
            di = self.d2i.get(did)
            ti = self.t2i.get(tid)
            if di is not None and ti is not None:
                self.adj[di, ti] = float(label)

    @classmethod
    def from_dataset(cls, dataset_name, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(_BASE_DIR, "dataset", dataset_name)
        drug_feat = np.load(os.path.join(data_dir, f"{dataset_name}_drug_features.npy"))
        target_feat = np.load(os.path.join(data_dir, f"{dataset_name}_target_features.npy"))
        drug_ids = np.load(os.path.join(data_dir, f"{dataset_name}_drug_ids.npy"), allow_pickle=True).tolist()
        target_ids = np.load(os.path.join(data_dir, f"{dataset_name}_target_ids.npy"), allow_pickle=True).tolist()
        df = pd.read_csv(os.path.join(_BASE_DIR, "dataset", f"{dataset_name}_dataset.csv"))
        interactions = list(zip(df["drug_id"], df["target_id"], df["label"]))
        env = cls(drug_feat, target_feat, drug_ids, target_ids)
        env.set_interactions(interactions)
        return env


class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001,
                 gamma=0.99, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        rng = np.random.RandomState(42)
        s = np.sqrt(2.0 / state_dim)
        self.W1 = (rng.randn(state_dim, hidden_dim) * s).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = (rng.randn(hidden_dim, action_dim) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(action_dim, dtype=np.float32)
        self._mW1 = np.zeros_like(self.W1)
        self._mb1 = np.zeros_like(self.b1)
        self._mW2 = np.zeros_like(self.W2)
        self._mb2 = np.zeros_like(self.b2)

    def _forward(self, state):
        h = np.maximum(0, state @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def select_action(self, state, valid_actions=None):
        if np.random.rand() < self.eps:
            return np.random.choice(valid_actions) if valid_actions else np.random.randint(0, self.action_dim)
        q = self._forward(state)
        if valid_actions:
            mask = np.full(self.action_dim, -np.inf)
            mask[valid_actions] = 0
            q = q + mask
        return int(np.argmax(q))

    def update(self, state, action, reward, ns, done):
        q_vals = self._forward(state)
        q_val = q_vals[action]
        nq = self._forward(ns)
        target = reward + (0 if done else self.gamma * np.max(nq))
        td_err = target - q_val
        grad = np.zeros(self.action_dim, dtype=np.float32)
        grad[action] = -td_err
        h = np.maximum(0, state @ self.W1 + self.b1)
        gW2 = np.outer(h, grad)
        gh = grad @ self.W2.T
        gh = gh * (h > 0)
        gW1 = np.outer(state, gh)
        mom = 0.9
        self._mW1 = mom * self._mW1 + self.lr * gW1
        self._mb1 = mom * self._mb1 + self.lr * gh
        self._mW2 = mom * self._mW2 + self.lr * gW2
        self._mb2 = mom * self._mb2 + self.lr * grad
        self.W1 -= self._mW1
        self.b1 -= self._mb1
        self.W2 -= self._mW2
        self.b2 -= self._mb2
        return float(td_err ** 2)

    def decay_eps(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def train(self, env, n_episodes=1000, max_steps=50):
        losses = []
        for _ in range(n_episodes):
            state = env.reset()
            e_loss = 0.0
            for _ in range(max_steps):
                action = self.select_action(state, env.valid_actions())
                ns, reward, done, _ = env.step(action)
                e_loss += self.update(state, action, reward, ns, done)
                state = ns
                if done:
                    break
            self.decay_eps()
            losses.append(e_loss)
        return losses

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, eps=np.array([self.eps]))

    def load(self, path):
        data = np.load(path)
        self.W1, self.b1 = data["W1"], data["b1"]
        self.W2, self.b2 = data["W2"], data["b2"]
        self.eps = float(data["eps"][0])


class MultiSpacePredictor:
    def __init__(self, drug_emb, target_emb, drug_ids, target_ids):
        self.drug_emb = drug_emb
        self.target_emb = target_emb
        self.drug_ids = drug_ids
        self.target_ids = target_ids
        self.d2i = {d: i for i, d in enumerate(drug_ids)}
        self.t2i = {t: i for i, t in enumerate(target_ids)}

    def predict_similarity(self, drug_id, k=10):
        di = self.d2i.get(drug_id)
        if di is None:
            return []
        dv = self.drug_emb[di]
        t_norm = np.linalg.norm(self.target_emb, axis=1)
        t_norm = np.where(t_norm == 0, 1.0, t_norm)
        tn = self.target_emb / t_norm[:, np.newaxis]
        dn = np.linalg.norm(dv)
        dn = 1.0 if dn == 0 else dv / dn
        scores = tn @ dn
        for i in np.argsort(-scores)[:k]:
            yield self.target_ids[i], float(scores[i])

    def predict_all(self, k=20):
        results = []
        for did in self.drug_ids:
            for tid, score in self.predict_similarity(did, k):
                results.append({"drug_id": did, "target_id": tid, "similarity_score": score})
        return results
