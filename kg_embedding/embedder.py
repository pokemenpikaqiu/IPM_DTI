# -*- coding: utf-8 -*-
import os
import json
import numpy as np

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KGE_OUTPUT_DIR = os.path.join(_BASE_DIR, "kg_embedding_output")
os.makedirs(KGE_OUTPUT_DIR, exist_ok=True)


class KGEmbedding:
    def __init__(self, n_e, n_r, dim=64, model_type="TransE", seed=42):
        self.n_e = n_e
        self.n_r = n_r
        self.dim = dim
        self.model_type = model_type
        rng = np.random.RandomState(seed)
        scale = 6.0 / np.sqrt(dim)
        self.E = rng.uniform(-scale, scale, (n_e, dim)).astype(np.float32)
        self.R = rng.uniform(-scale, scale, (n_r, dim)).astype(np.float32)
        self._norm_E()

    def _norm_E(self):
        norms = np.linalg.norm(self.E, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self.E = self.E / norms

    def _transe_score(self, h, r, t):
        return -np.linalg.norm(h + r - t, axis=-1)

    def _rotate_score(self, h, r, t):
        hd = self.dim // 2
        h_re, h_im = h[:, :hd], h[:, hd:]
        r_re, r_im = np.cos(r[:, :hd]), np.sin(r[:, :hd:])
        t_re, t_im = t[:, :hd], t[:, hd:]
        pr = h_re * r_re - h_im * r_im
        pi = h_re * r_im + h_im * r_re
        return -np.sqrt((pr - t_re) ** 2 + (pi - t_im) ** 2 + 1e-9).sum(axis=-1)

    def score(self, h_idx, r_idx, t_idx):
        h = self.E[h_idx]
        r = self.R[r_idx]
        t = self.E[t_idx]
        if self.model_type == "TransE":
            return self._transe_score(h, r, t)
        elif self.model_type == "RotatE":
            return self._rotate_score(h, r, t)
        return self._transe_score(h, r, t)

    def train_step(self, pos_h, pos_r, pos_t, neg_h, neg_t, lr=0.01, margin=1.0):
        pos_scores = self.score(pos_h, pos_r, pos_t)
        neg_scores = self.score(neg_h, pos_r, neg_t)
        losses = np.maximum(0, margin - pos_scores + neg_scores)
        loss = float(losses.mean())
        grad_mask = (losses > 0).astype(np.float32)
        for i in range(len(pos_h)):
            if grad_mask[i] == 0:
                continue
            hi, ri, ti = pos_h[i], pos_r[i], pos_t[i]
            nhi, nti = neg_h[i], neg_t[i]
            h, r, t = self.E[hi], self.R[ri], self.E[ti]
            nh, nt = self.E[nhi], self.E[nti]
            pd = h + r - t
            nd = nh + r - nt
            pn = np.linalg.norm(pd) + 1e-9
            nn = np.linalg.norm(nd) + 1e-9
            gp = pd / pn
            gn = nd / nn
            self.E[hi] -= lr * gp
            self.R[ri] -= lr * (gp - gn)
            self.E[ti] += lr * gp
            self.E[nhi] += lr * gn
            self.E[nti] -= lr * gn
        self._norm_E()
        return loss

    def fit(self, triples, n_epochs=100, batch_size=128, lr=0.01, margin=1.0, neg_ratio=1):
        triples = np.array(triples, dtype=np.int32)
        rng = np.random.RandomState(self.model_type + str(self.n_e))
        losses = []
        for _ in range(n_epochs):
            idx = rng.permutation(len(triples))
            triples = triples[idx]
            e_loss = 0.0
            n_batches = 0
            for start in range(0, len(triples), batch_size):
                batch = triples[start:start + batch_size]
                ph, pr, pt = batch[:, 0], batch[:, 1], batch[:, 2]
                nh = ph.copy()
                nt = rng.randint(0, self.n_e, size=len(batch))
                corrupt_head = rng.rand(len(batch)) < 0.5
                nh[corrupt_head] = rng.randint(0, self.n_e, size=corrupt_head.sum())
                nt[corrupt_head] = pt[corrupt_head]
                e_loss += self.train_step(ph, pr, pt, nh, nt, lr=lr, margin=margin)
                n_batches += 1
            losses.append(e_loss / max(n_batches, 1))
        return losses

    def save(self, out_dir, prefix="kge"):
        os.makedirs(out_dir, exist_ok=True)
        np.save(os.path.join(out_dir, f"{prefix}_entity_embeddings.npy"), self.E)
        np.save(os.path.join(out_dir, f"{prefix}_relation_embeddings.npy"), self.R)
        with open(os.path.join(out_dir, f"{prefix}_meta.json"), "w") as f:
            json.dump({"n_entities": self.n_e, "n_relations": self.n_r, "embed_dim": self.dim, "model_type": self.model_type}, f, indent=2)

    @classmethod
    def load(cls, out_dir, prefix="kge"):
        with open(os.path.join(out_dir, f"{prefix}_meta.json")) as f:
            meta = json.load(f)
        obj = cls(meta["n_entities"], meta["n_relations"], meta["embed_dim"], meta["model_type"])
        obj.E = np.load(os.path.join(out_dir, f"{prefix}_entity_embeddings.npy"))
        obj.R = np.load(os.path.join(out_dir, f"{prefix}_relation_embeddings.npy"))
        return obj


class LinkPredictor:
    def __init__(self, kge, e2i, r2i):
        self.kge = kge
        self.e2i = e2i
        self.r2i = r2i
        self.i2e = {v: k for k, v in e2i.items()}

    def predict(self, head, rel, tail):
        hi = self.e2i.get(head)
        ri = self.r2i.get(rel)
        ti = self.e2i.get(tail)
        if hi is None or ri is None or ti is None:
            return 0.0
        return float(self.kge.score(np.array([hi]), np.array([ri]), np.array([ti]))[0])

    def top_k(self, head, rel, candidates, k=10):
        hi = self.e2i.get(head)
        ri = self.r2i.get(rel)
        if hi is None or ri is None:
            return []
        valid = [(tid, self.e2i[tid]) for tid in candidates if tid in self.e2i]
        if not valid:
            return []
        tids = [c[0] for c in valid]
        tidx = np.array([c[1] for c in valid])
        scores = self.kge.score(np.full(len(tidx), hi), np.full(len(tidx), ri), tidx)
        for i in np.argsort(-scores)[:k]:
            yield tids[i], float(scores[i])

    def predict_all(self, drug_id, target_ids, rel="interacts_with", k=20):
        return [{"drug_id": drug_id, "target_id": tid, "kge_score": score}
                for tid, score in self.top_k(drug_id, rel, target_ids, k)]


def load_triples_for_cluster(kg_dir, dataset_name, cid):
    base = os.path.join(kg_dir, dataset_name, "clusters", f"cluster_{cid}",
                       f"cluster_{cid}_expanded_triples.txt")
    if not os.path.exists(base):
        base = os.path.join(kg_dir, dataset_name, "global", f"{dataset_name}_global_triples.txt")
    triples = []
    e2i = {}
    r2i = {}
    with open(base) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            h, r, t = line.split("\t")
            triples.append((h, r, t))
            for e in (h, t):
                if e not in e2i:
                    e2i[e] = len(e2i)
            if r not in r2i:
                r2i[r] = len(r2i)
    return triples, e2i, r2i


def run_kg_embedding_pipeline(dataset_name, kg_dir=None, out_dir=None,
                             model_type="TransE", dim=64, n_epochs=100,
                             batch_size=128, lr=0.01, cluster_ids=None):
    if kg_dir is None:
        kg_dir = os.path.join(_BASE_DIR, "knowledge_graph_output")
    if out_dir is None:
        out_dir = os.path.join(KGE_OUTPUT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    clusters_dir = os.path.join(kg_dir, dataset_name, "clusters")
    if cluster_ids is None:
        if os.path.exists(clusters_dir):
            cluster_ids = sorted([
                int(d.split("_")[1]) for d in os.listdir(clusters_dir)
                if d.startswith("cluster_")
            ])
        else:
            cluster_ids = [0]

    all_results = {}
    for cid in cluster_ids:
        try:
            triples, e2i, r2i = load_triples_for_cluster(kg_dir, dataset_name, cid)
        except FileNotFoundError:
            continue
        if not triples:
            continue
        triples_int = [(e2i[h], r2i[r], e2i[t]) for h, r, t in triples]
        kge = KGEmbedding(len(e2i), len(r2i), dim, model_type)
        losses = kge.fit(triples_int, n_epochs=n_epochs, batch_size=batch_size, lr=lr)

        c_dir = os.path.join(out_dir, f"cluster_{cid}")
        os.makedirs(c_dir, exist_ok=True)
        kge.save(c_dir, "kge")
        with open(os.path.join(c_dir, "entity_to_idx.json"), "w") as f:
            json.dump(e2i, f)
        with open(os.path.join(c_dir, "relation_to_idx.json"), "w") as f:
            json.dump(r2i, f)

        pred = LinkPredictor(kge, e2i, r2i)
        drugs = [e for e in e2i if e.startswith("DB")]
        targets = [e for e in e2i if e.startswith(("P", "Q"))]
        preds = []
        for did in drugs[:50]:
            preds.extend(pred.predict_all(did, targets, k=20))
        with open(os.path.join(c_dir, "predictions.json"), "w") as f:
            json.dump(preds, f, indent=2)

        all_results[cid] = {
            "n_entities": len(e2i), "n_relations": len(r2i), "n_triples": len(triples),
            "final_loss": losses[-1] if losses else None, "n_predictions": len(preds),
        }

    summary = {"dataset": dataset_name, "model_type": model_type, "embed_dim": dim,
               "n_epochs": n_epochs, "clusters": all_results, "output_dir": out_dir}
    with open(os.path.join(out_dir, "kge_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_type", type=str, default="TransE", choices=["TransE", "RotatE"])
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--kg_dir", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()
    run_kg_embedding_pipeline(args.dataset, args.kg_dir, args.out_dir,
                            args.model_type, args.dim, args.n_epochs, lr=args.lr)
