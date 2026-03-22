# -*- coding: utf-8 -*-
import os
import json
import random
import numpy as np
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KG_OUTPUT_DIR = os.path.join(_BASE_DIR, "knowledge_graph_output")
os.makedirs(KG_OUTPUT_DIR, exist_ok=True)


class Entity:
    def __init__(self, eid, etype, cid=None):
        self.eid = eid
        self.etype = etype
        self.cid = cid
        self.attrs = {}

    def to_dict(self):
        return {"entity_id": self.eid, "entity_type": self.etype, "cluster_id": self.cid, "attributes": self.attrs}


class Relation:
    def __init__(self, head, rel, tail, weight=1.0, cid=None):
        self.head = head
        self.rel = rel
        self.tail = tail
        self.weight = weight
        self.cid = cid

    def triple(self):
        return (self.head, self.rel, self.tail)

    def to_dict(self):
        return {"head": self.head, "relation": self.rel, "tail": self.tail, "weight": self.weight, "cluster_id": self.cid}


class KnowledgeGraph:
    def __init__(self, name="dti_kg"):
        self.name = name
        self.entities = {}
        self.relations = []
        self._adj = {}

    def add_entity(self, eid, etype, cid=None, attrs=None):
        if eid not in self.entities:
            e = Entity(eid, etype, cid)
            if attrs:
                e.attrs.update(attrs)
            self.entities[eid] = e
        return self.entities[eid]

    def add_relation(self, head, rel, tail, weight=1.0, cid=None):
        r = Relation(head, rel, tail, weight, cid)
        self.relations.append(r)
        if head not in self._adj:
            self._adj[head] = []
        self._adj[head].append((rel, tail))
        if tail not in self._adj:
            self._adj[tail] = []
        self._adj[tail].append((rel, head))
        return r

    def neighbors(self, eid):
        return self._adj.get(eid, [])

    def n_entities(self):
        return len(self.entities)

    def n_relations(self):
        return len(self.relations)

    def random_walk(self, start, max_steps=10, cont_prob=0.5, seed=None):
        if seed is not None:
            random.seed(seed)
        path = [start]
        cur = start
        for _ in range(max_steps):
            if random.random() > cont_prob:
                break
            nbrs = self.neighbors(cur)
            if not nbrs:
                break
            _, nxt = random.choice(nbrs)
            path.append(nxt)
            cur = nxt
        return path

    def expand_beta(self, seeds, beta=500, cont_prob=0.5, seed=42):
        random.seed(seed)
        vis = set(seeds)
        q = list(seeds)
        while len(vis) < beta and q:
            cur = q.pop(0)
            for _, nbr in self.neighbors(cur):
                if nbr not in vis and random.random() < cont_prob:
                    vis.add(nbr)
                    q.append(nbr)
                    if len(vis) >= beta:
                        break
        return vis

    def subgraph(self, eids):
        sub = KnowledgeGraph(f"{self.name}_sub")
        for eid in eids:
            if eid in self.entities:
                e = self.entities[eid]
                sub.add_entity(e.eid, e.etype, e.cid, e.attrs)
        for r in self.relations:
            if r.head in eids and r.tail in eids:
                sub.add_relation(r.head, r.rel, r.tail, r.weight, r.cid)
        return sub

    def triples(self):
        return [r.triple() for r in self.relations]

    def save_triples(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            for h, r, t in self.triples():
                f.write(f"{h}\t{r}\t{t}\n")

    def save_json(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "name": self.name,
            "entities": [e.to_dict() for e in self.entities.values()],
            "relations": [r.to_dict() for r in self.relations],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_triples(cls, path, name="loaded_kg"):
        kg = cls(name)
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) == 3:
                    h, r, t = parts
                    kg.add_entity(h, "unknown")
                    kg.add_entity(t, "unknown")
                    kg.add_relation(h, r, t)
        return kg


def build_kg_from_clustering(dataset_name, clustering_dir, output_dir=None,
                             beta=500, cont_prob=0.5, seed=42):
    if output_dir is None:
        output_dir = os.path.join(KG_OUTPUT_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    triplet_csv = os.path.join(clustering_dir, "triplets_clustered.csv")
    if not os.path.exists(triplet_csv):
        triplet_csv = os.path.join(clustering_dir, "interactions_clustered.csv")
    df = pd.read_csv(triplet_csv)

    kg = KnowledgeGraph(f"{dataset_name}_global")
    for _, row in df.iterrows():
        did = str(row["drug_id"])
        tid = str(row["target_id"])
        cid = int(row.get("cluster_id", -1))
        kg.add_entity(did, "drug", cid)
        kg.add_entity(tid, "target", cid)
        kg.add_relation(did, "interacts_with", tid, 1.0, cid)

    global_dir = os.path.join(output_dir, "global")
    os.makedirs(global_dir, exist_ok=True)
    kg.save_triples(os.path.join(global_dir, f"{dataset_name}_global_triples.txt"))
    kg.save_json(os.path.join(global_dir, f"{dataset_name}_global_kg.json"))

    cids = df["cluster_id"].unique().tolist()
    clusters_dir = os.path.join(output_dir, "clusters")
    for cid in sorted(cids):
        c_df = df[df["cluster_id"] == cid]
        seeds = c_df["drug_id"].tolist() + c_df["target_id"].tolist()
        expanded = kg.expand_beta(seeds, beta, cont_prob, seed)
        sub = kg.subgraph(expanded)
        c_dir = os.path.join(clusters_dir, f"cluster_{cid}")
        os.makedirs(c_dir, exist_ok=True)
        sub.save_triples(os.path.join(c_dir, f"cluster_{cid}_expanded_triples.txt"))
        sub.save_json(os.path.join(c_dir, f"cluster_{cid}_kg.json"))

    summary = {
        "dataset": dataset_name,
        "n_entities": kg.n_entities(),
        "n_relations": kg.n_relations(),
        "n_clusters": len(cids),
        "beta": beta,
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "kg_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return kg


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--clustering_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--beta", type=int, default=500)
    parser.add_argument("--cont_prob", type=float, default=0.5)
    args = parser.parse_args()
    if args.clustering_dir is None:
        args.clustering_dir = os.path.join(_BASE_DIR, "clustering_output", args.dataset)
    build_kg_from_clustering(args.dataset, args.clustering_dir, args.output_dir, args.beta, args.cont_prob)
