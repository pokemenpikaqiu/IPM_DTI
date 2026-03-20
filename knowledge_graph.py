"""
Knowledge Graph construction and management module
"""
import networkx as nx
import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import defaultdict
import random
from config import KG_CONFIG


class KnowledgeGraph:
    """Knowledge Graph for DTI"""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}

        self.n_entities = 0
        self.n_relations = 0
        self.n_edges = 0

        self.adjacency = defaultdict(list)
        self.reverse_adjacency = defaultdict(list)

        self.entity_types = defaultdict(str)

    def build_from_data(self, data: Dict):
        """Build knowledge graph from preprocessed data"""
        print("Building knowledge graph...")

        self.entity2id = data['entity2id']
        self.id2entity = data['id2entity']
        self.relation2id = data['relation2id']
        self.id2relation = data['id2relation']
        self.n_entities = data['n_entities']
        self.n_relations = data['n_relations']

        drugs = data.get('drugs', [])
        targets = data.get('targets', [])

        drug_set = set(drugs) if drugs else set()
        target_set = set(targets) if targets else set()

        for entity_id, entity_name in self.id2entity.items():
            if entity_name in drug_set:
                entity_type = 'drug'
            elif entity_name in target_set:
                entity_type = 'target'
            elif entity_name.startswith('DB') or entity_name.startswith('Drug'):
                entity_type = 'drug'
            elif entity_name.startswith('P') or entity_name.startswith('Q') or entity_name.startswith('Target'):
                entity_type = 'target'
            else:
                entity_type = 'drug'
            self.entity_types[entity_id] = entity_type
            self.graph.add_node(entity_id, name=entity_name, type=entity_type)

        all_data = data['train'] + data['val'] + data['test']

        for triple in all_data:
            head = self.entity2id[triple['head']]
            relation = self.relation2id[triple['relation']]
            tail = self.entity2id[triple['tail']]
            label = triple['label']

            self.graph.add_edge(head, tail, relation=relation, label=label)
            self.adjacency[head].append((relation, tail))
            self.reverse_adjacency[tail].append((relation, head))

            self.n_edges += 1

        print(f"Knowledge graph built: {self.n_entities} entities, {self.n_relations} relations, {self.n_edges} edges")

    def add_entity(self, entity_name: str, entity_type: str = 'unknown') -> int:
        """Add entity to graph"""
        if entity_name not in self.entity2id:
            entity_id = self.n_entities
            self.entity2id[entity_name] = entity_id
            self.id2entity[entity_id] = entity_name
            self.entity_types[entity_id] = entity_type
            self.graph.add_node(entity_id, name=entity_name, type=entity_type)
            self.n_entities += 1
        return self.entity2id[entity_name]

    def add_relation(self, relation_name: str) -> int:
        """Add relation to graph"""
        if relation_name not in self.relation2id:
            relation_id = self.n_relations
            self.relation2id[relation_name] = relation_id
            self.id2relation[relation_id] = relation_name
            self.n_relations += 1
        return self.relation2id[relation_name]

    def add_triple(self, head: str, relation: str, tail: str, label: int = 1):
        """Add triple to graph"""
        h_id = self.add_entity(head)
        r_id = self.add_relation(relation)
        t_id = self.add_entity(tail)

        self.graph.add_edge(h_id, t_id, relation=r_id, label=label)
        self.adjacency[h_id].append((r_id, t_id))
        self.reverse_adjacency[t_id].append((r_id, h_id))
        self.n_edges += 1

    def get_neighbors(self, entity_id: int, direction: str = 'out') -> List[Tuple[int, int]]:
        """Get neighbors of entity"""
        neighbors = []
        if direction in ['out', 'both']:
            neighbors.extend(self.adjacency[entity_id])
        if direction in ['in', 'both']:
            neighbors.extend(self.reverse_adjacency[entity_id])
        return neighbors

    def get_entity_info(self, entity_id: int) -> Dict:
        """Get entity information"""
        if entity_id in self.id2entity:
            return {
                'id': entity_id,
                'name': self.id2entity[entity_id],
                'type': self.entity_types[entity_id],
                'degree': self.graph.degree(entity_id),
                'in_degree': self.graph.in_degree(entity_id),
                'out_degree': self.graph.out_degree(entity_id)
            }
        return {}

    def random_walk(self, start_node: int, walk_length: int = None,
                    num_walks: int = None) -> List[List[int]]:
        """Random walk from start node"""
        if walk_length is None:
            walk_length = KG_CONFIG['walk_length']
        if num_walks is None:
            num_walks = KG_CONFIG['num_walks']

        walks = []

        for _ in range(num_walks):
            walk = [start_node]
            current = start_node

            for _ in range(walk_length - 1):
                neighbors = self.get_neighbors(current, 'out')

                if not neighbors:
                    break

                _, next_node = random.choice(neighbors)
                walk.append(next_node)
                current = next_node

            walks.append(walk)

        return walks

    def random_walk_all(self, walk_length: int = None,
                        num_walks: int = None) -> List[List[int]]:
        """Random walk from all nodes"""
        if walk_length is None:
            walk_length = KG_CONFIG['walk_length']
        if num_walks is None:
            num_walks = KG_CONFIG['num_walks']

        all_walks = []
        nodes = list(self.graph.nodes())

        print(f"Starting random walk: {len(nodes)} nodes, {num_walks} walks each, length {walk_length}")

        for node in nodes:
            walks = self.random_walk(node, walk_length, num_walks)
            all_walks.extend(walks)

        print(f"Random walk complete: {len(all_walks)} sequences generated")
        return all_walks

    def get_triples(self) -> List[Tuple[int, int, int]]:
        """Get all triples"""
        triples = []
        for u, v, data in self.graph.edges(data=True):
            triples.append((u, data['relation'], v))
        return triples

    def get_positive_triples(self) -> List[Tuple[int, int, int]]:
        """Get positive triples"""
        triples = []
        for u, v, data in self.graph.edges(data=True):
            if data.get('label', 0) == 1:
                triples.append((u, data['relation'], v))
        return triples

    def get_subgraph(self, entity_ids: List[int]) -> 'KnowledgeGraph':
        """Extract subgraph"""
        subgraph = KnowledgeGraph()

        for entity_id in entity_ids:
            if entity_id in self.id2entity:
                entity_name = self.id2entity[entity_id]
                entity_type = self.entity_types[entity_id]
                subgraph.add_entity(entity_name, entity_type)

        for u, v, data in self.graph.edges(data=True):
            if u in entity_ids and v in entity_ids:
                relation_name = self.id2relation[data['relation']]
                head_name = self.id2entity[u]
                tail_name = self.id2entity[v]
                subgraph.add_triple(head_name, relation_name, tail_name, data.get('label', 1))

        return subgraph

    def get_statistics(self) -> Dict:
        """Get graph statistics"""
        degrees = [d for n, d in self.graph.degree()]

        stats = {
            'n_entities': self.n_entities,
            'n_relations': self.n_relations,
            'n_edges': self.n_edges,
            'n_drugs': sum(1 for t in self.entity_types.values() if t == 'drug'),
            'n_targets': sum(1 for t in self.entity_types.values() if t == 'target'),
            'avg_degree': np.mean(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0,
            'density': nx.density(self.graph) if self.n_entities > 0 else 0
        }

        return stats

    def find_paths(self, source: int, target: int, max_length: int = 4) -> List[List[int]]:
        """Find paths between two entities"""
        try:
            paths = []
            for path in nx.all_simple_paths(self.graph, source, target, cutoff=max_length):
                paths.append(path)
            return paths[:100]
        except:
            return []

    def get_common_neighbors(self, entity1: int, entity2: int) -> Set[int]:
        """Get common neighbors of two entities"""
        neighbors1 = set(n for _, n in self.adjacency[entity1])
        neighbors2 = set(n for _, n in self.adjacency[entity2])
        return neighbors1 & neighbors2

    def calculate_similarity(self, entity1: int, entity2: int) -> float:
        """Calculate structural similarity between entities"""
        neighbors1 = set(n for _, n in self.adjacency[entity1])
        neighbors2 = set(n for _, n in self.adjacency[entity2])

        if not neighbors1 and not neighbors2:
            return 0.0

        intersection = len(neighbors1 & neighbors2)
        union = len(neighbors1 | neighbors2)

        return intersection / union if union > 0 else 0.0


if __name__ == "__main__":
    from data_loader import DataLoader

    loader = DataLoader()
    loader.generate_sample_data(n_drugs=100, n_targets=80, n_interactions=400)
    df = pd.DataFrame(loader.interactions)
    data = loader.preprocess_data(df)

    kg = KnowledgeGraph()
    kg.build_from_data(data)

    stats = kg.get_statistics()
    print("\nKnowledge Graph Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    walks = kg.random_walk(0, walk_length=10, num_walks=5)
    print(f"\nRandom walk example (from node 0):")
    for i, walk in enumerate(walks[:3]):
        print(f"  Walk {i+1}: {walk}")
