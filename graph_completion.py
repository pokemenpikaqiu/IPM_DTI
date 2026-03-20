"""
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import random
from knowledge_graph import KnowledgeGraph

class RandomWalker:
    """
    
    def __init__(self, kg: KnowledgeGraph, walk_length: int = 10, 
                 num_walks: int = 10, p: float = 1.0, q: float = 1.0):
        """
        self.kg = kg
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        
    def simple_walk(self, start_node: int) -> List[int]:
        """
        walk = [start_node]
        current = start_node
        
        for _ in range(self.walk_length - 1):
            neighbors = self.kg.get_neighbors(current)
            if not neighbors:
                break
            current = random.choice(neighbors)
            walk.append(current)
        
        return walk
    
    def biased_walk(self, start_node: int) -> List[int]:

        current = start_node
        
        for _ in range(self.walk_length - 1):

            edges = []
            for neighbor in self.kg.get_neighbors(current):

                edge_data = self.kg.graph.get_edge_data(current, neighbor)
                if edge_data:
                    for key, data in edge_data.items():
                        if 'relation' in data:
                            edges.append((neighbor, data['relation']))
            
            if not edges:
                break
            

            if target_relation is not None:
                target_edges = [e for e in edges if e[1] == target_relation]
                if target_edges:
                    edges = target_edges
            

            next_node, relation = random.choice(edges)
            walk.append((next_node, relation))
            current = next_node
        
        return walk
    
    def generate_walks(self, nodes: List[int], method: str = 'simple') -> List[List[int]]:

    
    def __init__(self, kg: KnowledgeGraph, min_entities: int = 50, 
                 min_relations: int = 3, walk_length: int = 10, 
                 num_walks: int = 20):

        target_entity_count = target_entity_count or self.min_entities
        target_relation_count = target_relation_count or self.min_relations
        
        new_entities = set()
        new_relations = set()
        new_triplets = []
        

        start_nodes = list(subspace_entities)
        walks = self.walker.generate_walks(start_nodes, method='biased')
        

        for walk in walks:
            for i, node in enumerate(walk):
                if len(subspace_entities | new_entities) >= target_entity_count:
                    break
                
                if node not in subspace_entities and node not in new_entities:
                    new_entities.add(node)
            

            for i in range(len(walk) - 1):
                head = walk[i]
                tail = walk[i + 1]
                

                edge_data = self.kg.graph.get_edge_data(head, tail)
                if edge_data:
                    for key, data in edge_data.items():
                        if 'relation' in data:
                            relation = data['relation']
                            if relation not in subspace_relations and relation not in new_relations:
                                new_relations.add(relation)
                            

                            if (head, tail) not in [(t['head'], t['tail']) for t in new_triplets]:
                                new_triplets.append({
                                    'head': head,
                                    'relation': relation,
                                    'tail': tail
                                })
        

        completed_entities = subspace_entities | new_entities
        completed_relations = subspace_relations | new_relations
        
        return completed_entities, completed_relations, new_triplets
    
    def random_walk(self, start_node: int, walk_length: int = None, 
                   num_walks: int = None) -> List[List[int]]:
        triplets = []
        triplet_set = set()
        
        for walk in walks:
            for i in range(len(walk) - 1):
                head = walk[i]
                tail = walk[i + 1]
                

                edge_data = self.kg.graph.get_edge_data(head, tail)
                if edge_data:
                    for key, data in edge_data.items():
                        if 'relation' in data:
                            relation = data['relation']
                            triplet_key = (head, relation, tail)
                            
                            if triplet_key not in triplet_set:
                                triplet_set.add(triplet_key)
                                triplets.append((head, relation, tail))
        
        return triplets
    
    def complete_all_subspaces(self, subspaces: Dict[int, List[Dict]], 
                              subspace_entities: Dict[int, Set[int]],
                              subspace_relations: Dict[int, Set[int]]) -> Dict:

        internal_edges = 0
        external_edges = 0
        
        for entity in subspace_entities:
            neighbors = self.kg.get_neighbors(entity)
            for neighbor in neighbors:
                if neighbor in subspace_entities:
                    internal_edges += 1
                else:
                    external_edges += 1
        
        return {

            'external_edges': external_edges,
            'connectivity_ratio': internal_edges / (internal_edges + external_edges) if (internal_edges + external_edges) > 0 else 0
        }

class SubspaceCompleter(GraphCompleter):
    """
    pass

class GraphAugmentor:
    """
    
    def __init__(self, kg: KnowledgeGraph):
        """
        self.kg = kg
        
    def augment_with_inverse_relations(self, triplets: List[Dict]) -> List[Dict]:
        """
        augmented = list(triplets)
        n_relations = len(self.kg.relation_to_id)
        
        for triplet in triplets:

            inverse_triplet = {
                'head': triplet['tail'],

                'tail': triplet['head']
            }
            augmented.append(inverse_triplet)
        
        return augmented
    
    def augment_with_self_loops(self, entities: Set[int], relation_id: int) -> List[Dict]:
        augmented = list(triplets)
        added_triplets = set()
        
        for triplet in triplets:
            head = triplet['head']
            tail = triplet['tail']
            

            head_neighbors = self.kg.get_neighbors(head)[:max_neighbors]
            for neighbor in head_neighbors:
                if neighbor != tail:
                    key = (head, neighbor)
                    if key not in added_triplets:
                        added_triplets.add(key)
                        augmented.append({
                            'head': head,
                            'relation': triplet['relation'],
                            'tail': neighbor
                        })
            

            tail_neighbors = self.kg.get_neighbors(tail)[:max_neighbors]
            for neighbor in tail_neighbors:
                if neighbor != head:
                    key = (tail, neighbor)
                    if key not in added_triplets:
                        added_triplets.add(key)
                        augmented.append({
                            'head': tail,
                            'relation': triplet['relation'],
                            'tail': neighbor
                        })
        
        return augmented

class WalkSequenceProcessor:
        triplets = []
        triplet_set = set()
        
        for walk in walks:
            for i in range(len(walk) - 1):
                head = walk[i]
                tail = walk[i + 1]
                

                edge_data = kg.graph.get_edge_data(head, tail)
                if edge_data:
                    for key, data in edge_data.items():
                        if 'relation' in data:
                            relation = data['relation']
                            triplet_key = (head, relation, tail)
                            
                            if triplet_key not in triplet_set:
                                triplet_set.add(triplet_key)
                                triplets.append({
                                    'head': head,
                                    'relation': relation,
                                    'tail': tail
                                })
        
        return triplets
    
    @staticmethod
    def extract_entities_from_walks(walks: List[List[int]]) -> Set[int]:
        relations = set()
        
        for walk in walks:
            for i in range(len(walk) - 1):
                head = walk[i]
                tail = walk[i + 1]
                
                edge_data = kg.graph.get_edge_data(head, tail)
                if edge_data:
                    for key, data in edge_data.items():
                        if 'relation' in data:
                            relations.add(data['relation'])
        
        return relations