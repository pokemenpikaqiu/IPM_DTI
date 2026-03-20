"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import random
from knowledge_graph import KnowledgeGraph

class DTIPredictionEnv:
    """
    
    def __init__(self, kg: KnowledgeGraph, subspaces: Dict[int, List[Dict]],
                 subspace_entities: Dict[int, Set[int]], 
                 subspace_relations: Dict[int, Set[int]],
                 entity_features: np.ndarray,
                 relation_features: Optional[np.ndarray] = None,
                 max_steps: int = 20):

        self.current_step = 0
        self.visited_subspaces = set()
        self.candidate_interactions = []
        self.prediction_scores = defaultdict(list)
        

        if query_entity is not None:
            self.query_entity = query_entity
            self.query_type = query_type or 'drug'
        else:

            self.query_entity = random.randint(0, self.n_entities - 1)
            self.query_type = random.choice(['drug', 'target'])
        

        if isinstance(self.query_entity, str):
            query_entity_id = self.kg.entity2id.get(self.query_entity, 0)
        else:
            query_entity_id = self.query_entity
        

        entity_subspaces = []
        for sid, entities in self.subspace_entities.items():

            if self.query_entity in entities or query_entity_id in entities:
                entity_subspaces.append(sid)
        

        if entity_subspaces:
            self.current_subspace = random.choice(entity_subspaces)
        else:
            self.current_subspace = random.randint(0, self.n_subspaces - 1)
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        self.current_step += 1
        reward = 0.0
        done = False
        info = {'action_type': '', 'success': False}
        

        if action < self.n_subspaces:

            self._jump_to_subspace(action)
            info['action_type'] = 'jump'
            info['success'] = True

            
        elif action == self.n_subspaces:

            query_result = self._query_current_subspace()
            info['action_type'] = 'query'
            info['success'] = query_result > 0

            
        elif action == self.n_subspaces + 1:

            prediction_result = self._make_prediction()
            info['action_type'] = 'predict'
            info['success'] = prediction_result > 0

            done = True
        

        if self.current_step >= self.max_steps:
            done = True
        

        next_state = self._get_state()
        
        return next_state, reward, done, info
    
    def _jump_to_subspace(self, target_subspace: int):

        triplets = self.subspaces.get(self.current_subspace, [])
        
        query_result = 0.0
        
        for triplet in triplets:
            head, tail = triplet['head'], triplet['tail']
            

            if self.query_type == 'drug':

                if head == self.query_entity:
                    self.candidate_interactions.append((head, tail))
                    self.prediction_scores[tail].append(1.0)
                    query_result += 1.0
            else:

                if tail == self.query_entity:
                    self.candidate_interactions.append((head, tail))
                    self.prediction_scores[head].append(1.0)
                    query_result += 1.0
        
        return query_result
    
    def _make_prediction(self) -> float:
        mask = np.ones(self.n_actions, dtype=np.float32)
        

        for sid in self.visited_subspaces:
            if sid < self.n_subspaces:
                mask[sid] = 0.0
        
        return mask
    
    def get_valid_actions(self) -> List[int]:

        info = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Query Entity: {self.query_entity} ({self.query_type})",
            f"Current Subspace: {self.current_subspace}",
            f"Visited Subspaces: {sorted(self.visited_subspaces)}",
            f"Candidate Interactions: {len(self.candidate_interactions)}",
            f"Prediction Scores: {len(self.prediction_scores)} entities"
        ]
        return "\n".join(info)
    
    def get_prediction_results(self) -> Dict:
      
    
    def __init__(self, kg: KnowledgeGraph, subspaces: Dict[int, List[Dict]],
                 subspace_entities: Dict[int, Set[int]],
                 entity_features: np.ndarray,
                 transition_matrix: Optional[np.ndarray] = None):
      
        transition_matrix = np.zeros((self.n_subspaces, self.n_subspaces))
        
        for i in range(self.n_subspaces):
            entities_i = self.subspace_entities.get(i, set())
            
            for j in range(self.n_subspaces):
                if i == j:
                    continue
                
                entities_j = self.subspace_entities.get(j, set())
                

                overlap = len(entities_i & entities_j)
                transition_matrix[i, j] = overlap
        

        row_sums = transition_matrix.sum(axis=1, keepdims=True)

        transition_matrix = transition_matrix / row_sums
        
        return transition_matrix
    
    def reset(self, start_subspace: Optional[int] = None,
              target_entity: Optional[int] = None) -> Tuple[int, np.ndarray]:
      

        current_entities = list(self.subspace_entities.get(self.current_subspace, set()))
        if current_entities:
            current_feature = np.mean(self.entity_features[current_entities], axis=0)
        else:
            current_feature = np.zeros(self.entity_features.shape[1])
        

        if self.target_entity is not None and self.target_entity < self.n_entities:
            target_feature = self.entity_features[self.target_entity]
        else:
            target_feature = np.zeros(self.entity_features.shape[1])
        

        visited_features = np.zeros(self.n_subspaces)
        for sid in self.visited_subspaces:
            if sid < self.n_subspaces:
                visited_features[sid] = 1.0
        

        state = np.concatenate([
            current_feature,
            target_feature,
            visited_features,
            [self.current_subspace / self.n_subspaces]
        ])
        
        return state.astype(np.float32)
    
    def step(self, action: int) -> Tuple[int, np.ndarray, float, bool]:
      
        transitions = self.transition_matrix[subspace_id]
        neighbors = np.where(transitions > 0)[0].tolist()
        return [n for n in neighbors if n != subspace_id]

if __name__ == "__main__":

    print("=" * 50)
    print("测试强化学习环境")
    print("=" * 50)
    

    from data_loader import DataLoader
    from knowledge_graph import KnowledgeGraph
    from subspace_clustering import TripletSubspaceManager
    

    loader = DataLoader()
    loader.generate_synthetic_data(n_drugs=50, n_targets=30, n_interactions=200)
    

    kg = KnowledgeGraph()
    for drug_id in loader.drugs:
        kg.add_entity(drug_id, 'drug', {'name': f'Drug_{drug_id}'})
    for target_id in loader.targets:
        kg.add_entity(target_id, 'target', {'name': f'Target_{target_id}'})
    for drug_id, target_id in loader.interactions:
        kg.add_interaction(drug_id, target_id, relation_type='interacts')
    

    triplets = []
    for drug_id, target_id in loader.interactions:
        triplets.append({
            'head': drug_id,
            'relation': 0,
            'tail': target_id
        })
    

    n_entities = kg.get_node_count()
    feature_dim = 128
    entity_features = np.random.randn(n_entities, feature_dim)
    

    manager = TripletSubspaceManager(n_clusters=5)
    subspaces = manager.cluster_triplets(triplets, entity_features[:len(triplets)])
    

    print("\n1. 测试DTI预测环境")
    env = DTIPredictionEnv(
        kg=kg,
        subspaces=subspaces,
        subspace_entities=manager.subspace_entities,
        subspace_relations=manager.subspace_relations,
        entity_features=entity_features,
        max_steps=15
    )
    

    state = env.reset(query_entity=0, query_type='drug')
    print(f"初始状态维度: {state.shape}")
    print(env.render())
    

    for i in range(5):
        valid_actions = env.get_valid_actions()
        action = random.choice(valid_actions)
        next_state, reward, done, info = env.step(action)
        print(f"\nStep {i+1}: Action={info['action_type']}, Reward={reward:.2f}, Done={done}")
        print(env.render())
        
        if done:
            break
    

    results = env.get_prediction_results()
    print(f"\n预测结果:")
    print(f"  查询实体: {results['query_entity']} ({results['query_type']})")
    print(f"  访问子空间: {results['visited_subspaces']}")
    print(f"  候选交互数: {len(results['candidate_interactions'])}")
    print(f"  排名前5: {results['rankings'][:5]}")
    

    print("\n2. 测试多子空间导航环境")
    nav_env = MultiSubspaceNavigationEnv(
        kg=kg,
        subspaces=subspaces,
        subspace_entities=manager.subspace_entities,
        entity_features=entity_features
    )
    

    current_subspace, state = nav_env.reset(target_entity=10)
    print(f"初始子空间: {current_subspace}")
    print(f"状态维度: {state.shape}")
    

    for i in range(5):
        neighbors = nav_env.get_neighbors(current_subspace)
        if neighbors:
            action = random.choice(neighbors)
        else:
            action = random.randint(0, nav_env.n_subspaces - 1)
        
        next_subspace, next_state, reward, done = nav_env.step(action)
        print(f"Step {i+1}: {current_subspace} -> {next_subspace}, Reward={reward:.2f}, Done={done}")
        current_subspace = next_subspace
        
        if done:
            print("找到目标!")
            break