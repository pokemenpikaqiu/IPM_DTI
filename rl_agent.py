"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
from config import RL_CONFIG

class ReplayBuffer:
    """
    
    def __init__(self, capacity: int = 10000):
        """
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)

class DQNetwork(nn.Module):
    """
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256):
        """
        super(DQNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, n_actions)
        
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class DuelingDQNetwork(nn.Module):
    """
    
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 256):
        """
        super(DuelingDQNetwork, self).__init__()
        

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )
        

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_actions)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        shared_features = self.shared(x)
        value = self.value_stream(shared_features)
        advantage = self.advantage_stream(shared_features)
        

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class DQNAgent:
    """
    
    def __init__(self, state_dim: int, n_actions: int, config: Dict = None):

        if training and random.random() < self.epsilon:
            if action_mask is not None:
                valid_actions = np.where(action_mask > 0)[0]
                return random.choice(valid_actions)
            else:
                return random.randint(0, self.n_actions - 1)
        

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).squeeze(0)
            

            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).to(self.device)
                q_values = q_values * mask_tensor - (1 - mask_tensor) * 1e9
            
            return q_values.argmax().item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """
        self.memory.push(state, action, reward, next_state, done)
        self.steps_done += 1
    
    def remember(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """
        self.store_transition(state, action, reward, next_state, done)
        
    def update(self, batch_size: int = None) -> Optional[float]:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes_done += 1
        
    def save(self, path: str):
      
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint['episodes_done']

class PolicyGradientAgent:
  
        self.config = config or RL_CONFIG
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.device = torch.device(self.config['device'])
        

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, self.config['hidden_dim']),
            nn.ReLU(),
            nn.BatchNorm1d(self.config['hidden_dim']),
            nn.Dropout(0.2),
            nn.Linear(self.config['hidden_dim'], self.config['hidden_dim']),
            nn.ReLU(),
            nn.BatchNorm1d(self.config['hidden_dim']),
            nn.Dropout(0.2),
            nn.Linear(self.config['hidden_dim'], n_actions)
        ).to(self.device)
        

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        

        self.saved_log_probs = []
        self.rewards = []
        
    def select_action(self, state: np.ndarray, action_mask: Optional[np.ndarray] = None) -> Tuple[int, torch.Tensor]:
        if not self.rewards:
            return 0.0
        

        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + self.config['gamma'] * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        

        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        

        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        

        del self.saved_log_probs[:]
        del self.rewards[:]
        
        return policy_loss.item()
    
    def save(self, path: str):
    
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

class MultiAgentPredictor:

        self.agents = agents
        self.n_subspaces = n_subspaces
        
    def predict(self, states: List[np.ndarray], action_masks: List[np.ndarray],
               training: bool = False) -> List[int]:

        aggregated = {
            'query_entity': predictions[0]['query_entity'] if predictions else None,
            'query_type': predictions[0]['query_type'] if predictions else None,
            'visited_subspaces': set(),
            'candidate_interactions': [],
            'prediction_scores': {},
            'rankings': []
        }
        

        for pred in predictions:
            aggregated['visited_subspaces'].update(pred.get('visited_subspaces', []))
            aggregated['candidate_interactions'].extend(pred.get('candidate_interactions', []))
            
            for entity, scores in pred.get('prediction_scores', {}).items():
                if entity not in aggregated['prediction_scores']:
                    aggregated['prediction_scores'][entity] = []
                aggregated['prediction_scores'][entity].extend(scores if isinstance(scores, list) else [scores])
        
        aggregated['visited_subspaces'] = list(aggregated['visited_subspaces'])
        

        final_scores = {}
        for entity, scores in aggregated['prediction_scores'].items():
            count = len(scores)
            avg_score = np.mean(scores)
            final_score = count * avg_score
            final_scores[entity] = {
                'count': count,
                'avg_score': avg_score,
                'final_score': final_score
            }
        

        rankings = sorted(
            final_scores.items(),
            key=lambda x: x[1]['final_score'],
            reverse=True
        )
        aggregated['rankings'] = [
            {'entity': entity, 'rank': rank + 1, **scores}
            for rank, (entity, scores) in enumerate(rankings)
        ]
        
        return aggregated

if __name__ == "__main__":

    print("=" * 50)
    print("测试强化学习智能体")
    print("=" * 50)
    

    from rl_environment import DTIPredictionEnv
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
    

    env = DTIPredictionEnv(
        kg=kg,
        subspaces=subspaces,
        subspace_entities=manager.subspace_entities,
        subspace_relations=manager.subspace_relations,
        entity_features=entity_features,
        max_steps=15
    )
    

    print("\n1. 测试DQN智能体")
    state = env.reset()
    state_dim = state.shape[0]
    n_actions = env.n_actions
    
    agent = DQNAgent(state_dim, n_actions)
    print(f"状态维度: {state_dim}")
    print(f"动作数量: {n_actions}")
    

    action_mask = env.get_action_mask()
    action = agent.select_action(state, action_mask, training=True)
    print(f"选择动作: {action}")
    

    for i in range(10):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action_mask = env.get_action_mask()
            action = agent.select_action(state, action_mask, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.update()
            
            total_reward += reward
            state = next_state
        
        agent.decay_epsilon()
        print(f"Episode {i+1}: Total Reward = {total_reward:.2f}, Epsilon = {agent.epsilon:.4f}")
    

    print("\n2. 测试策略梯度智能体")
    pg_agent = PolicyGradientAgent(state_dim, n_actions)
    
    for i in range(5):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action_mask = env.get_action_mask()
            action, log_prob = pg_agent.select_action(state, action_mask)
            next_state, reward, done, info = env.step(action)
            
            pg_agent.rewards.append(reward)
            total_reward += reward
            state = next_state
        
        loss = pg_agent.update()
        print(f"Episode {i+1}: Total Reward = {total_reward:.2f}, Loss = {loss:.4f}")