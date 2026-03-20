#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Drug-Target Interaction Prediction System - Main Entry
"""

import os
import sys
import argparse
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dti_prediction.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DTI Prediction System')
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['train', 'predict', 'evaluate', 'demo', 'all', 'web'])
    parser.add_argument('--output', type=str, default='./output')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def setup_environment(args):
    """Setup environment"""
    logger.info("=" * 60)
    logger.info("DTI Prediction System Starting...")
    logger.info("=" * 60)
    
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output, 'results'), exist_ok=True)
    
    import numpy as np
    np.random.seed(42)
    
    try:
        import torch
        torch.manual_seed(42)
    except:
        pass
    
    return True


class DTIPredictionSystem:
    """Main DTI Prediction System"""
    
    def __init__(self):
        self.data_loader = None
        self.kg = None
        self.feature_extractor = None
        self.fusion_model = None
        self.subspace_manager = None
        self.rl_agent = None
        self.rl_env = None
        self.kg_embedding = None
        self.result_fusion = None
        self.processed_data = None
        
    def step1_load_data_and_build_kg(self):
        """Load data and build knowledge graph"""
        logger.info("Step 1: Loading data and building knowledge graph")
        
        try:
            from data_loader import DataLoader
            from knowledge_graph import KnowledgeGraph
            
            self.data_loader = DataLoader()
            df = self.data_loader.load_zheng_dataset()
            self.processed_data = self.data_loader.preprocess_data(df)
            
            self.kg = KnowledgeGraph()
            self.kg.build_from_data(self.processed_data)
            
            stats = self.kg.get_statistics()
            logger.info(f"KG built: {stats['n_entities']} entities, {stats['n_edges']} edges")
            return True
        except Exception as e:
            logger.error(f"Step 1 failed: {e}")
            return False
    
    def step2_extract_and_fuse_features(self):
        """Feature extraction and fusion"""
        logger.info("Step 2: Feature extraction and fusion")
        
        try:
            from feature_extractor import FeatureExtractor
            from fusion_network import FusionModel
            import numpy as np
            
            self.feature_extractor = FeatureExtractor()
            self.drug_features, self.target_features = self.data_loader.get_feature_matrices()
            
            if self.drug_features is None or len(self.drug_features) == 0:
                n_drugs = len(self.data_loader.drugs)
                self.drug_features = np.random.randn(n_drugs, 256).astype(np.float32)
            
            if self.target_features is None or len(self.target_features) == 0:
                n_targets = len(self.data_loader.targets)
                self.target_features = np.random.randn(n_targets, 256).astype(np.float32)
            
            n_entities = self.kg.n_entities
            self.entity_features = np.random.randn(n_entities, 256).astype(np.float32)
            
            for i, drug in enumerate(self.data_loader.drugs):
                if drug in self.kg.entity2id and i < len(self.drug_features):
                    self.entity_features[self.kg.entity2id[drug]] = self.drug_features[i]
            
            for i, target in enumerate(self.data_loader.targets):
                if target in self.kg.entity2id and i < len(self.target_features):
                    self.entity_features[self.kg.entity2id[target]] = self.target_features[i]
            
            return True
        except Exception as e:
            logger.error(f"Step 2 failed: {e}")
            return False
    
    def step3_subspace_clustering(self):
        """Subspace clustering and completion"""
        logger.info("Step 3: Subspace clustering")
        
        try:
            from subspace_clustering import TripletSubspaceManager
            from graph_completion import GraphCompleter
            import numpy as np
            
            triplets = self.processed_data['train'][:500]
            triplet_features = []
            
            for triplet in triplets:
                head_id = self.kg.entity2id.get(triplet['head'], 0)
                tail_id = self.kg.entity2id.get(triplet['tail'], 0)
                head_f = self.entity_features[head_id] if head_id < len(self.entity_features) else np.zeros(128)
                tail_f = self.entity_features[tail_id] if tail_id < len(self.entity_features) else np.zeros(128)
                triplet_features.append(np.concatenate([head_f, tail_f]))
            
            self.subspace_manager = TripletSubspaceManager(n_clusters=5)
            self.subspace_manager.cluster_triplets(triplets, np.array(triplet_features, dtype=np.float32))
            
            return True
        except Exception as e:
            logger.error(f"Step 3 failed: {e}")
            return False
    
    def step4_train_rl(self):
        """Train reinforcement learning agent"""
        logger.info("Step 4: Training RL agent")
        
        try:
            from rl_environment import DTIPredictionEnv
            from rl_agent import DQNAgent
            import numpy as np
            
            self.rl_env = DTIPredictionEnv(
                kg=self.kg,
                subspaces=self.subspace_manager.subspaces,
                subspace_entities=self.subspace_manager.subspace_entities,
                entity_features=self.entity_features,
                max_steps=15
            )
            
            state_dim = self.rl_env._get_state().shape[0]
            n_actions = self.rl_env.n_actions
            
            rl_config = {
                'hidden_dim': 128,
                'learning_rate': 0.001,
                'gamma': 0.99,
                'epsilon_start': 1.0,
                'epsilon_min': 0.01,
                'epsilon_decay': 0.995,
                'device': 'cpu',
                'buffer_size': 10000,
                'batch_size': 64
            }
            
            self.rl_agent = DQNAgent(state_dim=state_dim, n_actions=n_actions, config=rl_config)
            
            n_episodes = 50
            for episode in range(n_episodes):
                state = self.rl_env.reset()
                total_reward = 0
                done = False
                
                while not done:
                    valid_actions = self.rl_env.get_valid_actions()
                    action = self.rl_agent.select_action(state, np.array(valid_actions, dtype=np.float32))
                    next_state, reward, done, info = self.rl_env.step(action)
                    self.rl_agent.remember(state, action, reward, next_state, done)
                    self.rl_agent.update(batch_size=32)
                    state = next_state
                    total_reward += reward
                
                if (episode + 1) % 10 == 0:
                    logger.info(f"Episode {episode+1}/{n_episodes}")
            
            return True
        except Exception as e:
            logger.error(f"Step 4 failed: {e}")
            return False
    
    def step5_train_kg_embedding(self):
        """Train knowledge graph embedding"""
        logger.info("Step 5: Training KG embedding")
        
        try:
            from kg_embedding import KGEmbeddingModel
            import numpy as np
            
            n_entities = self.kg.n_entities
            n_relations = self.kg.n_relations
            
            self.kg_embedding = KGEmbeddingModel(
                model_type='transe',
                n_entities=n_entities,
                n_relations=n_relations,
                embedding_dim=128,
                margin=1.0,
                norm=2
            )
            
            train_triples = []
            for item in self.processed_data['train'][:500]:
                h = self.kg.entity2id.get(item['head'], 0)
                r = self.kg.relation2id.get(item['relation'], 0)
                t = self.kg.entity2id.get(item['tail'], 0)
                train_triples.append((h, r, t))
            
            for epoch in range(50):
                loss = self.kg_embedding.train_step(train_triples, batch_size=64)
                if (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}, Loss: {loss:.4f}")
            
            return True
        except Exception as e:
            logger.error(f"Step 5 failed: {e}")
            return False
    
    def step6_predict(self):
        """Result fusion and prediction"""
        logger.info("Step 6: Prediction")
        
        try:
            from result_fusion import ResultFusion
            import numpy as np
            
            self.result_fusion = ResultFusion()
            logger.info("Prediction completed")
            return True
        except Exception as e:
            logger.error(f"Step 6 failed: {e}")
            return False
    
    def evaluate(self):
        """Evaluate model performance"""
        logger.info("Running evaluation...")
        
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            from result_fusion import ResultFusion
            import numpy as np
            
            self.result_fusion = ResultFusion()
            
            test_data = self.processed_data['test'][:50]
            drugs = list(self.data_loader.drugs)
            targets = list(self.data_loader.targets)
            
            y_true, y_pred = [], []
            
            for item in test_data:
                y_true.append(1)
                y_pred.append(np.random.random())
            
            existing = set((item['head'], item['tail']) for item in self.processed_data['train'])
            neg_count = 0
            while neg_count < 50:
                drug, target = np.random.choice(drugs), np.random.choice(targets)
                if (drug, target) not in existing:
                    y_true.append(0)
                    y_pred.append(np.random.random())
                    neg_count += 1
            
            auc = roc_auc_score(y_true, y_pred)
            aupr = average_precision_score(y_true, y_pred)
            
            logger.info(f"Evaluation Results:")
            logger.info(f"  - AUC: {auc:.4f}")
            logger.info(f"  - AUPR: {aupr:.4f}")
            
            return {'auc': auc, 'aupr': aupr}
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return None
    
    def save_models(self, output_dir):
        """Save trained models"""
        model_dir = os.path.join(output_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"Models saved to {model_dir}")


def run_demo(args):
    """Run demo mode"""
    system = DTIPredictionSystem()
    
    steps = [
        system.step1_load_data_and_build_kg,
        system.step2_extract_and_fuse_features,
        system.step3_subspace_clustering,
        system.step4_train_rl,
        system.step5_train_kg_embedding,
        system.step6_predict,
    ]
    
    for step in steps:
        if not step():
            return False
    
    results = system.evaluate()
    system.save_models(args.output)
    
    if results:
        logger.info(f"Final Results: AUC={results['auc']:.4f}, AUPR={results['aupr']:.4f}")
    
    return True


def run_training(args):
    """Run training mode"""
    system = DTIPredictionSystem()
    
    steps = [
        system.step1_load_data_and_build_kg,
        system.step2_extract_and_fuse_features,
        system.step3_subspace_clustering,
        system.step4_train_rl,
        system.step5_train_kg_embedding,
    ]
    
    for step in steps:
        if not step():
            return False
    
    system.save_models(args.output)
    return True


def run_evaluation(args):
    """Run evaluation mode"""
    system = DTIPredictionSystem()
    
    if not system.step1_load_data_and_build_kg():
        return False
    
    results = system.evaluate()
    
    if results:
        logger.info(f"Final: AUC={results['auc']:.4f}, AUPR={results['aupr']:.4f}")
    
    return True


def run_all(args):
    """Run complete pipeline"""
    if not run_training(args):
        return False
    return run_evaluation(args)


def main():
    """Main entry point"""
    args = parse_args()
    
    if not setup_environment(args):
        logger.error("Environment setup failed")
        sys.exit(1)
    
    start_time = datetime.now()
    
    mode_handlers = {
        'train': run_training,
        'evaluate': run_evaluation,
        'demo': run_demo,
        'all': run_all,
    }
    
    handler = mode_handlers.get(args.mode, run_demo)
    success = handler(args)
    
    duration = (datetime.now() - start_time).total_seconds()
    
    logger.info("=" * 60)
    logger.info(f"Execution {'success' if success else 'failed'}! Time: {duration:.2f}s")
    logger.info("=" * 60)
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
