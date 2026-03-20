"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import random
from config import EMBEDDING_CONFIG


class TransE(nn.Module):
    """
    
    def __init__(self, n_entities: int, n_relations: int, embedding_dim: int = 128,
                 margin: float = 1.0, norm: int = 2):
      
        h = self.entity_embeddings(heads)
        r = self.relation_embeddings(relations)
        t = self.entity_embeddings(tails)
        

        if self.norm == 1:
            score = torch.norm(h + r - t, p=1, dim=1)
        else:
            score = torch.norm(h + r - t, p=2, dim=1)
        
        return score
    
    def compute_loss(self, positive_triplets: Tuple, negative_triplets: Tuple) -> torch.Tensor:
        with torch.no_grad():
            self.entity_embeddings.weight.div_(
                self.entity_embeddings.weight.norm(p=2, dim=1, keepdim=True)
            )


class DistMult(nn.Module):
        super(DistMult, self).__init__()
        
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        

        self.entity_embeddings = nn.Embedding(n_entities, embedding_dim)

        self.relation_embeddings = nn.Embedding(n_relations, embedding_dim)
        

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
    def forward(self, heads: torch.Tensor, relations: torch.Tensor,
                tails: torch.Tensor) -> torch.Tensor:
        pos_heads, pos_relations, pos_tails = positive_triplets
        neg_heads, neg_relations, neg_tails = negative_triplets
        
        pos_score = self.forward(pos_heads, pos_relations, pos_tails)
        neg_score = self.forward(neg_heads, neg_relations, neg_tails)
        

        pos_loss = -F.logsigmoid(pos_score).mean()
        neg_loss = -F.logsigmoid(-neg_score).mean()
        
        return pos_loss + neg_loss


class ComplEx(nn.Module):
        super(ComplEx, self).__init__()
        
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        

        self.entity_embeddings_real = nn.Embedding(n_entities, embedding_dim)
        self.entity_embeddings_imag = nn.Embedding(n_entities, embedding_dim)
        

        self.relation_embeddings_real = nn.Embedding(n_relations, embedding_dim)
        self.relation_embeddings_imag = nn.Embedding(n_relations, embedding_dim)
        

        nn.init.xavier_uniform_(self.entity_embeddings_real.weight)
        nn.init.xavier_uniform_(self.entity_embeddings_imag.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_real.weight)
        nn.init.xavier_uniform_(self.relation_embeddings_imag.weight)
        
    def forward(self, heads: torch.Tensor, relations: torch.Tensor,
                tails: torch.Tensor) -> torch.Tensor:
        pos_heads, pos_relations, pos_tails = positive_triplets
        neg_heads, neg_relations, neg_tails = negative_triplets
        
        pos_score = self.forward(pos_heads, pos_relations, pos_tails)
        neg_score = self.forward(neg_heads, neg_relations, neg_tails)
        

        pos_loss = -F.logsigmoid(pos_score).mean()
        neg_loss = -F.logsigmoid(-neg_score).mean()
        
        return pos_loss + neg_loss


class RotatE(nn.Module):
        super(RotatE, self).__init__()
        
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.margin = margin
        self.epsilon = epsilon
        

        self.entity_embeddings = nn.Embedding(n_entities, embedding_dim)

        self.relation_embeddings = nn.Embedding(n_relations, embedding_dim)
        

        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.uniform_(self.relation_embeddings.weight, -np.pi, np.pi)
        
    def forward(self, heads: torch.Tensor, relations: torch.Tensor,
                tails: torch.Tensor) -> torch.Tensor:
       
        pos_heads, pos_relations, pos_tails = positive_triplets
        neg_heads, neg_relations, neg_tails = negative_triplets
        
        pos_score = self.forward(pos_heads, pos_relations, pos_tails)
        neg_score = self.forward(neg_heads, neg_relations, neg_tails)
        

        pos_loss = pos_score.mean()
        neg_loss = F.relu(self.margin - neg_score).mean()
        
        return pos_loss + neg_loss


class KGEmbeddingModel:
        self.model_type = model_type.lower()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        self.device = torch.device(device)
        

        if self.model_type == 'transe':
            self.model = TransE(n_entities, n_relations, embedding_dim, **kwargs)
        elif self.model_type == 'distmult':
            self.model = DistMult(n_entities, n_relations, embedding_dim)
        elif self.model_type == 'complex':
            self.model = ComplEx(n_entities, n_relations, embedding_dim)
        elif self.model_type == 'rotate':
            self.model = RotatE(n_entities, n_relations, embedding_dim, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def train_step(self, triples: List[Tuple], batch_size: int = 64) -> float:
        self.model.eval()
        
        with torch.no_grad():
            if self.model_type == 'complex':
                entity_real = self.model.entity_embeddings_real.weight.cpu().numpy()
                entity_imag = self.model.entity_embeddings_imag.weight.cpu().numpy()
                relation_real = self.model.relation_embeddings_real.weight.cpu().numpy()
                relation_imag = self.model.relation_embeddings_imag.weight.cpu().numpy()
                
                entity_embeddings = np.concatenate([entity_real, entity_imag], axis=1)
                relation_embeddings = np.concatenate([relation_real, relation_imag], axis=1)
            else:
                entity_embeddings = self.model.entity_embeddings.weight.cpu().numpy()
                relation_embeddings = self.model.relation_embeddings.weight.cpu().numpy()
        
        return entity_embeddings, relation_embeddings
    
    def predict_tail(self, heads: np.ndarray, relations: np.ndarray, top_k: int = 10) -> np.ndarray:
        self.model.eval()
        
        tails_tensor = torch.LongTensor(tails).to(self.device)
        relations_tensor = torch.LongTensor(relations).to(self.device)
        
        all_heads = torch.arange(self.n_entities).to(self.device)
        
        predictions = []
        
        with torch.no_grad():
            for t, r in zip(tails_tensor, relations_tensor):
                t_expanded = t.expand(self.n_entities)
                r_expanded = r.expand(self.n_entities)
                
                if self.model_type in ['transe', 'rotate']:

                    scores = self.model.forward(all_heads, r_expanded, t_expanded)
                    top_scores, top_indices = torch.topk(scores, k=top_k, largest=False)
                else:

                    scores = self.model.forward(all_heads, r_expanded, t_expanded)
                    top_scores, top_indices = torch.topk(scores, k=top_k, largest=True)
                
                predictions.append({
                    'indices': top_indices.cpu().numpy(),
                    'scores': top_scores.cpu().numpy()
                })
        
        return predictions
    
    def predict_relation(self, heads: np.ndarray, tails: np.ndarray, top_k: int = 10) -> np.ndarray:
        torch.save({
            'model_type': self.model_type,
            'model_state_dict': self.model.state_dict(),
            'n_entities': self.n_entities,
            'n_relations': self.n_relations,
            'embedding_dim': self.embedding_dim
        }, path)
        
    def load(self, path: str):

    
    def __init__(self, n_entities: int, n_relations: int, n_subspaces: int,
                 embedding_dim: int = 128, model_type: str = 'transe',
                 device: str = 'cpu'):
        if subspace_id < 0 or subspace_id >= self.n_subspaces:
            raise ValueError(f"Invalid subspace_id: {subspace_id}")
        
        return self.models[subspace_id].get_embeddings()
    
    def get_global_embeddings(self) -> Tuple[np.ndarray, np.ndarray]:
        if subspace_ids is None:
            subspace_ids = list(range(self.n_subspaces))
        
        results = {
            'subspace_predictions': [],
            'global_prediction': None,
            'aggregated_prediction': None
        }
        

        all_predictions = defaultdict(list)
        
        for sid in subspace_ids:
            pred = self.models[sid].predict_tail(np.array([head]), np.array([relation]), top_k)[0]
            results['subspace_predictions'].append({
                'subspace_id': sid,
                'indices': pred['indices'],
                'scores': pred['scores']
            })
            
            for idx, score in zip(pred['indices'], pred['scores']):
                all_predictions[idx].append(score)
        

        global_pred = self.global_model.predict_tail(np.array([head]), np.array([relation]), top_k)[0]
        results['global_prediction'] = {
            'indices': global_pred['indices'],
            'scores': global_pred['scores']
        }
        
        for idx, score in zip(global_pred['indices'], global_pred['scores']):

        

        aggregated_scores = []
        for idx, scores in all_predictions.items():
            avg_score = np.mean(scores)
            count = len(scores)

            aggregated_scores.append((idx, final_score, count))
        

        aggregated_scores.sort(key=lambda x: x[1], reverse=True)
        
        results['aggregated_prediction'] = {
            'indices': [x[0] for x in aggregated_scores[:top_k]],
            'scores': [x[1] for x in aggregated_scores[:top_k]],
            'counts': [x[2] for x in aggregated_scores[:top_k]]
        }
        
        return results
    
    def predict_head_multi_space(self, tail: int, relation: int,
                                  subspace_ids: List[int] = None,
                                  top_k: int = 10) -> Dict:

        for i, model in enumerate(self.models):
            model.save(f"{path}_subspace_{i}.pt")
        self.global_model.save(f"{path}_global.pt")
        
    def load(self, path: str):
        """
        for i, model in enumerate(self.models):
            model.load(f"{path}_subspace_{i}.pt")
        self.global_model.load(f"{path}_global.pt")