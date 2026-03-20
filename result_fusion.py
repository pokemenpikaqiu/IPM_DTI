"""Result Fusion Module"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from config import FUSION_CONFIG


class ResultFusion:
    """Result fusion for DTI prediction"""
    
    def __init__(self, config: Dict = None):
        self.config = config or FUSION_CONFIG
    
    def fuse_scores(self, rl_score: float, kg_score: float) -> float:
        """Fuse RL and KG scores"""
        rl_weight = self.config.get('rl_weight', 0.5)
        kg_weight = self.config.get('kg_weight', 0.5)
        
        rl_norm = 1 / (1 + np.exp(-rl_score))
        kg_norm = 1 / (1 + np.exp(-kg_score))
        
        fused = rl_weight * rl_norm + kg_weight * kg_norm
        return float(fused)
    
    def normalize_scores(self, scores: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """Normalize scores"""
        if len(scores) == 0:
            return scores
        if method == 'minmax':
            min_val, max_val = scores.min(), scores.max()
            if max_val > min_val:
                return (scores - min_val) / (max_val - min_val)
            return scores * 0
        return scores
    
    def rank_fusion(self, rankings: List[List[Tuple]], method: str = 'borda') -> List:
        """Fuse multiple rankings"""
        if method == 'borda':
            scores = defaultdict(float)
            for ranking in rankings:
                for idx, item in enumerate(ranking):
                    scores[item] += len(ranking) - idx
            return sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return rankings[0] if rankings else []


class PredictionEvaluator:
    """Evaluate prediction results"""
    
    def __init__(self):
        pass
    
    def evaluate(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score
        
        results = {}
        try:
            results['auc'] = roc_auc_score(y_true, y_pred)
        except:
            results['auc'] = 0.0
        
        try:
            results['aupr'] = average_precision_score(y_true, y_pred)
        except:
            results['aupr'] = 0.0
        
        y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
        try:
            results['accuracy'] = accuracy_score(y_true, y_pred_binary)
            results['f1'] = f1_score(y_true, y_pred_binary)
            results['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
            results['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
        except:
            results['accuracy'] = 0.0
            results['f1'] = 0.0
            results['precision'] = 0.0
            results['recall'] = 0.0
        
        return results


class PredictionOutput:
    """Output prediction results"""
    
    def __init__(self, output_dir: str = './output/results'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def save_predictions(self, predictions: List[Dict], filename: str = 'predictions.csv'):
        """Save predictions to file"""
        import pandas as pd
        df = pd.DataFrame(predictions)
        filepath = f'{self.output_dir}/{filename}'
        df.to_csv(filepath, index=False)
        return filepath
    
    def save_evaluation(self, metrics: Dict, filename: str = 'evaluation.txt'):
        """Save evaluation metrics"""
        filepath = f'{self.output_dir}/{filename}'
        with open(filepath, 'w') as f:
            for key, value in metrics.items():
                f.write(f'{key}: {value}\n')
        return filepath


if __name__ == '__main__':
    fusion = ResultFusion()
    print(fusion.fuse_scores(0.8, 0.6))
