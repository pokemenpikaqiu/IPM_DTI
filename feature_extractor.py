"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import hashlib
from config import FEATURE_CONFIG

class FeatureExtractor:
    """
    
    def __init__(self, feature_dim: int = None):
        self.feature_dim = feature_dim or FEATURE_CONFIG['drug_feature_dim']
        self.drug_features = {}
        self.target_features = {}
        
    def extract_drug_features(self, drug_id: str, drug_data: Optional[Dict] = None) -> np.ndarray:
        """
        if drug_id in self.drug_features:
            return self.drug_features[drug_id]
        

        if drug_data is not None:
            features = self._extract_real_drug_features(drug_data)
        else:

            features = self._generate_deterministic_features(drug_id, 'drug')
        
        self.drug_features[drug_id] = features
        return features
    
    def extract_target_features(self, target_id: str, target_data: Optional[Dict] = None) -> np.ndarray:
        """
        if target_id in self.target_features:
            return self.target_features[target_id]
        

        if target_data is not None:
            features = self._extract_real_target_features(target_data)
        else:

            features = self._generate_deterministic_features(target_id, 'target')
        
        self.target_features[target_id] = features
        return features
    
    def _extract_real_drug_features(self, drug_data: Dict) -> np.ndarray:
        features = np.zeros(self.feature_dim, dtype=np.float32)
        

        if 'features' in target_data:
            return np.array(target_data['features'], dtype=np.float32)
        

        if 'sequence' in target_data:
            features = self._compute_sequence_features(target_data['sequence'])
        

        if 'structure' in target_data:
            struct = target_data['structure']

            if 'helix_ratio' in struct:
                features[0] = struct['helix_ratio']
            if 'sheet_ratio' in struct:
                features[1] = struct['sheet_ratio']
            if 'coil_ratio' in struct:
                features[2] = struct['coil_ratio']
        
        return features
    
    def _compute_molecular_fingerprint(self, smiles: str) -> np.ndarray:
        """

        hash_obj = hashlib.md5(smiles.encode())
        hash_bytes = hash_obj.digest()
        

        features = np.zeros(self.feature_dim, dtype=np.float32)
        for i in range(self.feature_dim):
            byte_idx = i % len(hash_bytes)
            features[i] = hash_bytes[byte_idx] / 255.0
        
        return features
    
    def _compute_sequence_features(self, sequence: str) -> np.ndarray:
        """
        features = np.zeros(self.feature_dim, dtype=np.float32)
        

        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_count = defaultdict(int)
        
        for aa in sequence:
            aa_count[aa] += 1
        
        seq_len = len(sequence)
        if seq_len > 0:
            for i, aa in enumerate(amino_acids):
                if i < self.feature_dim:
                    features[i] = aa_count.get(aa, 0) / seq_len
        

        if self.feature_dim > 20:
            features[20] = min(seq_len / 1000.0, 1.0)
        

        hash_obj = hashlib.md5(sequence.encode())
        hash_bytes = hash_obj.digest()
        for i in range(21, min(50, self.feature_dim)):
            byte_idx = (i - 21) % len(hash_bytes)
            features[i] = hash_bytes[byte_idx] / 255.0
        
        return features
    
    def _generate_deterministic_features(self, entity_id: str, entity_type: str) -> np.ndarray:

        drug_matrix = np.zeros((len(drugs), self.feature_dim), dtype=np.float32)
        for i, drug_id in enumerate(drugs):
            d_data = drug_data.get(drug_id) if drug_data else None
            drug_matrix[i] = self.extract_drug_features(drug_id, d_data)
        

        target_matrix = np.zeros((len(targets), self.feature_dim), dtype=np.float32)
        for i, target_id in enumerate(targets):
            t_data = target_data.get(target_id) if target_data else None
            target_matrix[i] = self.extract_target_features(target_id, t_data)
        
        return drug_matrix, target_matrix
    
    def get_feature_matrix(self, entity_ids: List[str], entity_type: str) -> np.ndarray:
        if method == 'cosine':
            norm1 = np.linalg.norm(features1)
            norm2 = np.linalg.norm(features2)
            if norm1 > 0 and norm2 > 0:
                return np.dot(features1, features2) / (norm1 * norm2)
            return 0.0
        elif method == 'euclidean':
            return 1.0 / (1.0 + np.linalg.norm(features1 - features2))
        elif method == 'dot':
            return np.dot(features1, features2)
        else:
            return 0.0
    
    def save_features(self, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        self.drug_features = data['drug_features'].item()
        self.target_features = data['target_features'].item()
        print(f"特征已从 {filepath} 加载")

class InteractionFeatureBuilder:

        drug_features = self.feature_extractor.extract_drug_features(drug_id, drug_data)
        

        target_features = self.feature_extractor.extract_target_features(target_id, target_data)
        

        interaction_features = np.concatenate([drug_features, target_features])
        
        return interaction_features
    
    def build_batch_features(self, interactions: List[Dict]) -> np.ndarray:
        """
        features_list = []
        
        for interaction in interactions:
            drug_id = interaction['drug_id']
            target_id = interaction['target_id']
            drug_data = interaction.get('drug_data')
            target_data = interaction.get('target_data')
            
            features = self.build_interaction_features(drug_id, target_id, 
                                                       drug_data, target_data)
            features_list.append(features)
        
        return np.array(features_list, dtype=np.float32)

if __name__ == "__main__":

    extractor = FeatureExtractor(feature_dim=256)
    

    drug_features = extractor.extract_drug_features("Drug_0001")
    print(f"药物特征维度: {drug_features.shape}")
    print(f"药物特征前10维: {drug_features[:10]}")
    

    target_features = extractor.extract_target_features("Target_0001")
    print(f"\n靶点特征维度: {target_features.shape}")
    print(f"靶点特征前10维: {target_features[:10]}")
    

    builder = InteractionFeatureBuilder(feature_dim=256)
    interaction_features = builder.build_interaction_features("Drug_0001", "Target_0001")
    print(f"\n交互特征维度: {interaction_features.shape}")
    

    drug_features2 = extractor.extract_drug_features("Drug_0002")
    similarity = extractor.compute_similarity(drug_features, drug_features2)
    print(f"\n药物相似度: {similarity:.4f}")