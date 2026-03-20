import pandas as pd
import numpy as np
import os
import pickle
from typing import List, Tuple, Dict, Optional
import random
from config import DATA_CONFIG


class DataLoader:
    def __init__(self):
        self.data_dir = DATA_CONFIG['data_dir']
        self.raw_data_file = os.path.join(self.data_dir, DATA_CONFIG['raw_data_file'])
        self.processed_data_file = os.path.join(self.data_dir, DATA_CONFIG['processed_data_file'])
        
        # 数据存储
        self.drugs = []
        self.targets = []
        self.interactions = []
        self.drug_features = {}
        self.target_features = {}
        
        # 实体和关系映射
        self.entity2id = {}
        self.id2entity = {}
        self.relation2id = {}
        self.id2relation = {}
        
    def generate_sample_data(self, n_drugs: int = 200, n_targets: int = 150, n_interactions: int = 800):
        # 生成药物ID
        self.drugs = [f"Drug_{i:04d}" for i in range(n_drugs)]
        
        # 生成靶点ID
        self.targets = [f"Target_{i:04d}" for i in range(n_targets)]
        
        self.interactions = []
        interaction_set = set()
        
        while len(self.interactions) < n_interactions:
            drug = random.choice(self.drugs)
            target = random.choice(self.targets)
            pair = (drug, target)
            
            if pair not in interaction_set:
                interaction_set.add(pair)
                self.interactions.append({
                    'drug_id': drug,
                    'target_id': target,
                    'label': 1,
                    'drug_feature': np.random.randn(256).tolist(),  # 模拟药物特征
                    'target_feature': np.random.randn(256).tolist()  # 模拟靶点特征
                })
        
        # 生成负样本
        negative_samples = []
        while len(negative_samples) < n_interactions:
            drug = random.choice(self.drugs)
            target = random.choice(self.targets)
            pair = (drug, target)
            
            if pair not in interaction_set:
                negative_samples.append({
                    'drug_id': drug,
                    'target_id': target,
                    'label': 0,
                    'drug_feature': np.random.randn(256).tolist(),
                    'target_feature': np.random.randn(256).tolist()
                })
                interaction_set.add(pair)
        
        self.interactions.extend(negative_samples)
        

        self._build_entity_mapping()
    
        self._generate_feature_matrices()
        return self.interactions
    
    def _build_entity_mapping(self):
        """构建实体和关系的ID映射"""
        # 添加所有实体
        all_entities = list(set(self.drugs + self.targets))
        for idx, entity in enumerate(all_entities):
            self.entity2id[entity] = idx
            self.id2entity[idx] = entity
        
        # 添加关系类型
        self.relation2id['interacts'] = 0
        self.relation2id['not_interacts'] = 1
        self.id2relation[0] = 'interacts'
        self.id2relation[1] = 'not_interacts'
        
        print(f"实体数量: {len(self.entity2id)}, 关系数量: {len(self.relation2id)}")
    
    def _generate_feature_matrices(self):
        for drug in self.drugs:
            self.drug_features[drug] = np.random.randn(256).astype(np.float32)
        
        for target in self.targets:
            self.target_features[target] = np.random.randn(256).astype(np.float32)
    
    def load_zheng_dataset(self, file_path: Optional[str] = None) -> pd.DataFrame:
        if file_path is None:
            file_path = self.raw_data_file
        
        if os.path.exists(file_path):
            print(f"load_data: {file_path}")
            df = pd.read_csv(file_path)
        else:
            print("data file not exists!")
            self.generate_sample_data()
            df = pd.DataFrame(self.interactions)
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> Dict:
        print("preprocess_data...")
        
        if 'drug_id' in df.columns:
            self.drugs = list(df['drug_id'].unique())
            self.targets = list(df['target_id'].unique())
        
        self._build_entity_mapping()
        
        triples = []
        for _, row in df.iterrows():
            drug = row['drug_id']
            target = row['target_id']
            label = row.get('label', 1)
            
            relation = 'interacts' if label == 1 else 'not_interacts'
            
            triples.append({
                'head': drug,
                'relation': relation,
                'tail': target,
                'label': label
            })
        
        random.shuffle(triples)
        n_total = len(triples)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        train_data = triples[:n_train]
        val_data = triples[n_train:n_train+n_val]
        test_data = triples[n_train+n_val:]
        
        processed_data = {
            'train': train_data,
            'val': val_data,
            'test': test_data,
            'entity2id': self.entity2id,
            'id2entity': self.id2entity,
            'relation2id': self.relation2id,
            'id2relation': self.id2relation,
            'drug_features': self.drug_features,
            'target_features': self.target_features,
            'n_entities': len(self.entity2id),
            'n_relations': len(self.relation2id)
        }
        
        self.save_processed_data(processed_data)
        
        print(f"Finished preprocess: train{len(train_data)}, valid{len(val_data)}, test{len(test_data)}")
        return processed_data
    
    def save_processed_data(self, data: Dict):
        """保存预处理后的数据"""
        with open(self.processed_data_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"预处理数据已保存: {self.processed_data_file}")
    
    def load_processed_data(self) -> Optional[Dict]:
        """加载预处理后的数据"""
        if os.path.exists(self.processed_data_file):
            with open(self.processed_data_file, 'rb') as f:
                data = pickle.load(f)
            
            # 恢复属性
            self.entity2id = data['entity2id']
            self.id2entity = data['id2entity']
            self.relation2id = data['relation2id']
            self.id2relation = data['id2relation']
            self.drug_features = data['drug_features']
            self.target_features = data['target_features']
            
            print(f"load_processed_data: {len(data['train'])}训练样本")
            return data
        return None
    
    def get_feature_matrices(self) -> Tuple[np.ndarray, np.ndarray]:

        drug_feature_matrix = np.zeros((len(self.drugs), 256))
        for i, drug in enumerate(self.drugs):
            if drug in self.drug_features:
                drug_feature_matrix[i] = self.drug_features[drug]
        
        target_feature_matrix = np.zeros((len(self.targets), 256))
        for i, target in enumerate(self.targets):
            if target in self.target_features:
                target_feature_matrix[i] = self.target_features[target]
        
        return drug_feature_matrix, target_feature_matrix
    
    def get_triples(self, data_type: str = 'train') -> List[Tuple]:
        data = self.load_processed_data()
        if data is None:
            return []
        
        triples = []
        for item in data[data_type]:
            h = self.entity2id[item['head']]
            r = self.relation2id[item['relation']]
            t = self.entity2id[item['tail']]
            triples.append((h, r, t, item['label']))
        
        return triples
    
    def generate_synthetic_data(self, n_drugs: int = 200, n_targets: int = 150, 
                                 n_interactions: int = 800) -> Dict:

        print(f"生成合成数据: {n_drugs}个药物, {n_targets}个靶点, {n_interactions}个交互")
        
        drugs = list(range(n_drugs))
        
        targets = list(range(n_drugs, n_drugs + n_targets))
        
        entity2id = {}
        id2entity = {}
        
        for drug_id in drugs:
            entity_name = f"Drug_{drug_id}"
            entity2id[entity_name] = drug_id
            id2entity[drug_id] = entity_name
        
        for target_id in targets:
            entity_name = f"Target_{target_id}"
            entity2id[entity_name] = target_id
            id2entity[target_id] = entity_name
        
        relation2id = {'interacts': 0}
        id2relation = {0: 'interacts'}
        
        train_triplets = []
        interaction_set = set()
        
        while len(train_triplets) < n_interactions:
            drug = random.choice(drugs)
            target = random.choice(targets)
            pair = (drug, target)
            
            if pair not in interaction_set:
                interaction_set.add(pair)
                train_triplets.append((drug, 0, target)) 
 
        random.shuffle(train_triplets)
        n_total = len(train_triplets)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        train_data = train_triplets[:n_train]
        val_data = train_triplets[n_train:n_train + n_val]
        test_data = train_triplets[n_train + n_val:]
        

        drug_features = {}
        for drug_id in drugs:
            drug_features[drug_id] = np.random.randn(256).astype(np.float32)
        

        target_features = {}
        for target_id in targets:
            target_features[target_id] = np.random.randn(256).astype(np.float32)
        
        processed_data = {
            'drugs': drugs,
            'targets': targets,
            'train_triplets': train_data,
            'val_triplets': val_data,
            'test_triplets': test_data,
            'entity2id': entity2id,
            'id2entity': id2entity,
            'relation2id': relation2id,
            'id2relation': id2relation,
            'drug_features': drug_features,
            'target_features': target_features,
            'n_entities': n_drugs + n_targets,
            'n_relations': 1
        }
        
        print(f"合成数据生成完成:")
        print(f"  - 药物数量: {len(drugs)}")
        print(f"  - 靶点数量: {len(targets)}")
        print(f"  - 训练三元组: {len(train_data)}")
        print(f"  - 验证三元组: {len(val_data)}")
        print(f"  - 测试三元组: {len(test_data)}")
        print(f"  - 实体总数: {processed_data['n_entities']}")
        
        return processed_data


if __name__ == "__main__":

    loader = DataLoader()
    

    loader.generate_sample_data()

    df = pd.DataFrame(loader.interactions)
    processed_data = loader.preprocess_data(df)