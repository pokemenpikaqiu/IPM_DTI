"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
from config import MODEL_CONFIG

class FeatureFusionNetwork(nn.Module):
    """
    
    def __init__(self, drug_dim: int = 256, target_dim: int = 256, 
                 hidden_dim: int = 512, output_dim: int = 256,
                 dropout: float = 0.3):     


        residual = self.residual_proj(concat_features)
        

        fused_features = fused_features + residual
        

        fused_features = fused_features.unsqueeze(1)
        attn_output, _ = self.attention(fused_features, fused_features, fused_features)

        

        fused_features = F.normalize(fused_features, p=2, dim=1)
        
        return fused_features
    
    def get_similarity(self, features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
        """
        return F.cosine_similarity(features1, features2)

class DeepFusionNetwork(nn.Module):
    """
    
    def __init__(self, drug_dim: int = 256, target_dim: int = 256,
                 hidden_dims: List[int] = [512, 384, 256],
                 output_dim: int = 128, dropout: float = 0.3):

        x_drug = drug_features
        for layer in self.drug_layers:
            x_drug = layer(x_drug)
        

        x_target = target_features
        for layer in self.target_layers:
            x_target = layer(x_target)
        

        x = torch.cat([x_drug, x_target], dim=1)
        

        for layer in self.fusion_layers:
            x = layer(x)
        

        output = self.output_layer(x)
        output = F.normalize(output, p=2, dim=1)
        
        return output

class InteractionPredictor(nn.Module):
    """
    
    def __init__(self, fusion_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.3):
        """
        super(InteractionPredictor, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        """
        return self.predictor(fused_features)

class FusionModel(nn.Module):
    """
    
    def __init__(self, drug_dim: int = 256, target_dim: int = 256,
                 fusion_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.3):
       
        fused_features = self.fusion_network(drug_features, target_features)
        interaction_prob = self.predictor(fused_features)
        return fused_features, interaction_prob

class FusionTrainer:
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
    
    def train_step(self, drug_features: torch.Tensor, target_features: torch.Tensor,
                   labels: torch.Tensor) -> Tuple[float, torch.Tensor]:
      
        self.model.eval()
        with torch.no_grad():
            drug_features = drug_features.to(self.device)
            target_features = target_features.to(self.device)
            
            fused_features, predictions = self.model(drug_features, target_features)
            
            return fused_features.cpu().numpy(), predictions.cpu().numpy().squeeze()

if __name__ == "__main__":

    batch_size = 32
    drug_dim = 256
    target_dim = 256
    

    model = FusionModel(drug_dim=drug_dim, target_dim=target_dim, fusion_dim=256)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
    

    drug_features = torch.randn(batch_size, drug_dim)
    target_features = torch.randn(batch_size, target_dim)
    
    fused_features, predictions = model(drug_features, target_features)
    print(f"\n融合特征维度: {fused_features.shape}")
    print(f"预测概率维度: {predictions.shape}")
    print(f"预测概率范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    

    trainer = FusionTrainer(model)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    loss, _ = trainer.train_step(drug_features, target_features, labels)
    print(f"\n训练损失: {loss:.4f}")
    

    fused_np, pred_np = trainer.predict(drug_features, target_features)
    print(f"\n预测结果维度: {pred_np.shape}")
    print(f"前5个预测: {pred_np[:5]}")