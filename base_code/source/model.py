import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class HybridModel(nn.Module):
    """Modèle hybride CNN (ResNet50) + Transformer (Swin Tiny).
    Utilise ResNet50 car réduit le temps d'entraînement."""
    def __init__(self, num_classes=3, weights=None):
        super().__init__()
        # Backbone CNN
        self.cnn = resnet50(progress = True, weights=weights)    # Utilisation de ResNet50 pré-entraîné
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()  # Retirer la dernière couche
        
        # Couche Transformer (simplifiée)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=in_features, nhead=8, dim_feedforward=2048
        )
        # print(f"in_features: {in_features}")
        # print(f"num_classes type: {num_classes.shape}")
        self.layer_norm = nn.LayerNorm(in_features)
        self.classifier = nn.Linear(in_features, 3)  # 3 classes (lymphocyte, monocyte, inflammatory-cells)

    def forward(self, x):
        features = self.cnn(x)  # [batch_size, in_features]
        features = features.unsqueeze(1)  # [batch_size, 1, in_features]
        features = self.transformer(features)
        features = self.layer_norm(features.squeeze(1))  # Normalisation des caractéristiques
        return self.classifier(features.squeeze(1))
    
print("HybridModel loaded successfully.")