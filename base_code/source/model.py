import torch
import torch.nn as nn
from torchvision.models import resnet50

class HybridModel(nn.Module):
    """Modèle hybride CNN (ResNet50) + Transformer (Swin Tiny)."""
    def __init__(self, num_classes=3):
        super().__init__()
        # Backbone CNN
        self.cnn = resnet50(pretrained=True)
        in_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()  # Retirer la dernière couche
        
        # Couche Transformer (simplifiée)
        self.transformer = nn.TransformerEncoderLayer(
            d_model=in_features, nhead=8, dim_feedforward=2048
        )
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.cnn(x)  # [batch_size, in_features]
        features = features.unsqueeze(1)  # [batch_size, 1, in_features]
        features = self.transformer(features)
        return self.classifier(features.squeeze(1))