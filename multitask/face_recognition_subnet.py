import torch
import torch.nn as nn


class FaceRecognitionEmbeddingSubnet(nn.Module):
    def __init__(self, feature_embedding_dim=768, embedding_dim=512):
        super(FaceRecognitionEmbeddingSubnet, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.feature_head = nn.Sequential(
            nn.Linear(feature_embedding_dim, feature_embedding_dim, bias=False),
            nn.BatchNorm1d(feature_embedding_dim, eps=2e-5),
            nn.Linear(feature_embedding_dim, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim, eps=2e-5) 
        )

    def forward(self, multiscale_features):
        x = multiscale_features[-1]  # Shape: (B, feature_embedding_dim, h, w)
        
        x = self.avgpool(x) # Shape: (B, feature_embedding_dim, 1)
        x = self.flatten(x)  # Shape: (B, feature_embedding_dim)
        embedding = self.feature_head(x)
        
        embedding_norm = torch.norm(embedding, p=2, dim=1, keepdim=True)
        normalized_embedding = torch.div(embedding, embedding_norm)
        
        return normalized_embedding, embedding_norm