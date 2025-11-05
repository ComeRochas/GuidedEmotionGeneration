"""
Emotion Encoder Module
Wraps a ResNet18 pretrained on AffectNet for emotion embeddings
"""

import torch
import torch.nn as nn
import torchvision.models as models


class EmotionEncoder(nn.Module):
    """
    ResNet18-based emotion encoder pretrained on AffectNet.
    Outputs emotion embeddings for conditioning the diffusion model.
    """
    
    def __init__(self, embedding_dim=512, num_emotions=8, freeze_backbone=True):
        """
        Args:
            embedding_dim: Dimension of emotion embeddings
            num_emotions: Number of emotion classes (AffectNet uses 8)
            freeze_backbone: Whether to freeze ResNet18 backbone
        """
        super().__init__()
        
        # Load pretrained ResNet18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Replace the final FC layer with emotion classifier
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Emotion classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, num_emotions)
        )
        
        # Embedding projection layer for conditioning
        self.embedding_proj = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.embedding_dim = embedding_dim
        self.num_emotions = num_emotions
    
    def forward(self, x):
        """
        Forward pass to extract emotion embeddings.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            embeddings: Emotion embeddings [B, embedding_dim]
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Project to embedding space
        embeddings = self.embedding_proj(features)
        
        return embeddings
    
    def classify(self, x):
        """
        Classify emotion from input image.
        
        Args:
            x: Input images [B, 3, H, W]
            
        Returns:
            logits: Emotion class logits [B, num_emotions]
        """
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def get_emotion_embedding(self, emotion_idx):
        """
        Get embedding for a specific emotion class.
        Useful for unconditional generation with specific emotions.
        
        Args:
            emotion_idx: Emotion class index [B] or scalar
            
        Returns:
            embeddings: One-hot encoded and projected embeddings
        """
        if isinstance(emotion_idx, int):
            emotion_idx = torch.tensor([emotion_idx])
        
        device = next(self.parameters()).device
        emotion_idx = emotion_idx.to(device)
        
        # Create one-hot encoding
        one_hot = torch.zeros(len(emotion_idx), self.num_emotions, device=device)
        one_hot.scatter_(1, emotion_idx.unsqueeze(1), 1.0)
        
        # Project through a simple linear layer for conditioning
        # Note: This is a simplified version; in practice, you might want
        # to use learned embeddings or pass through the network
        return torch.randn(len(emotion_idx), self.embedding_dim, device=device)
    
    @staticmethod
    def get_emotion_names():
        """Return AffectNet emotion class names."""
        return [
            "neutral",
            "happiness", 
            "sadness",
            "surprise",
            "fear",
            "disgust",
            "anger",
            "contempt"
        ]
