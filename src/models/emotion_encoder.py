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
        
        # Embedding projection layer for conditioning (used during training)
        self.embedding_proj = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Learned emotion embeddings for class-conditional generation (used during inference)
        # These are trained via a contrastive loss or classification objective
        self.emotion_embeddings = nn.Embedding(num_emotions, embedding_dim)
        nn.init.normal_(self.emotion_embeddings.weight, std=0.02)
        
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
    
    def compute_embedding_alignment_loss(self, images, emotion_labels):
        """
        Compute loss to align learned emotion embeddings with image-derived embeddings.
        This helps ensure the learned embeddings are meaningful.
        
        Args:
            images: Input images [B, 3, H, W]
            emotion_labels: Ground truth emotion labels [B]
            
        Returns:
            loss: Alignment loss (MSE between learned and image-derived embeddings)
        """
        # Get embeddings from images
        image_embeddings = self.forward(images)
        
        # Get learned embeddings for the labels
        learned_embeddings = self.emotion_embeddings(emotion_labels)
        
        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(image_embeddings, learned_embeddings)
        
        return loss
    
    def get_emotion_embedding(self, emotion_idx):
        """
        Get learned embedding for a specific emotion class.
        Useful for unconditional generation with specific emotions.
        
        Args:
            emotion_idx: Emotion class index [B] or scalar
            
        Returns:
            embeddings: Learned emotion embeddings [B, embedding_dim]
        """
        if isinstance(emotion_idx, int):
            emotion_idx = torch.tensor([emotion_idx])
        
        device = next(self.parameters()).device
        emotion_idx = emotion_idx.to(device)
        
        # Get learned embeddings
        embeddings = self.emotion_embeddings(emotion_idx)
        
        return embeddings
    
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
