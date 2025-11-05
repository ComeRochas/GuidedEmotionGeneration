"""
UNet2DConditionModel for Emotion-Guided Diffusion
Wraps Diffusers UNet with emotion embedding conditioning
"""

import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


class EmotionConditionedUNet(nn.Module):
    """
    UNet2DConditionModel conditioned on emotion embeddings.
    Uses cross-attention to condition on emotion features.
    """
    
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        cross_attention_dim=512,
        pretrained_model_id=None,
        use_pretrained=False
    ):
        """
        Args:
            in_channels: Number of input latent channels (4 for SD VAE)
            out_channels: Number of output latent channels
            cross_attention_dim: Dimension of cross-attention (emotion embedding dim)
            pretrained_model_id: HuggingFace model ID if using pretrained weights
            use_pretrained: Whether to load pretrained UNet from SD
        """
        super().__init__()
        
        if use_pretrained and pretrained_model_id:
            # Load pretrained UNet from Stable Diffusion
            self.unet = UNet2DConditionModel.from_pretrained(
                pretrained_model_id,
                subfolder="unet"
            )
            print(f"Loaded pretrained UNet from {pretrained_model_id}")
        else:
            # Initialize UNet from scratch with custom config
            self.unet = UNet2DConditionModel(
                in_channels=in_channels,
                out_channels=out_channels,
                cross_attention_dim=cross_attention_dim,
                attention_head_dim=8,
                down_block_types=(
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                ),
                block_out_channels=(320, 640, 1280, 1280),
                layers_per_block=2,
                norm_num_groups=32,
            )
            print("Initialized UNet from scratch")
        
        self.cross_attention_dim = cross_attention_dim
    
    def forward(self, latents, timesteps, emotion_embeddings, return_dict=False):
        """
        Forward pass through the UNet.
        
        Args:
            latents: Noisy latent inputs [B, in_channels, H, W]
            timesteps: Diffusion timesteps [B]
            emotion_embeddings: Emotion embeddings for conditioning [B, cross_attention_dim]
            return_dict: Whether to return a dict or just the sample
            
        Returns:
            noise_pred: Predicted noise [B, out_channels, H, W]
        """
        # Reshape emotion embeddings for cross-attention: [B, 1, cross_attention_dim]
        if emotion_embeddings.dim() == 2:
            emotion_embeddings = emotion_embeddings.unsqueeze(1)
        
        # Forward through UNet with emotion conditioning
        output = self.unet(
            latents,
            timesteps,
            encoder_hidden_states=emotion_embeddings,
            return_dict=True
        )
        
        if return_dict:
            return output
        else:
            return output.sample
    
    def set_gradient_checkpointing(self, enable=True):
        """Enable or disable gradient checkpointing for memory efficiency."""
        if hasattr(self.unet, "enable_gradient_checkpointing"):
            if enable:
                self.unet.enable_gradient_checkpointing()
        else:
            self.unet.set_gradient_checkpointing(enable)
