"""
UNet2DModel for Diffusion
Basic UNet without conditioning
"""

import torch
import torch.nn as nn
from diffusers import UNet2DModel


class EmotionConditionedUNet(nn.Module):
    """
    Basic UNet2DModel for denoising.
    No conditioning applied.
    """
    
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        pretrained_model_id=None,
        use_pretrained=False
    ):
        """
        Args:
            in_channels: Number of input latent channels (4 for SD VAE)
            out_channels: Number of output latent channels
            pretrained_model_id: HuggingFace model ID if using pretrained weights
            use_pretrained: Whether to load pretrained UNet from SD
        """
        super().__init__()
        
        if use_pretrained and pretrained_model_id:
            # Load pretrained UNet from Stable Diffusion
            self.unet = UNet2DModel.from_pretrained(
                pretrained_model_id,
                subfolder="unet"
            )
            print(f"Loaded pretrained UNet from {pretrained_model_id}")
        else:
            # Initialize UNet from scratch with custom config
            self.unet = UNet2DModel(
                in_channels=in_channels,
                out_channels=out_channels,
                attention_head_dim=8,
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
                block_out_channels=(320, 640, 1280, 1280),
                layers_per_block=2,
                norm_num_groups=32,
            )
            print("Initialized UNet from scratch")
    
    def forward(self, latents, timesteps, return_dict=False):
        """
        Forward pass through the UNet.
        
        Args:
            latents: Noisy latent inputs [B, in_channels, H, W]
            timesteps: Diffusion timesteps [B]
            return_dict: Whether to return a dict or just the sample
            
        Returns:
            noise_pred: Predicted noise [B, out_channels, H, W]
        """
        # Forward through UNet without conditioning
        output = self.unet(
            latents,
            timesteps,
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
