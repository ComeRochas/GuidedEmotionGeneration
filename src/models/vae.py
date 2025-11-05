"""
VAE Encoder/Decoder Module
Wraps the VAE from Stable Diffusion v1-5 for latent encoding/decoding
"""

import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class VAEWrapper(nn.Module):
    """
    Wrapper for Stable Diffusion v1-5 VAE.
    Handles encoding images to latents and decoding latents to images.
    """
    
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5", subfolder="vae"):
        """
        Args:
            model_id: HuggingFace model ID for Stable Diffusion
            subfolder: Subfolder containing VAE weights
        """
        super().__init__()
        
        # Load pretrained VAE from Stable Diffusion v1-5
        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder=subfolder
        )
        
        # Freeze VAE parameters (typically we don't train the VAE)
        for param in self.vae.parameters():
            param.requires_grad = False
        
        self.scaling_factor = self.vae.config.scaling_factor
        self.latent_channels = self.vae.config.latent_channels
    
    @torch.no_grad()
    def encode(self, images):
        """
        Encode images to latent representations.
        
        Args:
            images: Input images [B, 3, H, W] in range [-1, 1]
            
        Returns:
            latents: Encoded latents [B, latent_channels, H//8, W//8]
        """
        latent_dist = self.vae.encode(images).latent_dist
        latents = latent_dist.sample()
        latents = latents * self.scaling_factor
        return latents
    
    @torch.no_grad()
    def decode(self, latents):
        """
        Decode latents to images.
        
        Args:
            latents: Latent representations [B, latent_channels, H//8, W//8]
            
        Returns:
            images: Decoded images [B, 3, H, W] in range [-1, 1]
        """
        latents = latents / self.scaling_factor
        images = self.vae.decode(latents).sample
        return images
    
    def get_latent_size(self, image_size):
        """
        Calculate latent size for given image size.
        
        Args:
            image_size: Input image size (H, W)
            
        Returns:
            latent_size: Latent size (H//8, W//8)
        """
        return (image_size[0] // 8, image_size[1] // 8)
