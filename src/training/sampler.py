"""
DDPM Sampling Script
Implements inference and image generation with emotion guidance
"""

import torch
from diffusers import DDPMScheduler
from tqdm import tqdm
from typing import Optional


class DDPMSampler:
    """
    Sampler for emotion-guided diffusion model.
    """
    
    def __init__(
        self,
        unet,
        vae,
        emotion_encoder,
        num_inference_steps=50,
        beta_schedule="linear"
    ):
        """
        Args:
            unet: Trained UNet model
            vae: VAE for decoding latents
            emotion_encoder: Emotion encoder
            num_inference_steps: Number of denoising steps
            beta_schedule: Noise schedule type
        """
        self.unet = unet
        self.vae = vae
        self.emotion_encoder = emotion_encoder
        
        # Initialize scheduler for inference
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule=beta_schedule
        )
        self.scheduler.set_timesteps(num_inference_steps)
        
        self.num_inference_steps = num_inference_steps
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        latent_size: tuple = (32, 32),
        device: str = "cuda",
        initial_latents: Optional[torch.Tensor] = None
    ):
        """
        Generate images.
        
        Args:
            batch_size: Number of images to generate
            latent_size: Size of latent space (H//8, W//8)
            device: Device to run on
            initial_latents: Optional initial latents to start denoising from [B, 4, H, W]
            
        Returns:
            images: Generated images [B, 3, H, W]
        """
        self.unet.eval()
        
        if initial_latents is not None:
            latents = initial_latents.to(device)
        else:
            # Initialize random latents
            latents = torch.randn(
                batch_size,
                4,  # Number of latent channels (SD VAE default)
                latent_size[0],
                latent_size[1],
                device=device
            )
        
        # Denoising loop
        for t in tqdm(self.scheduler.timesteps, desc="Sampling"):
            # Predict noise
            timestep = t.unsqueeze(0).repeat(batch_size).to(device)
            noise_pred = self.unet(latents, timestep)
            
            # Denoise step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents to images
        images = self.vae.decode(latents)
        
        return images

    
    @torch.no_grad()
    def emotion_transfer(
        self,
        source_image: torch.Tensor,
        target_emotion_idx: int,
        num_inference_steps: Optional[int] = None,
        noise_strength: float = 0.5,
        device: str = "cuda"
    ):
        """
        Transfer emotion from source image to target emotion.
        
        Args:
            source_image: Source image [1, 3, H, W]
            target_emotion_idx: Target emotion class index
            num_inference_steps: Number of denoising steps (uses default if None)
            noise_strength: Strength of noise addition (0.0 to 1.0)
            device: Device to run on
            
        Returns:
            edited_image: Image with transferred emotion [1, 3, H, W]
        """
        self.unet.eval()
        
        if num_inference_steps is not None:
            self.scheduler.set_timesteps(num_inference_steps)
        
        source_image = source_image.to(device)
        
        # Encode source image to latents
        latents = self.vae.encode(source_image)
        
        # Get target emotion embedding
        target_emotion_embeddings = self.emotion_encoder.get_emotion_embedding(
            torch.tensor([target_emotion_idx])
        ).to(device)
        
        # Add noise to latents based on noise_strength
        noise = torch.randn_like(latents)
        timestep_idx = int(noise_strength * len(self.scheduler.timesteps))
        timestep = self.scheduler.timesteps[timestep_idx]
        noisy_latents = self.scheduler.add_noise(latents, noise, timestep)
        
        # Denoise with target emotion conditioning
        for t in tqdm(
            self.scheduler.timesteps[timestep_idx:],
            desc="Emotion Transfer"
        ):
            timestep_tensor = t.unsqueeze(0).to(device)
            noise_pred = self.unet(
                noisy_latents,
                timestep_tensor,
                target_emotion_embeddings
            )
            noisy_latents = self.scheduler.step(
                noise_pred, t, noisy_latents
            ).prev_sample
        
        # Decode to image
        edited_image = self.vae.decode(noisy_latents)
        
        return edited_image
