"""
DDPM Training Loop
Implements Denoising Diffusion Probabilistic Models training
"""

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from tqdm import tqdm


class DDPMTrainer:
    """
    DDPM Trainer for emotion-guided diffusion model.
    """
    
    def __init__(
        self,
        unet,
        vae,
        emotion_encoder,
        num_train_timesteps=1000,
        beta_schedule="linear",
        prediction_type="epsilon"
    ):
        """
        Args:
            unet: UNet model for denoising
            vae: VAE for encoding/decoding
            emotion_encoder: Emotion encoder for conditioning
            num_train_timesteps: Number of diffusion timesteps
            beta_schedule: Noise schedule type
            prediction_type: What the model predicts ("epsilon" or "sample")
        """
        self.unet = unet
        self.vae = vae
        self.emotion_encoder = emotion_encoder
        
        # Initialize DDPM noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type
        )
        
        self.num_train_timesteps = num_train_timesteps
    
    def compute_loss(self, images, device):
        """
        Compute DDPM loss for a batch of images.
        
        Args:
            images: Batch of images [B, 3, H, W]
            device: Device to run on
            
        Returns:
            loss: DDPM loss
        """
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Encode images to latents
        with torch.no_grad():
            latents = self.vae.encode(images)
        
        # Extract emotion embeddings
        with torch.no_grad():
            emotion_embeddings = self.emotion_encoder(images)
        
        # Sample noise
        noise = torch.randn_like(latents)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, 
            self.num_train_timesteps,
            (batch_size,),
            device=device,
            dtype=torch.long
        )
        
        # Add noise to latents according to noise scheduler
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise with UNet
        noise_pred = self.unet(noisy_latents, timesteps, emotion_embeddings)
        
        # Compute loss (MSE between predicted and actual noise)
        loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")
        
        return loss
    
    def train_step(self, batch, optimizer, device, scaler=None):
        """
        Single training step.
        
        Args:
            batch: Batch of data from dataloader
            optimizer: Optimizer
            device: Device to run on
            scaler: GradScaler for mixed precision training
            
        Returns:
            loss: Training loss value
        """
        # Handle different batch formats (with or without labels)
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss = self.compute_loss(images, device)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = self.compute_loss(images, device)
            loss.backward()
            optimizer.step()
        
        return loss.item()
    
    def train_epoch(self, dataloader, optimizer, device, epoch, scaler=None):
        """
        Train for one epoch.
        
        Args:
            dataloader: Training dataloader
            optimizer: Optimizer
            device: Device to run on
            epoch: Current epoch number
            scaler: GradScaler for mixed precision training
            
        Returns:
            avg_loss: Average loss for the epoch
        """
        self.unet.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            loss = self.train_step(batch, optimizer, device, scaler)
            total_loss += loss
            pbar.set_postfix({"loss": f"{loss:.4f}"})
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
