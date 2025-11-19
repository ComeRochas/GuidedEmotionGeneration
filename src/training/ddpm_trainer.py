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
        prediction_type="epsilon",
        use_embedding_alignment=False,
        alignment_weight=0.1
    ):
        """
        Args:
            unet: UNet model for denoising
            vae: VAE for encoding/decoding
            emotion_encoder: Emotion encoder for conditioning
            num_train_timesteps: Number of diffusion timesteps
            beta_schedule: Noise schedule type
            prediction_type: What the model predicts ("epsilon" or "sample")
            use_embedding_alignment: Whether to use embedding alignment loss
            alignment_weight: Weight for alignment loss
        """
        self.unet = unet
        self.vae = vae
        self.emotion_encoder = emotion_encoder
        self.use_embedding_alignment = use_embedding_alignment
        self.alignment_weight = alignment_weight
        
        # Initialize DDPM noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type
        )
        
        self.num_train_timesteps = num_train_timesteps
    
    def compute_loss(self, images, device, emotion_labels=None):
        """
        Compute DDPM loss for a batch of images.
        
        Args:
            images: Batch of images [B, 3, H, W]
            device: Device to run on
            emotion_labels: Optional emotion labels [B] for alignment loss
            
        Returns:
            loss: Total loss (DDPM + optional alignment loss)
        """
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Encode images to latents (VAE is frozen)
        with torch.no_grad():
            latents = self.vae.encode(images)
        
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
        
        # Predict noise with UNet (unconditional)
        noise_pred = self.unet(noisy_latents, timesteps)
        
        # Compute DDPM loss (MSE between predicted and actual noise)
        ddpm_loss = nn.functional.mse_loss(noise_pred, noise, reduction="mean")
        
        loss = ddpm_loss
        
        # Add optional embedding alignment loss
        if self.use_embedding_alignment and emotion_labels is not None:
            alignment_loss = self.emotion_encoder.compute_embedding_alignment_loss(
                images, emotion_labels
            )
            loss = loss + self.alignment_weight * alignment_loss
        
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
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            images = batch[0]
            emotion_labels = batch[1] if len(batch) > 1 else None
        else:
            images = batch
            emotion_labels = None
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.amp.autocast(device.type):
                loss = self.compute_loss(images, device, emotion_labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = self.compute_loss(images, device, emotion_labels)
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
