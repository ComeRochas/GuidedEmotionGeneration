#!/usr/bin/env python3
"""
Main training script for emotion-guided diffusion model
"""

import argparse
import os
import yaml
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from src.models import EmotionEncoder, VAEWrapper, EmotionConditionedUNet
from src.data import get_dataloader
from src.training import DDPMTrainer
from src.utils import (
    save_checkpoint,
    load_checkpoint,
    get_device,
    count_parameters,
    save_images
)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_models(config, device):
    """Initialize all models."""
    print("Initializing models...")
    
    # Emotion encoder
    emotion_encoder = EmotionEncoder(
        embedding_dim=config['model']['emotion_encoder']['embedding_dim'],
        num_emotions=config['model']['emotion_encoder']['num_emotions'],
        freeze_backbone=config['model']['emotion_encoder']['freeze_backbone']
    ).to(device)
    
    # VAE
    vae = VAEWrapper(
        model_id=config['model']['vae']['model_id'],
        subfolder=config['model']['vae']['subfolder']
    ).to(device)
    vae.eval()  # VAE is not trained
    
    # UNet
    unet = EmotionConditionedUNet(
        in_channels=config['model']['unet']['in_channels'],
        out_channels=config['model']['unet']['out_channels'],
        use_pretrained=config['model']['unet']['use_pretrained'],
        pretrained_model_id=config['model']['unet'].get('pretrained_model_id')
    ).to(device)
    
    # Enable gradient checkpointing if specified
    if config['training'].get('gradient_checkpointing', False):
        unet.set_gradient_checkpointing(True)
    
    print(f"Emotion Encoder parameters: {count_parameters(emotion_encoder):,}")
    print(f"UNet parameters: {count_parameters(unet.unet):,}")
    
    return emotion_encoder, vae, unet


def setup_optimizer(unet, emotion_encoder, config):
    """Setup optimizer and learning rate scheduler."""
    # Train only UNet
    trainable_params = list(unet.parameters())
    
    optimizer = AdamW(
        trainable_params,
        lr=config['training']['learning_rate'],
        betas=config['training']['optimizer']['betas'],
        weight_decay=config['training']['optimizer']['weight_decay'],
        eps=config['training']['optimizer']['eps']
    )
    
    return optimizer


def train(config_path):
    """Main training function."""
    # Load configuration
    config = load_config(config_path)
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Setup models
    emotion_encoder, vae, unet = setup_models(config, device)
    
    # Setup optimizer
    optimizer = setup_optimizer(unet, emotion_encoder, config)
    
    # Setup dataloader
    train_loader = get_dataloader(
        data_root=config['data']['data_root'],
        batch_size=config['training']['batch_size'],
        image_size=config['data']['image_size'],
        num_workers=config['training']['num_workers'],
        split='train',
        shuffle=True,
        return_emotion_labels=config['data']['return_emotion_labels']
    )
    
    # Setup trainer
    trainer = DDPMTrainer(
        unet=unet,
        vae=vae,
        emotion_encoder=emotion_encoder,
        num_train_timesteps=config['training']['num_train_timesteps'],
        beta_schedule=config['training']['beta_schedule'],
        prediction_type=config['training']['prediction_type']
    )
    
    # Setup logging
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    if config['logging'].get('use_tensorboard', True):
        writer = SummaryWriter(log_dir=log_dir)
    else:
        writer = None
    
    # Setup mixed precision training
    scaler = None
    if config['training'].get('mixed_precision', False) and device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        print("Using mixed precision training")
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    save_dir = Path(config['checkpoint']['save_dir'])
    save_every = config['checkpoint']['save_every_n_epochs']
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Training samples: {len(train_loader.dataset)}")
    
    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        avg_loss = trainer.train_epoch(
            train_loader,
            optimizer,
            device,
            epoch,
            scaler=scaler
        )
        
        print(f"Epoch {epoch}/{num_epochs} - Average Loss: {avg_loss:.4f}")
        
        # Log to tensorboard
        if writer is not None:
            writer.add_scalar('Loss/train', avg_loss, epoch)
        
        # Save checkpoint
        if epoch % save_every == 0:
            save_checkpoint(
                unet=unet,
                emotion_encoder=emotion_encoder,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_loss,
                save_dir=save_dir
            )
    
    # Save final checkpoint
    save_checkpoint(
        unet=unet,
        emotion_encoder=emotion_encoder,
        optimizer=optimizer,
        epoch=num_epochs,
        loss=avg_loss,
        save_dir=save_dir,
        filename="checkpoint_final.pt"
    )
    
    if writer is not None:
        writer.close()
    
    print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Train emotion-guided diffusion model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to training configuration file"
    )
    
    args = parser.parse_args()
    
    train(args.config)


if __name__ == "__main__":
    main()
