#!/usr/bin/env python3
"""
Test script for VAE encoding/decoding and emotion classification
"""

import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path

from src.models import EmotionEncoder, VAEWrapper, EmotionConditionedUNet
from src.training.sampler import DDPMSampler
from src.utils import get_device, load_checkpoint
import yaml


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_models(config, device):
    """Initialize models."""
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
    
    print(f"Emotion Encoder parameters: {sum(p.numel() for p in emotion_encoder.parameters()):,}")
    print(f"VAE parameters: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"UNet parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    return emotion_encoder, vae, unet


def load_image(image_path, image_size=256):
    """Load and preprocess an image."""
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # to [-1, 1]
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def save_image(tensor, path, normalize=True):
    """Save a tensor as an image."""
    if normalize:
        # Convert from [-1, 1] to [0, 1]
        tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL Image
    to_pil = transforms.ToPILImage()
    image = to_pil(tensor.squeeze(0).cpu())
    image.save(path)


def main():
    # Load configuration
    config_path = "configs/train_config.yaml"
    config = load_config(config_path)
    
    # Setup device
    device = get_device()
    print(f"Using device: {device}")
    
    # Setup models
    emotion_encoder, vae, unet = setup_models(config, device)
    
    # Load checkpoint if exists
    checkpoint_dir = Path(config['checkpoint']['save_dir'])
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"Loading checkpoint: {latest_checkpoint}")
            load_checkpoint(str(latest_checkpoint), unet, emotion_encoder)
        else:
            print("No checkpoints found, using untrained models")
    else:
        print("Checkpoint directory does not exist, using untrained models")
    
    # Setup sampler
    sampler = DDPMSampler(
        unet=unet,
        vae=vae,
        emotion_encoder=emotion_encoder,
        num_inference_steps=3,  
        beta_schedule=config['training']['beta_schedule']
    )
    
    # Load a test image
    data_root = Path(config['data']['data_root'])
    image_dir = data_root / "images" if (data_root / "images").exists() else data_root
    image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    
    if not image_files:
        print("No images found in the dataset directory.")
        return
    
    # Take the first image
    test_image_path = image_files[1]
    print(f"Loading image: {test_image_path}")
    
    # Load and preprocess image
    image_tensor = load_image(test_image_path, config['data']['image_size']).to(device)
    
    # Save original image for comparison
    save_image(image_tensor, "test_image/original_image.jpg")
    print("Saved original image as 'original_image.jpg'")
    
    # Encode image to latents with VAE
    with torch.no_grad():
        latents = vae.encode(image_tensor)
    print(f"Encoded latents shape: {latents.shape}")
    

    # Decode latents back to image
    with torch.no_grad():
        reconstructed_image = vae.decode(latents)
    save_image(reconstructed_image, "test_image/reconstructed_image.jpg")
    print("Saved reconstructed image as 'reconstructed_image.jpg'")
    
    #Add noise to latents (for testing)
    noise = torch.randn_like(latents).to(device)
    noisy_latents = latents + noise * 0.1  # small noise
    
    # Denoise the clean latents using UNet
    with torch.no_grad():
        generated_images = sampler.sample(batch_size=1, device=device, initial_latents=latents)
    
    # Save generated image
    save_image(generated_images, "test_image/generated_image.jpg")
    print("Saved generated image as 'generated_image.jpg'")
    
    print("Test completed successfully!")


if __name__ == "__main__":
    main()