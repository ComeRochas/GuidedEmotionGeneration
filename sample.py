#!/usr/bin/env python3
"""
Sampling script for emotion-guided diffusion model
Generates images with specified emotions or performs emotion transfer
"""

import argparse
import yaml
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms

from src.models import EmotionEncoder, VAEWrapper, EmotionConditionedUNet
from src.training.sampler import DDPMSampler
from src.utils import get_device, load_checkpoint, save_images


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_models(config, device):
    """Load trained models from checkpoint."""
    print("Loading models...")
    
    # Initialize models
    emotion_encoder = EmotionEncoder(
        embedding_dim=512,
        num_emotions=8,
        freeze_backbone=True
    ).to(device)
    
    vae = VAEWrapper(
        model_id=config['model']['vae_model_id']
    ).to(device)
    
    unet = EmotionConditionedUNet(
        in_channels=4,
        out_channels=4,
        cross_attention_dim=512,
        use_pretrained=False
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = config['model']['checkpoint_path']
    load_checkpoint(checkpoint_path, unet, emotion_encoder)
    
    # Set to eval mode
    emotion_encoder.eval()
    vae.eval()
    unet.eval()
    
    return emotion_encoder, vae, unet


def sample_images(config_path, emotion_idx=None):
    """Generate images with specified emotion."""
    # Load config
    config = load_config(config_path)
    device = get_device()
    
    # Load models
    emotion_encoder, vae, unet = load_models(config, device)
    
    # Create sampler
    sampler = DDPMSampler(
        unet=unet,
        vae=vae,
        emotion_encoder=emotion_encoder,
        num_inference_steps=config['sampling']['num_inference_steps']
    )
    
    # Use emotion from config if not specified
    if emotion_idx is None:
        emotion_idx = config['sampling']['emotion_idx']
    
    # Get emotion name
    emotion_names = EmotionEncoder.get_emotion_names()
    emotion_name = emotion_names[emotion_idx]
    print(f"\nGenerating images with emotion: {emotion_name}")
    
    # Sample images
    batch_size = config['sampling']['batch_size']
    num_samples = config['output']['num_samples']
    image_size = config['sampling']['image_size']
    latent_size = (image_size // 8, image_size // 8)
    
    all_images = []
    for i in range(0, num_samples, batch_size):
        current_batch_size = min(batch_size, num_samples - i)
        
        images = sampler.sample(
            batch_size=current_batch_size,
            emotion_idx=emotion_idx,
            latent_size=latent_size,
            device=device,
            guidance_scale=config['sampling']['guidance_scale']
        )
        
        all_images.append(images)
    
    # Concatenate all images
    all_images = torch.cat(all_images, dim=0)
    
    # Save images
    output_dir = Path(config['output']['save_dir']) / f"emotion_{emotion_name}"
    save_images(all_images, output_dir, prefix="generated")
    
    print(f"\nGenerated {num_samples} images saved to {output_dir}")


def emotion_transfer(config_path, source_image_path, target_emotion_idx):
    """Transfer emotion from source image to target emotion."""
    # Load config
    config = load_config(config_path)
    device = get_device()
    
    # Load models
    emotion_encoder, vae, unet = load_models(config, device)
    
    # Create sampler
    sampler = DDPMSampler(
        unet=unet,
        vae=vae,
        emotion_encoder=emotion_encoder,
        num_inference_steps=config['sampling']['num_inference_steps']
    )
    
    # Load and preprocess source image
    image_size = config['sampling']['image_size']
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    
    source_image = Image.open(source_image_path).convert('RGB')
    source_image = transform(source_image).unsqueeze(0)
    
    # Get emotion names
    emotion_names = EmotionEncoder.get_emotion_names()
    target_emotion_name = emotion_names[target_emotion_idx]
    
    print(f"\nTransferring emotion to: {target_emotion_name}")
    
    # Perform emotion transfer
    edited_image = sampler.emotion_transfer(
        source_image=source_image,
        target_emotion_idx=target_emotion_idx,
        noise_strength=config['sampling']['noise_strength'],
        device=device
    )
    
    # Save result
    output_dir = Path(config['output']['save_dir']) / "emotion_transfer"
    save_images(edited_image, output_dir, prefix=f"transfer_{target_emotion_name}")
    
    # Also save source for comparison
    save_images(source_image, output_dir, prefix="source")
    
    print(f"\nEmotion transfer result saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample from emotion-guided diffusion model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sample_config.yaml",
        help="Path to sampling configuration file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample", "transfer"],
        default="sample",
        help="Sampling mode: 'sample' for generation, 'transfer' for emotion transfer"
    )
    parser.add_argument(
        "--emotion",
        type=int,
        default=None,
        help="Target emotion index (0-7)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Source image path for emotion transfer"
    )
    
    args = parser.parse_args()
    
    if args.mode == "sample":
        sample_images(args.config, args.emotion)
    elif args.mode == "transfer":
        if args.source is None:
            raise ValueError("Source image path required for emotion transfer mode")
        if args.emotion is None:
            raise ValueError("Target emotion index required for emotion transfer mode")
        emotion_transfer(args.config, args.source, args.emotion)


if __name__ == "__main__":
    main()
