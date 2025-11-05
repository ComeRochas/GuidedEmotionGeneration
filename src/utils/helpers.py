"""
Utility functions for training and inference
"""

import torch
import os
from pathlib import Path


def save_checkpoint(
    unet,
    emotion_encoder,
    optimizer,
    epoch,
    loss,
    save_dir,
    filename=None
):
    """
    Save training checkpoint.
    
    Args:
        unet: UNet model
        emotion_encoder: Emotion encoder model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        save_dir: Directory to save checkpoint
        filename: Optional custom filename
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if filename is None:
        filename = f"checkpoint_epoch_{epoch}.pt"
    
    checkpoint = {
        "epoch": epoch,
        "unet_state_dict": unet.state_dict(),
        "emotion_encoder_state_dict": emotion_encoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    
    save_path = save_dir / filename
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path,
    unet,
    emotion_encoder,
    optimizer=None
):
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        unet: UNet model
        emotion_encoder: Emotion encoder model
        optimizer: Optional optimizer to load state
        
    Returns:
        epoch: Epoch number from checkpoint
        loss: Loss from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    unet.load_state_dict(checkpoint["unet_state_dict"])
    emotion_encoder.load_state_dict(checkpoint["emotion_encoder_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", 0.0)
    
    print(f"Loaded checkpoint from epoch {epoch} with loss {loss:.4f}")
    
    return epoch, loss


def tensor_to_pil(images):
    """
    Convert tensor images to PIL images.
    
    Args:
        images: Tensor images [B, 3, H, W] in range [-1, 1]
        
    Returns:
        pil_images: List of PIL images
    """
    from PIL import Image
    import numpy as np
    
    # Denormalize from [-1, 1] to [0, 1]
    images = (images + 1.0) / 2.0
    images = torch.clamp(images, 0, 1)
    
    # Convert to numpy
    images = images.cpu().permute(0, 2, 3, 1).numpy()
    
    # Convert to uint8
    images = (images * 255).astype(np.uint8)
    
    # Convert to PIL
    pil_images = [Image.fromarray(img) for img in images]
    
    return pil_images


def save_images(images, save_dir, prefix="sample"):
    """
    Save tensor images to disk.
    
    Args:
        images: Tensor images [B, 3, H, W]
        save_dir: Directory to save images
        prefix: Filename prefix
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    pil_images = tensor_to_pil(images)
    
    for i, img in enumerate(pil_images):
        save_path = save_dir / f"{prefix}_{i}.png"
        img.save(save_path)
    
    print(f"Saved {len(pil_images)} images to {save_dir}")


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
