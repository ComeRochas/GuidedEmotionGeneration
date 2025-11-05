"""
CelebA-HQ Dataset Loader
Handles loading and preprocessing CelebA-HQ images for training
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CelebAHQDataset(Dataset):
    """
    CelebA-HQ dataset loader for emotion-guided diffusion training.
    
    Expected directory structure:
        data_root/
            images/
                00000.jpg
                00001.jpg
                ...
            (optional) emotions/
                00000.txt  # emotion label
                00001.txt
                ...
    """
    
    def __init__(
        self,
        data_root: str,
        image_size: int = 256,
        split: str = "train",
        transform: Optional[Callable] = None,
        return_emotion_labels: bool = False
    ):
        """
        Args:
            data_root: Root directory containing CelebA-HQ images
            image_size: Target image size for resizing
            split: Dataset split ('train', 'val', 'test')
            transform: Optional custom transform
            return_emotion_labels: Whether to return emotion labels if available
        """
        self.data_root = Path(data_root)
        self.image_size = image_size
        self.split = split
        self.return_emotion_labels = return_emotion_labels
        
        # Find images directory
        self.image_dir = self.data_root / "images"
        if not self.image_dir.exists():
            self.image_dir = self.data_root
        
        # Get all image paths
        self.image_paths = sorted(list(self.image_dir.glob("*.jpg")) + 
                                 list(self.image_dir.glob("*.png")))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.image_dir}")
        
        # Split dataset (simple split based on indices)
        total_images = len(self.image_paths)
        train_size = int(0.8 * total_images)
        val_size = int(0.1 * total_images)
        
        if split == "train":
            self.image_paths = self.image_paths[:train_size]
        elif split == "val":
            self.image_paths = self.image_paths[train_size:train_size + val_size]
        elif split == "test":
            self.image_paths = self.image_paths[train_size + val_size:]
        
        # Check for emotion labels
        self.emotion_dir = self.data_root / "emotions"
        self.has_emotions = self.emotion_dir.exists() and return_emotion_labels
        
        # Define transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # to [-1, 1]
            ])
        
        print(f"Loaded {len(self.image_paths)} images for {split} split")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """
        Get a sample from the dataset.
        
        Returns:
            image: Preprocessed image tensor [3, H, W]
            emotion_label (optional): Emotion class label if available
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms
        image = self.transform(image)
        
        if self.has_emotions:
            # Try to load emotion label
            emotion_file = self.emotion_dir / f"{image_path.stem}.txt"
            if emotion_file.exists():
                with open(emotion_file, 'r') as f:
                    emotion_label = int(f.read().strip())
            else:
                # Default to neutral if no label found
                emotion_label = 0
            
            return image, emotion_label
        else:
            # Return default emotion label (neutral) even when not loading labels
            return image, 0
    
    def get_sample_images(self, num_samples: int = 4):
        """
        Get a batch of sample images for visualization.
        
        Args:
            num_samples: Number of samples to return
            
        Returns:
            images: Batch of images [num_samples, 3, H, W]
        """
        indices = torch.randperm(len(self))[:num_samples]
        images = []
        
        for idx in indices:
            item = self[idx]
            images.append(item[0])
        
        return torch.stack(images)


def get_dataloader(
    data_root: str,
    batch_size: int = 8,
    image_size: int = 256,
    num_workers: int = 4,
    split: str = "train",
    shuffle: bool = True,
    return_emotion_labels: bool = False
):
    """
    Create a DataLoader for CelebA-HQ dataset.
    
    Args:
        data_root: Root directory containing CelebA-HQ images
        batch_size: Batch size
        image_size: Target image size
        num_workers: Number of data loading workers
        split: Dataset split ('train', 'val', 'test')
        shuffle: Whether to shuffle the data
        return_emotion_labels: Whether to return emotion labels
        
    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = CelebAHQDataset(
        data_root=data_root,
        image_size=image_size,
        split=split,
        return_emotion_labels=return_emotion_labels
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if split == "train" else False
    )
    
    return dataloader
