#!/usr/bin/env python3
"""
Visualization script for model outputs and training progress
"""

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from PIL import Image
import numpy as np


def visualize_samples(image_dir, num_images=8):
    """
    Visualize generated samples.
    
    Args:
        image_dir: Directory containing generated images
        num_images: Number of images to display
    """
    image_dir = Path(image_dir)
    image_paths = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    
    if len(image_paths) == 0:
        print(f"No images found in {image_dir}")
        return
    
    num_images = min(num_images, len(image_paths))
    
    # Calculate grid size
    cols = 4
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, image_path in enumerate(image_paths[:num_images]):
        row = idx // cols
        col = idx % cols
        
        # Load and display image
        img = Image.open(image_path)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
        axes[row, col].set_title(image_path.name)
    
    # Hide empty subplots
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = image_dir.parent / f"{image_dir.name}_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    plt.show()


def compare_emotions(output_dir):
    """
    Compare images generated with different emotions.
    
    Args:
        output_dir: Root output directory containing emotion subdirectories
    """
    output_dir = Path(output_dir)
    emotion_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
    
    if len(emotion_dirs) == 0:
        print(f"No emotion directories found in {output_dir}")
        return
    
    fig, axes = plt.subplots(len(emotion_dirs), 4, figsize=(12, 3 * len(emotion_dirs)))
    
    for row, emotion_dir in enumerate(emotion_dirs):
        emotion_name = emotion_dir.name.replace("emotion_", "")
        image_paths = sorted(list(emotion_dir.glob("generated_*.png")))[:4]
        
        for col, image_path in enumerate(image_paths):
            if len(emotion_dirs) == 1:
                ax = axes[col]
            else:
                ax = axes[row, col]
            
            img = Image.open(image_path)
            ax.imshow(img)
            ax.axis('off')
            
            if col == 0:
                ax.set_ylabel(emotion_name.capitalize(), fontsize=12, rotation=0, 
                             ha='right', va='center')
    
    plt.tight_layout()
    
    # Save comparison
    output_path = output_dir / "emotion_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Emotion comparison saved to {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize model outputs"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["samples", "compare"],
        default="samples",
        help="Visualization mode"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="outputs",
        help="Directory containing images"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=8,
        help="Number of images to display"
    )
    
    args = parser.parse_args()
    
    if args.mode == "samples":
        visualize_samples(args.image_dir, args.num_images)
    elif args.mode == "compare":
        compare_emotions(args.image_dir)


if __name__ == "__main__":
    main()
