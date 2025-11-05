#!/usr/bin/env python3
"""
Create a small synthetic dataset for testing the pipeline
"""

import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random


def create_synthetic_face(image_size=256, emotion_idx=0):
    """
    Create a simple synthetic face image.
    
    Args:
        image_size: Size of the image
        emotion_idx: Emotion index for color coding
        
    Returns:
        PIL Image
    """
    # Create colored background based on emotion
    colors = [
        (200, 200, 200),  # neutral - gray
        (255, 255, 150),  # happiness - yellow
        (150, 150, 255),  # sadness - blue
        (255, 200, 150),  # surprise - orange
        (200, 150, 255),  # fear - purple
        (150, 255, 150),  # disgust - green
        (255, 150, 150),  # anger - red
        (180, 150, 200),  # contempt - lavender
    ]
    
    color = colors[emotion_idx % len(colors)]
    img = Image.new('RGB', (image_size, image_size), color)
    draw = ImageDraw.Draw(img)
    
    # Draw simple face
    center_x, center_y = image_size // 2, image_size // 2
    
    # Face circle
    face_radius = image_size // 3
    draw.ellipse(
        [(center_x - face_radius, center_y - face_radius),
         (center_x + face_radius, center_y + face_radius)],
        fill=(250, 220, 180),
        outline=(100, 100, 100),
        width=2
    )
    
    # Eyes
    eye_y = center_y - face_radius // 3
    eye_offset = face_radius // 3
    eye_radius = face_radius // 8
    
    # Left eye
    draw.ellipse(
        [(center_x - eye_offset - eye_radius, eye_y - eye_radius),
         (center_x - eye_offset + eye_radius, eye_y + eye_radius)],
        fill=(0, 0, 0)
    )
    
    # Right eye
    draw.ellipse(
        [(center_x + eye_offset - eye_radius, eye_y - eye_radius),
         (center_x + eye_offset + eye_radius, eye_y + eye_radius)],
        fill=(0, 0, 0)
    )
    
    # Mouth (varies with emotion)
    mouth_y = center_y + face_radius // 3
    mouth_width = face_radius // 2
    
    if emotion_idx == 1:  # happiness
        draw.arc(
            [(center_x - mouth_width, mouth_y - 20),
             (center_x + mouth_width, mouth_y + 20)],
            start=0, end=180, fill=(0, 0, 0), width=3
        )
    elif emotion_idx == 2:  # sadness
        draw.arc(
            [(center_x - mouth_width, mouth_y - 40),
             (center_x + mouth_width, mouth_y)],
            start=180, end=0, fill=(0, 0, 0), width=3
        )
    else:  # neutral or others
        draw.line(
            [(center_x - mouth_width, mouth_y),
             (center_x + mouth_width, mouth_y)],
            fill=(0, 0, 0), width=3
        )
    
    return img


def create_synthetic_dataset(output_dir, num_images=100, image_size=256):
    """
    Create a synthetic dataset for testing.
    
    Args:
        output_dir: Directory to save the dataset
        num_images: Number of images to generate
        image_size: Size of images
    """
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    emotion_dir = output_dir / "emotions"
    
    image_dir.mkdir(parents=True, exist_ok=True)
    emotion_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating {num_images} synthetic images...")
    
    for i in range(num_images):
        # Random emotion
        emotion_idx = random.randint(0, 7)
        
        # Create image
        img = create_synthetic_face(image_size, emotion_idx)
        
        # Save image
        img.save(image_dir / f"{i:05d}.jpg", quality=95)
        
        # Save emotion label
        with open(emotion_dir / f"{i:05d}.txt", 'w') as f:
            f.write(str(emotion_idx))
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_images} images")
    
    print(f"\nDataset created at {output_dir}")
    print(f"  Images: {len(list(image_dir.glob('*.jpg')))}")
    print(f"  Emotion labels: {len(list(emotion_dir.glob('*.txt')))}")


def main():
    parser = argparse.ArgumentParser(
        description="Create synthetic dataset for testing"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/synthetic_test",
        help="Directory to save the synthetic dataset"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=100,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Size of images"
    )
    
    args = parser.parse_args()
    
    create_synthetic_dataset(args.output_dir, args.num_images, args.image_size)


if __name__ == "__main__":
    main()
