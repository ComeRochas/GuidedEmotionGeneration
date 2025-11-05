#!/usr/bin/env python3
"""
Script to download and prepare CelebA-HQ dataset
Note: This is a placeholder script. Actual implementation would require
authentication and proper dataset access.
"""

import argparse
from pathlib import Path


def download_celeba_hq(output_dir):
    """
    Download CelebA-HQ dataset.
    
    Args:
        output_dir: Directory to save the dataset
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("CelebA-HQ Dataset Preparation")
    print("=" * 60)
    print("\nNOTE: This is a placeholder script.")
    print("\nTo obtain CelebA-HQ dataset, you can:")
    print("\n1. Official CelebA-HQ:")
    print("   - Visit: https://github.com/tkarras/progressive_growing_of_gans")
    print("   - Follow instructions to create CelebA-HQ from CelebA")
    print("\n2. Pre-processed versions:")
    print("   - Kaggle: https://www.kaggle.com/datasets/lamsimon/celebahq")
    print("   - HuggingFace: https://huggingface.co/datasets/mattymchen/celeba-hq")
    print("\n3. Using HuggingFace datasets:")
    print("   pip install datasets")
    print("   from datasets import load_dataset")
    print("   dataset = load_dataset('mattymchen/celeba-hq')")
    print("\nAfter downloading, organize the dataset as:")
    print(f"  {output_dir}/")
    print("    ├── images/")
    print("    │   ├── 00000.jpg")
    print("    │   ├── 00001.jpg")
    print("    │   └── ...")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Download CelebA-HQ dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/celeba_hq",
        help="Directory to save the dataset"
    )
    
    args = parser.parse_args()
    
    download_celeba_hq(args.output_dir)


if __name__ == "__main__":
    main()
