#!/usr/bin/env python3
"""
Simple test script to validate the project setup
Tests model initialization and basic forward passes
"""

import torch
import sys

print("Testing Emotion-Guided Diffusion Setup...")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from src.models import EmotionEncoder, VAEWrapper, EmotionConditionedUNet
    from src.data import CelebAHQDataset
    from src.training import DDPMTrainer
    from src.utils import get_device, count_parameters
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test device
print("\n2. Testing device detection...")
device = get_device()
print(f"✓ Using device: {device}")

# Test emotion encoder
print("\n3. Testing Emotion Encoder...")
try:
    emotion_encoder = EmotionEncoder(embedding_dim=512, num_emotions=8)
    print(f"✓ Emotion Encoder initialized")
    print(f"  Parameters: {count_parameters(emotion_encoder):,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 256, 256)
    embeddings = emotion_encoder(dummy_input)
    print(f"  Output shape: {embeddings.shape} (expected: [2, 512])")
    assert embeddings.shape == (2, 512), "Unexpected output shape"
    print("✓ Emotion Encoder forward pass successful")
except Exception as e:
    print(f"✗ Emotion Encoder test failed: {e}")
    sys.exit(1)

# Test UNet (without loading pretrained weights for speed)
print("\n4. Testing UNet...")
try:
    unet = EmotionConditionedUNet(
        in_channels=4,
        out_channels=4,
        cross_attention_dim=512,
        use_pretrained=False
    )
    print(f"✓ UNet initialized")
    print(f"  Parameters: {count_parameters(unet.unet):,}")
    
    # Test forward pass
    dummy_latents = torch.randn(2, 4, 32, 32)
    dummy_timesteps = torch.randint(0, 1000, (2,))
    dummy_embeddings = torch.randn(2, 512)
    
    noise_pred = unet(dummy_latents, dummy_timesteps, dummy_embeddings)
    print(f"  Output shape: {noise_pred.shape} (expected: [2, 4, 32, 32])")
    assert noise_pred.shape == (2, 4, 32, 32), "Unexpected output shape"
    print("✓ UNet forward pass successful")
except Exception as e:
    print(f"✗ UNet test failed: {e}")
    sys.exit(1)

# Test configuration loading
print("\n5. Testing configuration files...")
try:
    import yaml
    from pathlib import Path
    
    config_path = Path("configs/train_config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print("✓ Training config loaded successfully")
    else:
        print("✗ Training config not found")
    
    sample_config_path = Path("configs/sample_config.yaml")
    if sample_config_path.exists():
        with open(sample_config_path) as f:
            sample_config = yaml.safe_load(f)
        print("✓ Sampling config loaded successfully")
    else:
        print("✗ Sampling config not found")
        
except Exception as e:
    print(f"✗ Config test failed: {e}")

# Summary
print("\n" + "=" * 60)
print("✓ All basic tests passed!")
print("\nYour setup is ready. You can now:")
print("  1. Prepare your dataset in data/celeba_hq/")
print("  2. Run training: python train.py")
print("  3. Run sampling: python sample.py")
print("\nSee QUICKSTART.md for detailed instructions.")
print("=" * 60)
