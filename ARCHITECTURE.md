# Architecture Documentation

## Overview

This document describes the architecture and design choices for the Emotion-Guided Facial Image Generation project.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Emotion-Guided Diffusion                 │
│                                                             │
│  ┌─────────────────┐      ┌──────────────┐               │
│  │  Input Image    │──────▶│   VAE        │               │
│  │  [B, 3, 256²]   │      │   Encoder    │               │
│  └─────────────────┘      └──────┬───────┘               │
│           │                       │                        │
│           │                       ▼                        │
│           │              ┌──────────────┐                 │
│           │              │   Latents    │                 │
│           │              │  [B, 4, 32²] │                 │
│           │              └──────┬───────┘                 │
│           │                     │                          │
│           ▼                     │                          │
│  ┌─────────────────┐           │                          │
│  │    Emotion      │           │                          │
│  │    Encoder      │           │                          │
│  │   (ResNet18)    │           │                          │
│  └────────┬────────┘           │                          │
│           │                     │                          │
│           ▼                     │                          │
│  ┌─────────────────┐           │                          │
│  │   Embeddings    │           │                          │
│  │   [B, 512]      │───────┐   │                          │
│  └─────────────────┘       │   │                          │
│                             │   │                          │
│                             ▼   ▼                          │
│                    ┌──────────────────┐                   │
│                    │      UNet        │                   │
│                    │  (Cross-Attn)    │                   │
│                    │  Conditioning    │                   │
│                    └────────┬─────────┘                   │
│                             │                              │
│                             ▼                              │
│                    ┌──────────────────┐                   │
│                    │  Denoised        │                   │
│                    │  Latents         │                   │
│                    └────────┬─────────┘                   │
│                             │                              │
│                             ▼                              │
│                    ┌──────────────────┐                   │
│                    │   VAE Decoder    │                   │
│                    └────────┬─────────┘                   │
│                             │                              │
│                             ▼                              │
│                    ┌──────────────────┐                   │
│                    │  Output Image    │                   │
│                    │  [B, 3, 256²]    │                   │
│                    └──────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. VAE (Variational Autoencoder)

**Source**: Stable Diffusion v1-5  
**Purpose**: Compress images to latent space for efficient diffusion  

**Architecture**:
- Encoder: RGB image (256×256) → Latent (32×32×4)
- Decoder: Latent (32×32×4) → RGB image (256×256)
- Compression ratio: 8×8 spatial, 0.75× channel

**Design Choices**:
- Use pretrained SD v1-5 VAE (no training needed)
- Frozen weights during training
- Provides stable latent space
- 8× spatial compression reduces computation

**Implementation**: `src/models/vae.py`

### 2. Emotion Encoder

**Base Model**: ResNet18 (ImageNet pretrained)  
**Purpose**: Extract emotion embeddings for conditioning  

**Architecture**:
- Backbone: ResNet18 (11M parameters)
- Projection head: Linear → ReLU → Dropout → Linear
- Output: 512-dimensional embeddings

**Design Choices**:
- ResNet18 chosen for efficiency and proven performance
- Pretrained on ImageNet, adaptable to AffectNet
- Backbone can be frozen to reduce training cost
- Separate classifier head for emotion classification
- Embedding projection for diffusion conditioning

**Emotions Supported** (AffectNet standard):
1. Neutral
2. Happiness
3. Sadness
4. Surprise
5. Fear
6. Disgust
7. Anger
8. Contempt

**Implementation**: `src/models/emotion_encoder.py`

### 3. UNet

**Base Model**: UNet2DConditionModel (Diffusers)  
**Purpose**: Denoise latents conditioned on emotion  

**Architecture**:
- Input channels: 4 (latent space)
- Output channels: 4 (denoised latent)
- Cross-attention: 512-dimensional emotion embeddings
- Blocks: 4 down, 4 up with skip connections
- Total parameters: ~853M

**Design Choices**:
- Cross-attention mechanism for emotion conditioning
- Can initialize from scratch or use pretrained SD weights
- Gradient checkpointing for memory efficiency
- Standard DDPM noise prediction objective

**Conditioning Method**:
- Emotion embeddings reshaped to [B, 1, 512]
- Passed as `encoder_hidden_states` to cross-attention
- Each attention block attends to emotion features

**Implementation**: `src/models/unet.py`

### 4. DDPM Training

**Algorithm**: Denoising Diffusion Probabilistic Models  
**Loss**: Mean Squared Error on predicted noise  

**Process**:
1. Encode image to latents using VAE
2. Extract emotion embeddings
3. Sample noise and timestep
4. Add noise to latents
5. Predict noise with UNet
6. Compute MSE loss

**Training Details**:
- Timesteps: 1000 (standard DDPM)
- Beta schedule: Linear
- Optimizer: AdamW with cosine schedule
- Mixed precision: FP16 for efficiency
- Gradient accumulation: Optional for large batches

**Implementation**: `src/training/ddpm_trainer.py`

### 5. Sampling

**Algorithm**: DDPM sampling with emotion guidance  
**Steps**: 50 (configurable)  

**Generation Modes**:

1. **Unconditional**: Random emotion embedding
2. **Class-guided**: Specific emotion index (0-7)
3. **Image-guided**: Extract emotion from reference image
4. **Emotion Transfer**: Edit existing image

**Guidance**:
- Optional classifier-free guidance
- Guidance scale controls conditioning strength
- Default: 1.0 (no guidance)

**Implementation**: `src/training/sampler.py`

## Data Pipeline

**Dataset**: CelebA-HQ (or compatible)  
**Format**: JPG/PNG images, optional emotion labels  

**Preprocessing**:
1. Resize to target size (default 256×256)
2. Center crop
3. Convert to tensor
4. Normalize to [-1, 1]

**DataLoader**:
- Batch size: Configurable (default 8)
- Num workers: 4
- Pin memory: Enabled
- Shuffle: Training only

**Implementation**: `src/data/celeba_dataset.py`

## Modular Design for Extensions

### Flow Matching Extension

The architecture is designed to easily support flow-matching methods:

**Required Changes**:
1. Replace `DDPMScheduler` with flow-matching scheduler
2. Change prediction target from noise to velocity
3. Update loss function to velocity matching
4. Modify sampling to use ODE solver

**Files to Modify**:
- `src/training/ddpm_trainer.py` → `src/training/flow_trainer.py`
- `src/training/sampler.py` → Update sampling loop

**No Changes Needed**:
- Emotion encoder (same conditioning)
- VAE (same latent space)
- UNet architecture (same backbone)

### Other Potential Extensions

1. **Classifier-Free Guidance**
   - Train with random dropout of condition
   - Interpolate between conditional/unconditional

2. **LoRA Fine-tuning**
   - Add low-rank adaptation layers
   - Fast adaptation to new domains

3. **Multi-scale Conditioning**
   - Condition at multiple resolutions
   - Better detail preservation

4. **Real-time Inference**
   - Distillation to fewer steps
   - Model quantization
   - TensorRT optimization

## Performance Considerations

### Memory Usage

**Training** (batch_size=8, image_size=256):
- UNet: ~6GB
- Emotion encoder: ~200MB
- VAE (frozen): ~400MB
- Activations: ~4GB
- **Total**: ~11GB

**Optimization Strategies**:
- Gradient checkpointing: -40% memory
- Mixed precision: -30% memory
- Smaller batch size: Linear scaling
- Freeze emotion encoder backbone: -50% gradients

### Compute Requirements

**Training Time** (single GPU):
- V100 (16GB): ~2-3 days for 100 epochs
- A100 (40GB): ~1-2 days for 100 epochs
- RTX 3090 (24GB): ~2-3 days for 100 epochs

**Inference Time**:
- 50 steps: ~5-10 seconds per image
- 25 steps: ~2-5 seconds per image
- Batch processing: Near-linear scaling

## Configuration System

### Training Config (`configs/train_config.yaml`)

```yaml
model:
  - emotion_encoder settings
  - vae settings  
  - unet settings

training:
  - hyperparameters
  - optimizer config
  - scheduler config

data:
  - dataset path
  - preprocessing options

checkpoint:
  - save frequency
  - retention policy

logging:
  - tensorboard/wandb
  - log frequency
```

### Sampling Config (`configs/sample_config.yaml`)

```yaml
model:
  - checkpoint path
  - model IDs

sampling:
  - num steps
  - guidance scale
  - emotion selection

output:
  - save directory
  - num samples
```

## Testing Strategy

### Unit Tests
- Model initialization: `test_setup.py`
- Forward passes: `test_setup.py`
- Config loading: `test_setup.py`

### Integration Tests
- Dataset loading: Synthetic dataset
- Training loop: Mock training
- Sampling: Mock inference

### Manual Testing
- Full training run with small dataset
- Sample generation with various emotions
- Emotion transfer on test images

## Future Work

1. **AffectNet Pre-training**
   - Train emotion encoder on AffectNet
   - Better emotion representation

2. **Larger Models**
   - Scale up UNet (more blocks, channels)
   - Larger emotion encoder (ResNet50)

3. **Advanced Conditioning**
   - Multi-emotion blending
   - Intensity control
   - Spatial emotion editing

4. **Quality Improvements**
   - Perceptual loss
   - Adversarial training
   - Progressive growing

## References

1. Stable Diffusion: https://github.com/CompVis/stable-diffusion
2. Diffusers: https://github.com/huggingface/diffusers
3. DDPM: https://arxiv.org/abs/2006.11239
4. AffectNet: http://mohammadmahoor.com/affectnet/

## Contributors

See LICENSE file for contribution guidelines.
