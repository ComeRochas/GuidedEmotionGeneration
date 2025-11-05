# Emotion-Guided Facial Image Generation

A PyTorch project for emotion-guided facial image editing using latent diffusion models. This project implements a conditional diffusion model that can generate facial images with specific emotions or transfer emotions between existing images.

## Overview

This project combines:
- **VAE Encoder/Decoder** from Stable Diffusion v1-5 for latent space compression
- **UNet2DConditionModel** from Diffusers with emotion conditioning via cross-attention
- **ResNet18 Emotion Encoder** pretrained on ImageNet (adaptable to AffectNet)
- **DDPM Training** with denoising diffusion probabilistic models
- **Modular Architecture** designed for future extension to flow-matching methods

## Features

- ✨ Generate facial images conditioned on specific emotions
- 🎭 Transfer emotions between existing images
- 🔧 Modular design for easy extension and experimentation
- 🚀 Support for mixed precision training
- 📊 TensorBoard logging
- 💾 Checkpoint management
- 🎨 CelebA-HQ dataset support

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- At least 16GB RAM

### Setup

1. Clone the repository:
```bash
git clone https://github.com/ComeRochas/GuidedEmotionGeneration.git
cd GuidedEmotionGeneration
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

The project uses CelebA-HQ dataset. Organize your data as follows:

```
data/celeba_hq/
├── images/
│   ├── 00000.jpg
│   ├── 00001.jpg
│   └── ...
└── emotions/ (optional)
    ├── 00000.txt
    ├── 00001.txt
    └── ...
```

Update the `data_root` path in `configs/train_config.yaml` to point to your dataset.

## Project Structure

```
GuidedEmotionGeneration/
├── src/
│   ├── models/
│   │   ├── emotion_encoder.py  # ResNet18-based emotion encoder
│   │   ├── vae.py              # Stable Diffusion VAE wrapper
│   │   └── unet.py             # Emotion-conditioned UNet
│   ├── data/
│   │   └── celeba_dataset.py   # CelebA-HQ dataset loader
│   ├── training/
│   │   ├── ddpm_trainer.py     # DDPM training loop
│   │   └── sampler.py          # Sampling and inference
│   └── utils/
│       └── helpers.py          # Utility functions
├── configs/
│   ├── train_config.yaml       # Training configuration
│   └── sample_config.yaml      # Sampling configuration
├── train.py                    # Main training script
├── sample.py                   # Sampling/inference script
└── requirements.txt            # Python dependencies
```

## Usage

### Training

Train the emotion-guided diffusion model:

```bash
python train.py --config configs/train_config.yaml
```

Key training parameters (in `configs/train_config.yaml`):
- `num_epochs`: Number of training epochs
- `batch_size`: Training batch size
- `learning_rate`: Learning rate for optimizer
- `num_train_timesteps`: Number of diffusion timesteps (default: 1000)
- `mixed_precision`: Enable mixed precision training

### Inference

#### Generate Images with Specific Emotions

Generate images with a target emotion:

```bash
python sample.py --config configs/sample_config.yaml --mode sample --emotion 1
```

Emotion indices:
- 0: Neutral
- 1: Happiness
- 2: Sadness
- 3: Surprise
- 4: Fear
- 5: Disgust
- 6: Anger
- 7: Contempt

#### Emotion Transfer

Transfer emotion from a source image:

```bash
python sample.py --config configs/sample_config.yaml --mode transfer \
    --source path/to/image.jpg --emotion 1
```

## Architecture Details

### Emotion Encoder
- Based on ResNet18 pretrained on ImageNet
- Outputs 512-dimensional emotion embeddings
- Includes emotion classifier head for 8 AffectNet classes
- Backbone can be frozen during training

### VAE
- Uses Stable Diffusion v1-5 VAE
- Encodes 256×256 images to 32×32×4 latents
- Frozen during training (pretrained weights)

### UNet
- UNet2DConditionModel with cross-attention
- Conditioned on emotion embeddings
- Supports gradient checkpointing for memory efficiency
- Can be initialized from scratch or pretrained SD weights

### Training
- DDPM loss (mean squared error on predicted noise)
- AdamW optimizer with cosine learning rate schedule
- Mixed precision training support
- Gradient accumulation for large batch sizes

## Configuration

### Training Configuration (`configs/train_config.yaml`)

Adjust hyperparameters for training:
- Model architecture settings
- Training hyperparameters
- Data loading options
- Checkpoint and logging settings

### Sampling Configuration (`configs/sample_config.yaml`)

Control inference behavior:
- Number of sampling steps
- Guidance scale
- Emotion selection
- Output settings

## Future Extensions

This project is designed with modularity in mind for easy extension:

### Flow Matching
The architecture can be extended to support flow-matching methods:
- Replace `DDPMScheduler` with flow matching scheduler
- Modify training loop to use velocity prediction
- Update sampling procedure for ODE-based generation

### Advanced Features
Potential enhancements:
- Classifier-free guidance for stronger conditioning
- LoRA fine-tuning for faster adaptation
- Multi-scale emotion control
- Real-time inference optimization
- StyleGAN inversion for better image editing

## Requirements

Core dependencies:
- PyTorch >= 2.0.0
- Diffusers >= 0.21.0
- Transformers >= 4.30.0
- Pillow, NumPy, OpenCV
- TensorBoard, Weights & Biases (optional)

See `requirements.txt` for complete list.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{guidedemotion2024,
  title={Emotion-Guided Facial Image Generation},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/ComeRochas/GuidedEmotionGeneration}
}
```

## License

This project is provided as-is for research and educational purposes.

## Acknowledgments

- Stable Diffusion team for the VAE architecture
- HuggingFace Diffusers library
- AffectNet dataset for emotion recognition research
