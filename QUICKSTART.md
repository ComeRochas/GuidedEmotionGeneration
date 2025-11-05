# Quick Start Guide

This guide will help you get started with training and using the emotion-guided diffusion model.

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare your dataset:**
   - Download CelebA-HQ dataset
   - Organize images in `data/celeba_hq/images/` directory
   - Update `data_root` in `configs/train_config.yaml`

## Training

### Quick Training Test

To verify the setup works, you can do a quick test with minimal settings:

1. Edit `configs/train_config.yaml`:
   - Set `num_epochs: 2`
   - Set `batch_size: 2`

2. Run training:
   ```bash
   python train.py --config configs/train_config.yaml
   ```

### Full Training

For production training:

1. Ensure you have sufficient GPU memory (16GB+ recommended)
2. Use default config or adjust based on your hardware:
   ```bash
   python train.py --config configs/train_config.yaml
   ```

Training progress will be saved in:
- Checkpoints: `./checkpoints/`
- Logs: `./logs/`

## Inference

### Generate Images

Generate images with a specific emotion:

```bash
# Generate happy faces
python sample.py --mode sample --emotion 1

# Generate sad faces  
python sample.py --mode sample --emotion 2

# Generate surprised faces
python sample.py --mode sample --emotion 3
```

### Emotion Transfer

Transfer emotion to an existing image:

```bash
python sample.py --mode transfer \
    --source path/to/your/image.jpg \
    --emotion 1
```

## Emotion Labels

The model supports 8 emotions (based on AffectNet):

| Index | Emotion  |
|-------|----------|
| 0     | Neutral  |
| 1     | Happiness|
| 2     | Sadness  |
| 3     | Surprise |
| 4     | Fear     |
| 5     | Disgust  |
| 6     | Anger    |
| 7     | Contempt |

## Configuration Tips

### For Limited GPU Memory

In `configs/train_config.yaml`:
- Reduce `batch_size` (e.g., 2 or 4)
- Enable `gradient_checkpointing: true`
- Reduce `image_size` to 128 or 192

### For Faster Training

- Increase `batch_size` if you have GPU memory
- Use `mixed_precision: true`
- Increase `num_workers` for data loading

### For Better Quality

- Train for more epochs (100+)
- Use larger `image_size` (256 or 512)
- Increase `num_inference_steps` during sampling

## Monitoring Training

### TensorBoard

View training progress:
```bash
tensorboard --logdir logs/
```

Then open http://localhost:6006 in your browser.

### Weights & Biases (Optional)

Enable in `configs/train_config.yaml`:
```yaml
logging:
  use_wandb: true
  wandb_project: "emotion-guided-diffusion"
```

## Troubleshooting

### Out of Memory Error
- Reduce batch size
- Enable gradient checkpointing
- Reduce image size
- Use mixed precision training

### Training is Slow
- Increase number of workers
- Check GPU utilization
- Ensure data is on fast storage (SSD)

### Poor Quality Results
- Train for more epochs
- Check if VAE loaded correctly
- Verify dataset quality
- Adjust learning rate

## Next Steps

1. **Fine-tune emotion encoder**: Train on AffectNet dataset for better emotion understanding
2. **Experiment with guidance**: Adjust `guidance_scale` in sampling config
3. **Try different architectures**: Modify UNet configuration
4. **Implement flow matching**: Extend to use continuous normalizing flows

For more details, see the main [README.md](README.md).
