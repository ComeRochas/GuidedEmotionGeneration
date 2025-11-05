# FAQ and Troubleshooting

Common questions and solutions for the Emotion-Guided Diffusion project.

## Installation Issues

### Q: I get "ModuleNotFoundError: No module named 'torch'"

**A:** Install PyTorch first:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Q: Installation hangs or is very slow

**A:** Try installing in batches:
```bash
pip install torch torchvision
pip install diffusers transformers accelerate
pip install -r requirements.txt
```

## Training Issues

### Q: "CUDA out of memory" error

**A:** Try these solutions in order:

1. Reduce batch size in `configs/train_config.yaml`:
   ```yaml
   training:
     batch_size: 2  # or even 1
   ```

2. Enable gradient checkpointing:
   ```yaml
   training:
     gradient_checkpointing: true
   ```

3. Reduce image size:
   ```yaml
   data:
     image_size: 128  # or 192
   ```

4. Use mixed precision:
   ```yaml
   training:
     mixed_precision: true
   ```

### Q: Training is very slow

**A:** Check these:

1. GPU utilization: `nvidia-smi`
   - Should be 80-100% during training
   - If low, increase `num_workers` or `batch_size`

2. Data loading: 
   - Ensure data is on fast storage (SSD, not HDD)
   - Increase `num_workers` to 4-8

3. Mixed precision:
   ```yaml
   training:
     mixed_precision: true
   ```

### Q: Loss is NaN or exploding

**A:** Solutions:

1. Reduce learning rate:
   ```yaml
   training:
     learning_rate: 5.0e-5  # lower from 1e-4
   ```

2. Check data normalization:
   - Images should be in [-1, 1] range
   - Verify with: `print(images.min(), images.max())`

3. Enable gradient clipping:
   ```python
   # In train.py, add:
   torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
   ```

### Q: "RuntimeError: Expected all tensors to be on the same device"

**A:** Ensure all models and data are on the same device:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)
```

## Dataset Issues

### Q: "No images found" error

**A:** Check:

1. Directory structure:
   ```
   data/celeba_hq/
   └── images/
       ├── 00000.jpg
       ├── 00001.jpg
       └── ...
   ```

2. File extensions (should be .jpg or .png)

3. Update `data_root` in config:
   ```yaml
   data:
     data_root: "/path/to/your/data"
   ```

### Q: Where can I get CelebA-HQ dataset?

**A:** Several options:

1. **Official source**: https://github.com/tkarras/progressive_growing_of_gans
2. **Kaggle**: https://www.kaggle.com/datasets/lamsimon/celebahq
3. **HuggingFace**: 
   ```python
   from datasets import load_dataset
   dataset = load_dataset('mattymchen/celeba-hq')
   ```

### Q: Can I use a different dataset?

**A:** Yes! Just organize it like CelebA-HQ:
```
your_dataset/
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

Then update `data_root` in config.

## Sampling Issues

### Q: Generated images are blurry or low quality

**A:** Try:

1. Increase inference steps:
   ```yaml
   sampling:
     num_inference_steps: 100  # up from 50
   ```

2. Train for more epochs (quality improves over time)

3. Check if model is properly loaded:
   ```python
   print(f"Checkpoint loaded from: {checkpoint_path}")
   ```

### Q: Emotion conditioning doesn't seem to work

**A:** Verify:

1. Emotion encoder is loaded correctly
2. Train for sufficient epochs (>50)
3. Try stronger emotions (happiness vs neutral)
4. Check emotion embeddings are not all zeros

### Q: "Checkpoint not found" error

**A:** Check:

1. Checkpoint path in `configs/sample_config.yaml`
2. File actually exists:
   ```bash
   ls -lh checkpoints/
   ```
3. Use correct filename (e.g., `checkpoint_epoch_100.pt`)

## Model Issues

### Q: VAE fails to load with authentication error

**A:** Login to HuggingFace:
```bash
huggingface-cli login
```

Or use local cache:
```python
vae = VAEWrapper(model_id="runwayml/stable-diffusion-v1-5", use_auth_token=False)
```

### Q: UNet initialization is very slow

**A:** This is normal for first time (downloading weights). Subsequent runs will be fast due to caching.

### Q: How much VRAM do I need?

**A:** Minimum requirements:

- **Training**: 11GB (with batch_size=8, image_size=256)
- **Inference**: 6GB (with batch_size=1)

For less VRAM:
- Reduce batch size
- Enable gradient checkpointing
- Use smaller image size

## Performance Issues

### Q: How long does training take?

**A:** Typical times:

- **100 epochs**, 10K images, V100: ~2-3 days
- **100 epochs**, 10K images, RTX 3090: ~2-3 days
- **100 epochs**, 10K images, A100: ~1-2 days

### Q: Can I speed up training?

**A:** Yes:

1. Use larger batch size (if memory allows)
2. Enable mixed precision
3. Use multiple GPUs (requires code modification)
4. Reduce number of timesteps

### Q: Can I speed up inference?

**A:** Yes:

1. Reduce inference steps:
   ```yaml
   sampling:
     num_inference_steps: 25  # down from 50
   ```

2. Batch process multiple images
3. Use model distillation (advanced)

## Configuration Issues

### Q: How do I change the emotion classes?

**A:** Modify `src/models/emotion_encoder.py`:

```python
@staticmethod
def get_emotion_names():
    return ["emotion1", "emotion2", ...]  # your emotions
```

And update `num_emotions` in config.

### Q: Can I use different image sizes?

**A:** Yes, but must be divisible by 8 (VAE requirement):
```yaml
data:
  image_size: 256  # or 128, 192, 384, 512
```

### Q: What's the best learning rate?

**A:** Depends on your setup:

- **From scratch**: 1e-4
- **Fine-tuning**: 5e-5 to 1e-5
- **Large batch**: 1e-4 to 5e-4

## Advanced Questions

### Q: Can I fine-tune on my own dataset?

**A:** Yes! 

1. Organize your dataset
2. (Optional) Start from pretrained checkpoint
3. Adjust learning rate to 1e-5 or lower
4. Train for fewer epochs

### Q: How do I implement flow matching?

**A:** Key changes needed:

1. Replace `DDPMScheduler` with flow scheduler
2. Change loss from noise to velocity prediction
3. Update sampling loop for ODE solving

See ARCHITECTURE.md for details.

### Q: Can I add more conditioning signals?

**A:** Yes! Modify UNet to accept additional inputs:

1. Add new encoder for your condition
2. Concatenate with emotion embeddings
3. Update cross-attention dimension

### Q: How do I evaluate model quality?

**A:** Metrics:

1. **FID** (Fréchet Inception Distance): Image quality
2. **Emotion accuracy**: Classify generated images
3. **Human evaluation**: Visual quality assessment

## Debugging Tips

### Enable Debug Mode

Add to your script:
```python
import torch
torch.autograd.set_detect_anomaly(True)
```

### Check Model Outputs

```python
# Print shapes at each stage
print(f"Input: {images.shape}")
print(f"Latents: {latents.shape}")
print(f"Embeddings: {embeddings.shape}")
print(f"Noise pred: {noise_pred.shape}")
```

### Visualize Intermediate Results

```python
from src.utils import save_images

# Save latents as images (for debugging)
save_images(latents, "debug/latents")

# Save noise predictions
save_images(noise_pred, "debug/noise")
```

### Monitor GPU Memory

```python
import torch

print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## Getting Help

If you're still stuck:

1. Check existing GitHub issues
2. Run `test_setup.py` to verify installation
3. Try with synthetic dataset first
4. Enable verbose logging
5. Share error traceback when asking for help

## Additional Resources

- **Diffusers docs**: https://huggingface.co/docs/diffusers
- **PyTorch docs**: https://pytorch.org/docs
- **DDPM paper**: https://arxiv.org/abs/2006.11239
- **Stable Diffusion**: https://github.com/CompVis/stable-diffusion

## Contributing

Found a bug or have a solution? Please contribute:

1. Fork the repository
2. Create a branch
3. Submit a pull request

See LICENSE for details.
