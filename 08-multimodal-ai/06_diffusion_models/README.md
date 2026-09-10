# Module 6: Diffusion Models

The architecture behind DALL-E, Midjourney, Stable Diffusion.

## Overview

Diffusion models learn to denoise, then use that skill to generate from pure noise.

## Key Topics

### Forward Diffusion
```
Gradually add noise over T steps:
x_t = √(α_t) * x_0 + √(1-α_t) * ε

After many steps: x_T ≈ pure noise
```

### Reverse Diffusion
```
Learn to predict and remove noise:
model(x_t, t) → predicts ε (the noise)

x_{t-1} = denoise(x_t, predicted_ε)

Repeat T times: noise → image
```

### U-Net Architecture
```
Encoder (downsample) → Bottleneck → Decoder (upsample)
With skip connections at each level

Processes image at multiple resolutions
Time embedding injected at each layer
```

### Latent Diffusion (Stable Diffusion)
```
Problem: Diffusion in pixel space is expensive
Solution: Diffuse in compressed latent space

VAE Encoder → Latent → Diffusion → Latent → VAE Decoder
             (64x64)            (64x64)

Much faster, same quality
```

### Text Conditioning
```
How text controls generation:
- Text → CLIP encoder → embedding
- Cross-attention: image features attend to text
- CFG: Classifier-Free Guidance amplifies text alignment
```

### ControlNet
```
Add spatial control (pose, edges, depth):
- Copy U-Net encoder
- Train on condition + image pairs
- Zero convolutions for stability
```

## Exercises

1. Implement simple DDPM on small images
2. Visualize denoising process step by step
3. Experiment with CFG scale

## Key Insight

Diffusion reframes generation as iterative refinement, making it stable and controllable.
