# Module 5: Image Generation Foundations

The evolution of generative models.

## Overview

Before diffusion, other approaches dominated. Understanding them reveals why diffusion won.

## Key Topics

### GANs (Generative Adversarial Networks)
```
Two networks competing:
- Generator: Creates fake images
- Discriminator: Distinguishes real vs fake

Training: Generator improves to fool discriminator

Problems: Mode collapse, training instability
```

### VAEs (Variational Autoencoders)
```
Encoder → Latent space → Decoder

Learn compressed representation
Can sample from latent space to generate

Problems: Blurry outputs (averaging effect)
```

### Autoregressive Models
```
Generate pixels one by one
- PixelCNN: Predict each pixel from previous
- ImageGPT: Transformer on pixel sequences

Problems: Slow generation, limited resolution
```

### Why Diffusion Won
```
Diffusion advantages:
- Stable training (no adversarial dynamics)
- High quality output (no blurring)
- Flexible conditioning (text, images, etc.)
- Controllable generation
```

## Exercises

1. Train a simple GAN on MNIST
2. Implement a VAE and visualize latent space
3. Compare generation quality across approaches

## Key Insight

Diffusion combined the stability of likelihood-based models with the quality of GANs.
