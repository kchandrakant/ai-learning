# Module 2: Image Encoders

How models see images.

## Overview

Image encoders convert pixel grids into feature vectors that can be processed by transformers.

## Key Topics

### CNN-based Encoders
```
ResNet, EfficientNet:
- Convolutional layers extract spatial features
- Pooling reduces dimensions
- Output: feature vector or grid

Still used for efficiency in some applications
```

### Vision Transformer (ViT)
```
The dominant approach:
1. Split image into patches (16x16)
2. Flatten each patch
3. Linear projection to embedding dimension
4. Add position embeddings
5. Process with transformer encoder

Image (224x224) → 196 patches → Transformer → Features
```

### Patch Embeddings
```
Image patch → Linear layer → Embedding

Each patch becomes a "token" like in text
Position embeddings tell model where each patch is
[CLS] token aggregates global information
```

### Pre-training Strategies
```
Supervised (ImageNet): Classify 1000 categories
Self-supervised (DINO, MAE): Predict masked patches
Contrastive (CLIP): Align with text descriptions
```

## Exercises

1. Implement patch embedding from scratch
2. Visualize ViT attention patterns
3. Compare CNN vs ViT features

## Key Insight

ViT's success shows: transformers work for images too, if you tokenize properly.
