# Module 4: CLIP & Contrastive Learning

Learning from image-text pairs.

## Overview

CLIP revolutionized vision-language by learning from 400M image-text pairs scraped from the web.

## Key Topics

### The CLIP Insight
```
Don't train separate image/text models
Train them together to align in shared space

Matching pairs: similar embeddings
Non-matching: different embeddings
```

### Architecture
```
Image Encoder (ViT or ResNet)
    ↓
Image Embedding (512 or 768 dim)
    ↓
Cosine Similarity ← → Text Embedding
                          ↑
                   Text Encoder (Transformer)
```

### Contrastive Loss (InfoNCE)
```
For batch of N image-text pairs:
- N positive pairs (matching)
- N² - N negative pairs (non-matching)

Loss pushes positives together, negatives apart
```

### Zero-Shot Classification
```
No training needed for new categories!

1. Encode all class names as text: "a photo of a dog"
2. Encode the image
3. Find closest text embedding
4. That's the predicted class

Works for any category with a name
```

### Limitations
```
- Counting objects (bad)
- Spatial relationships (weak)
- Fine-grained distinctions (varies)
- Compositionality (limited)
- Biases from web data
```

## Exercises

1. Implement CLIP zero-shot classification
2. Build text-to-image retrieval
3. Explore CLIP's failure modes

## Key Insight

CLIP shows the power of learning from natural supervision (captions) at scale.
