# Module 7: Video Understanding

Temporal reasoning over visual sequences.

## Overview

Video is images plus time. Understanding video requires reasoning about motion, causality, and long-range dependencies.

## Key Topics

### Video Representations
```
Video: Sequence of frames (images)
- 30 fps × 10 seconds = 300 frames
- Each frame: 224×224×3 = 150K values
- Total: 45M values for 10 seconds!

Challenge: Computational explosion
```

### Temporal Modeling Approaches
```
3D Convolutions:
- Extend 2D conv to time dimension
- Learn spatiotemporal features

Video Transformers (ViViT, TimeSformer):
- Factorized attention: space then time
- Or joint space-time attention

Two-stream networks:
- RGB stream: appearance
- Optical flow stream: motion
```

### Video Transformers
```
TimeSformer approach:
- Spatial attention within frame
- Temporal attention across frames
- Factorized = efficient

ViViT:
- Tubelet embeddings (3D patches)
- Various factorization strategies
```

### Video-Language Models
```
VideoChat, Video-LLaVA:
- Sample frames from video
- Encode each frame
- LLM processes frame sequence

Challenge: Very long contexts for long videos
```

### Video Generation
```
Sora (OpenAI), Runway Gen-3:
- Diffusion extended to video
- Temporal consistency is hard
- Massive compute requirements
```

## Exercises

1. Build a simple video classifier
2. Visualize temporal attention patterns
3. Experiment with frame sampling strategies

## Key Insight

Video understanding is unsolved. Models struggle with temporal reasoning and long-range dependencies.
