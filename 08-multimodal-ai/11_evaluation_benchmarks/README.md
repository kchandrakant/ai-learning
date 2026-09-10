# Module 11: Evaluation & Benchmarks

Measuring multimodal capabilities.

## Overview

Multimodal evaluation is complex — we must measure both understanding and generation quality.

## Key Topics

### Vision-Language Benchmarks
```
VQA (Visual Question Answering):
- VQAv2: Real images, open-ended questions
- GQA: Compositional reasoning
- TextVQA: Reading text in images
- OKVQA: Outside knowledge needed

Document Understanding:
- DocVQA: Document questions
- ChartQA: Chart interpretation
- InfographicVQA: Infographic questions
```

### Image Generation Metrics
```
FID (Fréchet Inception Distance):
- Compare generated vs real image distributions
- Lower is better
- Standard metric, but not perfect

Inception Score:
- Quality and diversity
- Higher is better

CLIP Score:
- Text-image alignment
- Does generated image match prompt?

Human Evaluation:
- Gold standard
- Expensive and slow
```

### Hallucination Detection
```
VLMs hallucinate:
- Objects not in image
- Wrong spatial relationships
- Fabricated text in images

POPE, CHAIR benchmarks measure this
```

### Video Benchmarks
```
- ActivityNet: Activity recognition
- Kinetics: Human actions
- MSRVTT: Video captioning
- Video-QA datasets
```

## Exercises

1. Evaluate a VLM on VQA benchmarks
2. Compute FID for generated images
3. Test for hallucinations in VLM responses

## Key Insight

No single metric captures multimodal quality. Use multiple benchmarks and human evaluation.
