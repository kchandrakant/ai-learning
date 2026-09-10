# Module 3: Vision-Language Models

Combining sight and language.

## Overview

Vision-language models understand both images and text, enabling visual question answering, image captioning, and more.

## Key Topics

### Architecture Patterns
```
Pattern 1: Concatenation
- Image tokens + text tokens
- Both processed by same transformer
- GPT-4V, Gemini approach

Pattern 2: Cross-attention
- Separate image/text encoders
- Text attends to image features
- Flamingo approach

Pattern 3: Projection (LLaVA)
- Frozen vision encoder
- Projection layer to LLM space
- LLM processes visual tokens
```

### LLaVA Architecture
```
Image → CLIP ViT → MLP Projection → [Visual Tokens]
                                          ↓
Text  → Tokenizer → [Text Tokens] + [Visual Tokens] → LLM → Response
```

### Visual Tokens
```
How many tokens per image?
- Low res: 85 tokens
- High res: 1000+ tokens
- Trade-off: detail vs context length vs cost
```

### Capabilities
- Visual Question Answering
- Image Captioning
- OCR / Document Understanding
- Visual Reasoning
- Image-guided Generation

## Exercises

1. Run a VLM on various images
2. Analyze visual token representations
3. Test VLM limitations (counting, spatial reasoning)

## Key Insight

VLMs extend LLM capabilities to vision, but inherit LLM limitations too.
