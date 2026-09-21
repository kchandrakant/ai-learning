# Deep Learning: Beyond the Basics

## Beyond the Basics

### Efficient Attention
- Flash Attention: memory-efficient attention
- Linear attention: O(n) instead of O(n²)
- Sparse attention: attend to subset of positions

### Position Encoding Advances
- RoPE (Rotary Position Embedding)
- ALiBi (Attention with Linear Biases)
- Relative position encodings

### Architecture Innovations
- Pre-norm vs Post-norm
- Mixture of Experts (MoE)
- State Space Models (Mamba)

## Training Improvements

| Technique | Benefit |
|-----------|---------|
| Mixed precision (FP16/BF16) | 2x faster, less memory |
| Gradient checkpointing | Trade compute for memory |
| DeepSpeed / FSDP | Distributed training |
| LoRA | Efficient fine-tuning |

## Modern Models

### Language
- GPT series (decoder-only)
- BERT/RoBERTa (encoder-only)
- T5/BART (encoder-decoder)
- LLaMA, Mistral, Qwen

### Vision
- Vision Transformer (ViT)
- Swin Transformer
- DINO, DINOv2

### Multimodal
- CLIP (vision + language)
- Flamingo, LLaVA
- Stable Diffusion

## Resources

- "The Illustrated Transformer" (Jay Alammar)
- Andrej Karpathy's "Let's build GPT"
- Hugging Face Transformers library
- Papers With Code

## Continue Learning

→ **transformer-architecture** course: detailed implementation
→ **fine-tuning-adaptation** course: adapting models
→ **prompt-engineering-rag** course: using LLMs effectively
