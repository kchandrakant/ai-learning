# Transformer Research: Key Papers & References

A curated collection of the most influential papers in transformer architecture evolution, from the original 2017 paper to the frontier research of 2026.

---

## 📚 How to Use This Document

- **🔧 Hands-on**: Papers we implement in this learning path
- **📖 Reading**: Important papers for conceptual understanding
- **🔬 Frontier**: Cutting-edge research, still evolving

---

## The Original (2017)

### Attention Is All You Need
**Vaswani et al., Google Brain, June 2017**

The paper that started it all. Introduced the transformer architecture, replacing recurrence with self-attention.

- **Key innovations**: Self-attention, multi-head attention, positional encoding
- **Impact**: Foundation for BERT, GPT, and all modern LLMs
- **Link**: [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- **Status**: 🔧 Hands-on (Steps 1-8 of this learning path)

---

## Era 1: Foundation Models (2018-2019)

### GPT: Improving Language Understanding by Generative Pre-Training
**Radford et al., OpenAI, June 2018**

First demonstration that pre-training a transformer decoder on large text corpora creates powerful general-purpose representations.

- **Key innovations**: Decoder-only architecture, unsupervised pre-training + supervised fine-tuning
- **Impact**: Established the pre-training paradigm
- **Link**: [OpenAI Blog](https://openai.com/research/language-unsupervised)
- **Status**: 📖 Reading

---

### BERT: Pre-training of Deep Bidirectional Transformers
**Devlin et al., Google, October 2018**

Introduced bidirectional pre-training using masked language modeling.

- **Key innovations**: Masked LM objective, bidirectional context, [CLS] token
- **Impact**: Dominated NLP benchmarks, spawned RoBERTa, ALBERT, DistilBERT
- **Link**: [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- **Status**: 📖 Reading

---

### GPT-2: Language Models are Unsupervised Multitask Learners
**Radford et al., OpenAI, February 2019**

Scaled up GPT, demonstrated emergent zero-shot capabilities.

- **Key innovations**: Pre-LayerNorm, larger scale (1.5B params), zero-shot learning
- **Impact**: Showed scaling enables emergent capabilities
- **Link**: [OpenAI Blog](https://openai.com/research/better-language-models)
- **Status**: 🔧 Hands-on (Pre-LayerNorm in evolutions/)

---

### RMSNorm: Root Mean Square Layer Normalization
**Zhang & Sennrich, September 2019**

Simplified LayerNorm by removing mean centering.

- **Key innovations**: Faster normalization, equivalent quality
- **Impact**: Used in LLaMA, Mistral, and most modern models
- **Link**: [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- **Status**: 🔧 Hands-on (evolutions/rmsnorm.py)

---

## Era 2: Scaling & Efficiency (2020-2021)

### GPT-3: Language Models are Few-Shot Learners
**Brown et al., OpenAI, May 2020**

Massive scale (175B params) unlocked in-context learning.

- **Key innovations**: Few-shot prompting, emergent abilities at scale
- **Impact**: Launched the "prompting" paradigm, inspired ChatGPT
- **Link**: [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- **Status**: 📖 Reading

---

### GLU Variants Improve Transformer (SwiGLU)
**Shazeer, Google, February 2020**

Introduced gated linear units with Swish activation for FFN layers.

- **Key innovations**: SwiGLU activation, better optimization landscape
- **Impact**: Used in PaLM, LLaMA, Mistral
- **Link**: [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- **Status**: 🔧 Hands-on (evolutions/swiglu.py)

---

### Longformer: The Long-Document Transformer
**Beltagy et al., Allen AI, April 2020**

Addressed quadratic attention complexity with sparse patterns.

- **Key innovations**: Sliding window + global attention, O(n) complexity
- **Impact**: Enabled long document processing
- **Link**: [arXiv:2004.05150](https://arxiv.org/abs/2004.05150)
- **Status**: 📖 Reading

---

### RoFormer: Enhanced Transformer with Rotary Position Embedding
**Su et al., April 2021**

Introduced Rotary Position Embeddings (RoPE).

- **Key innovations**: Position encoded via rotation, relative position naturally emerges
- **Impact**: Used in LLaMA, Mistral, GPT-NeoX, PaLM
- **Link**: [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- **Status**: 🔧 Hands-on (evolutions/rope.py)

---

### LoRA: Low-Rank Adaptation of Large Language Models
**Hu et al., Microsoft, June 2021**

Efficient fine-tuning by training low-rank adapter matrices.

- **Key innovations**: Freeze base model, train small adapters
- **Impact**: Made LLM fine-tuning accessible, QLoRA followed
- **Link**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Status**: 📖 Reading

---

## Era 3: Alignment & Instruction (2022)

### InstructGPT: Training language models to follow instructions
**Ouyang et al., OpenAI, March 2022**

Aligned language models with human preferences using RLHF.

- **Key innovations**: RLHF (Reinforcement Learning from Human Feedback)
- **Impact**: Foundation for ChatGPT, established alignment techniques
- **Link**: [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **Status**: 📖 Reading

---

### Chinchilla: Training Compute-Optimal Large Language Models
**Hoffmann et al., DeepMind, March 2022**

Established optimal scaling laws for model size vs. training data.

- **Key innovations**: Compute-optimal scaling, "train smaller models on more data"
- **Impact**: Changed how models are sized, influenced LLaMA design
- **Link**: [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- **Status**: 📖 Reading

---

### FlashAttention: Fast and Memory-Efficient Exact Attention
**Dao et al., Stanford, May 2022**

Hardware-aware attention that's both faster and uses less memory.

- **Key innovations**: Tiling, kernel fusion, IO-aware algorithm
- **Impact**: Standard in all modern inference/training stacks
- **Link**: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- **Status**: 📖 Reading (implementation is CUDA-specific)

---

## Era 4: Open Models & Efficiency (2023)

### LLaMA: Open and Efficient Foundation Language Models
**Touvron et al., Meta, February 2023**

Open-weights models competitive with much larger closed models.

- **Key innovations**: Combined best practices (RoPE, SwiGLU, RMSNorm, Pre-LN)
- **Impact**: Sparked open-source LLM movement, basis for Alpaca, Vicuna, etc.
- **Link**: [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- **Status**: 🔧 Hands-on (Modern architecture in evolutions/)

---

### GQA: Training Generalized Multi-Query Transformer Models
**Ainslie et al., Google, May 2023**

Grouped-Query Attention for efficient KV cache during inference.

- **Key innovations**: Share KV heads across query groups
- **Impact**: Used in LLaMA 2, Mistral, Gemma
- **Link**: [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- **Status**: 🔧 Hands-on (evolutions/grouped_query_attention.py)

---

### LLaMA 2: Open Foundation and Fine-Tuned Chat Models
**Touvron et al., Meta, July 2023**

Improved LLaMA with GQA, longer context, and safety tuning.

- **Key innovations**: GQA integration, 4K→128K context, RLHF for chat
- **Impact**: Most-used open foundation model
- **Link**: [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)
- **Status**: 📖 Reading

---

### Mistral 7B
**Jiang et al., Mistral AI, October 2023**

Highly efficient 7B model outperforming larger models.

- **Key innovations**: Sliding window attention, efficient architecture
- **Impact**: Showed small models can compete with larger ones
- **Link**: [arXiv:2310.06825](https://arxiv.org/abs/2310.06825)
- **Status**: 📖 Reading

---

## Era 5: Reasoning & Mixture of Experts (2024)

### Mixtral: Mixture of Experts
**Jiang et al., Mistral AI, January 2024**

Sparse mixture of experts for efficient scaling.

- **Key innovations**: 8 experts, 2 active per token, efficient routing
- **Impact**: Revived MoE approach for LLMs
- **Link**: [arXiv:2401.04088](https://arxiv.org/abs/2401.04088)
- **Status**: 📖 Reading

---

### LLaMA 3: Herd of Models
**Meta, April 2024**

Scaled to 405B, improved tokenizer and training.

- **Key innovations**: 128K vocabulary, better data curation, 15T tokens
- **Impact**: New open-source SOTA
- **Link**: [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)
- **Status**: 📖 Reading

---

### DeepSeek-V2: A Strong, Economical, and Efficient MoE Model
**DeepSeek AI, May 2024**

Multi-head Latent Attention (MLA) for extreme KV cache compression.

- **Key innovations**: MLA reduces KV cache by 90%+, efficient MoE
- **Impact**: Showed path to affordable large-scale inference
- **Link**: [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- **Status**: 📖 Reading

---

## Era 6: Looped & Latent Reasoning (2025-2026)

### Reasoning with Latent Thoughts: On the Power of Looped Models
**March 2025**

Formally proves that looped shallow networks can match or exceed deep networks on reasoning.

- **Key innovations**: Mathematical proof of loop power, depth via iteration
- **Impact**: Theoretical foundation for looped transformers
- **Link**: [OpenReview](https://openreview.net/forum?id=din0lGfZFd)
- **Status**: 🔬 Frontier (evolutions/advanced/)

---

### Simply Stabilizing the Loop: Fully Looped Transformer
**May 2026**

Solved gradient explosion in looped transformers.

- **Key innovations**: Parameter-free stabilization for inter-loop signals
- **Impact**: Made looped transformers trainable at scale
- **Link**: [arXiv:2605.18797](https://arxiv.org/abs/2605.18797)
- **Status**: 🔬 Frontier

---

### ReLIT: Recursive Latent Implicit Transformer Framework
**August 2026**

Architecture for internal "pondering" before output.

- **Key innovations**: Trainable recursive block, latent thinking vector refinement
- **Impact**: Decouples reasoning depth from output token count
- **Link**: [arXiv:2608.08113](https://arxiv.org/html/2608.08113v1)
- **Status**: 🔬 Frontier

---

### Loop, Think, & Generalize: Implicit Reasoning in Recurrent-Depth Transformers
**September 2026**

Proves recurrent-depth transformers generalize to longer reasoning chains.

- **Key innovations**: Training on 5-step → generalizing to 10-step problems
- **Impact**: Shows looped models have better inductive bias for reasoning
- **Link**: [OpenReview](https://openreview.net/forum?id=8fz7WRThKL)
- **Status**: 🔬 Frontier

---

## Curated Collections

### Awesome Loop Models
**Living Repository, September 2026**

142+ papers on looped models, latent reasoning, and test-time compute.

- **Link**: [GitHub](https://huskydoge.github.io/Awesome-Loop-Models/)
- **Status**: 📖 Reference

---

## Additional Resources

### Tutorials & Explanations

| Resource | Description | Link |
|----------|-------------|------|
| The Illustrated Transformer | Visual explanation of attention | [jalammar.github.io](https://jalammar.github.io/illustrated-transformer/) |
| The Annotated Transformer | Line-by-line code walkthrough | [Harvard NLP](https://nlp.seas.harvard.edu/annotated-transformer/) |
| Attention? Attention! | Lilian Weng's comprehensive overview | [lilianweng.github.io](https://lilianweng.github.io/posts/2018-06-24-attention/) |
| nanoGPT | Minimal GPT implementation by Karpathy | [GitHub](https://github.com/karpathy/nanoGPT) |

### Video Lectures

| Resource | Description | Link |
|----------|-------------|------|
| Andrej Karpathy - GPT from Scratch | Building GPT step-by-step | [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY) |
| 3Blue1Brown - Attention | Visual intuition for attention | [YouTube](https://www.youtube.com/watch?v=eMlx5fFNoYc) |

### Specifications & Standards

| Resource | Description | Link |
|----------|-------------|------|
| Hugging Face Transformers | Reference implementations | [GitHub](https://github.com/huggingface/transformers) |
| PyTorch nn.Transformer | Official PyTorch implementation | [Docs](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html) |

---

## Reading Order Recommendation

### If you're new to transformers:
1. Original "Attention Is All You Need" paper
2. The Illustrated Transformer (visual guide)
3. This learning path (hands-on implementation)

### If you understand the basics:
1. BERT & GPT papers (foundation models)
2. LLaMA paper (modern best practices)
3. FlashAttention (efficiency)

### If you're exploring the frontier:
1. Scaling Laws (Chinchilla)
2. Mixture of Experts (Mixtral)
3. Looped Transformers (2025-2026 papers)

---

*Last updated: September 2026*
