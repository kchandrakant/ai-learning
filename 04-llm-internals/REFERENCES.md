# LLM Internals: Key References

A curated collection of papers covering tokenization, pretraining, scaling laws, and alignment.

---

## 📚 How to Use This Document

- **📖 Essential**: Core papers everyone should read
- **🔧 Hands-on**: Papers with practical implementations
- **🔬 Frontier**: Cutting-edge research

---

## Tokenization

### Neural Machine Translation of Rare Words with Subword Units (BPE)
**Sennrich, Haddow, Birch, ACL 2016**

Introduced Byte Pair Encoding for neural machine translation.

- **Key innovations**: Subword tokenization, handling rare words
- **Impact**: Foundation for GPT tokenizers
- **Link**: [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)
- **Status**: 🔧 Hands-on

---

### SentencePiece: A simple and language independent subword tokenizer
**Kudo & Richardson, EMNLP 2018**

Language-agnostic tokenization directly on raw text.

- **Key innovations**: Unigram LM, language-independent processing
- **Impact**: Used in T5, LLaMA, and many multilingual models
- **Link**: [arXiv:1808.06226](https://arxiv.org/abs/1808.06226)
- **Status**: 🔧 Hands-on

---

### Byte Pair Encoding is Suboptimal for Language Model Pretraining
**Bostrom & Durrett, 2020**

Analysis showing unigram LM can outperform BPE.

- **Key findings**: Tokenization affects downstream performance significantly
- **Link**: [arXiv:2004.03720](https://arxiv.org/abs/2004.03720)
- **Status**: 📝 Reference

---

## Pretraining Foundations

### Language Models are Unsupervised Multitask Learners (GPT-2)
**Radford et al., OpenAI 2019**

Demonstrated zero-shot capabilities of large language models.

- **Key innovations**: Pre-LayerNorm, scaling to 1.5B parameters
- **Impact**: Established the GPT architecture
- **Link**: [OpenAI](https://openai.com/research/better-language-models)
- **Status**: 📖 Essential

---

### Language Models are Few-Shot Learners (GPT-3)
**Brown et al., NeurIPS 2020**

175B parameter model demonstrating in-context learning.

- **Key innovations**: Few-shot prompting, emergent capabilities
- **Impact**: Launched the era of prompting
- **Link**: [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- **Status**: 📖 Essential

---

### BERT: Pre-training of Deep Bidirectional Transformers
**Devlin et al., NAACL 2019**

Bidirectional pretraining using masked language modeling.

- **Key innovations**: MLM objective, [CLS] token, bidirectional context
- **Impact**: Dominated NLP benchmarks
- **Link**: [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
- **Status**: 📖 Essential

---

## Scaling Laws

### Scaling Laws for Neural Language Models
**Kaplan et al., OpenAI 2020**

Discovered power-law relationships between compute, data, and performance.

- **Key findings**: Performance scales predictably with resources
- **Impact**: Guided GPT-3 and subsequent model scaling
- **Link**: [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- **Status**: 📖 Essential

---

### Training Compute-Optimal Large Language Models (Chinchilla)
**Hoffmann et al., DeepMind 2022**

Revised scaling laws: train smaller models on more data.

- **Key findings**: Most models are undertrained for their size
- **Impact**: Changed how models are sized (influenced LLaMA)
- **Link**: [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- **Status**: 📖 Essential

---

## Alignment & RLHF

### Training language models to follow instructions with human feedback (InstructGPT)
**Ouyang et al., OpenAI 2022**

Introduced RLHF for aligning LLMs with human preferences.

- **Key innovations**: RLHF pipeline, reward modeling, PPO fine-tuning
- **Impact**: Foundation for ChatGPT
- **Link**: [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **Status**: 📖 Essential

---

### Constitutional AI: Harmlessness from AI Feedback
**Bai et al., Anthropic 2022**

Training helpful and harmless AI using AI feedback.

- **Key innovations**: RLAIF, constitutional principles
- **Impact**: Scalable alignment without human labeling
- **Link**: [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
- **Status**: 📖 Essential

---

### Direct Preference Optimization (DPO)
**Rafailov et al., NeurIPS 2023**

Simpler alternative to RLHF without explicit reward modeling.

- **Key innovations**: Direct optimization from preferences
- **Impact**: Widely adopted for fine-tuning
- **Link**: [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- **Status**: 🔧 Hands-on

---

## Modern Architectures

### LLaMA: Open and Efficient Foundation Language Models
**Touvron et al., Meta 2023**

Open-weights models combining best practices.

- **Key innovations**: RoPE, SwiGLU, RMSNorm, Pre-LN
- **Impact**: Sparked open-source LLM movement
- **Link**: [arXiv:2302.13971](https://arxiv.org/abs/2302.13971)
- **Status**: 🔧 Hands-on

---

### RoFormer: Enhanced Transformer with Rotary Position Embedding (RoPE)
**Su et al., 2021**

Rotary position embeddings for better length generalization.

- **Key innovations**: Position encoded via rotation
- **Impact**: Used in LLaMA, Mistral, GPT-NeoX
- **Link**: [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- **Status**: 🔧 Hands-on

---

### GLU Variants Improve Transformer (SwiGLU)
**Shazeer, Google 2020**

Gated linear units for better FFN layers.

- **Key innovations**: SwiGLU activation
- **Impact**: Used in PaLM, LLaMA, Mistral
- **Link**: [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- **Status**: 🔧 Hands-on

---

### RMSNorm: Root Mean Square Layer Normalization
**Zhang & Sennrich, 2019**

Simplified normalization without mean centering.

- **Key innovations**: Faster, equivalent quality
- **Impact**: Used in most modern LLMs
- **Link**: [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)
- **Status**: 🔧 Hands-on

---

## Efficiency

### FlashAttention: Fast and Memory-Efficient Exact Attention
**Dao et al., NeurIPS 2022**

Hardware-aware attention algorithm.

- **Key innovations**: Tiling, kernel fusion, IO-awareness
- **Impact**: Standard in all modern training/inference
- **Link**: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- **Status**: 📖 Essential

---

### GQA: Training Generalized Multi-Query Transformer Models
**Ainslie et al., Google 2023**

Grouped-Query Attention for efficient inference.

- **Key innovations**: Share KV heads across query groups
- **Impact**: Used in LLaMA 2, Mistral, Gemma
- **Link**: [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- **Status**: 🔧 Hands-on

---

## Online Resources

| Resource | Description | Link |
|----------|-------------|------|
| Hugging Face Tokenizers | Tokenization library | [GitHub](https://github.com/huggingface/tokenizers) |
| nanoGPT | Minimal GPT training | [GitHub](https://github.com/karpathy/nanoGPT) |
| LLM Course | Comprehensive LLM curriculum | [GitHub](https://github.com/mlabonne/llm-course) |
| Lilian Weng's Blog | Excellent LLM explanations | [lilianweng.github.io](https://lilianweng.github.io/) |

---

*Last updated: September 2026*
