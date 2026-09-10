# Multimodal AI: Key References

A curated collection of papers on vision-language models, diffusion, and audio AI.

---

## 📚 How to Use This Document

- **📖 Essential**: Core papers everyone should read
- **🔧 Hands-on**: Papers with practical implementations
- **🔬 Frontier**: Cutting-edge research

---

## Vision-Language Models

### Learning Transferable Visual Models From Natural Language Supervision (CLIP)
**Radford et al., OpenAI, ICML 2021**

Contrastive learning for vision-language alignment.

- **Key innovations**: 400M image-text pairs, zero-shot transfer
- **Impact**: Foundation for all vision-language models
- **Link**: [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- **Status**: 📖 Essential

---

### ALIGN: Scaling Up Visual and Vision-Language Representation Learning
**Jia et al., Google, ICML 2021**

Scaling CLIP-style training to 1.8B image-text pairs.

- **Key innovations**: Noisy web data at scale
- **Link**: [arXiv:2102.05918](https://arxiv.org/abs/2102.05918)
- **Status**: 📝 Reference

---

### BLIP: Bootstrapping Language-Image Pre-training
**Li et al., Salesforce 2022**

Unified vision-language understanding and generation.

- **Key innovations**: Captioning + filtering bootstrap, multimodal mixture of encoder-decoder
- **Link**: [arXiv:2201.12086](https://arxiv.org/abs/2201.12086)
- **Status**: 🔧 Hands-on

---

### BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Models
**Li et al., ICML 2023**

Efficient vision-language training with frozen models.

- **Key innovations**: Q-Former, frozen LLM integration
- **Impact**: Efficient multimodal architecture
- **Link**: [arXiv:2301.12597](https://arxiv.org/abs/2301.12597)
- **Status**: 📖 Essential

---

### LLaVA: Visual Instruction Tuning
**Liu et al., NeurIPS 2023**

Instruction-following for vision-language models.

- **Key innovations**: Visual instruction tuning, GPT-4 generated data
- **Impact**: Popular open-source VLM
- **Link**: [arXiv:2304.08485](https://arxiv.org/abs/2304.08485)
- **Status**: 🔧 Hands-on

---

### Flamingo: A Visual Language Model for Few-Shot Learning
**Alayrac et al., DeepMind, NeurIPS 2022**

Few-shot multimodal learning.

- **Key innovations**: Perceiver resampler, interleaved image-text
- **Link**: [arXiv:2204.14198](https://arxiv.org/abs/2204.14198)
- **Status**: 📖 Essential

---

## Image Generation

### Zero-Shot Text-to-Image Generation (DALL-E)
**Ramesh et al., OpenAI 2021**

Transformer-based text-to-image generation.

- **Key innovations**: dVAE + autoregressive transformer
- **Impact**: Launched text-to-image era
- **Link**: [arXiv:2102.12092](https://arxiv.org/abs/2102.12092)
- **Status**: 📖 Essential

---

### Denoising Diffusion Probabilistic Models (DDPM)
**Ho et al., NeurIPS 2020**

Foundation of modern diffusion models.

- **Key innovations**: Denoising score matching, noise schedule
- **Impact**: Foundation for Stable Diffusion, DALL-E 2
- **Link**: [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- **Status**: 📖 Essential

---

### High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion)
**Rombach et al., CVPR 2022**

Diffusion in latent space for efficient generation.

- **Key innovations**: VAE + latent diffusion, cross-attention conditioning
- **Impact**: Democratized image generation
- **Link**: [arXiv:2112.10752](https://arxiv.org/abs/2112.10752)
- **Status**: 🔧 Hands-on

---

### Denoising Diffusion Implicit Models (DDIM)
**Song et al., ICLR 2021**

Faster sampling for diffusion models.

- **Key innovations**: Deterministic sampling, fewer steps
- **Link**: [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)
- **Status**: 🔧 Hands-on

---

### Classifier-Free Diffusion Guidance
**Ho & Salimans, 2022**

Improved conditioning without classifiers.

- **Key innovations**: Jointly train conditional and unconditional
- **Impact**: Standard for all guided diffusion
- **Link**: [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
- **Status**: 📖 Essential

---

## Video Generation

### VideoGPT: Video Generation using VQ-VAE and Transformers
**Yan et al., 2021**

Autoregressive video generation.

- **Key innovations**: 3D VQ-VAE, GPT for video
- **Link**: [arXiv:2104.10157](https://arxiv.org/abs/2104.10157)
- **Status**: 📝 Reference

---

### Make-A-Video: Text-to-Video Generation without Text-Video Data
**Singer et al., Meta 2022**

Text-to-video without paired data.

- **Key innovations**: Extend image diffusion to video
- **Link**: [arXiv:2209.14792](https://arxiv.org/abs/2209.14792)
- **Status**: 📝 Reference

---

## Audio & Speech

### Whisper: Robust Speech Recognition via Large-Scale Weak Supervision
**Radford et al., OpenAI 2022**

Universal speech recognition.

- **Key innovations**: 680K hours weak supervision, multilingual
- **Impact**: State-of-the-art ASR
- **Link**: [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)
- **Status**: 📖 Essential

---

### AudioLM: A Language Modeling Approach to Audio Generation
**Borsos et al., Google 2022**

Audio generation via language modeling.

- **Key innovations**: Hierarchical tokens, semantic + acoustic
- **Link**: [arXiv:2209.03143](https://arxiv.org/abs/2209.03143)
- **Status**: 📝 Reference

---

### MusicLM: Generating Music From Text
**Agostinelli et al., Google 2023**

Text-to-music generation.

- **Key innovations**: MuLan embeddings, hierarchical generation
- **Link**: [arXiv:2301.11325](https://arxiv.org/abs/2301.11325)
- **Status**: 📝 Reference

---

## Multimodal LLMs

### GPT-4V(ision) System Card
**OpenAI, 2023**

Multimodal GPT-4 capabilities and limitations.

- **Key value**: Understanding MLLM capabilities
- **Link**: [OpenAI](https://openai.com/research/gpt-4v-system-card)
- **Status**: 📖 Essential

---

### Gemini: A Family of Highly Capable Multimodal Models
**Team Gemini, Google 2023**

Natively multimodal foundation models.

- **Key innovations**: Native multimodal training
- **Link**: [arXiv:2312.11805](https://arxiv.org/abs/2312.11805)
- **Status**: 📖 Essential

---

## Online Resources

| Resource | Description | Link |
|----------|-------------|------|
| Hugging Face Diffusers | Diffusion models library | [GitHub](https://github.com/huggingface/diffusers) |
| LAION | Open datasets for multimodal AI | [laion.ai](https://laion.ai/) |
| OpenCLIP | Open-source CLIP training | [GitHub](https://github.com/mlfoundations/open_clip) |
| Lilian Weng's Blog | Excellent diffusion explanations | [lilianweng.github.io](https://lilianweng.github.io/) |

---

## Implementation Resources

| Resource | Description | Link |
|----------|-------------|------|
| ComfyUI | Node-based diffusion interface | [GitHub](https://github.com/comfyanonymous/ComfyUI) |
| AUTOMATIC1111 | Stable Diffusion WebUI | [GitHub](https://github.com/AUTOMATIC1111/stable-diffusion-webui) |
| transformers | Multimodal model library | [HuggingFace](https://huggingface.co/docs/transformers/) |

---

*Last updated: September 2026*
