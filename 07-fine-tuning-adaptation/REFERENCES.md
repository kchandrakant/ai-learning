# Fine-Tuning & Adaptation: Key References

A curated collection of papers on parameter-efficient fine-tuning, RLHF, and preference optimization.

---

## 📚 How to Use This Document

- **📖 Essential**: Core papers everyone should read
- **🔧 Hands-on**: Papers with practical implementations
- **🔬 Frontier**: Cutting-edge research

---

## Parameter-Efficient Fine-Tuning (PEFT)

### LoRA: Low-Rank Adaptation of Large Language Models
**Hu et al., ICLR 2022**

The breakthrough paper for efficient fine-tuning.

- **Key innovations**: Low-rank weight updates, frozen base model
- **Impact**: Enables fine-tuning on consumer hardware
- **Link**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Status**: 📖 Essential

---

### QLoRA: Efficient Finetuning of Quantized LLMs
**Dettmers et al., NeurIPS 2023**

Combines 4-bit quantization with LoRA.

- **Key innovations**: NF4 quantization, double quantization, paged optimizers
- **Impact**: Fine-tune 65B models on single 48GB GPU
- **Link**: [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)
- **Status**: 🔧 Hands-on

---

### Prefix-Tuning: Optimizing Continuous Prompts for Generation
**Li & Liang, ACL 2021**

Prepend learnable vectors to attention layers.

- **Key innovations**: Continuous prompt optimization
- **Link**: [arXiv:2101.00190](https://arxiv.org/abs/2101.00190)
- **Status**: 📝 Reference

---

### LLaMA-Adapter: Efficient Fine-tuning of Language Models
**Zhang et al., 2023**

Adapters with zero-init attention.

- **Key innovations**: Zero-init attention gating
- **Link**: [arXiv:2303.16199](https://arxiv.org/abs/2303.16199)
- **Status**: 🔧 Hands-on

---

### DoRA: Weight-Decomposed Low-Rank Adaptation
**Liu et al., 2024**

Decomposes LoRA into magnitude and direction.

- **Key innovations**: Better parameter efficiency than LoRA
- **Link**: [arXiv:2402.09353](https://arxiv.org/abs/2402.09353)
- **Status**: 🔬 Frontier

---

## Reinforcement Learning from Human Feedback

### Training language models to follow instructions with human feedback (InstructGPT)
**Ouyang et al., OpenAI 2022**

The foundational RLHF paper.

- **Key innovations**: SFT → Reward Model → PPO pipeline
- **Impact**: Foundation for ChatGPT
- **Link**: [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
- **Status**: 📖 Essential

---

### Deep Reinforcement Learning from Human Preferences
**Christiano et al., NeurIPS 2017**

Original paper on learning from human preferences.

- **Key innovations**: Learning reward models from comparisons
- **Link**: [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
- **Status**: 📝 Reference

---

### Proximal Policy Optimization Algorithms (PPO)
**Schulman et al., 2017**

The RL algorithm used in RLHF.

- **Key innovations**: Clipped surrogate objective, stable training
- **Link**: [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- **Status**: 📖 Essential

---

## Direct Preference Optimization

### Direct Preference Optimization: Your Language Model is Secretly a Reward Model
**Rafailov et al., NeurIPS 2023**

Eliminates need for explicit reward model.

- **Key innovations**: Closed-form solution from Bradley-Terry model
- **Impact**: Simpler, more stable than PPO
- **Link**: [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- **Status**: 📖 Essential

---

### Identity Preference Optimization (IPO)
**Azar et al., 2023**

Addresses overfitting issues in DPO.

- **Key innovations**: Regularized preference optimization
- **Link**: [arXiv:2310.12036](https://arxiv.org/abs/2310.12036)
- **Status**: 🔧 Hands-on

---

### KTO: Model Alignment as Prospect Theoretic Optimization
**Ethayarajh et al., 2024**

Alignment using Kahneman-Tversky prospect theory.

- **Key innovations**: Works without paired preferences
- **Link**: [arXiv:2402.01306](https://arxiv.org/abs/2402.01306)
- **Status**: 🔬 Frontier

---

### ORPO: Monolithic Preference Optimization without Reference Model
**Hong et al., 2024**

Combines SFT and preference alignment.

- **Key innovations**: No reference model needed
- **Link**: [arXiv:2403.07691](https://arxiv.org/abs/2403.07691)
- **Status**: 🔬 Frontier

---

## Constitutional AI

### Constitutional AI: Harmlessness from AI Feedback
**Bai et al., Anthropic 2022**

Training AI using AI-generated feedback.

- **Key innovations**: RLAIF, constitutional principles, self-critique
- **Impact**: Scalable alignment without human labeling
- **Link**: [arXiv:2212.08073](https://arxiv.org/abs/2212.08073)
- **Status**: 📖 Essential

---

### RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback
**Lee et al., Google 2023**

Analysis of AI feedback vs human feedback.

- **Key findings**: AI feedback can match human feedback quality
- **Link**: [arXiv:2309.00267](https://arxiv.org/abs/2309.00267)
- **Status**: 📝 Reference

---

## Data & Evaluation

### LIMA: Less Is More for Alignment
**Zhou et al., Meta 2023**

High-quality data matters more than quantity.

- **Key findings**: 1000 carefully curated examples sufficient
- **Link**: [arXiv:2305.11206](https://arxiv.org/abs/2305.11206)
- **Status**: 📖 Essential

---

### Self-Instruct: Aligning Language Models with Self-Generated Instructions
**Wang et al., ACL 2023**

Generate training data from LLM itself.

- **Key innovations**: Bootstrapping instruction data
- **Link**: [arXiv:2212.10560](https://arxiv.org/abs/2212.10560)
- **Status**: 🔧 Hands-on

---

### Alpaca: A Strong, Replicable Instruction-Following Model
**Taori et al., Stanford 2023**

Fine-tuning LLaMA on GPT-generated instructions.

- **Key value**: Reproducible fine-tuning recipe
- **Link**: [GitHub](https://github.com/tatsu-lab/stanford_alpaca)
- **Status**: 🔧 Hands-on

---

## Online Resources

| Resource | Description | Link |
|----------|-------------|------|
| Hugging Face PEFT | PEFT library | [GitHub](https://github.com/huggingface/peft) |
| TRL | Transformer Reinforcement Learning | [GitHub](https://github.com/huggingface/trl) |
| Axolotl | Fine-tuning toolkit | [GitHub](https://github.com/OpenAccess-AI-Collective/axolotl) |
| LLaMA-Factory | Easy fine-tuning | [GitHub](https://github.com/hiyouga/LLaMA-Factory) |

---

## Implementation Resources

| Resource | Description | Link |
|----------|-------------|------|
| unsloth | Fast fine-tuning | [GitHub](https://github.com/unslothai/unsloth) |
| MLX | Apple silicon training | [GitHub](https://github.com/ml-explore/mlx) |
| DeepSpeed | Distributed training | [deepspeed.ai](https://www.deepspeed.ai/) |

---

*Last updated: September 2026*
