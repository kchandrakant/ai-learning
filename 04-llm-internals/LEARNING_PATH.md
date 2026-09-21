# LLM Internals: A Step-by-Step Learning Journey

This course goes deep into how Large Language Models are built from scratch — tokenization, pre-training, scaling laws, RLHF, and the engineering that makes billion-parameter models possible.

---

## 🎯 Why Study LLM Internals?

Understanding LLM internals helps you:
- **Debug and improve** — Know why models behave certain ways
- **Make informed choices** — Select models based on architecture, not just benchmarks
- **Contribute to research** — Understand the frontier
- **Build better systems** — Design around model strengths/weaknesses

This course is for practitioners who want to go beyond using APIs to understanding what's inside.

---

## 🎯 Prerequisites

- Completed Transformer Architecture course (or equivalent)
- PyTorch proficiency
- Understanding of gradient descent, loss functions
- Familiarity with distributed computing concepts

---

## 📚 Part 1: Foundations of LLM Training

How raw text becomes a language model.

### Module 1: Tokenization
**Breaking text into learnable units**

**What we'll cover:**
- Why tokenization matters for model behavior
- Byte-Pair Encoding (BPE) algorithm
- WordPiece and Unigram alternatives
- SentencePiece implementation
- Tokenizer training from scratch
- Vocabulary size trade-offs
- Multilingual tokenization challenges

**Key insight:**
```
Tokenization shapes what the model can learn:
- "tokenization" → ["token", "ization"] (compositional)
- Rare words → many tokens (inefficient)
- Numbers/code → often poorly tokenized
```

---

### Module 2: Pre-training Objectives
**What loss functions teach models**

**What we'll cover:**
- Causal Language Modeling (GPT-style)
- Masked Language Modeling (BERT-style)
- Prefix LM and T5-style objectives
- Next sentence prediction (and why it was dropped)
- Span corruption
- Fill-in-the-middle (FIM) for code models

**Key comparison:**
```
Causal LM:    Predict next token (left-to-right)
              "The cat sat on the ___"
              
Masked LM:    Predict masked tokens (bidirectional)
              "The [MASK] sat on the mat"
              
Prefix LM:    Bidirectional prefix, causal generation
```

---

### Module 3: Scaling Laws
**The science of scaling**

**What we'll cover:**
- Chinchilla scaling laws
- Compute-optimal training
- Over-training for inference efficiency
- The relationship: params, data, compute
- Power law predictions
- When scaling breaks down

**Key equations:**
```
L(N, D) ≈ (N_c/N)^α + (D_c/D)^β + L_∞

Where:
- N = parameters
- D = data tokens
- L = loss
- Chinchilla: N and D should scale together
```

---

### Module 4: Data Curation
**Data is the secret sauce**

**What we'll cover:**
- Web crawl processing (Common Crawl)
- Deduplication strategies
- Quality filtering (perplexity, classifiers)
- Data mixing ratios
- Synthetic data generation
- Contamination detection
- Legal and ethical considerations

**The stack:**
```
Raw web → Language filter → Dedup → Quality filter → 
PII removal → Mix with curated data → Final dataset
```

---

## 📚 Part 2: Training at Scale

Engineering billion-parameter models.

### Module 5: Training Infrastructure
**Hardware and systems**

**What we'll cover:**
- GPU architecture for LLMs (A100, H100)
- Memory hierarchy (HBM, SRAM)
- Interconnects (NVLink, InfiniBand)
- Cloud vs on-prem trade-offs
- Training cluster design
- Cost estimation

**Key numbers:**
```
Llama-2-70B training:
- ~1.7M GPU hours
- ~2T tokens
- Estimated $2-5M compute cost
```

---

### Module 6: Distributed Training
**Parallelism strategies**

**What we'll cover:**
- Data parallelism (DDP)
- Model/Tensor parallelism (Megatron-style)
- Pipeline parallelism
- ZeRO optimization stages
- Fully Sharded Data Parallel (FSDP)
- Gradient checkpointing
- Mixed precision training

**Parallelism strategies:**
```
Data Parallel:    Same model on each GPU, different data
Tensor Parallel:  Split layers across GPUs
Pipeline Parallel: Different layers on different GPUs
ZeRO:             Shard optimizer state, gradients, params
```

---

### Module 7: Architecture Decisions
**Design choices in modern LLMs**

**What we'll cover:**
- Depth vs width trade-offs
- Attention variants (MHA, GQA, MQA)
- Activation functions (ReLU → GELU → SwiGLU)
- Normalization (LayerNorm → RMSNorm)
- Position encodings (Sinusoidal → RoPE → ALiBi)
- Context length scaling
- Mixture of Experts (MoE)

**Architecture evolution:**
```
GPT-2 (2019): Post-LN, learned position, ReLU
GPT-3 (2020): Sparse attention patterns
LLaMA (2023): Pre-LN, RoPE, SwiGLU, RMSNorm, GQA
Mixtral (2024): MoE with 8 experts, 2 active
```

---

## 📚 Part 3: Alignment & Capabilities

Making models useful and safe.

### Module 8: Emergent Abilities
**What appears at scale**

**What we'll cover:**
- Definition of emergence
- In-context learning
- Chain-of-thought reasoning
- Instruction following
- The "capability overhang" debate
- Grokking and phase transitions
- Are emergent abilities real?

**Emergent examples:**
```
Small models:  Can't do 3-digit addition
Large models:  Suddenly can (sharp transition)
               
But: Some "emergence" is metric artifacts
```

---

### Module 9: RLHF from Scratch
**Reinforcement Learning from Human Feedback**

**What we'll cover:**
- Why RLHF? The alignment problem
- Preference data collection
- Reward model training
- PPO for language models
- KL divergence constraints
- Reward hacking and mitigation
- Implementation with TRL

**The RLHF pipeline:**
```
1. SFT: Train on demonstrations
2. RM: Train reward model on preferences
3. RL: Optimize policy with PPO against RM
```

---

### Module 10: Constitutional AI
**Alignment without human labels**

**What we'll cover:**
- Self-supervision for alignment
- Constitutional principles
- Critique and revision
- RLAIF (AI feedback)
- Red teaming and adversarial training
- Comparison to RLHF

**Constitutional AI loop:**
```
Generate → Self-critique → Revise → Repeat
"Is this response harmful?" → "Yes, because..." → "Better version..."
```

---

## 📚 Part 4: Evaluation & Analysis

Understanding what models know.

### Module 11: Evaluation & Benchmarks
**Measuring model capabilities**

**What we'll cover:**
- Perplexity and loss
- Standard benchmarks (MMLU, HellaSwag, ARC, etc.)
- Reasoning benchmarks (GSM8K, MATH, BBH)
- Code benchmarks (HumanEval, MBPP)
- Safety evaluations
- Benchmark contamination
- Limitations of benchmarks

**Benchmark categories:**
```
Knowledge:    MMLU, ARC, TriviaQA
Reasoning:    GSM8K, MATH, BBH, ARC-Challenge  
Code:         HumanEval, MBPP, SWE-bench
Safety:       TruthfulQA, toxicity evals
```

---

### Module 12: Model Analysis
**Looking inside the black box**

**What we'll cover:**
- Probing classifiers
- Activation analysis
- Attention pattern visualization
- Mechanistic interpretability
- Feature circuits
- Sparse autoencoders
- Model editing

**Analysis techniques:**
```
Probing:       Train classifier on hidden states
Attention:     Visualize what tokens attend to what
Circuits:      Find minimal subgraphs for behaviors
Editing:       Modify weights to change specific facts
```

---

## 🗂️ Project Structure

```
llm-internals/
├── LEARNING_PATH.md
├── requirements.txt
├── verify_setup.py
│
├── 01_tokenization/
├── 02_pretraining_objectives/
├── 03_scaling_laws/
├── 04_data_curation/
├── 05_training_infrastructure/
├── 06_distributed_training/
├── 07_architecture_decisions/
├── 08_emergent_abilities/
├── 09_rlhf_from_scratch/
├── 10_constitutional_ai/
├── 11_evaluation_benchmarks/
├── 12_model_analysis/
│
├── demo/
└── beyond/
```

---

## 📅 Recommended Learning Order

```
Week 1-2: Foundations
├── Module 1: Tokenization
├── Module 2: Pre-training Objectives
└── Module 3: Scaling Laws

Week 3-4: Training Engineering
├── Module 4: Data Curation
├── Module 5: Training Infrastructure
└── Module 6: Distributed Training

Week 5: Architecture
└── Module 7: Architecture Decisions

Week 6-7: Alignment
├── Module 8: Emergent Abilities
├── Module 9: RLHF from Scratch
└── Module 10: Constitutional AI

Week 8: Evaluation
├── Module 11: Evaluation & Benchmarks
└── Module 12: Model Analysis
```

---

## 🚀 Let's Begin!

Start with **Module 1: Tokenization** — the foundation of how models see text.

---

## 📖 References

### Key Papers
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Few-Shot Learners" (GPT-3, 2020)
- "Training Compute-Optimal LLMs" (Chinchilla, 2022)
- "LLaMA: Open and Efficient LLMs" (2023)
- "Training Language Models to Follow Instructions with Human Feedback" (InstructGPT)
- "Constitutional AI" (Anthropic, 2022)

### Resources
- Andrej Karpathy: "Let's build GPT"
- Lilian Weng: "Large Language Model Course"
- EleutherAI documentation
- Hugging Face course on NLP
