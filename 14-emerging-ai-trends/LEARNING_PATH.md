# Emerging AI Trends: A Step-by-Step Learning Journey

This capstone course explores the cutting-edge developments shaping AI's future — from reasoning paradigms and alternative architectures to domain-specific transformers and frontier research.

---

## 🎯 Why Study Emerging Trends?

The AI landscape evolves rapidly. This course helps you:

- **Stay current** — Understand what's new and why it matters
- **Anticipate shifts** — Know where the field is heading
- **Make informed decisions** — Choose technologies that will last
- **Explore frontiers** — Engage with research, not just products

This is a capstone course — it assumes you've built strong foundations and are ready to explore what's next.

---

## 🎯 Prerequisites

- Completed Transformer Architecture course (03)
- Familiarity with LLM internals (04)
- Experience with agents and RAG (05, 06)
- Understanding of multimodal AI helps (08)

---

## 📚 Part 1: Reasoning Paradigms

How AI systems think — fast intuition vs deliberate reasoning.

### Module 1: System 1 vs System 2 Thinking
**Two modes of AI cognition**

**What we'll cover:**
- Kahneman's dual-process theory applied to AI
- System 1: Fast, intuitive (standard autoregressive generation)
- System 2: Slow, deliberate (chain-of-thought, verification)
- When each mode is appropriate
- The reasoning-compute tradeoff
- Examples: GPT-4 vs o1, Claude vs Claude with extended thinking

**Key insight:**
```
System 1:  Input → Model → Output (one pass)
           Fast, cheap, often sufficient

System 2:  Input → Think → Verify → Revise → Think more → Output
           Slow, expensive, better for hard problems
```

---

### Module 2: Test-Time Compute
**Trading inference cost for reasoning quality**

**What we'll cover:**
- What is test-time compute?
- Chain-of-thought as compute
- Beam search over reasoning paths
- Self-consistency and voting
- Verification and backtracking
- Budget allocation strategies
- o1, DeepSeek-R1 approaches
- When to invest more compute

**The scaling dimension:**
```
Training compute:   Fixed at training time
Test-time compute:  Can scale per-query

Key insight: Some problems benefit more from thinking
longer than from larger models.
```

---

### Module 3: Chain-of-Thought & Beyond
**Structured reasoning techniques**

**What we'll cover:**
- Chain-of-thought prompting
- Tree-of-thought exploration
- Graph-of-thought reasoning
- Self-reflection and critique
- Decomposition strategies
- When explicit reasoning helps (and when it doesn't)
- Reasoning traces as training data

**Evolution of reasoning:**
```
2022: Chain-of-thought (linear reasoning)
2023: Tree-of-thought (branching exploration)
2024: Verification loops (check your work)
2025: Learned reasoning (trained to think)
2026: Adaptive reasoning (know when to think more)
```

---

### Module 4: Looped & Adaptive Transformers
**Beyond fixed-depth architectures**

**What we'll cover:**
- Fixed depth vs adaptive computation
- Universal Transformers (2018)
- Looped Transformers (2025-2026)
- Pondering and halting mechanisms
- Training stability challenges
- Extrapolation to harder problems
- Connections to System 2 thinking

**The loop idea:**
```
Traditional: x → Layer1 → Layer2 → ... → LayerN → output
             (fixed N layers)

Looped:      x → LoopBlock → LoopBlock → ... → output
             (repeat until "done thinking")
             
Benefit: Simple problems → few iterations
         Hard problems → more iterations
```

---

## 📚 Part 2: Alternative Architectures

Challenging transformer dominance.

### Module 5: State Space Models (Mamba)
**Linear-time sequence modeling**

**What we'll cover:**
- Limitations of attention (quadratic complexity)
- State space model fundamentals
- S4 and structured state spaces
- Mamba: selective state spaces
- Mamba-2 improvements
- When SSMs beat transformers
- When transformers still win
- Hybrid architectures

**The complexity tradeoff:**
```
Attention: O(n²) in sequence length — every token attends to every token
SSM:       O(n) in sequence length — linear recurrence

Mamba achieves 5× higher throughput on long sequences
But: Transformers still better at precise recall tasks
```

---

### Module 6: Hybrid Architectures
**Best of both worlds**

**What we'll cover:**
- Why hybrid? Strengths of each architecture
- Mamba-Transformer hybrids (Jamba)
- Attention for precision, SSM for efficiency
- Layer composition strategies
- When to use what
- Production considerations
- Future directions

**Hybrid patterns:**
```
Interleaved:  SSM → Attention → SSM → Attention → ...
Parallel:     Input → [SSM branch, Attention branch] → Merge
Selective:    Route to SSM or Attention based on input
```

---

## 📚 Part 3: Domain-Specific Transformers

Transformers beyond text and images.

### Module 7: Tabular Transformers
**Deep learning for structured data**

**What we'll cover:**
- Why tabular is hard for deep learning
- TabTransformer: embedding categoricals
- FT-Transformer: feature tokenization
- TabPFN: in-context learning for tables
- When to use vs XGBoost/random forests
- Handling mixed types, missing values
- Production considerations

**The tabular challenge:**
```
Text/Images: Homogeneous, spatial/sequential structure
Tabular:     Heterogeneous columns, no natural order

Traditional ML (XGBoost) still often wins!
But: TabPFN achieves SOTA on small datasets with zero training
```

---

### Module 8: Time Series Transformers
**Attention for temporal data**

**What we'll cover:**
- Time series forecasting landscape
- Temporal Fusion Transformer
- PatchTST and patching strategies
- Informer and efficient attention
- Foundation models for time series (TimesFM, Chronos)
- Multivariate vs univariate
- When transformers help vs hurt

---

### Module 9: Graph Transformers
**Attention on graph-structured data**

**What we'll cover:**
- Graphs in ML: molecules, social networks, knowledge graphs
- Message passing neural networks (baseline)
- Graph attention mechanisms
- Positional encodings for graphs
- GraphGPS and hybrid approaches
- Scalability challenges
- Applications: drug discovery, recommendations

---

## 📚 Part 4: Scaling Paradigms

Efficient scaling beyond dense transformers.

### Module 10: Mixture of Experts (MoE)
**Sparse scaling**

**What we'll cover:**
- Dense vs sparse models
- MoE architecture: router + experts
- Routing strategies (top-k, soft routing)
- Load balancing challenges
- Training stability
- Mixtral, Switch Transformer, GPT-4 (rumored)
- Inference considerations (memory vs compute)

**The MoE insight:**
```
Dense 70B:  70B parameters, 70B compute per token
MoE 8×7B:   56B parameters, ~14B compute per token (2 experts active)

More parameters, same compute budget
But: All parameters must fit in memory
```

---

### Module 11: Efficient Architectures
**Doing more with less**

**What we'll cover:**
- Small Language Models (SLMs)
- When SLMs beat LLMs
- Distillation techniques
- Quantization (INT8, INT4, GGUF)
- Pruning and sparsity
- Architecture search for efficiency
- Edge deployment considerations
- Phi, Gemma, Qwen-small families

**The efficiency frontier:**
```
2023: Bigger is better
2024: Efficient at scale (GQA, FlashAttention)
2025: Small but mighty (Phi-3, Gemma-2)
2026: Right-sized for the task
```

---

## 📚 Part 5: Frontier Research

What's coming next.

### Module 12: World Models
**Learning physics and dynamics**

**What we'll cover:**
- What is a world model?
- Learning from video (Sora, Runway)
- Physical understanding and prediction
- Simulation and planning
- Robotics applications
- Current limitations
- The path to embodied AI

**World model vision:**
```
Current LLMs: Predict next token (words)
World models: Predict next state (physics, dynamics)

"Imagination" for planning and reasoning
```

---

### Module 13: Multi-head Latent Attention (MLA)
**Extreme KV cache compression**

**What we'll cover:**
- KV cache as the inference bottleneck
- DeepSeek-V2's MLA approach
- Latent compression of keys/values
- 90%+ memory reduction
- Tradeoffs with standard attention
- Enabling million-token contexts
- Implementation considerations

---

### Module 14: Frontier Directions (2027+)
**Speculative but important**

**What we'll cover:**
- Continual learning without forgetting
- Self-improving systems
- Neurosymbolic integration
- Truly multimodal reasoning
- Embodied intelligence
- AI-AI collaboration
- What might surprise us

---

## 🗂️ Project Structure

```
14-emerging-ai-trends/
├── LEARNING_PATH.md          # This file
├── requirements.txt          # Dependencies
├── verify_setup.py           # Environment verification
│
├── 01_system1_system2/
│   └── README.md
│
├── 02_test_time_compute/
│   └── README.md
│
├── 03_chain_of_thought/
│   └── README.md
│
├── 04_looped_transformers/
│   └── README.md
│
├── 05_state_space_models/
│   └── README.md
│
├── 06_hybrid_architectures/
│   └── README.md
│
├── 07_tabular_transformers/
│   └── README.md
│
├── 08_time_series/
│   └── README.md
│
├── 09_graph_transformers/
│   └── README.md
│
├── 10_mixture_of_experts/
│   └── README.md
│
├── 11_efficient_architectures/
│   └── README.md
│
├── 12_world_models/
│   └── README.md
│
├── 13_mla_kv_compression/
│   └── README.md
│
├── 14_frontier_directions/
│   └── README.md
│
├── demo/
│   └── README.md             # Hands-on demonstrations
│
└── beyond/
    └── README.md             # Meta: how to stay current
```

---

## 📅 Recommended Learning Order

```
Week 1-2: Reasoning
├── Module 1: System 1 vs System 2
├── Module 2: Test-Time Compute
├── Module 3: Chain-of-Thought & Beyond
└── Module 4: Looped & Adaptive Transformers

Week 3-4: Architectures
├── Module 5: State Space Models (Mamba)
├── Module 6: Hybrid Architectures
└── Module 10: Mixture of Experts

Week 5-6: Domain-Specific
├── Module 7: Tabular Transformers
├── Module 8: Time Series Transformers
└── Module 9: Graph Transformers

Week 7: Efficiency
└── Module 11: Efficient Architectures

Week 8: Frontiers
├── Module 12: World Models
├── Module 13: Multi-head Latent Attention
└── Module 14: Frontier Directions
```

---

## 🚀 Let's Begin!

Start with **Module 1: System 1 vs System 2 Thinking** — understanding the fundamental shift in how we think about AI reasoning.

---

## 📖 References

### Key Papers

**Reasoning:**
- "Chain-of-Thought Prompting Elicits Reasoning" (Wei et al., 2022)
- "Tree of Thoughts" (Yao et al., 2023)
- "Test-Time Compute: From System-1 to System-2" (Ji et al., 2025)
- "Looped Transformers for Algorithmic Reasoning" (2026)

**Architectures:**
- "Mamba: Linear-Time Sequence Modeling" (Gu & Dao, 2023)
- "Jamba: Hybrid Transformer-Mamba" (AI21, 2024)
- "Mixtral of Experts" (Mistral AI, 2024)

**Domain-Specific:**
- "TabPFN: A Transformer That Solves Small Tabular Problems" (2022)
- "FT-Transformer: Revisiting Deep Learning for Tabular Data" (2021)
- "Temporal Fusion Transformers" (Lim et al., 2021)

**Efficiency:**
- "Phi-3 Technical Report" (Microsoft, 2024)
- "DeepSeek-V2: Multi-head Latent Attention" (2024)

### Resources
- arXiv cs.LG, cs.CL for latest papers
- Papers With Code for implementations
- Hugging Face Blog for practical guides
- Sebastian Raschka's newsletter

