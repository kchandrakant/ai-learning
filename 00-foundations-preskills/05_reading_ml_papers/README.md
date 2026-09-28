# Module 5: Reading ML Papers

Every course in this learning path references research papers. "Attention Is All You Need," "BERT," "GPT-3," "LoRA" — these aren't just names, they're papers that define how modern AI works. Reading papers efficiently is a skill that unlocks deeper understanding and keeps you connected to the source of new ideas.

---

## 🎯 Why This Matters

When you read a blog post about transformers, you're reading someone's interpretation. When you read "Attention Is All You Need," you're learning from the source.

```
┌─────────────────────────────────────────────────────────────────────┐
│                 Primary vs Secondary Sources                        │
│                                                                     │
│   Paper (Primary):                Secondary Sources:                │
│   ────────────────                ─────────────────                 │
│   • Exact method details          • Blog posts: may simplify/error  │
│   • What didn't work              • Tutorials: often outdated       │
│   • Assumptions & limitations     • Videos: can misinterpret        │
│   • Why design choices            • Docs: focus on "how" not "why"  │
│   • Ablation studies                                                │
│                                                                     │
│   Reading papers = Learning what the authors actually discovered    │
│   Reading blogs  = Learning what someone thinks they discovered     │
│                                                                     │
│   Papers tell you: "We tried X, Y, Z. X failed. Y worked but only   │
│   under condition A. Z worked best, here's why we think so."        │
│   Blogs say: "Use Z, it's great!"                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Papers Reveal What Blogs Hide

- **Limitations:** Every method has failure modes. Papers document them. Blogs often don't.
- **Ablations:** "What happens if we remove component X?" Papers show this. Critical for understanding what actually matters.
- **Context:** Why was this problem hard? What had been tried before? Papers establish this.
- **Details:** Implementation specifics that matter for reproduction.

---

## 📚 Part 1: Anatomy of an ML Paper

Most ML papers follow this structure. Knowing it helps you navigate efficiently.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ML Paper Structure                              │
│                                                                     │
│   ┌─────────────┐                                                   │
│   │  Abstract   │  ← Read FIRST. Whole paper in one paragraph.      │
│   └─────────────┘                                                   │
│   ┌─────────────┐                                                   │
│   │Introduction │  ← Problem context, motivation, contribution      │
│   └─────────────┘    summary. Read intro first/last paragraphs.     │
│   ┌─────────────┐                                                   │
│   │Related Work │  ← Prior approaches. SKIM. Return if needed.      │
│   └─────────────┘                                                   │
│   ┌─────────────┐                                                   │
│   │   Method    │  ← THE CORE. How it works. Read CAREFULLY.        │
│   └─────────────┘                                                   │
│   ┌─────────────┐                                                   │
│   │ Experiments │  ← Does it work? How much better? Ablations.      │
│   └─────────────┘                                                   │
│   ┌─────────────┐                                                   │
│   │ Conclusion  │  ← Summary + future directions. Read EARLY.       │
│   └─────────────┘                                                   │
│   ┌─────────────┐                                                   │
│   │  Appendix   │  ← Details, proofs. Read only if implementing.    │
│   └─────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Section-by-Section Guide

| Section | Length | Read When | Key Questions |
|---------|--------|-----------|---------------|
| Abstract | ~200 words | Always first | What problem? What approach? What results? |
| Introduction | 1-2 pages | Pass 1 & 2 | Why does this matter? What's the gap? |
| Related Work | ~1 page | If confused about context | What came before? How is this different? |
| Method | 3-5 pages | Pass 2 (core focus) | How does it work? Why these choices? |
| Experiments | 3-5 pages | Pass 2 | Does it work? Under what conditions? |
| Conclusion | ~0.5 page | Early (Pass 1) | What did they achieve? What's next? |
| Appendix | Varies | Only if implementing | Implementation details, extra results |

---

## 📚 Part 2: The Three-Pass Strategy

Don't read papers linearly. Use multiple passes at increasing depth.

```
┌─────────────────────────────────────────────────────────────────────┐
│                     The Three-Pass Strategy                         │
│                                                                     │
│   Pass 1: Survey (5-10 min)                                         │
│   ─────────────────────────                                         │
│   → Title, abstract, section headings                               │
│   → Figures & tables (with captions)                                │
│   → Conclusion                                                      │
│   Goal: Decide if worth reading further                             │
│                                                                     │
│   Pass 2: Comprehension (30-60 min)                                 │
│   ──────────────────────────────────                                │
│   → Full intro                                                      │
│   → Method (focus on intuition, skim math)                          │
│   → Key results & figures                                           │
│   Goal: Explain the paper to someone                                │
│                                                                     │
│   Pass 3: Mastery (1-4 hours)                                       │
│   ──────────────────────────                                        │
│   → Work through all math                                           │
│   → Verify claims                                                   │
│   → Think about implementation                                      │
│   Goal: Could implement or extend this                              │
│                                                                     │
│   Most papers: Pass 1 only                                          │
│   Important papers: Pass 1 + 2                                      │
│   Papers you'll use: Pass 1 + 2 + 3                                 │
└─────────────────────────────────────────────────────────────────────┘
```

### Pass 1: Survey (5-10 minutes)

**Goal:** Decide if the paper is worth reading.

**Read:**
1. Title, abstract, keywords
2. Introduction (first and last paragraphs only)
3. Section headings (scan the structure)
4. Figures and tables (with captions)
5. Conclusion

**After Pass 1, you should know:**
- What type of paper (new method, analysis, survey)
- What problem it addresses
- The claimed contribution
- Whether to continue

> **Tip:** The figures often tell the whole story. In a good ML paper, Figure 1 is the method overview, and a table shows the main results.

### Pass 2: Comprehension (30-60 minutes)

**Goal:** Understand the main ideas without verifying details.

**Read:**
- Full introduction
- Method section (focus on intuition, skim heavy math)
- Key figures and results
- Conclusion

**Active reading techniques:**
- Circle/highlight key terms
- Note concepts you don't understand (look up later)
- Summarize each section in 1-2 sentences
- Draw a simplified version of the main figure

**After Pass 2, you should be able to:**
- Explain the paper to someone in 5 minutes
- Identify the key contribution
- Know the main limitations
- Decide if you need Pass 3

### Pass 3: Mastery (1-4 hours)

**Goal:** Deep understanding, suitable for implementation or extension.

**Actions:**
- Work through all math derivations
- Verify claims and reasoning
- Recreate figures mentally
- Compare with related work
- Think: "How would I implement this?"

**Only do Pass 3 for papers you'll:**
- Implement
- Build upon in your own work
- Present to others
- Cite significantly

---

## 📚 Part 3: Reading Mathematical Notation

ML papers use standard notation. Here's a quick reference:

### Common Symbols

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ML Notation Cheat Sheet                         │
│                                                                     │
│   Scalars, Vectors, Matrices:                                       │
│   ───────────────────────────                                       │
│   x, y, z        Scalars (lowercase)                                │
│   x, w, b        Vectors (bold lowercase) — sometimes just x        │
│   W, X, A        Matrices (bold uppercase) — sometimes just W       │
│   X ∈ ℝ^(n×d)    X is an n×d matrix of real numbers                │
│                                                                     │
│   Parameters:                                                       │
│   ───────────                                                       │
│   θ (theta)      All model parameters                               │
│   W, b           Weights and biases                                 │
│   α, β, γ        Hyperparameters                                    │
│                                                                     │
│   Operations:                                                       │
│   ───────────                                                       │
│   ∑              Sum                                                │
│   ∏              Product                                            │
│   ∂L/∂θ          Partial derivative of L with respect to θ          │
│   ∇L             Gradient of L (vector of all partials)             │
│   ‖x‖            Norm (length) of x                                 │
│   x^T            Transpose                                          │
│   x · y or x^T y Dot product                                        │
│                                                                     │
│   Special Notation:                                                 │
│   ────────────────                                                  │
│   x̂ (x-hat)      Estimate or prediction                             │
│   x*             Optimal value                                      │
│   x'             Derivative or alternative                          │
│   x̄ (x-bar)      Mean                                               │
└─────────────────────────────────────────────────────────────────────┘
```

### Probability Notation

| Symbol | Meaning | Example |
|--------|---------|---------|
| p(x) | Probability of x | p(y=1) = 0.7 |
| p(x\|y) | Probability of x given y | p(word\|context) |
| 𝔼[x] | Expected value | 𝔼[loss] |
| 𝔼_p[x] | Expectation under distribution p | 𝔼_data[x] |
| ∼ | "distributed as" | x ∼ N(0, 1) |
| argmax | Value that maximizes | argmax_θ p(y\|x; θ) |
| argmin | Value that minimizes | argmin_θ L(θ) |

### Common Functions

| Notation | Meaning | Formula |
|----------|---------|---------|
| σ(x) | Sigmoid | 1 / (1 + e^(-x)) |
| softmax(x)_i | Softmax | e^(x_i) / Σ_j e^(x_j) |
| ReLU(x) | Rectified Linear Unit | max(0, x) |
| log | Natural log (usually) | ln(x) |
| ‖x‖_2 | L2 norm | √(Σ x_i²) |
| ‖x‖_1 | L1 norm | Σ \|x_i\| |

### Reading Equations: A Process

When you encounter a complex equation:

```
┌─────────────────────────────────────────────────────────────────────┐
│              How to Read an ML Equation                             │
│                                                                     │
│   Example: L = -∑ᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]                 │
│                                                                     │
│   Step 1: Identify inputs and outputs                               │
│   ───────────────────────────────────                               │
│   • What goes in? yᵢ (true labels), ŷᵢ (predictions)               │
│   • What comes out? L (a scalar loss value)                         │
│                                                                     │
│   Step 2: Break into components                                     │
│   ──────────────────────────────                                    │
│   • -∑ᵢ: Sum over all samples, negate at end                        │
│   • yᵢ log(ŷᵢ): When y=1, penalize if ŷ is small                   │
│   • (1-yᵢ) log(1-ŷᵢ): When y=0, penalize if ŷ is large            │
│                                                                     │
│   Step 3: Check dimensions                                          │
│   ────────────────────────                                          │
│   • yᵢ: scalar (0 or 1)                                             │
│   • ŷᵢ: scalar (0 to 1)                                             │
│   • Sum over i: produces scalar                                     │
│   ✓ Dimensions make sense                                           │
│                                                                     │
│   Step 4: Find the intuition                                        │
│   ──────────────────────────                                        │
│   "Penalize predictions that disagree with labels"                  │
│   This is binary cross-entropy loss.                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Part 4: Guided Reading — "Attention Is All You Need"

Let's apply these techniques to the foundational transformer paper.

### Paper Info
- **Title:** Attention Is All You Need
- **Authors:** Vaswani et al. (Google)
- **Year:** 2017
- **Link:** [arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

### Pass 1: Survey (10 minutes)

**Abstract — Key sentences:**
> "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."
> "The Transformer... achieves 28.4 BLEU on the WMT 2014 English-to-German translation task..."
> "training for 3.5 days on eight GPUs"

**What I learned from abstract:**
- New architecture called "Transformer"
- Uses only attention (no RNN, no CNN)
- State-of-the-art translation results
- Trains fast (3.5 days vs weeks for RNNs)

**Figure 1 (THE figure to study):**
```
┌─────────────────────────────────────────────────────────────────────┐
│              Transformer Architecture (Simplified)                  │
│                                                                     │
│   Encoder Stack:                  Decoder Stack:                    │
│   ┌─────────────┐                ┌─────────────┐                   │
│   │ Multi-Head  │                │ Multi-Head  │                   │
│   │  Attention  │                │  Attention  │ ← Masked          │
│   └─────────────┘                └─────────────┘                   │
│         ↓                              ↓                           │
│   ┌─────────────┐                ┌─────────────┐                   │
│   │ Feed Forward│                │Cross-Attend │ ← Encoder output  │
│   └─────────────┘                └─────────────┘                   │
│         ↓                              ↓                           │
│   (repeat N times)               ┌─────────────┐                   │
│                                  │ Feed Forward│                   │
│                                  └─────────────┘                   │
│                                        ↓                           │
│                                  (repeat N times)                   │
│                                                                     │
│   Key insight: No recurrence! All positions processed in parallel.  │
└─────────────────────────────────────────────────────────────────────┘
```

**Conclusion — Key points:**
- "The Transformer is the first transduction model relying entirely on self-attention"
- Plans to apply to other modalities (images, audio, video)
- Code available

**Pass 1 verdict:** Highly relevant — this is the foundation of modern NLP. Proceed to Pass 2.

### Pass 2: Comprehension (45 minutes)

**Introduction insights:**
- RNNs process sequentially → slow, hard to parallelize
- Attention was previously used WITH RNNs
- Key insight: **attention alone is sufficient**

**Method — The Core Components:**

**1. Scaled Dot-Product Attention (Section 3.2.1)**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Breaking it down:
- Q (Query), K (Key), V (Value) are matrices
- QK^T computes similarity scores between all query-key pairs
- √d_k scales down large values (prevents softmax saturation)
- softmax normalizes to weights that sum to 1
- Multiply by V to get weighted combination of values

> **Intuition:** "Look up relevant information. Q asks 'what am I looking for?', K says 'here's what I have', V says 'here's the actual content'. Attention returns a weighted mix of V based on Q-K similarity."

**2. Multi-Head Attention (Section 3.2.2)**
- Run attention h times in parallel with different learned projections
- Each "head" can attend to different aspects
- Concatenate outputs and project

> **Intuition:** "One attention head might focus on syntax, another on semantics, another on position. Multiple heads capture different types of relationships."

**3. Positional Encoding (Section 3.5)**
- Transformers have no inherent sense of position
- Add sinusoidal functions to encode position
- sin/cos at different frequencies

> **Intuition:** "Since we process all positions at once (no sequence), we need to tell the model where each token is. We encode position as a unique 'fingerprint' using sine waves."

**Key Results (Table 2):**
- 28.4 BLEU on English-German (new state-of-the-art)
- 41.0 BLEU on English-French
- Training: 3.5 days on 8 GPUs

### Pass 2 Summary

> The Transformer replaces recurrent networks with pure attention, enabling parallel processing of all sequence positions. The core mechanism is scaled dot-product attention, which computes relevance between queries and keys, then retrieves weighted values. Multi-head attention runs this multiple times to capture different relationship types. Positional encodings inject sequence order information. This achieves state-of-the-art translation while training much faster than RNNs.

---

## 📚 Part 5: Guided Reading — "BERT"

### Paper Info
- **Title:** BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- **Authors:** Devlin et al. (Google)
- **Year:** 2018
- **Link:** [arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)

### Quick Pass 1+2

**Core Problem:**
Previous models (including GPT) were *unidirectional* — they could only look left-to-right or right-to-left, not both simultaneously.

**Key Innovation — Bidirectional:**
```
┌─────────────────────────────────────────────────────────────────────┐
│              Unidirectional vs Bidirectional                        │
│                                                                     │
│   GPT (Left-to-Right):                                              │
│   "The cat sat on the ___"                                          │
│    ←──────────────────                                              │
│   Can only see words before the blank                               │
│                                                                     │
│   BERT (Bidirectional):                                             │
│   "The cat ___ on the mat"                                          │
│    ←─────────→  ←─────────                                          │
│   Sees context from BOTH directions                                 │
│                                                                     │
│   Why this matters: "bank" means different things in                │
│   "river bank" vs "bank account" — you need full context.           │
└─────────────────────────────────────────────────────────────────────┘
```

**Pre-training Tasks:**

1. **Masked Language Model (MLM):**
   - Randomly mask 15% of tokens
   - Model predicts the masked tokens
   - Forces bidirectional understanding

2. **Next Sentence Prediction (NSP):**
   - Given sentence A, is B the actual next sentence?
   - Teaches sentence-level relationships

**Why BERT Matters:**
- One pre-trained model → fine-tune for many tasks
- Much less labeled data needed for downstream tasks
- Established "pre-train then fine-tune" paradigm

### Summary

> BERT pre-trains a bidirectional transformer using masked language modeling, where it learns to predict randomly hidden words from full context. Unlike GPT's left-to-right approach, BERT sees both directions simultaneously. Fine-tuning this pre-trained model on downstream tasks achieves state-of-the-art across NLP benchmarks with minimal task-specific data.

---

## 📚 Part 6: Building a Paper Reading Habit

### Sustainable Goals

| Level | Papers/Week | Strategy |
|-------|-------------|----------|
| Minimum | 1 deep read | Pick one important paper, do Pass 1+2 |
| Better | 1 deep + 5 abstracts | Scan more, deep dive on one |
| Advanced | 2-3 deep reads | For active researchers |

### When to Read

- **Morning:** Fresh mind for complex papers
- **Commute:** Abstracts and Pass 1 on phone (use Semantic Scholar app)
- **Lunch:** Pass 2 reading
- **Before bed:** Light reading of interesting papers

### Active Reading Template

Use this template for every paper you read seriously:

```markdown
# [Paper Title]

**Authors:** 
**Year:** 
**Link:** 

## One-Sentence Summary
[What is this paper about in one sentence?]

## Problem
What problem does this solve? Why does it matter?

## Key Idea
What's the core insight or contribution?

## Method (How)
How does it work? (High-level, 3-5 bullet points)

## Results
Does it work? How well? Key numbers.

## Limitations
What doesn't work? What assumptions are made?

## Connections
How does this relate to other papers I know?

## My Thoughts
What did I learn? How might I use this?
```

### Paper Annotation Strategy

```
┌─────────────────────────────────────────────────────────────────────┐
│              Annotation System                                      │
│                                                                     │
│   While reading, mark:                                              │
│                                                                     │
│   ★  Key contribution (what's new)                                  │
│   ?  Didn't understand (look up later)                              │
│   !  Surprising or important                                        │
│   →  Implication or connection                                      │
│   ✗  Limitation or weakness                                         │
│                                                                     │
│   At section ends, write 1-sentence summary in margin               │
│                                                                     │
│   Tools: PDF annotations, Zotero, Notion, paper printouts           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Part 7: Paper Sources

### Where to Find Papers

| Source | Best For | URL |
|--------|----------|-----|
| arXiv | Latest preprints | arxiv.org/list/cs.LG/recent |
| Semantic Scholar | Smart search + recommendations | semanticscholar.org |
| Papers With Code | Papers + implementations | paperswithcode.com |
| Google Scholar | Comprehensive search | scholar.google.com |
| Connected Papers | Visualize relationships | connectedpapers.com |

### Finding Good Papers

**High-signal filters:**
- Conference best paper awards (NeurIPS, ICML, ICLR)
- Papers With Code leaderboards
- Citations from papers you trust
- Recommendations from researchers you follow

**Red flags:**
- No code released
- Results on unusual/proprietary datasets only
- Claims without ablations
- No comparison to recent baselines

### Essential Papers to Read

**Must-read foundational papers:**

| Paper | Year | Why It Matters |
|-------|------|----------------|
| Attention Is All You Need | 2017 | Transformers — foundation of modern AI |
| BERT | 2018 | Pre-training paradigm for NLP |
| GPT-3 (Language Models are Few-Shot Learners) | 2020 | Scaling laws, few-shot learning |
| ResNet (Deep Residual Learning) | 2015 | Skip connections, training deep networks |
| Adam | 2014 | The optimizer everyone uses |
| Dropout | 2014 | Regularization that works |
| Batch Normalization | 2015 | Training stability |
| LoRA | 2021 | Efficient fine-tuning |

---

## 🏋️ Exercises

### Exercise 1: Abstract Analysis
Read the abstract of "Language Models are Few-Shot Learners" (GPT-3 paper):

> Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by task-specific fine-tuning. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions – something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches.

**Answer:**
1. What problem does this paper address?
2. What is their approach?
3. What is the claimed result?

### Exercise 2: Equation Reading
Given this equation from a paper:

```
L = -∑_{i=1}^{N} [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)]
```

1. What type of loss function is this?
2. What do y_i and ŷ_i represent?
3. When is this loss minimized?

### Exercise 3: Paper Summary
Find and read the abstract and conclusion of the "LoRA" paper (arxiv.org/abs/2106.09685).

Write a 3-sentence summary covering:
- What problem does it solve?
- What's the key idea?
- Why does it matter?

### Exercise 4: First Full Paper
Apply the three-pass strategy to "Attention Is All You Need":
1. Do Pass 1 (10 minutes)
2. Do Pass 2 on sections 1, 3, and 6 (30 minutes)
3. Write a one-paragraph summary in your own words

---

## ✅ Solutions

<details>
<summary>Click to reveal solutions</summary>

### Exercise 1: Abstract Analysis
1. **Problem:** Current NLP requires task-specific fine-tuning with thousands of examples, unlike humans who can learn from few examples or instructions.
2. **Approach:** Scale up language models (make them much larger) and test few-shot performance without fine-tuning.
3. **Result:** Large models achieve strong few-shot performance, sometimes matching fine-tuned models, without any gradient updates on the task.

### Exercise 2: Equation Reading
1. **Binary cross-entropy loss** (also called log loss)
2. **y_i** = true label (0 or 1), **ŷ_i** = predicted probability (0 to 1)
3. Minimized when predictions match labels perfectly: ŷ_i → 1 when y_i = 1, and ŷ_i → 0 when y_i = 0

### Exercise 3: LoRA Summary
> LoRA (Low-Rank Adaptation) addresses the prohibitive cost of fine-tuning large language models by freezing pre-trained weights and only training small low-rank decomposition matrices injected into each layer. Instead of updating all millions of parameters, LoRA adds trainable matrices of much smaller rank (typically 4-64), reducing trainable parameters by 10,000x and memory requirements accordingly. This enables fine-tuning models like GPT-3 on consumer hardware while matching or exceeding full fine-tuning performance.

### Exercise 4: Paper Summary
(Your summary will vary, but should capture these key points)

> The Transformer architecture replaces recurrent connections with self-attention, allowing the model to process all positions in a sequence simultaneously rather than sequentially. The core innovation is scaled dot-product attention, which computes relevance scores between all pairs of positions using learned query, key, and value projections. Multi-head attention runs this mechanism multiple times in parallel to capture different types of relationships. Combined with positional encodings that inject sequence order information, this achieves state-of-the-art translation quality while training significantly faster than RNN-based models due to parallelization.

</details>

---

## 🎯 Key Takeaways

1. **Papers are primary sources.** Blogs simplify and sometimes distort. Papers contain what actually happened, including failures and limitations.

2. **Use the three-pass strategy.** Pass 1 (5 min) decides if it's worth reading. Pass 2 (30-60 min) builds understanding. Pass 3 (hours) is for papers you'll implement.

3. **Figures tell the story.** In good ML papers, Figure 1 shows the method, and tables show results. Start there.

4. **Learn the notation.** θ for parameters, ∇ for gradient, softmax, argmax — these appear everywhere. Build fluency.

5. **Read the ablations.** "What happens when we remove X?" tells you what actually matters in the method.

6. **Build a system.** Use a template, annotate consistently, track what you've read. Paper reading is a skill that improves with practice.

7. **Start with the classics.** Transformers, BERT, ResNet, Adam — these papers are referenced everywhere. Reading them unlocks understanding of everything built on top.

---

## 🔗 What's Next?

You can now read the papers that define modern AI. Move on to **Module 6: Staying Current** to learn how to keep up with the fast-moving field.

---

## 📖 References

- [How to Read a Paper](http://ccr.sigcomm.org/online/files/p83-keshavA.pdf) — S. Keshav (the three-pass approach)
- [Yannic Kilcher's YouTube](https://www.youtube.com/@YannicKilcher) — Paper explanations
- [Two Minute Papers](https://www.youtube.com/@TwoMinutePapers) — Quick summaries
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Visual guide to the paper
