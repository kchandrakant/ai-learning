# Paper Reading Template

Use this template for every paper you read. Fill it out during your reading passes.

---

## Paper Information

**Title:**  
**Authors:**  
**Year:**  
**Venue:** (Conference/Journal)  
**Link:** (arXiv, PDF, etc.)  

**Date Read:**  
**Reading Time:** (total across all passes)

---

## Pass 1: Survey (5-10 minutes)

### First Impressions

**Paper type:** (New method / Analysis / Survey / Application)

**Main claim/contribution:**
> (One sentence from abstract)

**Key figure:** (Figure number and what it shows)


**Should I continue?** [ ] Yes / [ ] No

**Why/why not:**

---

## Pass 2: Comprehension (30-60 minutes)

### Problem Statement

**What problem does this paper solve?**


**Why is this problem important?**


**What are existing approaches and their limitations?**


### Proposed Method

**What is the key idea/insight?**


**How does the method work?** (High-level description)


**Key equations/algorithms:** (Copy the most important ones)


**What's novel compared to prior work?**


### Experimental Setup

**Datasets used:**

| Dataset | Size | Description |
|---------|------|-------------|
|         |      |             |

**Baselines compared:**

| Baseline | Why included? |
|----------|---------------|
|          |               |

**Evaluation metrics:**


### Key Results

**Main result table/figure:** (Summarize key numbers)


**Ablation studies:** (What happens when components are removed?)


**Does it actually work?** [ ] Yes / [ ] Partially / [ ] No

---

## Pass 3: Mastery (1-4 hours) [Optional]

*Only complete this for papers you'll implement, cite, or present.*

### Detailed Analysis

**Mathematical details I now understand:**


**Implementation details:**


**Hyperparameters that matter:**

| Parameter | Value | Sensitivity |
|-----------|-------|-------------|
|           |       |             |

**Reproduce key result:** (Did you try? What happened?)


### Critical Analysis

**Strengths:**
1. 
2. 
3. 

**Weaknesses/Limitations:**
1. 
2. 
3. 

**Questions I still have:**
1. 
2. 

**What would I do differently?**


---

## Summary

### One-Paragraph Summary
*(Write this after completing your reading. Should be understandable by someone who hasn't read the paper.)*


### Key Takeaways
1. 
2. 
3. 

### How might I use this?


### Related Papers to Read
- 
- 
- 

---

## Quick Reference

### Math Notation Used in This Paper

| Symbol | Meaning |
|--------|---------|
|        |         |

### Key Terms Defined

| Term | Definition |
|------|------------|
|      |            |

---

## Tags

**Topics:** #transformers #attention #nlp  
**Techniques:** #self-supervision #pretraining  
**Application:** #language-modeling #translation  
**Quality:** ⭐⭐⭐⭐⭐ (1-5 stars)  
**Difficulty:** Easy / Medium / Hard  
**Must-read:** [ ] Yes / [ ] No

---

# Example: Completed Template

## Paper Information

**Title:** Attention Is All You Need  
**Authors:** Vaswani, Shazeer, Parmar, et al.  
**Year:** 2017  
**Venue:** NeurIPS  
**Link:** https://arxiv.org/abs/1706.03762  

**Date Read:** 2024-01-15  
**Reading Time:** 2.5 hours (across 3 passes)

---

## Pass 1: Survey (10 minutes)

### First Impressions

**Paper type:** New method

**Main claim/contribution:**
> "We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely."

**Key figure:** Figure 1 - The Transformer architecture diagram showing encoder-decoder structure with attention blocks.

**Should I continue?** [x] Yes / [ ] No

**Why/why not:** This is THE foundational paper for modern NLP. Must understand.

---

## Pass 2: Comprehension (45 minutes)

### Problem Statement

**What problem does this paper solve?**
Sequence-to-sequence modeling (like translation) using architectures that can:
1. Train faster (parallelizable, unlike RNNs)
2. Handle long-range dependencies better

**Why is this problem important?**
RNNs are slow (sequential) and struggle with long sequences. Faster, better models enable more applications.

**What are existing approaches and their limitations?**
- RNNs/LSTMs: Sequential processing prevents parallelization; gradient issues with long sequences
- ConvNets for sequences: Limited receptive field unless very deep

### Proposed Method

**What is the key idea/insight?**
Attention alone (without recurrence or convolution) is sufficient for sequence modeling. Self-attention lets every position attend to every other position in one step.

**How does the method work?**
1. Input embeddings + positional encoding
2. Stack of encoder blocks (self-attention + feed-forward)
3. Stack of decoder blocks (masked self-attention + cross-attention + feed-forward)
4. Each attention uses Query, Key, Value projections

**Key equations:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

**What's novel compared to prior work?**
- No recurrence at all (fully parallelizable)
- Multi-head attention (attend to different things in parallel)
- Scaled dot-product attention (√d_k prevents saturation)

### Experimental Setup

**Datasets used:**

| Dataset | Size | Description |
|---------|------|-------------|
| WMT 2014 En-De | 4.5M pairs | English-German translation |
| WMT 2014 En-Fr | 36M pairs | English-French translation |

**Baselines compared:**

| Baseline | Why included? |
|----------|---------------|
| ConvS2S | Previous CNN-based SOTA |
| GNMT (Google) | Previous RNN-based SOTA |
| Deep-Att + PosUnk | Strong attention + RNN |

**Evaluation metrics:** BLEU score

### Key Results

**Main results:**
- En-De: 28.4 BLEU (vs 25.8 previous SOTA) - **new record**
- En-Fr: 41.0 BLEU (vs 40.4 previous SOTA) - **new record**
- Training time: 3.5 days on 8 P100 GPUs (much faster than RNNs)

**Ablation studies:**
- Removing positional encoding hurts significantly
- More attention heads better (up to a point)
- Larger d_model helps, but diminishing returns

**Does it actually work?** [x] Yes

---

## Summary

### One-Paragraph Summary

The Transformer is a sequence-to-sequence architecture that uses only attention mechanisms, completely eliminating recurrence and convolution. The key innovation is self-attention, which allows every position in a sequence to attend to every other position in a single step, enabling massive parallelization during training. The architecture uses multi-head attention (multiple attention functions in parallel) and positional encodings (since there's no recurrence to provide position information). On machine translation benchmarks, the Transformer achieves state-of-the-art results while training significantly faster than RNN-based models.

### Key Takeaways
1. Self-attention can replace recurrence for sequence modeling
2. Multi-head attention lets the model attend to different aspects simultaneously
3. Positional encoding is necessary since attention is permutation-invariant

### How might I use this?
Foundation for understanding BERT, GPT, and all modern NLP. Will implement from scratch for learning.

### Related Papers to Read
- BERT (bidirectional pretraining)
- GPT (decoder-only, language modeling)
- Vision Transformer (applying to images)

---

## Tags

**Topics:** #transformers #attention #nlp  
**Techniques:** #encoder-decoder #self-attention  
**Application:** #machine-translation  
**Quality:** ⭐⭐⭐⭐⭐  
**Difficulty:** Medium  
**Must-read:** [x] Yes

---

# Reading Checklist

## Before Reading
- [ ] Check paper length and allocate appropriate time
- [ ] Have notebook/template ready for notes
- [ ] Review related concepts if needed

## During Pass 1
- [ ] Read title, abstract, keywords
- [ ] Look at all figures and captions
- [ ] Read introduction first/last paragraphs
- [ ] Read conclusion
- [ ] Decide: continue or stop?

## During Pass 2
- [ ] Read introduction fully
- [ ] Read method section (skim heavy math)
- [ ] Study key figures in detail
- [ ] Read results and key tables
- [ ] Mark sections/equations to revisit

## During Pass 3 (if needed)
- [ ] Work through all math
- [ ] Try to implement key algorithm
- [ ] Read related work section
- [ ] Check appendix for details
- [ ] Compare to other papers on same topic

## After Reading
- [ ] Write summary in own words
- [ ] Identify 3 key takeaways
- [ ] Add to paper collection/database
- [ ] Schedule review in 1 week
