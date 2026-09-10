# Module 2: Pre-training Objectives

What loss functions teach language models.

## Overview

The pre-training objective determines what the model learns. Different objectives lead to different capabilities.

## Key Topics

### Causal Language Modeling (GPT-style)
```
Objective: Predict the next token given all previous tokens
Loss: Cross-entropy on next token prediction

"The cat sat on the" → predict "mat"

Used by: GPT series, LLaMA, Claude, most modern LLMs
Strength: Natural for generation
```

### Masked Language Modeling (BERT-style)
```
Objective: Predict masked tokens from context (both directions)
Loss: Cross-entropy on masked tokens only

"The [MASK] sat on the mat" → predict "cat"

Used by: BERT, RoBERTa
Strength: Bidirectional understanding
Weakness: Not natural for generation
```

### Other Objectives
- **Prefix LM**: Bidirectional on prefix, causal on suffix
- **Span Corruption**: T5-style, predict corrupted spans
- **Fill-in-the-Middle (FIM)**: For code models, predict middle given start/end

## Exercises

1. Implement causal LM loss computation
2. Compare MLM vs CLM on a small dataset
3. Experiment with FIM for code

## Key Insight

Causal LM's simplicity is its strength — just predict the next token, but that's enough to learn language.
