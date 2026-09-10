# Module 1: Tokenization

Breaking text into learnable units.

## Overview

Tokenization is how models "see" text. The choices made here fundamentally shape what models can and cannot learn.

## Key Topics

### Why Tokenization Matters
- Vocabulary size affects model capacity and efficiency
- Token boundaries shape compositional understanding
- Poor tokenization → poor performance on specific domains

### Byte-Pair Encoding (BPE)
```
Algorithm:
1. Start with character vocabulary
2. Count adjacent pair frequencies
3. Merge most frequent pair into new token
4. Repeat until vocabulary size reached

Example:
"low lower lowest" → ["l", "o", "w", " ", "e", "r", "s", "t"]
→ merge "lo" → ["lo", "w", " ", "e", "r", "s", "t", "lo"]
→ merge "low" → ["low", " ", "e", "r", "s", "t", "low"]
```

### Vocabulary Size Trade-offs
```
Small vocab (32K):  More tokens per text, compositional
Large vocab (100K): Fewer tokens, but larger embedding table
Sweet spot:         50K-100K for most LLMs
```

### Multilingual Challenges
- Non-Latin scripts often over-tokenized
- "hello" = 1 token, "नमस्ते" = 5+ tokens
- Affects cost and capability for non-English users

## Exercises

1. Train a BPE tokenizer from scratch
2. Compare GPT-2 vs Llama tokenizers
3. Analyze tokenization of code vs prose
4. Visualize vocabulary distribution

## Key Insight

Tokenization is not neutral. It encodes assumptions about language that propagate through training and inference.
