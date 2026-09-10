# Module 12: Model Analysis

Looking inside the black box.

## Overview

Interpretability techniques reveal what models learn. This matters for trust, debugging, and safety.

## Key Topics

### Probing
```
Idea: Train simple classifier on hidden states
Question: What information is encoded at each layer?

Example: Does layer 5 encode part-of-speech?
- Extract hidden states for many sentences
- Train linear classifier to predict POS
- High accuracy → information is there
```

### Attention Analysis
```
Visualize attention patterns:
- What tokens attend to what?
- Do attention heads specialize?
- "Previous token" heads, "rare token" heads, etc.

Caution: Attention ≠ importance (not always)
```

### Mechanistic Interpretability
```
Goal: Reverse-engineer the algorithm
Find "circuits" — minimal subgraphs for behaviors

Example: Indirect Object Identification
"John gave Mary a book. Mary gave _ a book"
→ Specific attention heads track subjects/objects
```

### Sparse Autoencoders
```
Train autoencoder on activations with sparsity
- Decompose activations into interpretable features
- Each feature may correspond to a concept
- Anthropic's work on this

Example features: "code", "French", "deception"
```

### Model Editing
```
Goal: Change specific facts without retraining
"The Eiffel Tower is in Paris" → "London"

Techniques:
- ROME: Rank-one model editing
- MEMIT: Mass editing
- Locating and patching specific weights
```

## Exercises

1. Train probing classifiers on different layers
2. Visualize attention patterns
3. Attempt a simple model edit

## Key Insight

Interpretability is hard but essential. We're still early in understanding what models learn.
