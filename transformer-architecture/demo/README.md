# Demo: Transformer in Action

## Overview

This directory contains demonstrations of the complete transformer, bringing together all the components we built.

## Demos

### 1. Forward Pass Visualization
See data flow through the transformer:
- Token → Embedding → Position → Encoder → Decoder → Output

### 2. Attention Visualization
Visualize what the model attends to:
- Self-attention patterns in encoder
- Causal attention in decoder
- Cross-attention between encoder/decoder

### 3. Simple Training Example
Train a small transformer on a toy task:
- Copy task (learn to repeat input)
- Sorting task (learn to sort numbers)

### 4. Text Generation
Autoregressive generation with a decoder-only model.

## Running the Demos

```bash
# Forward pass demo
python demo/forward_pass_demo.py

# Attention visualization
python demo/attention_viz.py

# Training example
python demo/train_copy_task.py
```

## Expected Outputs

Each demo generates:
- Console output explaining what's happening
- Visualizations saved as PNG files
- Learned model checkpoints (for training demos)

## Files

- `forward_pass_demo.py` - Step-by-step forward pass
- `attention_viz.py` - Attention pattern visualization
- `train_copy_task.py` - Training on a simple task

## What You'll Learn

1. How data flows through the transformer
2. What attention patterns look like
3. How the model learns from examples
4. The difference between training and inference

## Prerequisites

Complete Steps 1-8 first to build all the components!
