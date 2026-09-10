# LLM Internals - Demonstrations

Hands-on demonstrations of LLM training and analysis concepts.

## Included Demos

### Demo 1: Tokenizer Training
Train a BPE tokenizer from scratch on a small corpus.
```bash
python demo/01_tokenizer_training.py
```

### Demo 2: Pre-training Loop
Minimal pre-training on a toy dataset to see the training dynamics.
```bash
python demo/02_pretraining_loop.py
```

### Demo 3: Scaling Analysis
Visualize scaling laws with experiments at different sizes.
```bash
python demo/03_scaling_analysis.py
```

### Demo 4: RLHF Mini
Simplified RLHF pipeline with a small model.
```bash
python demo/04_rlhf_mini.py
```

### Demo 5: Probing Experiments
Probe model internals to understand what's learned.
```bash
python demo/05_probing.py
```

### Demo 6: Attention Visualization
Visualize attention patterns across layers and heads.
```bash
python demo/06_attention_viz.py
```

## Requirements

Most demos require GPU for reasonable runtime. Some demos use small models that can run on CPU for educational purposes.
