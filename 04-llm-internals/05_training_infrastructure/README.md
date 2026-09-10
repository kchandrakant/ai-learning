# Module 5: Training Infrastructure

Hardware and systems for LLM training.

## Overview

Training large models requires understanding hardware constraints. Infrastructure decisions shape what's possible.

## Key Topics

### GPU Architecture
```
Key specs (A100 vs H100):
- Memory: 40-80GB HBM
- Bandwidth: 2-3 TB/s
- Compute: 300-1000 TFLOPS (FP16)
- Interconnect: NVLink (900 GB/s)
```

### Memory Hierarchy
```
Registers → L1/L2 Cache → HBM → CPU RAM → NVMe

LLM training is memory-bound:
- Model weights
- Optimizer states (2x weights for Adam)
- Activations (for backward pass)
- Gradients
```

### Interconnects
```
Within node:  NVLink (600-900 GB/s)
Across nodes: InfiniBand (100-400 Gb/s)

Bandwidth affects parallelism strategy:
- High bandwidth → can split model across GPUs
- Low bandwidth → prefer data parallelism
```

### Cost Estimation
```
Llama-2-70B approximate:
- ~1.7M GPU hours
- At $2/GPU-hour: ~$3.4M
- Real cost higher (failed runs, experiments)
```

## Exercises

1. Calculate memory requirements for a model
2. Estimate training cost for different configurations
3. Profile GPU utilization during training

## Key Insight

Understanding hardware lets you design feasible training runs. Most failures are resource-related.
