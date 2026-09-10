# Module 6: Distributed Training

Parallelism strategies for large models.

## Overview

Large models don't fit on one GPU. Distributed training strategies enable training at scale.

## Key Topics

### Data Parallelism
```
Same model on each GPU, different data batches
- Simple to implement
- All-reduce for gradient sync
- Memory: model replicated on each GPU

Limitation: Model must fit on one GPU
```

### Tensor/Model Parallelism
```
Split layers across GPUs (Megatron-style)
- Attention heads on different GPUs
- FFN split across GPUs
- Requires high-bandwidth interconnect

Used when: Model too large for one GPU
```

### Pipeline Parallelism
```
Different layers on different GPUs
- Micro-batching to hide latency
- Bubble overhead (some GPUs idle)
- Lower bandwidth requirements

GPipe, PipeDream patterns
```

### ZeRO Optimization
```
Zero Redundancy Optimizer (DeepSpeed):
- Stage 1: Shard optimizer states
- Stage 2: + Shard gradients
- Stage 3: + Shard parameters

Trade-off: Communication vs memory
```

### FSDP (Fully Sharded Data Parallel)
```
PyTorch's ZeRO implementation
- Shards parameters across GPUs
- All-gather before forward
- Reduce-scatter after backward
```

## Exercises

1. Implement basic DDP training
2. Configure FSDP for a medium model
3. Analyze communication patterns

## Key Insight

The right parallelism strategy depends on model size, hardware, and interconnect. Often combine multiple strategies.
