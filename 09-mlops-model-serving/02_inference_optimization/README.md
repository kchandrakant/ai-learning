# Step 2: Inference Optimization

## Why Optimize?

Raw inference is slow and expensive. Optimization can give 2-10x improvements.

## Quantization

Reduce precision to save memory and speed up computation.

```
FP32 (32-bit)  → Baseline, highest quality
FP16 (16-bit)  → 2x memory reduction, minimal loss
INT8 (8-bit)   → 4x memory reduction, small loss
INT4 (4-bit)   → 8x memory reduction, noticeable loss
```

**Methods:**
- **GPTQ**: Post-training quantization
- **AWQ**: Activation-aware quantization
- **GGUF**: llama.cpp format

## KV Cache Optimization

The KV cache stores key/value pairs to avoid recomputation.

```
Without cache: Recompute attention for all tokens
With cache:    Only compute for new token

Memory: O(batch_size × seq_len × hidden_dim × num_layers)
```

**PagedAttention (vLLM):**
- Allocates KV cache in pages
- Eliminates fragmentation
- Enables larger batches

## Continuous Batching

Traditional batching waits for all requests to finish.
Continuous batching adds new requests as slots free up.

```
Traditional:  [req1, req2, req3] → wait for all → [req4, req5, req6]
Continuous:   [req1, req2, req3] → req1 done → [req2, req3, req4]
```

## Flash Attention

Optimized attention computation:
- Fuses operations
- Reduces memory I/O
- 2-4x speedup

## Files

- `inference_optimization.py` - Optimization techniques
