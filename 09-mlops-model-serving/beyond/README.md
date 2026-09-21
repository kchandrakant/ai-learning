# MLOps & Model Serving: Beyond the Basics

## Current Trends (2024-2026)

### 1. Speculative Decoding
Use small model to draft, large model to verify.

### 2. Disaggregated Serving
Separate prefill and decode phases.

### 3. Multi-LoRA Serving
Serve multiple fine-tuned adapters efficiently.

### 4. Edge Inference
Run models on devices with WebGPU, WASM.

## Emerging Technologies

| Technology | Status | Impact |
|------------|--------|--------|
| Speculative decoding | Production | 2-3x speedup |
| PagedAttention v2 | Production | Better memory |
| FP8 inference | Early | 2x throughput |
| NPU inference | Emerging | Mobile/edge |

## Resources

- vLLM Blog
- NVIDIA TensorRT-LLM
- MLSys Conference Papers
