# Step 3: Inference Engines

## Don't Reinvent the Wheel

Production inference engines include years of optimization.

## vLLM

PagedAttention + continuous batching for high throughput.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")
sampling = SamplingParams(temperature=0.7, max_tokens=100)

outputs = llm.generate(["Hello, how are you?"], sampling)
```

**Key features:**
- PagedAttention (efficient KV cache)
- Continuous batching
- OpenAI-compatible API server

## TensorRT-LLM

NVIDIA's optimized inference for NVIDIA GPUs.

**Key features:**
- Quantization (INT8, FP8)
- Tensor parallelism
- In-flight batching

## llama.cpp

CPU and edge inference.

```bash
./main -m model.gguf -p "Hello" -n 100
```

**Key features:**
- GGUF format (quantized)
- CPU, Metal, CUDA support
- Low memory footprint

## Comparison

| Engine | Throughput | Latency | Hardware | Ease |
|--------|------------|---------|----------|------|
| vLLM | High | Low | GPU | Easy |
| TensorRT | Highest | Lowest | NVIDIA | Medium |
| llama.cpp | Medium | Medium | Any | Easy |
| TGI | High | Low | GPU | Easy |

## When to Use What

- **vLLM**: General purpose GPU serving
- **TensorRT-LLM**: Maximum NVIDIA performance
- **llama.cpp**: Local/edge, no GPU
- **TGI**: HuggingFace ecosystem

## Files

- `inference_engines.py` - Engine comparisons
