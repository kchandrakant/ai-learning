# Module 11: Efficient Architectures

## The Efficiency Imperative

Not every use case needs 70B parameters:
- Edge deployment (phones, IoT)
- Cost-sensitive applications
- Latency requirements
- Privacy (local inference)

## Small Language Models (SLMs)

### The Phi Family (Microsoft)

Phi-3: Tiny but capable
- **Phi-3-mini**: 3.8B params, matches GPT-3.5 on many benchmarks
- **Phi-3-small**: 7B params
- **Phi-3-medium**: 14B params

**How?** Carefully curated training data > raw scale

### Gemma (Google)
- **Gemma-2B**: 2B params, open weights
- **Gemma-7B**: 7B params

### Qwen-small (Alibaba)
- Multiple sizes from 0.5B to 7B
- Strong multilingual capabilities

## When SLMs Beat LLMs

| Scenario | SLM Advantage |
|----------|---------------|
| Latency-critical | 10-100× faster |
| Cost-sensitive | $0.001 vs $0.01+ per query |
| Privacy-required | Runs locally |
| Edge deployment | Fits in memory |
| High throughput | More requests/GPU |

## Efficiency Techniques

### Quantization

Reduce precision of weights:

```python
# Original: float32 (4 bytes per param)
# Quantized: int8 (1 byte) or int4 (0.5 bytes)

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    load_in_4bit=True,  # 8× memory reduction
    device_map="auto",
)
```

**Tradeoffs**:
- INT8: ~1% quality loss, 4× memory savings
- INT4: ~3-5% quality loss, 8× memory savings

### Knowledge Distillation

Train small model to mimic large model:

```
Large teacher → Generate outputs → 
                Small student learns to match → 
                Smaller model with similar quality
```

### Pruning

Remove unimportant weights:

```python
# Structured pruning: remove entire neurons/heads
# Unstructured pruning: remove individual weights

# After pruning:
# - Fewer computations
# - Sparse matrices
# - Need hardware support for speedup
```

### Architecture Search

Automatically find efficient architectures:
- Neural Architecture Search (NAS)
- Once-for-all networks
- Hardware-aware optimization

## GGUF and Local Inference

**GGUF**: Popular format for local LLM inference

```bash
# Using llama.cpp
./main -m model.gguf -p "Hello, world!"

# Quantization levels
# Q4_0: 4-bit, fastest
# Q5_1: 5-bit, balanced
# Q8_0: 8-bit, highest quality
```

### Ollama
Run models locally with simple commands:
```bash
ollama run phi3
ollama run gemma:2b
```

## Cost Comparison

| Model | Params | API Cost | Local Cost |
|-------|--------|----------|------------|
| GPT-4 | ~1.8T | $30/1M tokens | N/A |
| GPT-3.5 | ~175B | $0.50/1M tokens | N/A |
| Llama-2-70B | 70B | $0.90/1M tokens | Free (self-host) |
| Phi-3-mini | 3.8B | - | Runs on laptop |

## Right-Sizing

**Don't default to biggest model.** Ask:
1. What's the task complexity?
2. What's the latency requirement?
3. What's the cost budget?
4. What's the deployment environment?

Often a well-tuned small model beats a generic large one.

## Exercises

1. Quantize a model and measure quality vs speed tradeoff
2. Compare Phi-3-mini vs GPT-3.5 on a task
3. Deploy a model locally with Ollama

## Resources

- Phi-3 technical report (Microsoft)
- llama.cpp project
- Hugging Face Optimum library

## What's Next?

Module 12 explores **World Models** — learning physics and dynamics from video.
