# Step 1: Inference Basics

## Training vs Inference

Training: Update model weights to minimize loss
Inference: Use trained weights to generate predictions

```
Training:   Input → Forward → Loss → Backward → Update weights
Inference:  Input → Forward → Output (no backward pass)
```

## Key Metrics

### Latency
- **Time to First Token (TTFT)**: How long until first output
- **Time Per Output Token (TPOT)**: Speed of generation
- **End-to-End Latency**: Total request time

### Throughput
- **Tokens/second**: Generation speed
- **Requests/second**: Service capacity

## Memory Components

```
Total GPU Memory = Model Weights + KV Cache + Activations

Model Weights:  Fixed, depends on model size
KV Cache:       Grows with sequence length × batch size
Activations:    Temporary, during computation
```

## Batch vs Streaming

**Batch**: Wait for complete response
```python
response = model.generate(prompt, max_tokens=100)
return response  # All at once
```

**Streaming**: Return tokens as generated
```python
for token in model.generate_stream(prompt):
    yield token  # One at a time
```

## Files

- `inference_basics.py` - Basic inference examples

## Key Takeaways

1. Inference = forward pass only
2. Measure TTFT, TPOT, and throughput
3. Memory = weights + KV cache + activations
4. Streaming improves perceived latency
