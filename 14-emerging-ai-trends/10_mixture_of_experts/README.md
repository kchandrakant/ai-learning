# Module 10: Mixture of Experts (MoE)

## The Scaling Challenge

Dense transformers: Every parameter activates for every token.
```
70B model = 70B parameters computed per token = expensive
```

**MoE insight**: What if only some parameters activated per token?

## How MoE Works

Replace dense FFN with multiple "expert" FFNs:

```
         ┌─ Expert 1 ──┐
Input →  │  Expert 2   │ → Router selects top-k → Output
         │  Expert 3   │
         └─ Expert N ──┘

Only k experts (typically 2) activate per token
```

### The Router
Learned gating network that decides which experts handle each token:

```python
def router(x, expert_weights):
    # Compute scores for each expert
    scores = x @ expert_weights  # (batch, num_experts)
    
    # Select top-k experts
    top_k_scores, top_k_indices = scores.topk(k=2)
    
    # Normalize weights
    weights = softmax(top_k_scores)
    
    return weights, top_k_indices
```

## Key MoE Models

| Model | Experts | Active | Total Params | Active Params |
|-------|---------|--------|--------------|---------------|
| Mixtral 8×7B | 8 | 2 | 47B | ~13B |
| Switch-XXL | 2048 | 1 | 1.6T | ~1B |
| DeepSeek-MoE | 64 | 6 | 145B | ~22B |
| GPT-4 (rumored) | ? | ? | ~1.8T | ~200B |

## Advantages

### More Parameters, Same Compute
```
Dense 13B:  13B params, 13B compute per token
MoE 47B:    47B params, 13B compute per token

3.6× more parameters, same inference cost!
```

### Specialization
Experts can specialize:
- Expert 1: Handles code
- Expert 2: Handles math
- Expert 3: Handles languages

(In practice, specialization is more subtle)

## Challenges

### Memory
All parameters must fit in memory (even if not all compute):
```
Mixtral 8×7B: Needs ~100GB to load (all 47B params)
             But only ~13B compute per token
```

### Load Balancing
If router always picks the same experts:
- Some experts overloaded
- Others never train
- Wasted capacity

**Solution**: Auxiliary loss to encourage balanced routing

### Training Stability
MoE training can be unstable:
- Router collapse (all tokens → same experts)
- Expert collapse (some experts never used)

Requires careful hyperparameter tuning.

## Inference Considerations

### Batching
Different tokens may route to different experts:
- Harder to batch efficiently
- Expert parallelism needed

### Memory vs Compute Tradeoff
```
MoE: More memory, same compute per token
Dense: Less memory, more compute per token
```

For memory-constrained settings, dense may be better.

## Mixtral Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    device_map="auto",
    load_in_4bit=True,  # Quantize to fit in memory
)
```

## When to Use MoE

**Good fit:**
- Large-scale deployment with sufficient memory
- Tasks benefiting from diverse expertise
- When you need capacity but not proportional compute

**Not ideal:**
- Memory-constrained settings
- Small-scale deployment
- When dense model fits your needs

## Exercises

1. Run Mixtral and analyze expert activation patterns
2. Compare Mixtral vs Llama-2-70B on reasoning tasks
3. Measure memory usage vs effective compute

## Resources

- "Mixtral of Experts" (Mistral AI, 2024)
- "Switch Transformers" (Fedus et al., 2021)
- "Outrageously Large Neural Networks" (Shazeer et al., 2017)

## What's Next?

Module 11 covers **Efficient Architectures** — small but capable models.
