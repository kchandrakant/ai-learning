# Step 1: When to Fine-tune

## The Decision Framework

Fine-tuning is expensive. Make sure it's the right choice.

```
┌─────────────────────────────────────────────────────────────┐
│            Do You Need Fine-tuning?                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   Need new knowledge/facts?                                  │
│   ├── YES → Use RAG (not fine-tuning!)                      │
│   └── NO  ↓                                                  │
│                                                              │
│   Need different behavior/style?                            │
│   ├── YES → Try few-shot prompting first                    │
│   │         └── Still not working? → Fine-tune              │
│   └── NO  ↓                                                  │
│                                                              │
│   Need consistent output format?                            │
│   ├── YES → Try structured output / function calling        │
│   │         └── Still not working? → Fine-tune              │
│   └── NO  → Probably don't need fine-tuning                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Good Reasons to Fine-tune

| Use Case | Why Fine-tuning Helps |
|----------|----------------------|
| Domain-specific language | Learn jargon, terminology |
| Consistent output format | Enforce structure reliably |
| Specific writing style | Match brand voice |
| Task specialization | Outperform general models |
| Latency reduction | Smaller fine-tuned model = faster |
| Cost reduction | Smaller model with same quality |

## Bad Reasons to Fine-tune

| Use Case | Better Alternative |
|----------|-------------------|
| Add new facts | RAG |
| Update information | RAG |
| One-off tasks | Prompting |
| Quick experiments | Few-shot prompting |

## The Fine-tuning Spectrum

```
Prompting ← → RAG ← → Fine-tuning ← → Pre-training

Cost:     Low        Medium       High           Very High
Effort:   Minutes    Hours        Days           Weeks+
Data:     0          Documents    100s-1000s     Billions tokens
```

## Cost-Benefit Analysis

```python
def should_finetune(use_case):
    factors = {
        "prompting_works": -3,      # If prompting works, don't fine-tune
        "need_consistency": +2,      # Fine-tuning helps consistency
        "have_quality_data": +2,     # Need good data
        "volume_of_requests": +1,    # High volume justifies cost
        "latency_critical": +1,      # Smaller fine-tuned model is faster
        "domain_specific": +1,       # Domain knowledge helps
    }
    
    score = sum(factors[f] for f in use_case.applicable_factors)
    return score > 2  # Threshold for "yes, fine-tune"
```

## Files

- `when_to_finetune.py` - Decision framework examples

## Key Takeaways

1. Try prompting and RAG first
2. Fine-tune for behavior, not knowledge
3. Need quality data (garbage in = garbage out)
4. Consider the cost-benefit tradeoff
5. Start small, validate, then scale
