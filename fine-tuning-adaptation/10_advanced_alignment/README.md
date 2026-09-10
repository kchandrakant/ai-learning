# Step 10: Advanced Alignment Methods

## Beyond DPO

The field evolves rapidly. Know the alternatives.

## ORPO (Odds Ratio Preference Optimization)

No reference model needed!

```python
from trl import ORPOTrainer, ORPOConfig

config = ORPOConfig(
    beta=0.1,
    learning_rate=5e-6,
)

trainer = ORPOTrainer(
    model=model,
    args=config,
    train_dataset=preference_data,
    tokenizer=tokenizer,
)
```

**Key advantage:** Simpler setup, no reference model.

## SimPO (Simple Preference Optimization)

Simplified DPO variant with length normalization.

```python
# SimPO adds length normalization to the loss
# Prevents model from preferring shorter responses
```

## KTO (Kahneman-Tversky Optimization)

Works with binary feedback (good/bad) instead of pairs.

```python
# Data format
{
    "prompt": "...",
    "response": "...",
    "label": True  # or False
}

# No need for paired preferences!
```

## Comparison

| Method | Data Format | Reference Model | Complexity |
|--------|-------------|-----------------|------------|
| DPO | Pairs | Required | Medium |
| ORPO | Pairs | Not needed | Low |
| SimPO | Pairs | Required | Low |
| KTO | Binary | Required | Medium |

## When to Use What

```
Have paired preferences?     → DPO or ORPO
Only have good/bad labels?   → KTO
Want simplest setup?         → ORPO
Length bias issues?          → SimPO
```

## Files

- `advanced_alignment.py` - Various alignment methods

## Key Takeaways

1. DPO is the current standard
2. ORPO eliminates reference model
3. KTO works with simpler feedback
4. Choose based on your data format
5. The field changes fast—stay updated
