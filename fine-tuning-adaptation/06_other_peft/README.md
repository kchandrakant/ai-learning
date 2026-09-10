# Step 6: Other PEFT Methods

## Beyond LoRA

LoRA isn't the only option. Different methods suit different scenarios.

## Method Comparison

| Method | Parameters | Memory | Quality | Use Case |
|--------|------------|--------|---------|----------|
| LoRA | ~0.1% | Low | High | General |
| QLoRA | ~0.1% | Very Low | High | Limited memory |
| Prefix Tuning | ~0.1% | Low | Medium | Long contexts |
| P-Tuning v2 | ~1% | Low | High | NLU tasks |
| IA³ | ~0.01% | Very Low | Medium | Extreme efficiency |
| Adapters | ~1-3% | Medium | High | Multi-task |

## Prefix Tuning

Add learnable "virtual tokens" to the input.

```python
from peft import PrefixTuningConfig

config = PrefixTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,       # Number of prefix tokens
    prefix_projection=True,       # Use MLP to project
)
```

## P-Tuning v2

Deep prompt tuning at every layer.

```python
from peft import PromptTuningConfig

config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    num_virtual_tokens=20,
    prompt_tuning_init="TEXT",
    prompt_tuning_init_text="Classify the sentiment:",
)
```

## IA³ (Infused Adapter)

Scale activations with learned vectors.

```python
from peft import IA3Config

config = IA3Config(
    task_type="CAUSAL_LM",
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],
)
```

**Ultra-efficient:** Only ~0.01% parameters!

## Adapters

Add small modules between layers.

```python
from peft import AdaLoraConfig

config = AdaLoraConfig(
    init_r=12,
    target_r=8,
    beta1=0.85,
    beta2=0.85,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    target_modules=["q_proj", "v_proj"],
)
```

## When to Use What

```
Need maximum quality?         → LoRA
Extreme memory constraints?   → QLoRA or IA³
Multi-task learning?          → Adapters
Very long sequences?          → Prefix Tuning
Classification/NLU?           → P-Tuning v2
```

## Files

- `other_peft.py` - Examples of each method

## Key Takeaways

1. LoRA is the default choice
2. QLoRA for memory constraints
3. IA³ for extreme efficiency
4. Prefix Tuning for long contexts
5. Mix and match based on needs
