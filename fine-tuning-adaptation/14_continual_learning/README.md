# Step 14: Continual Learning

## The Problem

Fine-tuning on new data can cause **catastrophic forgetting**.

```
Model v1: Good at task A
Fine-tune on task B data
Model v2: Good at task B, BAD at task A  ← Catastrophic forgetting!
```

## Solutions

### 1. Replay Buffer

Mix old data with new data.

```python
def create_mixed_dataset(old_data, new_data, old_ratio=0.2):
    # Sample from old data
    old_sample = old_data.shuffle().select(range(int(len(new_data) * old_ratio)))
    
    # Combine
    return concatenate_datasets([old_sample, new_data])
```

### 2. Elastic Weight Consolidation (EWC)

Penalize changes to important weights.

```python
def ewc_loss(model, fisher_info, old_params, lambda_ewc=1000):
    loss = 0
    for name, param in model.named_parameters():
        if name in fisher_info:
            loss += (fisher_info[name] * (param - old_params[name]) ** 2).sum()
    return lambda_ewc * loss
```

### 3. Multi-Task Learning

Train on all tasks simultaneously.

```python
# Combine all task data with task prefixes
{
    "input": "[TASK_A] What is the capital of France?",
    "output": "Paris"
}
{
    "input": "[TASK_B] Summarize: ...",
    "output": "Summary: ..."
}
```

### 4. Separate Adapters

Keep separate LoRA adapters per task.

```python
# Load different adapters for different tasks
if task == "A":
    model.load_adapter("adapter_A")
elif task == "B":
    model.load_adapter("adapter_B")
```

## Best Practices

1. **Always keep base model** - Don't modify original weights
2. **Version your adapters** - Track what each adapter was trained on
3. **Evaluate on all tasks** - Check for regression
4. **Use replay** - Mix in old data when fine-tuning on new

## Files

- `continual_learning.py` - Continual learning strategies

## Key Takeaways

1. Catastrophic forgetting is real
2. Replay buffers are the simplest fix
3. Separate adapters prevent interference
4. Always test for regression
5. Keep the base model frozen
