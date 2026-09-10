# Step 3: Training Fundamentals

## The Language Model Loss

Predict the next token, minimize cross-entropy loss.

```python
# Causal LM loss
loss = CrossEntropyLoss(logits, labels)

# Only compute loss on completion, not prompt
labels[prompt_indices] = -100  # Ignored by loss
```

## Learning Rate Schedule

```
Warmup → Peak → Decay

|        /‾‾‾‾‾‾‾\
|       /          \
|      /            \____
|_____/                   
  warmup   training   
```

```python
# Cosine schedule with warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000
)
```

## Batch Size & Gradient Accumulation

```
Effective batch size = batch_size × gradient_accumulation × num_gpus

GPU memory limited to batch_size=2?
Use gradient_accumulation_steps=8 for effective batch of 16.
```

## Mixed Precision Training

```python
# FP16 (older GPUs)
training_args = TrainingArguments(fp16=True)

# BF16 (Ampere+ GPUs, recommended)
training_args = TrainingArguments(bf16=True)
```

**Why it matters:**
- 2x memory reduction
- Faster training
- BF16 more stable than FP16

## Key Hyperparameters

| Parameter | Typical Range | Notes |
|-----------|--------------|-------|
| Learning rate | 1e-5 to 2e-4 | Lower for larger models |
| Batch size | 4-32 | Larger is usually better |
| Epochs | 1-5 | More can overfit |
| Warmup ratio | 0.03-0.1 | Stabilizes early training |
| Weight decay | 0.01-0.1 | Regularization |

## Checkpointing

```python
training_args = TrainingArguments(
    save_strategy="steps",
    save_steps=500,
    save_total_limit=3,  # Keep only last 3
)
```

## Files

- `training_fundamentals.py` - Training utilities

## Key Takeaways

1. Loss = next token prediction (cross-entropy)
2. Use warmup + cosine decay schedule
3. Gradient accumulation for larger effective batches
4. BF16 preferred over FP16
5. Checkpoint frequently
