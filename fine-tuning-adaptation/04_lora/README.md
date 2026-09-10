# Step 4: LoRA (Low-Rank Adaptation)

## The Big Idea

Instead of updating all weights, add small trainable matrices.

```
Original:  W (frozen)
LoRA:      W + BA where B and A are small matrices

W: 4096 × 4096 = 16M parameters (frozen)
B: 4096 × 16   = 65K parameters (trainable)
A: 16 × 4096   = 65K parameters (trainable)

Total trainable: 130K vs 16M (0.8%!)
```

## Why It Works

Neural network weight updates during fine-tuning are low-rank.
LoRA exploits this by only learning the low-rank update.

## Implementation with PEFT

```python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# Configure LoRA
lora_config = LoraConfig(
    r=16,                      # Rank
    lora_alpha=32,             # Scaling factor
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.062
```

## Key Hyperparameters

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `r` | Rank of decomposition | 8, 16, 32, 64 |
| `lora_alpha` | Scaling factor | 16, 32, 64 |
| `target_modules` | Which layers to adapt | q, k, v, o projections |
| `lora_dropout` | Dropout for regularization | 0.0 - 0.1 |

**Rule of thumb:** `lora_alpha = 2 * r`

## Training

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./lora-output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

## Merging Adapters

```python
# Option 1: Keep separate (swap adapters at runtime)
model.save_pretrained("./lora-adapter")

# Option 2: Merge into base model
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./merged-model")
```

## Files

- `lora.py` - Complete LoRA training example

## Key Takeaways

1. LoRA trains <1% of parameters
2. Achieves near full fine-tuning quality
3. Rank 16-32 works for most cases
4. Target attention projections (q, k, v, o)
5. Can merge or keep adapters separate
