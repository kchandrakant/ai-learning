# Step 7: Supervised Fine-tuning (SFT)

## What is SFT?

Train a model to follow instructions using (instruction, response) pairs.

```
Base Model → SFT → Instruction-following Model
            ↑
    (instruction, response) pairs
```

## Data Format

```json
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."}
    ]
}
```

## Implementation with TRL

```python
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

sft_config = SFTConfig(
    output_dir="./sft-output",
    max_seq_length=2048,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

## Chat Template

```python
# Apply chat template
def format_chat(example):
    return tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

dataset = dataset.map(lambda x: {"text": format_chat(x)})
```

## Multi-turn Conversations

```python
# Only compute loss on assistant responses
def mask_prompt(labels, tokenizer, messages):
    # Find where assistant responses start
    # Set labels to -100 for non-assistant tokens
    for i, msg in enumerate(messages):
        if msg["role"] != "assistant":
            # Find token positions and mask
            labels[start:end] = -100
    return labels
```

## SFT Best Practices

1. **Quality over quantity** - 1K good examples > 100K bad ones
2. **Diverse instructions** - Cover many task types
3. **Consistent format** - Use same template throughout
4. **Include system prompts** - Train with them if using them
5. **Balance lengths** - Mix short and long responses

## Files

- `sft.py` - SFT training implementation

## Key Takeaways

1. SFT = supervised learning on (instruction, response)
2. Use chat templates for consistency
3. Mask prompts (only train on responses)
4. Quality data is crucial
5. First step before DPO/RLHF
