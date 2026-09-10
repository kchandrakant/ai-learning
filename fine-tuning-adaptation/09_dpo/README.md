# Step 9: DPO (Direct Preference Optimization)

## RLHF Without RL

DPO achieves alignment without:
- Training a reward model
- Running PPO
- Complex RL infrastructure

## The Idea

Instead of reward modeling + RL, directly optimize from preferences.

```
RLHF Pipeline:
SFT → Reward Model → PPO Training → Aligned Model
      (complex)      (unstable)

DPO Pipeline:
SFT → DPO Training → Aligned Model
      (simple, stable)
```

## Preference Data Format

```json
{
    "prompt": "Write a poem about AI",
    "chosen": "In silicon dreams, we find...",
    "rejected": "AI is a computer thing that does stuff..."
}
```

## Implementation with TRL

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("your-sft-model")
tokenizer = AutoTokenizer.from_pretrained("your-sft-model")

# Load reference model (usually the SFT model)
ref_model = AutoModelForCausalLM.from_pretrained("your-sft-model")

# DPO config
dpo_config = DPOConfig(
    beta=0.1,                    # KL penalty coefficient
    learning_rate=5e-7,          # Lower than SFT
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_length=512,
    max_prompt_length=256,
)

# Create trainer
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    args=dpo_config,
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
)

trainer.train()
```

## The DPO Loss

```python
# Simplified DPO loss
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
    return loss
```

## Key Hyperparameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| `beta` | KL penalty strength | 0.1 - 0.5 |
| `learning_rate` | Much lower than SFT | 1e-7 to 5e-6 |
| `max_length` | Sequence length | 512 - 2048 |

## Creating Preference Data

```python
def create_preference_pair(prompt: str, model_a, model_b) -> dict:
    response_a = model_a.generate(prompt)
    response_b = model_b.generate(prompt)
    
    # Get human preference (or use another model)
    chosen, rejected = human_preference(prompt, response_a, response_b)
    
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }
```

## Files

- `dpo.py` - DPO training implementation

## Key Takeaways

1. DPO = simpler alternative to RLHF
2. Directly optimizes from preferences
3. No reward model or PPO needed
4. Use lower learning rate than SFT
5. Beta controls KL penalty strength
