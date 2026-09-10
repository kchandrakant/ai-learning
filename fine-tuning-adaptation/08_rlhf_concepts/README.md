# Step 8: RLHF Concepts

## What is RLHF?

Reinforcement Learning from Human Feedback aligns models with human preferences.

```
┌─────────────────────────────────────────────────────────────┐
│                    RLHF Pipeline                             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   1. Pre-training   →  Base language model                  │
│                                                              │
│   2. SFT            →  Instruction-following model          │
│                                                              │
│   3. Reward Model   →  Model that predicts human preference │
│                                                              │
│   4. PPO Training   →  Optimize policy using reward model   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Step 1: Reward Model Training

Train a model to predict which response humans prefer.

```python
# Preference data
{
    "prompt": "Write a poem",
    "chosen": "A beautiful, thoughtful poem...",
    "rejected": "poem poem poem words..."
}

# Reward model learns: chosen > rejected
```

## Step 2: PPO Training

Use reward model to guide policy optimization.

```python
# Simplified PPO loop
for prompt in prompts:
    # Generate response from policy
    response = policy.generate(prompt)
    
    # Get reward from reward model
    reward = reward_model(prompt, response)
    
    # PPO update
    policy.update(prompt, response, reward)
    
    # KL penalty to stay close to reference
    kl_penalty = KL(policy || reference_policy)
    total_reward = reward - beta * kl_penalty
```

## Why RLHF is Hard

1. **Reward hacking** - Model finds loopholes
2. **Instability** - PPO is finicky
3. **Compute** - Requires reward model + policy + reference
4. **Complexity** - Many hyperparameters

## Constitutional AI

Alternative: Have the model critique itself.

```
Generate → Self-critique → Revise → Final response

"Is this response harmful?"
"How could it be improved?"
```

## When to Use RLHF

- Training frontier models
- Need fine-grained preference control
- Have resources for complexity

**Most use cases:** DPO is simpler and often better.

## Files

- `rlhf_concepts.py` - Conceptual examples

## Key Takeaways

1. RLHF = Reward Model + PPO
2. Aligns model with human preferences
3. Complex and resource-intensive
4. DPO often preferred for simplicity
5. Constitutional AI is an alternative
