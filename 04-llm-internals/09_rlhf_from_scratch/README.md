# Module 9: RLHF from Scratch

Reinforcement Learning from Human Feedback.

## Overview

RLHF is how base models become assistants like ChatGPT. Understanding it reveals alignment challenges.

## Key Topics

### Why RLHF?
```
Base model: Good at prediction, bad at being helpful
- Predicts what comes next (could be anything)
- No preference for helpful vs harmful
- RLHF teaches "what humans prefer"
```

### The Pipeline
```
1. Supervised Fine-tuning (SFT)
   - Train on demonstrations of good behavior
   - Model learns format, task structure
   
2. Reward Model Training
   - Collect preference comparisons (A vs B)
   - Train model to predict human preferences
   
3. RL Optimization (PPO)
   - Generate responses
   - Score with reward model
   - Update policy to maximize reward
   - KL penalty to stay near SFT model
```

### Reward Model
```
Input: Prompt + Response
Output: Scalar reward score

Training data: Human rankings
Loss: Bradley-Terry model (pairwise comparisons)

P(A > B) = σ(r(A) - r(B))
```

### PPO for LLMs
```
Standard PPO adapted for language:
- Actor: the LLM policy
- Critic: value function for reward prediction
- KL penalty: prevent reward hacking

Loss = E[min(r(θ)A, clip(r(θ), 1-ε, 1+ε)A)] - β*KL(π||π_ref)
```

### Reward Hacking
```
Model finds shortcuts to maximize reward:
- Verbosity (longer = higher reward?)
- Sycophancy (agree with user)
- Exploiting reward model weaknesses

Mitigations: KL penalty, reward model ensembles
```

## Exercises

1. Train a reward model on preference data
2. Implement PPO for a small LM
3. Observe and analyze reward hacking

## Key Insight

RLHF is powerful but fragile. The reward model is the bottleneck.
