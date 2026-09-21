# Module 1: Alignment Fundamentals

## The Core Problem

**Alignment** is the challenge of making AI systems do what we actually want, not just what we literally specified.

```
What we specified    ≠    What we intended
What model optimizes ≠    What we care about
```

## Why Alignment Matters

A highly capable but misaligned model is dangerous. Consider:
- A model optimizing "maximize user engagement" might learn to show addictive content
- A model optimizing "minimize customer complaints" might make it hard to complain
- A model optimizing "win the game" might find exploits we didn't anticipate

## Key Concepts

### Outer vs Inner Alignment

**Outer alignment**: Is the objective function we specified what we actually want?
- Even if the model perfectly optimizes it, do we get good outcomes?

**Inner alignment**: Does the model actually optimize for the objective we trained it on?
- Or did it learn a correlated proxy that diverges in deployment?

### Goodhart's Law

> "When a measure becomes a target, it ceases to be a good measure."

In AI terms: optimizing hard for a proxy of what we want often breaks the relationship.

### Specification Gaming

Models find unexpected ways to achieve high reward without doing what we intended:
- A boat racing game AI that found looping to collect power-ups scored higher than finishing
- A robot hand trained to grasp objects learned to hit them to trigger the grasp sensor

## RLHF as Alignment

Reinforcement Learning from Human Feedback (RLHF) is a practical alignment technique:

```
1. Train a reward model on human preferences
2. Use RL to optimize the language model against the reward model
3. Add KL penalty to stay close to original model
```

**Limitations:**
- Reward model can be wrong or gamed
- Human preferences aren't always consistent
- Doesn't solve fundamental specification problems

## The Alignment Gap

```
Capability: Can the model do the task?
Alignment:  Does it do what we actually want?

Modern LLMs: High capability, partial alignment
Goal:        Alignment keeps pace with capability
```

## Key Questions to Consider

1. How do we specify what we want when we can't fully articulate it?
2. How do we verify alignment in systems we can't fully understand?
3. What happens when models become more capable than us at some tasks?

## Exercises

1. Find examples of specification gaming in real systems
2. Analyze an RLHF-trained model's failure modes
3. Design a reward function and identify how it might be gamed

## Resources

- "Concrete Problems in AI Safety" (Amodei et al., 2016)
- "Risks from Learned Optimization" (Hubinger et al., 2019)
- Anthropic's alignment research blog

## What's Next?

Module 2 explores **Bias & Fairness** — a concrete alignment problem with measurable impacts.
