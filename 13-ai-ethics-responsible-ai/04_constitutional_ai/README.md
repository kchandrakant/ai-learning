# Module 4: Constitutional AI & Self-Alignment

## The RLHF Bottleneck

RLHF requires massive human feedback:
- Expensive to collect
- Hard to scale
- Humans may be inconsistent or biased

**Constitutional AI** (Anthropic, 2022) offers an alternative: use AI to provide feedback based on explicit principles.

## The Constitutional AI Approach

### Step 1: Define Constitutional Principles

A "constitution" of rules the model should follow:

```
1. Choose the response that is most helpful while being harmless
2. Choose the response that is most truthful
3. Choose the response that sounds most similar to what a peaceful person would say
4. Choose the response that is least likely to be seen as harmful
```

### Step 2: Self-Critique and Revision

```
Original response: [potentially problematic output]

Critique prompt: "Identify ways this response could be harmful 
                  according to the constitution."

Revision prompt: "Rewrite the response to address the identified issues."
```

### Step 3: Train on Revised Responses

Use the self-revised responses as training data, with RLAIF (RL from AI Feedback) instead of human feedback.

## RLAIF vs RLHF

| Aspect | RLHF | RLAIF |
|--------|------|-------|
| Feedback source | Humans | AI model |
| Scalability | Limited | High |
| Cost | Expensive | Lower |
| Consistency | Variable | More consistent |
| Captures nuance | Yes | Depends on constitution |

## The Constitutional Loop

```
Generate → Critique → Revise → Generate better
              ↓
         Train on revised outputs
              ↓
         Better base model
              ↓
         Better critiques
              ↓
         [Repeat]
```

## Writing Good Principles

Principles should be:
- **Specific enough** to guide behavior
- **General enough** to cover many situations
- **Consistent** with each other
- **Prioritized** when they conflict

### Example Principles (simplified)

```
1. Avoid content that could enable violence or illegal acts
2. Be truthful and acknowledge uncertainty
3. Respect user privacy and autonomy
4. Be helpful within ethical bounds
5. When principles conflict, prioritize safety over helpfulness
```

## Scaling Oversight

Constitutional AI is part of a broader vision: **scalable oversight**.

As AI systems become more capable:
- Humans can't evaluate every output
- AI can help supervise AI
- But we need to bootstrap from human values

```
Human values → Constitution → AI feedback → Better AI → Better feedback
```

## Limitations

- **Constitution must be well-designed**: Garbage principles in, garbage behavior out
- **Doesn't solve fundamental alignment**: Just moves the problem to constitution design
- **AI feedback has limits**: Model may not catch subtle issues
- **Principles can conflict**: Requires implicit or explicit prioritization

## Debate and Alternatives

**Other approaches to scalable oversight:**
- **Debate**: Two AIs argue, human judges
- **Iterated distillation**: Amplify human judgment through multiple rounds
- **Recursive reward modeling**: AI helps improve reward model

## Exercises

1. Write a constitution for a customer service chatbot
2. Implement a simple critique-and-revise loop
3. Identify edge cases where your constitution gives unclear guidance

## Resources

- "Constitutional AI: Harmlessness from AI Feedback" (Bai et al., 2022)
- Anthropic's Claude system prompt approaches
- "Debate" paper (Irving et al., 2018)

## What's Next?

Module 5 covers **Privacy & Data Rights** — respecting individuals in AI systems.
