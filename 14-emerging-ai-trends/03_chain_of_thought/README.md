# Module 3: Chain-of-Thought & Beyond

## Chain-of-Thought (CoT)

The foundational reasoning technique (Wei et al., 2022).

### Zero-Shot CoT
Just add "Let's think step by step":

```
Q: A bat and ball cost $1.10. The bat costs $1 more than the ball. 
   How much does the ball cost?

A: Let's think step by step.
   Let x = ball price
   Bat price = x + 1
   Total: x + (x + 1) = 1.10
   2x + 1 = 1.10
   2x = 0.10
   x = 0.05
   The ball costs $0.05.
```

### Few-Shot CoT
Provide examples with reasoning:

```
Q: [Example problem 1]
A: [Step-by-step reasoning] → [Answer]

Q: [Example problem 2]  
A: [Step-by-step reasoning] → [Answer]

Q: [Your actual problem]
A: [Model generates reasoning]
```

## Beyond Linear Chains

### Tree of Thoughts (ToT)
Explore multiple reasoning paths:

```
        Problem
       /   |   \
     Path1 Path2 Path3
     /  \    |    /  \
   ...  ... ...  ... ...
   
Evaluate each path, backtrack if needed
```

```python
def tree_of_thoughts(problem, depth=3, branching=3):
    thoughts = [generate_thought(problem)]
    
    for d in range(depth):
        new_thoughts = []
        for thought in thoughts:
            # Generate multiple continuations
            continuations = [
                generate_continuation(thought) 
                for _ in range(branching)
            ]
            # Evaluate and keep best
            scored = [(c, evaluate(c)) for c in continuations]
            new_thoughts.extend(top_k(scored, k=branching))
        thoughts = new_thoughts
    
    return best(thoughts)
```

### Graph of Thoughts
Allow merging and cycles in reasoning:
```
     A ──→ B
     ↓   ↗ ↓
     C ──→ D
     
Thoughts can combine, form loops
```

### Self-Consistency
Sample multiple chains, vote on answer:

```python
answers = []
for _ in range(10):
    chain = model.generate(prompt, temperature=0.7)
    answer = extract_final_answer(chain)
    answers.append(answer)

final = majority_vote(answers)
```

Improves robustness — different reasoning paths should converge.

## Reasoning + Verification

### Self-Critique
Model checks its own work:

```
Generate answer → "Is this correct? Let me verify..." → 
Identify errors → Regenerate if needed
```

### Process Reward Models
Train a model to evaluate reasoning quality:

```
Step 1: [Good step] → Score: 0.9
Step 2: [Reasonable] → Score: 0.7
Step 3: [Error!]     → Score: 0.2

Use scores to guide search
```

## When CoT Helps

**Helps a lot:**
- Math word problems
- Multi-step reasoning
- Code debugging
- Logical puzzles

**Helps less:**
- Simple factual recall
- Creative writing
- Tasks without clear reasoning structure

**Can hurt:**
- Very simple tasks (overthinking)
- When reasoning is wrong (confident errors)

## Reasoning as Training Data

Modern insight: Use CoT traces as training data.

```
1. Generate many reasoning traces
2. Filter for correct answers
3. Train model on successful traces
4. Model learns to reason, not just produce reasoning text
```

This is how o1 and DeepSeek-R1 are trained.

## Exercises

1. Implement Tree of Thoughts for a puzzle task
2. Compare zero-shot vs few-shot CoT on math problems
3. Build a self-consistency voting system

## Resources

- "Chain-of-Thought Prompting" (Wei et al., 2022)
- "Tree of Thoughts" (Yao et al., 2023)
- "Self-Consistency Improves CoT" (Wang et al., 2022)

## What's Next?

Module 4 covers **Looped & Adaptive Transformers** — architectures that reason iteratively.
