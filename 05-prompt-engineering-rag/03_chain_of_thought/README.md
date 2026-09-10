# Step 3: Chain-of-Thought Prompting

## The Problem with Direct Answers

For reasoning tasks, asking for a direct answer often fails:

```
Q: If a store has 45 apples and sells 3/5 of them, how many are left?
A: 27  ← Often wrong without reasoning
```

## The Solution: Think Step by Step

Chain-of-Thought (CoT) prompting elicits intermediate reasoning steps:

```
Q: If a store has 45 apples and sells 3/5 of them, how many are left?
A: Let me think step by step:
   1. The store starts with 45 apples
   2. They sell 3/5 of them: 45 × (3/5) = 27 apples sold
   3. Remaining: 45 - 27 = 18 apples
   
   Answer: 18 apples
```

## Types of CoT Prompting

### 1. Zero-Shot CoT
Just add "Let's think step by step":

```python
prompt = """
Q: {question}

Let's think step by step:
"""
```

Surprisingly effective! The magic phrase triggers reasoning behavior.

### 2. Few-Shot CoT
Provide examples WITH reasoning:

```
Q: Roger has 5 tennis balls. He buys 2 more cans of 3. How many does he have?
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 balls.
   5 + 6 = 11. The answer is 11.

Q: {new_question}
A:
```

### 3. Self-Consistency
Generate multiple CoT paths, take the majority answer:

```
                    ┌─ Path 1 → Answer: 18
Question ──────────├─ Path 2 → Answer: 18
                    └─ Path 3 → Answer: 27
                    
Majority vote → Final Answer: 18
```

```python
def self_consistency(question, n_paths=5):
    answers = []
    for _ in range(n_paths):
        response = llm(question, temperature=0.7)  # Higher temp for diversity
        answer = extract_final_answer(response)
        answers.append(answer)
    return most_common(answers)
```

## When to Use CoT

✅ **Use CoT for:**
- Math and arithmetic problems
- Multi-step reasoning
- Logic puzzles
- Complex decision making
- Code debugging

❌ **Skip CoT for:**
- Simple factual questions
- Classification tasks
- Translation
- Tasks where reasoning doesn't help

## The Cost-Benefit Tradeoff

```
┌────────────────────────────────────────┐
│ Direct Answer: Fast, cheap, less accurate │
│ CoT:           Slower, more tokens, more accurate │
└────────────────────────────────────────┘
```

CoT uses more tokens (= more cost), so use it strategically.

## Implementation Pattern

```python
def solve_with_cot(question: str, use_self_consistency: bool = False) -> str:
    cot_prompt = f"""
    Solve this problem step by step. Show your reasoning clearly,
    then provide the final answer on its own line starting with "Answer:".
    
    Problem: {question}
    
    Solution:
    """
    
    if use_self_consistency:
        answers = []
        for _ in range(5):
            response = llm(cot_prompt, temperature=0.7)
            answer = extract_answer(response)
            answers.append(answer)
        return majority_vote(answers)
    else:
        return llm(cot_prompt, temperature=0.0)
```

## Key Research Finding

From the original paper:
- **62% → 79%** accuracy improvement on GSM8K (math problems)
- Larger models benefit more from CoT
- Works across multiple reasoning domains

## Files

- `chain_of_thought.py` - CoT implementation with self-consistency

## Key Takeaways

1. "Let's think step by step" triggers reasoning
2. Show reasoning examples for best results
3. Self-consistency improves accuracy (at cost of more API calls)
4. Use CoT for reasoning, skip for simple tasks
5. More tokens = higher cost, but often worth it for complex tasks

## What's Next?

Step 4: **Advanced Prompting Techniques** — Tree-of-Thought, ReAct, structured outputs.
