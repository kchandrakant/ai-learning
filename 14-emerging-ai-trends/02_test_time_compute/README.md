# Module 2: Test-Time Compute

## What Is Test-Time Compute?

**Training-time compute**: Fixed when model is trained
**Test-time compute**: Can scale per query at inference

Test-time compute lets models "think longer" on harder problems.

## Techniques

### 1. Chain-of-Thought (CoT)
Generate explicit reasoning steps:
```
Q: If John has 3 apples and gives 2 to Mary, how many does he have?

A: Let me think step by step.
   John starts with 3 apples.
   He gives 2 to Mary.
   3 - 2 = 1.
   John has 1 apple.
```

**Compute cost**: More output tokens = more compute

### 2. Self-Consistency
Sample multiple reasoning paths, vote on answer:
```python
answers = []
for _ in range(5):
    response = model.generate(prompt, temperature=0.7)
    answer = extract_answer(response)
    answers.append(answer)

final_answer = majority_vote(answers)
```

**Compute cost**: N× single generation

### 3. Verification and Backtracking
Generate → Check → Revise if wrong:
```
Generate answer → Verify (is this right?) → 
If wrong: Try different approach → Verify again
```

**Compute cost**: Variable, depends on verification results

### 4. Beam Search over Reasoning
Explore multiple reasoning paths in parallel:
```
             ┌─ Path A ──┬─ Continue
Start ──────┼─ Path B ──┴─ Prune (low score)
             └─ Path C ────── Best answer
```

### 5. Budget Allocation
Dynamically allocate compute based on problem difficulty:
```python
def adaptive_reasoning(problem):
    # Quick attempt
    answer = model.generate(problem, max_tokens=100)
    confidence = assess_confidence(answer)
    
    if confidence > 0.9:
        return answer
    
    # Need more thinking
    answer = model.generate(problem, max_tokens=1000, cot=True)
    return answer
```

## Scaling Laws for Test-Time Compute

Research shows:
- Accuracy improves with more test-time compute (log-linear)
- Diminishing returns eventually
- Different problems have different scaling curves

```
Accuracy = f(log(test_time_compute))
```

## Cost-Quality Tradeoff

| Technique | Compute | Quality Gain |
|-----------|---------|--------------|
| Direct answer | 1× | Baseline |
| Chain-of-thought | 2-5× | +10-20% on reasoning |
| Self-consistency (5) | 5× | +5-10% additional |
| Verification loop | Variable | Problem-dependent |

## Implementation Considerations

### Latency
More compute = more latency
- Streaming helps user experience
- Background processing for non-interactive

### Cost
Test-time compute scales with usage
- Budget limits per query
- Tiered pricing

### When It Doesn't Help
- Simple factual queries
- Creative tasks (subjectivity)
- Tasks without verifiable answers

## Exercises

1. Implement self-consistency with majority voting
2. Build a verification loop that retries on detected errors
3. Compare accuracy vs compute on a reasoning benchmark

## Resources

- "Test-Time Compute: From System-1 to System-2" (Ji et al., 2025)
- "Self-Consistency Improves Chain of Thought" (Wang et al., 2022)
- OpenAI o1 technical details

## What's Next?

Module 3 dives deeper into **Chain-of-Thought & Beyond** — structured reasoning techniques.
