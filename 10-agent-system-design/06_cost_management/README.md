# Module 6: Cost Management & Token Economics

Making agent systems economically viable.

## Overview

LLM inference is expensive. A poorly designed agent can burn through budgets in minutes. Cost-aware design is essential for production systems.

## Key Topics

### Token Counting & Attribution
```python
def count_tokens(text: str, model: str) -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def calculate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    rates = {
        "gpt-4": {"input": 0.03/1000, "output": 0.06/1000},
        "gpt-3.5-turbo": {"input": 0.0005/1000, "output": 0.0015/1000},
    }
    return input_tokens * rates[model]["input"] + output_tokens * rates[model]["output"]
```

### Cost-Aware Model Routing
```
Simple query → GPT-3.5 ($0.001)
Complex reasoning → GPT-4 ($0.10)
Code generation → Claude ($0.05)
```

**Decision factors:**
- Task complexity classification
- Quality requirements
- Latency constraints
- Budget remaining

### Caching Strategies

**Exact match cache:**
- Cache identical prompts
- High hit rate for repeated queries

**Semantic cache:**
- Embed queries, find similar cached responses
- Works for paraphrased questions

**Partial cache:**
- Cache tool outputs
- Cache intermediate reasoning steps

### Context Pruning
- Remove old conversation turns
- Summarize instead of full history
- Prioritize recent context
- Drop low-relevance retrieved documents

### Budget Injection
```
System: "You have $0.50 remaining budget. Prioritize efficiency."
```

Agents can adapt behavior based on budget:
- Use cheaper models
- Reduce exploration
- Skip optional steps

### Cost Dashboards
Track:
- Cost per user/task/session
- Token usage breakdown (input vs output)
- Cache hit rate
- Cost trends over time

## Exercises

1. Implement token counting and cost tracking
2. Build a semantic caching layer
3. Design a cost-aware model router
4. Create a budget management system with alerts

## Key Insight

A 10x cheaper solution that's 90% as good often wins. Optimize for cost per success, not just success rate.

## Cost Comparison (Order of Magnitude)

| Approach | Cost per Query |
|----------|---------------|
| GPT-4 + full context | $0.10 |
| GPT-3.5 + pruned context | $0.001 |
| Cached response | $0.00001 |
| Local model | $0.0001 (compute) |
