# Step 2: Few-Shot Learning

## The Power of Examples

Few-shot learning lets you "program" an LLM through demonstrations rather than instructions. Often more effective than verbose explanations.

## The Pattern

```
[Context/System instruction]

Example 1:
Input: {example_input_1}
Output: {example_output_1}

Example 2:
Input: {example_input_2}
Output: {example_output_2}

Now complete this:
Input: {actual_input}
Output:
```

## When Few-Shot Beats Zero-Shot

| Scenario | Best Approach |
|----------|---------------|
| Custom output format | Few-shot (show the format) |
| Domain-specific language | Few-shot (demonstrate style) |
| Complex classification | Few-shot (show edge cases) |
| Simple, well-known tasks | Zero-shot (examples unnecessary) |

## Example Selection Strategies

### 1. Representative Examples
Choose examples that cover the range of expected inputs.

```python
# For sentiment analysis, include positive, negative, AND neutral
examples = [
    ("I love this product!", "positive"),
    ("This is the worst purchase ever.", "negative"),
    ("The package arrived on Tuesday.", "neutral"),  # ← Often forgotten!
]
```

### 2. Similar Examples (Dynamic)
Retrieve examples most similar to the current input.

```python
def get_similar_examples(query, example_pool, k=3):
    """Find k most similar examples to the query."""
    query_embedding = embed(query)
    similarities = [
        (example, cosine_similarity(query_embedding, embed(example.input)))
        for example in example_pool
    ]
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:k]
```

### 3. Diverse Examples
Ensure examples cover different patterns/categories.

## How Many Examples?

```
1-2 examples: Minimal guidance, may be inconsistent
3-5 examples: Usually optimal balance
6+  examples: Diminishing returns, uses context window
```

**Note:** More isn't always better — too many examples waste tokens and can confuse the model.

## Few-Shot for Different Tasks

### Classification
```
Text: "The movie was a masterpiece"
Category: Entertainment

Text: "Stock prices fell 3% today"
Category: Finance

Text: "New study links coffee to longevity"
Category: ?
```

### Extraction
```
Input: "Contact John at john@email.com or 555-1234"
Output: {"name": "John", "email": "john@email.com", "phone": "555-1234"}

Input: "Reach out to Sarah (sarah@company.org)"
Output: {"name": "Sarah", "email": "sarah@company.org", "phone": null}

Input: "Call Mike at 555-9876"
Output: ?
```

### Transformation
```
Formal: "We regret to inform you that your application was unsuccessful."
Casual: "Sorry, but your application wasn't accepted this time."

Formal: "Please be advised that the meeting has been rescheduled."
Casual: ?
```

## Anti-Patterns

❌ **Inconsistent format across examples**
```
Example 1: positive
Example 2: POSITIVE  
Example 3: Positive sentiment
```

❌ **Examples that contradict each other**

❌ **Too many examples for a simple task**

## Files

- `few_shot.py` - Implementation with dynamic example selection

## Key Takeaways

1. Examples often work better than instructions
2. 3-5 examples is usually the sweet spot
3. Choose representative AND diverse examples
4. Keep format consistent across all examples
5. Consider dynamic example selection for complex tasks

## What's Next?

Step 3: **Chain-of-Thought Prompting** — eliciting step-by-step reasoning.
