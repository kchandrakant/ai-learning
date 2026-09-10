# Step 5: Prompt Optimization

## The Problem

Prompts need iteration. Intuition-based tweaking is slow and unreliable.

## Systematic Prompt Development

```
┌──────────────────────────────────────────────────────┐
│  1. Define success metrics                           │
│  2. Create evaluation dataset                        │
│  3. Establish baseline                               │
│  4. Iterate systematically                           │
│  5. A/B test in production                           │
└──────────────────────────────────────────────────────┘
```

## Step 1: Define Success Metrics

What does "good" look like?

| Task Type | Possible Metrics |
|-----------|-----------------|
| Classification | Accuracy, F1, Precision, Recall |
| Generation | BLEU, ROUGE, human preference |
| Extraction | Exact match, partial match |
| Q&A | Correctness, faithfulness |
| Code | Passes tests, syntax correctness |

## Step 2: Create Evaluation Dataset

```python
eval_dataset = [
    {
        "input": "The movie was absolutely fantastic!",
        "expected": "positive",
        "difficulty": "easy"
    },
    {
        "input": "It wasn't bad, but not great either.",
        "expected": "neutral",
        "difficulty": "hard"  # Edge case
    },
    # Include edge cases!
]
```

**Rules for eval datasets:**
- Minimum 50-100 examples
- Include edge cases and hard examples
- Label with expected outputs
- Version control your dataset

## Step 3: Establish Baseline

```python
def evaluate_prompt(prompt_template, eval_dataset):
    results = []
    for item in eval_dataset:
        prompt = prompt_template.format(input=item["input"])
        response = llm(prompt)
        correct = response.strip() == item["expected"]
        results.append({
            "input": item["input"],
            "expected": item["expected"],
            "actual": response,
            "correct": correct
        })
    
    accuracy = sum(r["correct"] for r in results) / len(results)
    return accuracy, results

# Baseline
baseline_prompt = "Classify the sentiment: {input}"
baseline_score, _ = evaluate_prompt(baseline_prompt, eval_dataset)
print(f"Baseline accuracy: {baseline_score:.2%}")
```

## Step 4: Iterate Systematically

### Variation Strategies

**1. Instruction Clarity**
```
v1: "Classify the sentiment"
v2: "Classify the sentiment as positive, negative, or neutral"
v3: "Analyze the emotional tone and classify as exactly one of: positive, negative, neutral"
```

**2. Output Format**
```
v1: Return one word
v2: Return JSON: {"sentiment": "..."}
v3: Think step by step, then provide sentiment on last line
```

**3. Few-shot Examples**
```
v1: Zero-shot
v2: 3 basic examples
v3: 3 examples including edge cases
```

### Tracking Results

```python
experiments = []

for prompt_version, prompt_template in prompt_versions.items():
    score, results = evaluate_prompt(prompt_template, eval_dataset)
    experiments.append({
        "version": prompt_version,
        "prompt": prompt_template,
        "score": score,
        "timestamp": datetime.now()
    })
    
# Find best
best = max(experiments, key=lambda x: x["score"])
print(f"Best: {best['version']} with {best['score']:.2%}")
```

## Automated Prompt Optimization (DSPy Concepts)

Manual iteration is slow. Automated approaches:

```python
# Conceptual example (DSPy-inspired)
class SentimentClassifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("text -> sentiment")
    
    def forward(self, text):
        return self.classify(text=text)

# Compile (optimize) the prompt automatically
optimizer = dspy.BootstrapFewShot(metric=accuracy_metric)
optimized_classifier = optimizer.compile(
    SentimentClassifier(),
    trainset=train_examples
)
```

**Key idea:** Let the system find the best prompt/examples automatically.

## Step 5: A/B Testing in Production

```python
import random

def classify_with_ab_test(text, user_id):
    # Deterministic assignment based on user
    bucket = hash(user_id) % 100
    
    if bucket < 50:
        prompt = prompt_v1  # Control
        version = "v1"
    else:
        prompt = prompt_v2  # Treatment
        version = "v2"
    
    result = llm(prompt.format(input=text))
    
    # Log for analysis
    log_event({
        "user_id": user_id,
        "version": version,
        "input": text,
        "output": result,
        "timestamp": datetime.now()
    })
    
    return result
```

## Cost-Quality Tradeoffs

```
┌─────────────────────────────────────────────────────┐
│                                                      │
│   Quality ▲                                         │
│           │            ● GPT-4 + CoT                │
│           │       ● GPT-4                           │
│           │   ● GPT-3.5 + CoT                       │
│           │ ● GPT-3.5                               │
│           │● Local model                            │
│           └──────────────────────────────▶ Cost     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

Consider:
- Can a cheaper model + better prompt match expensive model?
- Is CoT worth the extra tokens?
- What's the minimum acceptable quality?

## Files

- `prompt_optimization.py` - Evaluation framework implementation

## Key Takeaways

1. Define measurable success metrics first
2. Create a versioned evaluation dataset with edge cases
3. Establish a baseline before optimizing
4. Track all experiments systematically
5. A/B test before full production rollout
6. Consider cost-quality tradeoffs

## What's Next?

Step 6: **RAG Fundamentals** — grounding LLMs with external knowledge.
