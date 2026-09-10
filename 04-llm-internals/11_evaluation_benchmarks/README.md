# Module 11: Evaluation & Benchmarks

Measuring model capabilities.

## Overview

Benchmarks drive progress but have limitations. Understanding them helps interpret model comparisons.

## Key Topics

### Standard Benchmarks
```
Knowledge:
- MMLU: 57 subjects, multiple choice
- ARC: Science questions
- TriviaQA: Factual recall

Reasoning:
- GSM8K: Grade school math
- MATH: Competition math
- BBH: BIG-Bench Hard subset

Code:
- HumanEval: Function completion
- MBPP: Python problems
- SWE-bench: Real GitHub issues
```

### Evaluation Metrics
```
Accuracy:    % correct (for classification)
Pass@k:      Solve in k attempts (for code)
Perplexity:  How "surprised" by test data
F1/BLEU:     Token overlap (for generation)
```

### Benchmark Limitations
```
Contamination: Test data in training set
Saturation:    Top models near 100%
Gaming:        Optimizing metric, not capability
Distribution:  Benchmark ≠ real use cases
```

### Human Evaluation
```
When benchmarks fail:
- Chatbot Arena (Elo ratings from users)
- Blind A/B comparisons
- Task-specific human judges

Gold standard but expensive and slow
```

### What Benchmarks Miss
```
- Open-ended creativity
- Long-horizon reasoning
- Real-world robustness
- Safety and alignment
- Efficiency (cost, speed)
```

## Exercises

1. Run evaluations on multiple benchmarks
2. Analyze contamination in a dataset
3. Design a custom evaluation for a use case

## Key Insight

Benchmarks are necessary but insufficient. Real-world performance requires broader evaluation.
