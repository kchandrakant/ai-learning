# Step 12: Evaluation & Benchmarking

## Did Fine-tuning Work?

Measure both during and after training.

## Training Metrics

### Loss & Perplexity

```python
# Track during training
def compute_perplexity(loss):
    return torch.exp(loss)

# Lower is better (but watch for overfitting!)
```

### Validation Loss

```python
trainer = Trainer(
    model=model,
    train_dataset=train,
    eval_dataset=val,
    compute_metrics=compute_metrics,
)
```

## Task-Specific Evaluation

### Classification

```python
from sklearn.metrics import accuracy_score, f1_score

def evaluate_classification(model, test_data):
    predictions = []
    labels = []
    
    for example in test_data:
        pred = model.generate(example["input"])
        predictions.append(pred)
        labels.append(example["label"])
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="macro")
    }
```

### Generation (BLEU, ROUGE)

```python
from evaluate import load

bleu = load("bleu")
rouge = load("rouge")

def evaluate_generation(model, test_data):
    predictions = [model.generate(x["input"]) for x in test_data]
    references = [x["output"] for x in test_data]
    
    return {
        "bleu": bleu.compute(predictions=predictions, references=references),
        "rouge": rouge.compute(predictions=predictions, references=references)
    }
```

## LLM-as-Judge

```python
def judge_response(instruction, response, judge_model):
    prompt = f"""
    Rate this response from 1-10.
    
    Instruction: {instruction}
    Response: {response}
    
    Consider: accuracy, helpfulness, clarity.
    Score:
    """
    return int(judge_model.generate(prompt))
```

## A/B Testing

```python
def ab_test(model_a, model_b, test_prompts, judge):
    wins_a = 0
    wins_b = 0
    
    for prompt in test_prompts:
        resp_a = model_a.generate(prompt)
        resp_b = model_b.generate(prompt)
        
        winner = judge.compare(prompt, resp_a, resp_b)
        if winner == "a":
            wins_a += 1
        else:
            wins_b += 1
    
    return {"model_a_wins": wins_a, "model_b_wins": wins_b}
```

## Evaluation Checklist

- [ ] Validation loss decreasing?
- [ ] Not overfitting (train << val)?
- [ ] Task-specific metrics improving?
- [ ] Human evaluation positive?
- [ ] Compared to baseline?

## Files

- `evaluation.py` - Evaluation utilities

## Key Takeaways

1. Track loss and perplexity during training
2. Use task-specific metrics
3. LLM-as-judge for subjective quality
4. A/B test against baseline
5. Human evaluation is the gold standard
