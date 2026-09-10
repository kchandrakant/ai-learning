# Step 2: Data Preparation

## Data Quality > Data Quantity

A few thousand high-quality examples beat millions of low-quality ones.

## Dataset Formats

### Instruction Format (Alpaca-style)

```json
{
    "instruction": "Write a haiku about programming",
    "input": "",
    "output": "Code flows like water\nBugs emerge from the shadows\nDebug, compile, run"
}
```

### Chat Format (ShareGPT-style)

```json
{
    "conversations": [
        {"from": "human", "value": "What is Python?"},
        {"from": "gpt", "value": "Python is a programming language..."},
        {"from": "human", "value": "Show me an example"},
        {"from": "gpt", "value": "Here's a simple example:\n```python\nprint('Hello')\n```"}
    ]
}
```

### OpenAI Format

```json
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"}
    ]
}
```

## Data Cleaning

```python
def clean_example(example: dict) -> dict | None:
    # Remove empty responses
    if not example["output"].strip():
        return None
    
    # Remove too short responses
    if len(example["output"]) < 10:
        return None
    
    # Remove duplicates (by hash)
    # Check for quality issues
    
    return example

# Apply cleaning
dataset = dataset.filter(lambda x: clean_example(x) is not None)
```

## Deduplication

```python
from datasketch import MinHash, MinHashLSH

def deduplicate(examples: list, threshold: float = 0.8) -> list:
    lsh = MinHashLSH(threshold=threshold, num_perm=128)
    unique = []
    
    for i, ex in enumerate(examples):
        mh = MinHash(num_perm=128)
        for word in ex["output"].split():
            mh.update(word.encode())
        
        # Check for near-duplicates
        if not lsh.query(mh):
            lsh.insert(str(i), mh)
            unique.append(ex)
    
    return unique
```

## Quality Filtering

```python
def quality_score(example: dict) -> float:
    score = 1.0
    
    # Penalize very short responses
    if len(example["output"]) < 50:
        score *= 0.5
    
    # Penalize repetition
    words = example["output"].split()
    unique_ratio = len(set(words)) / len(words)
    score *= unique_ratio
    
    # Penalize bad formatting
    if example["output"].count("```") % 2 != 0:  # Unclosed code blocks
        score *= 0.3
    
    return score

# Filter low quality
dataset = dataset.filter(lambda x: quality_score(x) > 0.5)
```

## Train/Validation Split

```python
from sklearn.model_selection import train_test_split

train, val = train_test_split(dataset, test_size=0.1, random_state=42)

# Or with HuggingFace
dataset = dataset.train_test_split(test_size=0.1)
```

## Files

- `data_preparation.py` - Data processing utilities

## Key Takeaways

1. Quality over quantity
2. Use consistent format
3. Deduplicate aggressively
4. Filter low-quality examples
5. Always have a validation set
