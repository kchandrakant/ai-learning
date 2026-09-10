# Step 10: RAG Evaluation

## Why Evaluation Matters

Without evaluation, you're guessing. RAG has multiple failure modes:
1. **Retrieval fails** → Wrong documents retrieved
2. **Generation fails** → Ignores context or hallucinates
3. **Both fail** → Complete miss

**Evaluate retrieval and generation separately.**

## The Evaluation Framework

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Evaluation                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Query                                                     │
│     │                                                       │
│     ▼                                                       │
│   Retrieval ────────▶ Retrieval Metrics                    │
│     │                  • Recall@K                           │
│     │                  • MRR                                │
│     │                  • NDCG                               │
│     ▼                                                       │
│   Generation ───────▶ Generation Metrics                   │
│     │                  • Faithfulness                       │
│     │                  • Answer Relevance                   │
│     │                  • Context Precision                  │
│     ▼                                                       │
│   Final Answer ─────▶ End-to-End Metrics                   │
│                        • Correctness                        │
│                        • Human Preference                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Retrieval Metrics

### Recall@K
"Did we retrieve the relevant documents?"

```python
def recall_at_k(retrieved_docs: list, relevant_docs: list, k: int) -> float:
    """
    What fraction of relevant docs were retrieved in top-k?
    """
    retrieved_set = set(doc.id for doc in retrieved_docs[:k])
    relevant_set = set(doc.id for doc in relevant_docs)
    
    if not relevant_set:
        return 0.0
    
    found = len(retrieved_set & relevant_set)
    return found / len(relevant_set)

# Example: 3 relevant docs, retrieved 2 of them in top-5
# Recall@5 = 2/3 = 0.67
```

### Mean Reciprocal Rank (MRR)
"How high is the first relevant document?"

```python
def mrr(retrieved_docs: list, relevant_docs: list) -> float:
    """
    1/rank of first relevant document.
    """
    relevant_set = set(doc.id for doc in relevant_docs)
    
    for rank, doc in enumerate(retrieved_docs, 1):
        if doc.id in relevant_set:
            return 1.0 / rank
    
    return 0.0

# Example: First relevant doc at position 3
# MRR = 1/3 = 0.33
```

### NDCG (Normalized Discounted Cumulative Gain)
"How good is the ranking quality?"

```python
import numpy as np

def ndcg_at_k(retrieved_docs: list, relevance_scores: dict, k: int) -> float:
    """
    Considers graded relevance (not just binary).
    """
    # DCG: sum of relevance / log2(rank + 1)
    dcg = 0.0
    for i, doc in enumerate(retrieved_docs[:k]):
        rel = relevance_scores.get(doc.id, 0)
        dcg += rel / np.log2(i + 2)
    
    # Ideal DCG: perfect ranking
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg
```

## Generation Metrics

### Faithfulness
"Is the answer grounded in the retrieved context?"

```python
def evaluate_faithfulness(answer: str, context: str) -> float:
    """
    LLM-as-judge: Does the answer only use information from context?
    """
    prompt = f"""
    Evaluate if the answer is faithful to the given context.
    The answer should ONLY contain information present in the context.
    
    Context: {context}
    
    Answer: {answer}
    
    Score from 0 (completely unfaithful) to 1 (completely faithful):
    Provide just the number.
    """
    
    score = float(llm(prompt).strip())
    return score
```

### Answer Relevance
"Does the answer address the question?"

```python
def evaluate_relevance(question: str, answer: str) -> float:
    """
    LLM-as-judge: Does the answer address the question?
    """
    prompt = f"""
    Evaluate if the answer addresses the question.
    
    Question: {question}
    
    Answer: {answer}
    
    Score from 0 (completely irrelevant) to 1 (perfectly relevant):
    Provide just the number.
    """
    
    score = float(llm(prompt).strip())
    return score
```

### Context Precision
"Was the retrieved context useful?"

```python
def context_precision(question: str, context_chunks: list) -> float:
    """
    What fraction of retrieved chunks were actually useful?
    """
    useful_count = 0
    
    for chunk in context_chunks:
        prompt = f"""
        Is this context useful for answering the question?
        
        Question: {question}
        Context: {chunk}
        
        Answer YES or NO:
        """
        
        if "YES" in llm(prompt).upper():
            useful_count += 1
    
    return useful_count / len(context_chunks)
```

## Using RAGAS Framework

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from datasets import Dataset

# Prepare evaluation data
eval_data = {
    "question": ["What is X?", "How does Y work?"],
    "answer": ["X is...", "Y works by..."],
    "contexts": [["context1", "context2"], ["context3"]],
    "ground_truths": [["X is..."], ["Y works..."]]
}

dataset = Dataset.from_dict(eval_data)

# Run evaluation
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]
)

print(results)
```

## Creating an Evaluation Dataset

```python
eval_dataset = [
    {
        "question": "What is the return policy?",
        "ground_truth_answer": "30 days full refund",
        "relevant_doc_ids": ["policy_doc_1", "faq_returns"],
        "difficulty": "easy"
    },
    {
        "question": "Can I return opened electronics?",
        "ground_truth_answer": "Only within 14 days, restocking fee applies",
        "relevant_doc_ids": ["policy_doc_1", "electronics_policy"],
        "difficulty": "hard"  # Requires combining info from multiple docs
    },
]
```

**Best practices:**
- 50-100 questions minimum
- Include easy, medium, and hard questions
- Include questions with no answer in corpus (to test "I don't know")
- Have humans label relevant documents

## End-to-End Evaluation Pipeline

```python
def evaluate_rag_system(rag_fn, eval_dataset):
    results = []
    
    for item in eval_dataset:
        # Run RAG
        retrieved_docs = rag_fn.retrieve(item["question"])
        answer = rag_fn.generate(item["question"], retrieved_docs)
        
        # Retrieval metrics
        recall = recall_at_k(retrieved_docs, item["relevant_doc_ids"], k=5)
        mrr = mrr(retrieved_docs, item["relevant_doc_ids"])
        
        # Generation metrics
        faithful = evaluate_faithfulness(answer, retrieved_docs)
        relevant = evaluate_relevance(item["question"], answer)
        
        # Correctness (if ground truth available)
        correct = evaluate_correctness(answer, item["ground_truth_answer"])
        
        results.append({
            "question": item["question"],
            "recall@5": recall,
            "mrr": mrr,
            "faithfulness": faithful,
            "relevance": relevant,
            "correctness": correct
        })
    
    # Aggregate
    return {
        "avg_recall@5": np.mean([r["recall@5"] for r in results]),
        "avg_mrr": np.mean([r["mrr"] for r in results]),
        "avg_faithfulness": np.mean([r["faithfulness"] for r in results]),
        "avg_relevance": np.mean([r["relevance"] for r in results]),
        "avg_correctness": np.mean([r["correctness"] for r in results]),
    }
```

## Files

- `rag_evaluation.py` - Complete evaluation framework

## Key Takeaways

1. Evaluate retrieval and generation separately
2. Retrieval: Recall@K, MRR, NDCG
3. Generation: Faithfulness, Relevance, Precision
4. Use LLM-as-judge for nuanced evaluation
5. Create a labeled evaluation dataset
6. Track metrics over time as you iterate

## What's Next?

Step 11: **Production RAG Patterns** — from demo to production.
