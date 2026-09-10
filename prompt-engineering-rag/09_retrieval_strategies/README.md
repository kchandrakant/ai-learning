# Step 9: Retrieval Strategies

## Beyond Basic Vector Search

Basic RAG: embed query → find similar vectors → done.

**Problem:** Vector search alone often misses relevant documents.

## The Retrieval Pipeline

```
Query
  │
  ▼
┌──────────────────┐
│ Query Processing │ ← Rewrite, expand, decompose
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Candidate Recall │ ← Hybrid search (vector + keyword)
│   (High recall)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    Reranking     │ ← Cross-encoder scoring
│ (High precision) │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Compression    │ ← Extract relevant parts
└────────┬─────────┘
         │
         ▼
    Top-K Results
```

## Strategy 1: Hybrid Search

Combine vector (semantic) and keyword (lexical) search:

```python
def hybrid_search(query: str, alpha: float = 0.5, k: int = 10):
    """
    alpha=1.0: pure vector search
    alpha=0.0: pure keyword search
    alpha=0.5: balanced
    """
    # Vector search
    vector_results = vector_store.similarity_search(query, k=k*2)
    
    # Keyword search (BM25)
    keyword_results = bm25_search(query, k=k*2)
    
    # Combine scores (Reciprocal Rank Fusion)
    combined = reciprocal_rank_fusion(
        [vector_results, keyword_results],
        weights=[alpha, 1-alpha]
    )
    
    return combined[:k]

def reciprocal_rank_fusion(result_lists, weights, k=60):
    """RRF: score = sum(1 / (k + rank))"""
    scores = {}
    for results, weight in zip(result_lists, weights):
        for rank, doc in enumerate(results):
            if doc.id not in scores:
                scores[doc.id] = 0
            scores[doc.id] += weight * (1 / (k + rank + 1))
    
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**When to use hybrid:**
- Technical terms (API names, product IDs)
- Exact phrases matter
- Domain-specific vocabulary

## Strategy 2: Reranking

First stage retrieves candidates (high recall, lower precision).
Reranker scores each candidate against the query (high precision).

```python
from sentence_transformers import CrossEncoder

# Cross-encoder reranker
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank(query: str, documents: list, top_k: int = 5):
    # Score each document against the query
    pairs = [[query, doc.text] for doc in documents]
    scores = reranker.predict(pairs)
    
    # Sort by score
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in ranked[:top_k]]

# Usage
candidates = vector_store.similarity_search(query, k=20)  # Recall
final = rerank(query, candidates, top_k=5)                # Precision
```

**Why reranking works:**
- Bi-encoders (embedding models): encode query and docs separately
- Cross-encoders (rerankers): encode query+doc together, see interactions

## Strategy 3: Multi-Query Retrieval

Generate multiple query variations, retrieve for each, combine:

```python
def multi_query_retrieval(original_query: str, k: int = 5):
    # Generate query variations
    variations_prompt = f"""
    Generate 3 different versions of this question to help with search:
    Original: {original_query}
    
    Variations:
    """
    variations = llm(variations_prompt).split("\n")
    
    # Retrieve for each variation
    all_docs = set()
    for query in [original_query] + variations:
        docs = vector_store.similarity_search(query, k=k)
        all_docs.update(docs)
    
    # Rerank combined results
    return rerank(original_query, list(all_docs), top_k=k)
```

**Why it works:** Different phrasings match different relevant documents.

## Strategy 4: Query Decomposition

Break complex queries into sub-queries:

```python
def decompose_and_retrieve(complex_query: str):
    # Decompose
    decompose_prompt = f"""
    Break this complex question into simpler sub-questions:
    Question: {complex_query}
    
    Sub-questions:
    """
    sub_questions = llm(decompose_prompt).split("\n")
    
    # Retrieve for each sub-question
    all_context = []
    for sq in sub_questions:
        docs = retrieve(sq, k=3)
        all_context.append({
            "sub_question": sq,
            "context": docs
        })
    
    # Generate answer using all context
    return generate_answer(complex_query, all_context)
```

## Strategy 5: Contextual Compression

Extract only the relevant parts of retrieved documents:

```python
def compress_context(query: str, documents: list) -> list:
    compressed = []
    
    for doc in documents:
        prompt = f"""
        Extract only the parts of this document that are relevant to the question.
        If nothing is relevant, respond with "NOT_RELEVANT".
        
        Question: {query}
        
        Document:
        {doc.text}
        
        Relevant extract:
        """
        
        extract = llm(prompt)
        if extract != "NOT_RELEVANT":
            compressed.append(extract)
    
    return compressed
```

**Benefit:** Fit more relevant information in the context window.

## Strategy 6: Parent Document Retrieval

Retrieve small chunks, return larger parent context:

```python
def parent_document_retrieval(query: str, k: int = 3):
    # Search against small chunks
    child_results = vector_store.similarity_search(
        query, 
        k=k,
        collection="child_chunks"
    )
    
    # Get parent documents
    parent_ids = set(doc.metadata["parent_id"] for doc in child_results)
    
    parents = []
    for parent_id in parent_ids:
        parent = document_store.get(parent_id)
        parents.append(parent)
    
    return parents
```

## Strategy Comparison

| Strategy | Best For | Cost |
|----------|----------|------|
| Vector only | General semantic search | Low |
| Hybrid | Technical terms, exact matches | Low |
| Reranking | Improving precision | Medium |
| Multi-query | Broad coverage | Medium |
| Decomposition | Complex questions | High |
| Compression | Long documents | High |

## Combining Strategies

```python
def advanced_retrieval(query: str, k: int = 5):
    # 1. Multi-query expansion
    queries = expand_query(query)
    
    # 2. Hybrid search for each
    all_candidates = []
    for q in queries:
        candidates = hybrid_search(q, k=10)
        all_candidates.extend(candidates)
    
    # 3. Deduplicate
    unique_candidates = deduplicate(all_candidates)
    
    # 4. Rerank
    reranked = rerank(query, unique_candidates, top_k=k*2)
    
    # 5. Compress
    compressed = compress_context(query, reranked[:k])
    
    return compressed
```

## Files

- `retrieval_strategies.py` - All strategies implemented

## Key Takeaways

1. Basic vector search is often not enough
2. Hybrid search (vector + keyword) catches more relevant docs
3. Reranking dramatically improves precision
4. Multi-query helps with query variation
5. Combine strategies for best results
6. More sophisticated = higher latency/cost

## What's Next?

Step 10: **RAG Evaluation** — measuring what matters.
