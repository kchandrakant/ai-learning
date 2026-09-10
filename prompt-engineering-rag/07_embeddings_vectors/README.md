# Step 7: Embeddings & Vector Stores

## What Are Embeddings?

Embeddings convert text into dense numerical vectors that capture semantic meaning.

```
"The cat sat on the mat"  →  [0.02, -0.15, 0.83, ..., 0.41]  (1536 dimensions)
"A feline rested on the rug" →  [0.03, -0.14, 0.81, ..., 0.39]  (similar vector!)
```

**Key insight:** Similar meanings = similar vectors = closer in vector space.

## How Similarity Works

### Cosine Similarity (Most Common)

```
similarity = (A · B) / (||A|| × ||B||)
```

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Example
embed_cat = get_embedding("cat")
embed_dog = get_embedding("dog")
embed_car = get_embedding("car")

# cat ↔ dog: ~0.85 (similar - both animals)
# cat ↔ car: ~0.45 (less similar - different domains)
```

### Distance Metrics Comparison

| Metric | Formula | When to Use |
|--------|---------|-------------|
| Cosine | 1 - cos(θ) | Normalized vectors, text similarity |
| Euclidean | ||A - B|| | Absolute differences matter |
| Dot Product | A · B | Already normalized, fast |

## Embedding Models

### OpenAI
```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list:
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding
```

### Open Source (Sentence Transformers)
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str) -> list:
    return model.encode(text).tolist()
```

### Model Comparison

| Model | Dimensions | Speed | Quality | Cost |
|-------|------------|-------|---------|------|
| text-embedding-3-small | 1536 | Fast | Good | $0.02/1M tokens |
| text-embedding-3-large | 3072 | Medium | Excellent | $0.13/1M tokens |
| all-MiniLM-L6-v2 | 384 | Very Fast | Good | Free |
| bge-large-en | 1024 | Medium | Excellent | Free |

## Vector Stores

Vector databases store embeddings and enable fast similarity search.

### Chroma (Local, Easy)
```python
import chromadb

# Create client
client = chromadb.Client()

# Create collection
collection = client.create_collection("my_docs")

# Add documents
collection.add(
    documents=["doc1 text", "doc2 text"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    metadatas=[{"source": "file1"}, {"source": "file2"}],
    ids=["id1", "id2"]
)

# Query
results = collection.query(
    query_embeddings=[[0.15, 0.25, ...]],
    n_results=3
)
```

### Pinecone (Cloud, Scalable)
```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-key")
index = pc.Index("my-index")

# Upsert
index.upsert(vectors=[
    {"id": "doc1", "values": [0.1, 0.2, ...], "metadata": {"source": "file1"}}
])

# Query
results = index.query(
    vector=[0.15, 0.25, ...],
    top_k=3,
    include_metadata=True
)
```

### pgvector (PostgreSQL Extension)
```sql
-- Enable extension
CREATE EXTENSION vector;

-- Create table
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536)
);

-- Query
SELECT content, embedding <=> '[0.1, 0.2, ...]' AS distance
FROM documents
ORDER BY distance
LIMIT 3;
```

### Comparison

| Store | Type | Scale | Best For |
|-------|------|-------|----------|
| Chroma | Local | Small-Medium | Prototyping, dev |
| Pinecone | Cloud | Large | Production, scale |
| Weaviate | Both | Large | Hybrid search |
| pgvector | Local | Medium | Existing Postgres |
| Qdrant | Both | Large | High performance |

## Index Types

### Flat (Exact)
- Compares query to every vector
- 100% accurate, slow for large datasets

### HNSW (Approximate)
- Hierarchical Navigable Small World
- ~95-99% accurate, very fast
- Most common choice

### IVF (Inverted File)
- Clusters vectors, searches relevant clusters
- Good balance of speed/accuracy

```python
# Chroma with HNSW (default)
collection = client.create_collection(
    "my_docs",
    metadata={"hnsw:space": "cosine"}
)
```

## Embedding Best Practices

1. **Use the right model for your domain**
   - General: OpenAI, BGE
   - Code: CodeBERT, StarCoder embeddings
   - Multi-lingual: multilingual-e5-large

2. **Normalize vectors** (if using dot product)
   ```python
   embedding = embedding / np.linalg.norm(embedding)
   ```

3. **Batch embedding calls**
   ```python
   # Good: One call
   embeddings = client.embeddings.create(input=texts_list, model=model)
   
   # Bad: Many calls
   for text in texts_list:
       embedding = client.embeddings.create(input=text, model=model)
   ```

4. **Cache embeddings**
   - Don't re-embed the same text
   - Store embeddings with documents

## Files

- `embeddings_vectors.py` - Embedding and vector store implementations

## Key Takeaways

1. Embeddings capture semantic meaning as vectors
2. Similar texts → similar vectors → close in vector space
3. Cosine similarity is the standard metric
4. Choose embedding model based on domain and cost
5. Vector stores enable fast similarity search at scale

## What's Next?

Step 8: **Chunking Strategies** — how you split documents matters.
