# Step 6: RAG Fundamentals

## The Problem with Pure LLMs

1. **Knowledge cutoff** — Training data has a date limit
2. **Hallucination** — Confident but wrong answers
3. **No access to private data** — Can't answer about your documents
4. **Generic responses** — Not specific to your domain

## The Solution: RAG

**Retrieval-Augmented Generation (RAG)** = Retrieve relevant documents + Generate answer using them

```
┌─────────────────────────────────────────────────────────────┐
│                       RAG Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Query                                                 │
│      │                                                      │
│      ▼                                                      │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────┐        │
│  │ Embed   │───▶│ Vector Store │───▶│ Top-K Docs  │        │
│  │ Query   │    │   Search     │    │  Retrieved  │        │
│  └─────────┘    └──────────────┘    └──────┬──────┘        │
│                                            │               │
│                                            ▼               │
│                               ┌────────────────────┐       │
│                               │ Query + Retrieved  │       │
│                               │ Context → LLM      │       │
│                               └─────────┬──────────┘       │
│                                         │                  │
│                                         ▼                  │
│                                   Final Answer             │
│                                 (grounded in docs)         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## The RAG Formula

```
Answer = LLM(Query + Retrieved_Context)
```

Where:
- **Query**: User's question
- **Retrieved_Context**: Relevant chunks from your documents
- **LLM**: Generates answer using the context

## RAG vs Fine-tuning vs Prompting

| Approach | Best For | Limitations |
|----------|----------|-------------|
| **Prompting** | General knowledge, reasoning | No access to new data |
| **RAG** | Factual Q&A, up-to-date info, citations | Retrieval quality limits answer |
| **Fine-tuning** | Style adaptation, specialized behavior | Expensive, static knowledge |

**Rule of thumb:**
- Need facts from documents → RAG
- Need different behavior/style → Fine-tuning
- Need reasoning about general knowledge → Prompting

## Basic RAG Implementation

```python
from openai import OpenAI
import chromadb

# 1. Setup
client = OpenAI()
chroma = chromadb.Client()
collection = chroma.create_collection("docs")

# 2. Index documents (one-time)
def index_documents(documents):
    for i, doc in enumerate(documents):
        # Embed and store
        embedding = get_embedding(doc["text"])
        collection.add(
            documents=[doc["text"]],
            embeddings=[embedding],
            metadatas=[doc["metadata"]],
            ids=[f"doc_{i}"]
        )

# 3. RAG query
def rag_query(question: str, k: int = 3) -> str:
    # Retrieve
    query_embedding = get_embedding(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    
    # Build context
    context = "\n\n".join(results["documents"][0])
    
    # Generate
    prompt = f"""Answer the question based only on the following context:

Context:
{context}

Question: {question}

Answer:"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content
```

## The RAG Pipeline Components

### 1. Ingestion
- Load documents (PDF, HTML, Markdown, etc.)
- Clean and preprocess text
- Split into chunks

### 2. Indexing
- Generate embeddings for each chunk
- Store in vector database
- Store metadata (source, page number, etc.)

### 3. Retrieval
- Embed the user query
- Search for similar chunks
- Return top-k most relevant

### 4. Generation
- Build prompt with query + retrieved context
- Generate answer with LLM
- Optionally include citations

## Why RAG Works

1. **Grounding**: Answer comes from actual documents
2. **Updateable**: Add new documents anytime (no retraining)
3. **Transparent**: Can show which documents were used
4. **Domain-specific**: Works with your private data

## Common RAG Pitfalls

| Pitfall | Symptom | Solution |
|---------|---------|----------|
| Bad chunking | Incomplete answers | Optimize chunk size |
| Retrieval misses | "I don't know" when answer exists | Better embedding model, reranking |
| Context overflow | Truncated context | Smarter selection, compression |
| Hallucination | Makes up facts not in docs | Stricter prompting, lower temp |

## Files

- `rag_fundamentals.py` - Basic RAG implementation

## Key Takeaways

1. RAG = Retrieve relevant docs + Generate using them
2. Solves hallucination, knowledge cutoff, and private data access
3. Four stages: Ingest → Index → Retrieve → Generate
4. Choose RAG for factual Q&A over documents
5. Quality depends on retrieval quality

## What's Next?

Step 7: **Embeddings & Vector Stores** — the heart of semantic search.
