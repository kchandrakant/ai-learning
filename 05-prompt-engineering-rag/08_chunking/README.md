# Step 8: Chunking Strategies

## Why Chunking Matters

You can't embed entire documents — they're too long for embedding models and LLM context windows. Chunking splits documents into retrieval-friendly pieces.

**Bad chunking = bad retrieval = bad answers.**

## The Chunking Tradeoff

```
Small chunks (100-200 tokens):
  ✅ Precise retrieval
  ✅ Less noise
  ❌ Lost context
  ❌ Incomplete information

Large chunks (1000+ tokens):
  ✅ Complete context
  ✅ Full information
  ❌ Retrieval noise
  ❌ Diluted relevance
```

**Sweet spot:** 200-500 tokens for most use cases.

## Chunking Strategies

### 1. Fixed-Size Chunking

Split every N characters/tokens:

```python
def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # Overlap for continuity
    return chunks
```

**Pros:** Simple, predictable size
**Cons:** Splits mid-sentence, mid-paragraph

### 2. Recursive Character Splitting

Split by natural boundaries, fall back to smaller units:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]  # Try in order
)

chunks = splitter.split_text(document)
```

**Hierarchy:** Paragraphs → Sentences → Words → Characters

### 3. Semantic Chunking

Split at topic boundaries (requires embedding):

```python
from sentence_transformers import SentenceTransformer
import numpy as np

def semantic_chunk(text: str, threshold: float = 0.5) -> list:
    # Split into sentences
    sentences = split_into_sentences(text)
    
    # Embed each sentence
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    
    # Find breakpoints (low similarity = topic change)
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(1, len(sentences)):
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])
        
        if similarity < threshold:
            # Topic change - start new chunk
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    
    chunks.append(" ".join(current_chunk))
    return chunks
```

**Pros:** Respects natural topic boundaries
**Cons:** More complex, requires embeddings

### 4. Document-Structure-Aware Chunking

Use document structure (headers, sections):

```python
def markdown_chunk(markdown_text: str) -> list:
    """Chunk by markdown headers."""
    chunks = []
    current_section = {"title": "", "content": ""}
    
    for line in markdown_text.split("\n"):
        if line.startswith("#"):
            # Save previous section
            if current_section["content"]:
                chunks.append(current_section)
            # Start new section
            current_section = {"title": line, "content": ""}
        else:
            current_section["content"] += line + "\n"
    
    if current_section["content"]:
        chunks.append(current_section)
    
    return chunks
```

### 5. Parent-Child Chunking

Store both small chunks (for retrieval) and large chunks (for context):

```
Document
    │
    ├── Parent Chunk (1000 tokens) ← Returned to LLM
    │       │
    │       ├── Child Chunk (200 tokens) ← Used for retrieval
    │       ├── Child Chunk (200 tokens)
    │       └── Child Chunk (200 tokens)
    │
    └── Parent Chunk (1000 tokens)
            │
            └── ...
```

```python
def parent_child_chunk(text: str) -> tuple:
    """Return both parent and child chunks."""
    # Large parent chunks
    parents = fixed_size_chunk(text, chunk_size=1000, overlap=0)
    
    # Small child chunks with parent reference
    children = []
    for i, parent in enumerate(parents):
        child_chunks = fixed_size_chunk(parent, chunk_size=200, overlap=50)
        for child in child_chunks:
            children.append({
                "text": child,
                "parent_id": i
            })
    
    return parents, children
```

**Retrieval:** Search children → Return corresponding parent

## Overlap Strategy

Overlap prevents information loss at boundaries:

```
Without overlap:
[Chunk 1: "The cat sat on the mat"]|[Chunk 2: "It was sleeping peacefully"]
                                  ^ Information could be split awkwardly

With overlap:
[Chunk 1: "The cat sat on the mat. It was"]
                        [Chunk 2: "on the mat. It was sleeping peacefully"]
                                  ^ Context preserved
```

**Rule of thumb:** 10-20% overlap (e.g., 50-100 tokens for 500-token chunks)

## Chunk Size Optimization

```python
def evaluate_chunk_size(documents, questions, sizes=[200, 500, 1000]):
    """Find optimal chunk size for your use case."""
    results = []
    
    for size in sizes:
        # Index with this chunk size
        index = create_index(documents, chunk_size=size)
        
        # Evaluate retrieval quality
        recall_at_k = evaluate_recall(index, questions, k=5)
        
        results.append({
            "chunk_size": size,
            "recall@5": recall_at_k
        })
    
    return results
```

## Metadata Enrichment

Always store metadata with chunks:

```python
chunk = {
    "text": "The actual chunk content...",
    "metadata": {
        "source": "document.pdf",
        "page": 5,
        "section": "Introduction",
        "chunk_index": 3,
        "total_chunks": 15,
        "doc_id": "abc123"
    }
}
```

This enables:
- Source citation
- Filtering by document/section
- Reconstructing context

## Files

- `chunking.py` - All chunking strategies implemented

## Key Takeaways

1. Chunking directly impacts retrieval quality
2. 200-500 tokens is a good starting point
3. Use overlap to preserve context (10-20%)
4. Match strategy to document type
5. Consider parent-child for best of both worlds
6. Always store metadata with chunks

## What's Next?

Step 9: **Retrieval Strategies** — beyond basic vector search.
