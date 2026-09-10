# Step 11: Production RAG Patterns

## Demo vs Production

Demo RAG works on happy paths. Production RAG handles:
- Bad queries
- Missing information
- Latency requirements
- Cost constraints
- Source attribution
- Failure modes

## Pattern 1: Query Preprocessing

### Intent Classification
Route queries to appropriate handlers:

```python
def classify_intent(query: str) -> str:
    """Classify query intent for routing."""
    prompt = f"""
    Classify this query into one of:
    - FACTUAL: Asking for specific information
    - COMPARISON: Comparing options
    - HOWTO: Asking for instructions
    - CHITCHAT: Small talk, greetings
    - UNCLEAR: Cannot determine intent
    
    Query: {query}
    Intent:
    """
    return llm(prompt).strip()

def route_query(query: str):
    intent = classify_intent(query)
    
    if intent == "CHITCHAT":
        return handle_chitchat(query)
    elif intent == "UNCLEAR":
        return ask_clarification(query)
    else:
        return rag_pipeline(query)
```

### Query Rewriting
Improve queries before retrieval:

```python
def rewrite_query(query: str, conversation_history: list = None) -> str:
    """Make query self-contained and clear."""
    
    if conversation_history:
        # Handle follow-up questions
        prompt = f"""
        Given the conversation history, rewrite the follow-up question
        to be self-contained (include context from history).
        
        History: {conversation_history}
        Follow-up: {query}
        
        Rewritten query:
        """
    else:
        prompt = f"""
        Rewrite this query to be clear and specific for search:
        Original: {query}
        Rewritten:
        """
    
    return llm(prompt).strip()

# Example:
# History: "What is Python?" -> "Python is a programming language..."
# Follow-up: "What about its main uses?"
# Rewritten: "What are the main uses of Python programming language?"
```

## Pattern 2: Context Window Management

LLM context windows are limited. Manage wisely:

```python
def fit_context(retrieved_docs: list, max_tokens: int = 3000) -> list:
    """Select documents that fit in context budget."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    
    selected = []
    current_tokens = 0
    
    for doc in retrieved_docs:
        doc_tokens = len(enc.encode(doc.text))
        
        if current_tokens + doc_tokens > max_tokens:
            break
        
        selected.append(doc)
        current_tokens += doc_tokens
    
    return selected

def summarize_if_needed(docs: list, max_tokens: int) -> str:
    """Summarize if too long, otherwise concatenate."""
    total = sum(count_tokens(doc.text) for doc in docs)
    
    if total <= max_tokens:
        return "\n\n".join(doc.text for doc in docs)
    
    # Summarize each doc
    summaries = [summarize(doc.text) for doc in docs]
    return "\n\n".join(summaries)
```

## Pattern 3: Caching

Cache expensive operations:

```python
from functools import lru_cache
import hashlib

# Cache embeddings
@lru_cache(maxsize=10000)
def cached_embed(text: str) -> tuple:
    """Cache embeddings (return tuple for hashability)."""
    return tuple(embed(text))

# Cache RAG responses
def get_cache_key(query: str, doc_ids: list) -> str:
    content = query + "".join(sorted(doc_ids))
    return hashlib.md5(content.encode()).hexdigest()

class RAGCache:
    def __init__(self, ttl_seconds: int = 3600):
        self.cache = {}
        self.ttl = ttl_seconds
    
    def get(self, query: str, doc_ids: list):
        key = get_cache_key(query, doc_ids)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                return entry["response"]
        return None
    
    def set(self, query: str, doc_ids: list, response: str):
        key = get_cache_key(query, doc_ids)
        self.cache[key] = {
            "response": response,
            "timestamp": time.time()
        }
```

## Pattern 4: Fallback Chains

Handle failures gracefully:

```python
def rag_with_fallbacks(query: str) -> str:
    # Try primary RAG
    try:
        docs = retrieve(query, k=5)
        
        if not docs or all(doc.score < 0.5 for doc in docs):
            # Low confidence retrieval - fallback to broader search
            docs = retrieve(query, k=10, threshold=0.3)
        
        if not docs:
            # No relevant docs found
            return generate_no_context_response(query)
        
        response = generate(query, docs)
        
        # Verify response isn't "I don't know" type
        if is_non_answer(response):
            return fallback_to_llm_knowledge(query)
        
        return response
        
    except RetrievalError:
        # Vector store down - fallback to LLM
        return fallback_to_llm_knowledge(query)
    
    except GenerationError:
        # LLM error - return graceful message
        return "I'm having trouble generating a response. Please try again."

def is_non_answer(response: str) -> bool:
    """Detect non-informative responses."""
    non_answers = [
        "I don't have information",
        "I cannot find",
        "not mentioned in the context",
    ]
    return any(phrase in response.lower() for phrase in non_answers)
```

## Pattern 5: Citation and Attribution

Show your sources:

```python
def generate_with_citations(query: str, docs: list) -> dict:
    """Generate answer with source citations."""
    
    # Number the documents
    numbered_context = ""
    for i, doc in enumerate(docs, 1):
        numbered_context += f"[{i}] {doc.text}\n\n"
    
    prompt = f"""
    Answer the question using ONLY the provided sources.
    Cite sources using [1], [2], etc.
    
    Sources:
    {numbered_context}
    
    Question: {query}
    
    Answer (with citations):
    """
    
    answer = llm(prompt)
    
    # Extract cited sources
    cited_nums = set(re.findall(r'\[(\d+)\]', answer))
    cited_sources = [
        {"index": i, "source": docs[int(i)-1].metadata["source"]}
        for i in cited_nums
    ]
    
    return {
        "answer": answer,
        "sources": cited_sources
    }
```

## Pattern 6: Cost Optimization

```python
class CostAwareRAG:
    def __init__(self, budget_per_query: float = 0.01):
        self.budget = budget_per_query
    
    def query(self, question: str) -> str:
        # Try cheap model first
        docs = self.retrieve(question, k=3)
        
        response = self.generate(
            question, 
            docs, 
            model="gpt-3.5-turbo"  # Cheap
        )
        
        # Check if response is confident
        if self.is_confident(response):
            return response
        
        # Escalate to expensive model
        if self.remaining_budget() > 0.005:
            return self.generate(
                question,
                docs,
                model="gpt-4"  # Expensive
            )
        
        return response
    
    def remaining_budget(self) -> float:
        return self.budget - self.spent
```

## Pattern 7: Observability

Log everything for debugging:

```python
import logging
from dataclasses import dataclass
from datetime import datetime

@dataclass
class RAGTrace:
    query_id: str
    timestamp: datetime
    original_query: str
    rewritten_query: str
    retrieved_docs: list
    retrieval_scores: list
    context_tokens: int
    response: str
    latency_ms: float
    model: str
    cost: float

def traced_rag(query: str) -> tuple[str, RAGTrace]:
    start = time.time()
    trace = RAGTrace(
        query_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        original_query=query,
        # ... fill in as we go
    )
    
    # Rewrite
    rewritten = rewrite_query(query)
    trace.rewritten_query = rewritten
    
    # Retrieve
    docs = retrieve(rewritten)
    trace.retrieved_docs = [d.id for d in docs]
    trace.retrieval_scores = [d.score for d in docs]
    
    # Generate
    response = generate(rewritten, docs)
    trace.response = response
    
    trace.latency_ms = (time.time() - start) * 1000
    
    # Log trace
    logger.info(f"RAG trace: {trace}")
    
    return response, trace
```

## Files

- `production_rag.py` - Production patterns implemented

## Key Takeaways

1. Preprocess queries (intent, rewriting)
2. Manage context window budget
3. Cache aggressively (embeddings, responses)
4. Implement fallback chains
5. Always cite sources
6. Optimize for cost
7. Log everything for debugging

## What's Next?

Step 12: **Advanced RAG Architectures** — Self-RAG, Graph RAG, and beyond.
