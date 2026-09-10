# Step 14: Caching

## Why Cache?

- Reduce latency
- Lower costs
- Handle repeated queries

## Semantic Caching

Cache based on meaning, not exact match:

```python
import chromadb
import hashlib

class SemanticCache:
    def __init__(self, similarity_threshold=0.95):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("cache")
        self.threshold = similarity_threshold
    
    def get(self, query: str) -> str | None:
        results = self.collection.query(
            query_texts=[query],
            n_results=1
        )
        
        if results["distances"][0][0] < (1 - self.threshold):
            return results["metadatas"][0][0]["response"]
        return None
    
    def set(self, query: str, response: str):
        key = hashlib.md5(query.encode()).hexdigest()
        self.collection.add(
            documents=[query],
            metadatas=[{"response": response}],
            ids=[key]
        )
```

## Exact Match Cache

```python
import redis

class ExactCache:
    def __init__(self, ttl=3600):
        self.redis = redis.Redis()
        self.ttl = ttl
    
    def get(self, key: str) -> str | None:
        return self.redis.get(key)
    
    def set(self, key: str, value: str):
        self.redis.setex(key, self.ttl, value)
```

## Cache Strategy

```python
def generate_with_cache(query: str) -> str:
    # Try exact match first
    cache_key = hash_query(query)
    if cached := exact_cache.get(cache_key):
        return cached
    
    # Try semantic match
    if cached := semantic_cache.get(query):
        return cached
    
    # Generate and cache
    response = model.generate(query)
    exact_cache.set(cache_key, response)
    semantic_cache.set(query, response)
    
    return response
```

## Files

- `caching.py` - Caching implementations
