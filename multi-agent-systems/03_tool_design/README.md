# Step 3: Tool Design

## Tools Are the Agent's Hands

An agent is only as capable as its tools. Well-designed tools = effective agents.

## Tool Definition Anatomy

```python
{
    "name": "get_weather",
    "description": "Get current weather for a location. Use when user asks about weather conditions.",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g., 'San Francisco' or 'London, UK'"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units (default: celsius)"
            }
        },
        "required": ["location"]
    }
}
```

## The Golden Rules

### 1. Clear, Descriptive Names
```
❌ fn1, doThing, process
✅ get_weather, send_email, search_documents
```

### 2. Single Responsibility
```
❌ search_and_summarize_and_email()  # Does too much
✅ search() → summarize() → send_email()  # Composable
```

### 3. Explicit Documentation
```python
def search_documents(
    query: str,           # What to search for
    max_results: int = 5, # How many to return
    filters: dict = None  # Optional metadata filters
) -> list[Document]:
    """
    Search the document database for relevant content.
    
    Use this tool when the user asks questions that require
    looking up information from our knowledge base.
    
    Returns a list of documents with title, content, and relevance score.
    """
```

### 4. Graceful Error Messages
```python
def get_user(user_id: str) -> dict:
    try:
        return database.get_user(user_id)
    except UserNotFound:
        return {"error": f"No user found with ID '{user_id}'. Verify the ID is correct."}
    except DatabaseError as e:
        return {"error": f"Database error: {e}. Try again in a moment."}
```

## Tool Design Patterns

### Pattern 1: Atomic Tools
Small, focused tools that do one thing:

```python
tools = [
    search_web,      # Search the internet
    read_webpage,    # Get content from URL
    search_docs,     # Search internal documents
    get_doc,         # Get specific document
    write_file,      # Create/update a file
    run_code,        # Execute code
]
```

### Pattern 2: Parameterized Tools
One tool with mode/action parameter:

```python
def database_tool(action: str, **kwargs):
    """
    Interact with the database.
    
    Actions:
    - "query": Run a SQL query (kwargs: sql)
    - "insert": Insert a record (kwargs: table, data)
    - "update": Update records (kwargs: table, data, where)
    """
    if action == "query":
        return db.execute(kwargs["sql"])
    elif action == "insert":
        return db.insert(kwargs["table"], kwargs["data"])
    # ...
```

### Pattern 3: Confirmation Tools
For dangerous operations:

```python
def delete_file(path: str, confirm: bool = False) -> str:
    """Delete a file. Requires confirm=True for safety."""
    if not confirm:
        return f"Are you sure you want to delete '{path}'? Call again with confirm=True"
    
    os.remove(path)
    return f"Deleted {path}"
```

## Common Mistakes

### 1. Too Many Tools
```
Problem: Agent has 50 tools → Confused about which to use
Solution: Start with 5-10 essential tools, add as needed
```

The "tool cliff" — performance drops sharply after ~15 tools.

### 2. Ambiguous Overlap
```
Problem: search_web() vs search_internet() vs google_search()
Solution: One tool per capability, clear distinctions
```

### 3. Missing Error Context
```
❌ return {"error": "Failed"}
✅ return {"error": "API rate limit exceeded. Wait 60 seconds before retrying."}
```

### 4. Implicit Dependencies
```
❌ Tool assumes global state exists
✅ Tool receives all needed context as parameters
```

## Tool Implementation Template

```python
from pydantic import BaseModel, Field
from typing import Optional

class SearchParams(BaseModel):
    """Parameters for document search."""
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, description="Maximum results to return", ge=1, le=20)
    filters: Optional[dict] = Field(None, description="Metadata filters")

class SearchResult(BaseModel):
    """A single search result."""
    title: str
    content: str
    score: float
    source: str

def search_documents(params: SearchParams) -> list[SearchResult]:
    """
    Search internal documents for relevant content.
    
    Use when:
    - User asks about company policies
    - User needs information from documentation
    - User references a specific document
    
    Do NOT use when:
    - User asks about current events (use search_web instead)
    - User asks general knowledge questions
    """
    try:
        results = vector_store.search(
            query=params.query,
            k=params.max_results,
            filters=params.filters
        )
        return [
            SearchResult(
                title=r.metadata["title"],
                content=r.content,
                score=r.score,
                source=r.metadata["source"]
            )
            for r in results
        ]
    except Exception as e:
        return [SearchResult(
            title="Error",
            content=f"Search failed: {str(e)}",
            score=0,
            source="error"
        )]
```

## Testing Tools

```python
def test_search_documents():
    # Happy path
    result = search_documents(SearchParams(query="vacation policy"))
    assert len(result) > 0
    assert all(isinstance(r, SearchResult) for r in result)
    
    # Edge cases
    result = search_documents(SearchParams(query=""))
    assert "error" in result[0].content.lower() or len(result) == 0
    
    # Error handling
    result = search_documents(SearchParams(query="x" * 10000))  # Very long query
    # Should handle gracefully, not crash
```

## Files

- `tool_design.py` - Tool implementation examples and templates

## Key Takeaways

1. Clear names, single responsibility, explicit docs
2. Graceful error messages with actionable info
3. Start with few tools, expand as needed
4. Test tools independently before giving to agents
5. Document when to use AND when NOT to use

## What's Next?

Step 4: **Memory Systems** — giving agents the ability to remember.
