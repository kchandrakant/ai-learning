# Step 3: Building MCP Servers

## FastMCP: The Easy Way

FastMCP is a high-level framework for building MCP servers quickly.

```python
from fastmcp import FastMCP

mcp = FastMCP("My Tools")

@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return str(eval(expression))  # Use safer eval in production!

@mcp.resource("config://app")
def get_config() -> str:
    """Return application configuration."""
    return json.dumps({"version": "1.0", "env": "prod"})

if __name__ == "__main__":
    mcp.run()
```

## Tool Definition Best Practices

### Clear Names and Descriptions

```python
@mcp.tool()
def search_documents(
    query: str,
    max_results: int = 10,
    include_metadata: bool = False
) -> str:
    """
    Search the document database.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        include_metadata: Include document metadata in results
    
    Returns:
        JSON array of matching documents
    """
    ...
```

### Input Validation

```python
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    max_results: int = Field(default=10, ge=1, le=100)

@mcp.tool()
def search(params: SearchParams) -> str:
    """Search with validated parameters."""
    ...
```

### Error Handling

```python
@mcp.tool()
def fetch_user(user_id: str) -> str:
    """Fetch user by ID."""
    try:
        user = db.get_user(user_id)
        if not user:
            return json.dumps({"error": "User not found", "user_id": user_id})
        return json.dumps(user)
    except DatabaseError as e:
        return json.dumps({"error": "Database error", "details": str(e)})
```

## Resource Patterns

### Static Resources

```python
@mcp.resource("schema://database")
def database_schema() -> str:
    """Return database schema for context."""
    return read_file("schema.sql")
```

### Dynamic Resources with Parameters

```python
@mcp.resource("users://{user_id}/profile")
def user_profile(user_id: str) -> str:
    """Fetch user profile."""
    return json.dumps(fetch_profile(user_id))
```

### List Available Resources

```python
@mcp.list_resources()
def list_resources():
    return [
        Resource(uri="schema://database", name="Database Schema"),
        Resource(uri="config://app", name="App Config"),
    ]
```

## Prompt Templates

```python
@mcp.prompt()
def sql_query_prompt(table_name: str, question: str) -> list:
    """Generate SQL query from natural language."""
    schema = get_table_schema(table_name)
    return [
        {"role": "system", "content": f"You are a SQL expert. Schema:\n{schema}"},
        {"role": "user", "content": f"Write SQL for: {question}"}
    ]
```

## Async Operations

```python
import asyncio
import httpx

@mcp.tool()
async def fetch_url(url: str) -> str:
    """Fetch content from URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.text[:5000]  # Limit response size
```

## Complete Server Example

```python
from fastmcp import FastMCP
import json
import sqlite3

mcp = FastMCP("Database Tools")

# Database connection
conn = sqlite3.connect("app.db")

@mcp.resource("schema://tables")
def get_schema() -> str:
    """Return database schema."""
    cursor = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table'"
    )
    return "\n\n".join(row[0] for row in cursor if row[0])

@mcp.tool()
def query_database(sql: str) -> str:
    """
    Execute a SELECT query on the database.
    Only SELECT queries are allowed for safety.
    """
    sql = sql.strip()
    if not sql.upper().startswith("SELECT"):
        return json.dumps({"error": "Only SELECT queries allowed"})
    
    try:
        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return json.dumps(rows, indent=2)
    except sqlite3.Error as e:
        return json.dumps({"error": str(e)})

@mcp.tool()
def list_tables() -> str:
    """List all tables in the database."""
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    tables = [row[0] for row in cursor]
    return json.dumps(tables)

if __name__ == "__main__":
    mcp.run()
```

## Testing Your Server

```python
# test_server.py
import pytest
from my_server import mcp

@pytest.mark.asyncio
async def test_list_tables():
    result = await mcp.call_tool("list_tables", {})
    tables = json.loads(result)
    assert isinstance(tables, list)

@pytest.mark.asyncio
async def test_query():
    result = await mcp.call_tool("query_database", {
        "sql": "SELECT * FROM users LIMIT 1"
    })
    assert "error" not in result.lower()
```

## Files

- `fastmcp_server.py` - FastMCP example
- `database_server.py` - Database tools server
- `test_server.py` - Server tests

## Key Takeaways

1. FastMCP simplifies server development
2. Use clear names, descriptions, and validation
3. Handle errors gracefully with informative messages
4. Resources for read-only data, Tools for actions
5. Test your servers!

## What's Next?

Step 4: **MCP in Production** — deploy reliable MCP servers.
