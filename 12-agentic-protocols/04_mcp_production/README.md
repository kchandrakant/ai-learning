# Step 4: MCP in Production

## Production Considerations

Local development is easy. Production requires:
- Authentication
- Rate limiting
- Logging & monitoring
- Error handling
- Scalability

## HTTP Transport for Remote Servers

```python
from fastmcp import FastMCP
from fastmcp.server import HTTPTransport

mcp = FastMCP("Production Server")

# ... define tools ...

if __name__ == "__main__":
    transport = HTTPTransport(host="0.0.0.0", port=8000)
    mcp.run(transport=transport)
```

## Authentication

### API Key Authentication

```python
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

@mcp.tool()
async def secure_tool(data: str, api_key: str = Depends(verify_api_key)):
    """Tool that requires authentication."""
    ...
```

### OAuth Integration

```python
from authlib.integrations.starlette_client import OAuth

oauth = OAuth()
oauth.register(
    name='provider',
    client_id='...',
    client_secret='...',
)

# Verify tokens on each request
```

## Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@mcp.tool()
@limiter.limit("10/minute")
async def rate_limited_tool(query: str):
    """Tool with rate limiting."""
    ...
```

## Logging & Observability

```python
import structlog
from opentelemetry import trace

logger = structlog.get_logger()
tracer = trace.get_tracer(__name__)

@mcp.tool()
async def observed_tool(query: str):
    """Tool with full observability."""
    with tracer.start_as_current_span("tool_execution") as span:
        span.set_attribute("query", query)
        
        logger.info("tool_called", tool="observed_tool", query=query)
        
        try:
            result = await execute_query(query)
            logger.info("tool_success", result_size=len(result))
            return result
        except Exception as e:
            logger.error("tool_error", error=str(e))
            span.record_exception(e)
            raise
```

## Error Handling

```python
from mcp.types import McpError, ErrorCode

@mcp.tool()
async def robust_tool(params: dict):
    """Tool with proper error handling."""
    try:
        # Validate input
        if not params.get("required_field"):
            raise McpError(
                ErrorCode.InvalidParams,
                "required_field is missing"
            )
        
        # Execute
        result = await process(params)
        return json.dumps(result)
        
    except ValidationError as e:
        raise McpError(ErrorCode.InvalidParams, str(e))
    except PermissionError:
        raise McpError(ErrorCode.PermissionDenied, "Access denied")
    except Exception as e:
        logger.exception("Unexpected error")
        raise McpError(ErrorCode.InternalError, "Internal server error")
```

## Deployment Patterns

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "server.py"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    spec:
      containers:
      - name: mcp-server
        image: my-mcp-server:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            memory: "512Mi"
            cpu: "500m"
```

## Multi-Server Architecture

```
┌─────────────────────────────────────────┐
│              Load Balancer              │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    ▼             ▼             ▼
┌───────┐   ┌───────┐   ┌───────┐
│Server │   │Server │   │Server │
│  (1)  │   │  (2)  │   │  (3)  │
└───────┘   └───────┘   └───────┘
    │             │             │
    └─────────────┼─────────────┘
                  ▼
          ┌─────────────┐
          │   Database  │
          └─────────────┘
```

## Health Checks

```python
@app.get("/health")
async def health_check():
    checks = {
        "database": await check_db_connection(),
        "external_api": await check_external_api(),
    }
    
    healthy = all(checks.values())
    return {
        "status": "healthy" if healthy else "unhealthy",
        "checks": checks
    }
```

## Files

- `production_server.py` - Production-ready server
- `docker-compose.yml` - Local deployment
- `kubernetes.yaml` - K8s deployment

## Key Takeaways

1. HTTP transport for remote servers
2. Always authenticate production endpoints
3. Rate limit to prevent abuse
4. Log and trace everything
5. Handle errors with proper MCP error codes

## What's Next?

Step 5: **Agent Communication Problem** — why agents need to talk to agents.
