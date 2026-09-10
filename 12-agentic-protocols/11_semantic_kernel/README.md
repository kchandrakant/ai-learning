# Step 11: Semantic Kernel & Enterprise Patterns

## What is Semantic Kernel?

Microsoft's SDK for building AI applications with enterprise patterns.

## Core Concepts

### Kernel

Central orchestrator:

```python
import semantic_kernel as sk

kernel = sk.Kernel()

# Add AI service
kernel.add_service(
    sk.connectors.OpenAIChatCompletion(
        service_id="chat",
        api_key=os.getenv("OPENAI_API_KEY")
    )
)
```

### Plugins

Collections of functions:

```python
from semantic_kernel.functions import kernel_function

class WeatherPlugin:
    @kernel_function(
        name="get_weather",
        description="Get weather for a location"
    )
    def get_weather(self, location: str) -> str:
        return fetch_weather(location)
    
    @kernel_function(
        name="get_forecast",
        description="Get 5-day forecast"
    )
    def get_forecast(self, location: str) -> str:
        return fetch_forecast(location)

kernel.add_plugin(WeatherPlugin(), "weather")
```

### Prompt Functions

```python
# Inline prompt
summarize = kernel.add_function(
    function_name="summarize",
    plugin_name="text",
    prompt="""
    Summarize the following text in 3 bullet points:
    
    {{$input}}
    
    Summary:
    """,
    description="Summarize text into bullet points"
)

# From file
kernel.add_plugin(parent_directory="./plugins", plugin_name="writer")
# Loads from ./plugins/writer/write_article/skprompt.txt
```

### Function Calling

```python
from semantic_kernel.functions import KernelArguments

# Manual invocation
result = await kernel.invoke(
    kernel.get_function("weather", "get_weather"),
    KernelArguments(location="Seattle")
)

# Auto function calling
settings = sk.PromptExecutionSettings(
    function_choice_behavior="auto"
)

result = await kernel.invoke_prompt(
    "What's the weather in Seattle?",
    settings=settings
)
```

## Planners

Automatic task decomposition:

```python
from semantic_kernel.planners import FunctionCallingStepwisePlanner

planner = FunctionCallingStepwisePlanner(
    service_id="chat",
    max_iterations=10
)

result = await planner.invoke(
    kernel,
    "Research AI news and write a summary report"
)
```

## Memory Connectors

```python
from semantic_kernel.connectors.memory import AzureAISearchMemoryStore

# Connect to vector store
memory = AzureAISearchMemoryStore(
    endpoint="https://search.azure.com",
    api_key=os.getenv("SEARCH_API_KEY")
)

# Save memory
await memory.create_collection("documents")
await memory.upsert(
    collection="documents",
    record=MemoryRecord(
        id="doc1",
        text="Important information...",
        embedding=embeddings
    )
)

# Recall
results = await memory.search(
    collection="documents",
    query="What is important?",
    limit=5
)
```

## Enterprise Patterns

### Dependency Injection

```python
from semantic_kernel import Kernel
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    config = providers.Configuration()
    
    kernel = providers.Singleton(
        Kernel,
        services=[
            providers.Singleton(OpenAIChatCompletion)
        ]
    )
    
    agent_service = providers.Factory(
        AgentService,
        kernel=kernel
    )
```

### Logging and Telemetry

```python
import logging
from opentelemetry import trace

logging.basicConfig(level=logging.INFO)
tracer = trace.get_tracer(__name__)

# Semantic Kernel integrates with OpenTelemetry
kernel = sk.Kernel()
# Logs and traces are automatically captured
```

### Filters (Middleware)

```python
from semantic_kernel.filters import FunctionInvocationFilter

class LoggingFilter(FunctionInvocationFilter):
    async def on_function_invocation(
        self,
        context: FunctionInvocationContext,
        next: Callable
    ):
        logging.info(f"Calling: {context.function.name}")
        result = await next(context)
        logging.info(f"Result: {result}")
        return result

kernel.add_filter(LoggingFilter())
```

### Multi-Tenant

```python
class TenantKernelFactory:
    def __init__(self, config: dict):
        self.config = config
    
    def create_kernel(self, tenant_id: str) -> Kernel:
        tenant_config = self.config[tenant_id]
        
        kernel = Kernel()
        kernel.add_service(
            OpenAIChatCompletion(
                api_key=tenant_config["api_key"]
            )
        )
        
        # Add tenant-specific plugins
        for plugin in tenant_config["plugins"]:
            kernel.add_plugin(plugin)
        
        return kernel
```

## Files

- `semantic_kernel_basics.py` - SK fundamentals
- `plugins/` - Example plugins
- `enterprise_patterns.py` - Enterprise integration

## Key Takeaways

1. Kernel orchestrates AI services and plugins
2. Plugins group related functions
3. Planners decompose complex tasks
4. Memory connectors for RAG patterns
5. Enterprise-ready with DI, logging, filters

## What's Next?

Step 12: **Protocol Comparison** — choosing the right protocol.
