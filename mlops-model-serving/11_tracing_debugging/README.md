# Step 11: Tracing & Debugging

## Why Tracing?

Multi-step LLM applications need end-to-end visibility.

## OpenTelemetry Setup

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Setup
provider = TracerProvider()
processor = BatchSpanProcessor(OTLPSpanExporter())
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)
```

## Tracing LLM Calls

```python
@tracer.start_as_current_span("llm_generate")
def generate(messages: list) -> str:
    span = trace.get_current_span()
    span.set_attribute("model", "llama-2")
    span.set_attribute("input_tokens", count_tokens(messages))
    
    result = model.generate(messages)
    
    span.set_attribute("output_tokens", count_tokens(result))
    span.set_attribute("finish_reason", "stop")
    
    return result
```

## FastAPI Integration

```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)
```

## LangSmith Integration

```python
from langsmith import Client
from langchain.callbacks import LangChainTracer

client = Client()
tracer = LangChainTracer(project_name="my-project")

# Use with chain
result = chain.invoke(input, config={"callbacks": [tracer]})
```

## Debugging Tips

1. **Log full request/response** for failed requests
2. **Capture prompts** with sensitive data redacted
3. **Track token counts** at each step
4. **Store traces** for replay

## Files

- `tracing_debugging.py` - Tracing setup and examples
