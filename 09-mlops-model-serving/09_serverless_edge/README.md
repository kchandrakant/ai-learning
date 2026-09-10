# Step 9: Serverless & Edge

## Serverless Options

### AWS Lambda + Bedrock

```python
import boto3

bedrock = boto3.client('bedrock-runtime')

def handler(event, context):
    response = bedrock.invoke_model(
        modelId='anthropic.claude-3-sonnet',
        body=json.dumps({
            "messages": event["messages"],
            "max_tokens": 1000
        })
    )
    return json.loads(response['body'].read())
```

### Modal

```python
import modal

app = modal.App("llm-api")

@app.function(gpu="A10G")
def generate(prompt: str) -> str:
    from vllm import LLM
    llm = LLM(model="meta-llama/Llama-2-7b-hf")
    return llm.generate([prompt])[0].outputs[0].text
```

### Replicate

```python
import replicate

output = replicate.run(
    "meta/llama-2-7b-chat",
    input={"prompt": "Hello, how are you?"}
)
```

## Edge Deployment

### Ollama (Local)

```bash
# Install and run
ollama run llama2

# API
curl http://localhost:11434/api/generate -d '{
  "model": "llama2",
  "prompt": "Hello"
}'
```

### Cloudflare Workers AI

```javascript
export default {
  async fetch(request, env) {
    const response = await env.AI.run('@cf/meta/llama-2-7b-chat-int8', {
      messages: [{ role: 'user', content: 'Hello' }]
    });
    return Response.json(response);
  }
}
```

## When to Use What

| Option | Latency | Cost Model | Best For |
|--------|---------|------------|----------|
| Lambda+Bedrock | Medium | Per-token | Variable load |
| Modal | Low | Per-second | Batch jobs |
| Ollama | Lowest | Fixed | Local dev |
| Workers AI | Low | Per-request | Edge apps |

## Files

- `serverless_edge.py` - Deployment examples
