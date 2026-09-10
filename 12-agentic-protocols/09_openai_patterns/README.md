# Step 9: OpenAI Patterns

## De Facto Standards

OpenAI's patterns often become industry standards due to their massive adoption.

## Function Calling Evolution

### Original (2023)

```python
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather?"}],
    functions=[{
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }]
)
```

### Modern (2024+)

```python
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "What's the weather?"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }],
    tool_choice="auto"
)
```

## Structured Outputs

Force JSON schema compliance:

```python
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "Extract person info"}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "person",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                },
                "required": ["name", "age"]
            }
        }
    }
)
```

## Assistants API

Stateful, tool-using assistants:

```python
# Create assistant
assistant = client.beta.assistants.create(
    name="Data Analyst",
    instructions="You analyze data and create visualizations",
    tools=[
        {"type": "code_interpreter"},
        {"type": "file_search"}
    ],
    model="gpt-4-turbo"
)

# Create thread
thread = client.beta.threads.create()

# Add message
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="Analyze this data"
)

# Run assistant
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

# Wait for completion
while run.status != "completed":
    run = client.beta.threads.runs.retrieve(
        thread_id=thread.id,
        run_id=run.id
    )
    time.sleep(1)

# Get messages
messages = client.beta.threads.messages.list(thread_id=thread.id)
```

## Assistants with Custom Tools

```python
assistant = client.beta.assistants.create(
    name="Research Assistant",
    tools=[
        {"type": "code_interpreter"},
        {
            "type": "function",
            "function": {
                "name": "search_papers",
                "description": "Search academic papers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    }
                }
            }
        }
    ],
    model="gpt-4-turbo"
)

# Handle tool calls during run
while run.status == "requires_action":
    tool_calls = run.required_action.submit_tool_outputs.tool_calls
    
    outputs = []
    for tool_call in tool_calls:
        if tool_call.function.name == "search_papers":
            args = json.loads(tool_call.function.arguments)
            result = search_papers(args["query"])
            outputs.append({
                "tool_call_id": tool_call.id,
                "output": json.dumps(result)
            })
    
    run = client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread.id,
        run_id=run.id,
        tool_outputs=outputs
    )
```

## Realtime API (Voice Agents)

```python
import asyncio
import websockets

async def voice_agent():
    async with websockets.connect(
        "wss://api.openai.com/v1/realtime",
        extra_headers={"Authorization": f"Bearer {api_key}"}
    ) as ws:
        # Configure session
        await ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": "alloy",
                "tools": [...]
            }
        }))
        
        # Send audio
        await ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }))
        
        # Receive responses
        async for message in ws:
            data = json.loads(message)
            if data["type"] == "response.audio.delta":
                # Play audio chunk
                pass
```

## Parallel Tool Calls

```python
# Model can call multiple tools in one response
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "Get weather for NYC and SF"}],
    tools=[weather_tool],
    parallel_tool_calls=True
)

# Response may contain multiple tool calls
for tool_call in response.choices[0].message.tool_calls:
    # Execute each in parallel
    pass
```

## Key Patterns

### Tool Validation

```python
def execute_tool(tool_call):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    
    # Validate against schema
    validate(args, tool_schemas[name])
    
    # Execute
    return tools[name](**args)
```

### Error Handling

```python
try:
    result = execute_tool(tool_call)
    output = {"tool_call_id": tool_call.id, "output": json.dumps(result)}
except Exception as e:
    output = {
        "tool_call_id": tool_call.id,
        "output": json.dumps({"error": str(e)})
    }
```

## Files

- `function_calling.py` - Modern function calling
- `assistants.py` - Assistants API usage
- `realtime.py` - Voice agent example

## Key Takeaways

1. OpenAI patterns become industry standards
2. Tools API supersedes functions API
3. Structured outputs ensure valid JSON
4. Assistants API manages state and files
5. Realtime API enables voice agents

## What's Next?

Step 10: **Framework Protocols** — LangChain, CrewAI, and more.
