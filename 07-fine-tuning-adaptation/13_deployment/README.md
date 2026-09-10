# Step 13: Deployment

## Deployment Options

### Option 1: Merge and Deploy

```python
from peft import PeftModel

# Load adapter
model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# Merge into base
merged_model = model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged-model")
```

### Option 2: Serve with Adapters

```python
from peft import PeftModel

# Load base once
base_model = AutoModelForCausalLM.from_pretrained("base-model")

# Load different adapters on demand
adapter_a = PeftModel.from_pretrained(base_model, "./adapter-a")
adapter_b = PeftModel.from_pretrained(base_model, "./adapter-b")
```

## Quantization for Deployment

```python
from transformers import AutoModelForCausalLM

# GPTQ quantization
model = AutoModelForCausalLM.from_pretrained(
    "./merged-model",
    device_map="auto",
    load_in_4bit=True,
)

# Or use AutoGPTQ
from auto_gptq import AutoGPTQForCausalLM

quantized = AutoGPTQForCausalLM.from_pretrained(
    "./merged-model",
    quantize_config=quantize_config
)
```

## Serving with vLLM

```python
from vllm import LLM

# Serve merged model
llm = LLM(model="./merged-model")

# Or with LoRA adapters (vLLM supports this)
llm = LLM(
    model="base-model",
    enable_lora=True,
    max_loras=4,
)
```

## Version Management

```
models/
├── base-llama-2-7b/
├── adapters/
│   ├── customer-support-v1/
│   ├── customer-support-v2/
│   └── code-assistant-v1/
└── merged/
    ├── customer-support-v2-merged/
    └── code-assistant-v1-merged/
```

## Files

- `deployment.py` - Deployment utilities

## Key Takeaways

1. Merge for simplicity, keep separate for flexibility
2. Quantize for production efficiency
3. vLLM supports multi-LoRA serving
4. Version your adapters and models
5. Test thoroughly before production
