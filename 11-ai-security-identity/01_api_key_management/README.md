# Step 1: API Key Management

## The Starting Point

Most AI integrations begin with API keys:

```python
import openai
client = openai.OpenAI(api_key="sk-...")
```

Simple, but fraught with security risks.

## The Problems with API Keys

```
1. Shared secret — anyone with the key can use it
2. No identity — key doesn't say WHO is using it
3. No scope — key grants full access
4. No expiration — keys often live forever
5. Hard to audit — which call came from where?
```

## Best Practices

### Never Hardcode Keys

```python
# BAD
api_key = "sk-abc123..."

# GOOD
import os
api_key = os.environ.get("OPENAI_API_KEY")

# BETTER
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]
```

### Use Secret Managers

```python
# AWS Secrets Manager
import boto3

def get_api_key():
    client = boto3.client('secretsmanager')
    response = client.get_secret_value(SecretId='openai-api-key')
    return response['SecretString']

# HashiCorp Vault
import hvac

client = hvac.Client(url='https://vault.example.com')
secret = client.secrets.kv.read_secret_version(path='ai/openai')
api_key = secret['data']['data']['api_key']
```

### Rotate Keys Regularly

```python
# Key rotation pattern
class APIKeyManager:
    def __init__(self):
        self.primary_key = get_secret("openai_key_primary")
        self.secondary_key = get_secret("openai_key_secondary")
    
    def get_key(self):
        # Use primary, fallback to secondary during rotation
        return self.primary_key or self.secondary_key
    
    def rotate(self):
        # 1. Generate new key in provider
        # 2. Store as secondary
        # 3. Promote secondary to primary
        # 4. Revoke old primary
        pass
```

### Rate Limiting

```python
from ratelimit import limits, sleep_and_retry

@sleep_and_retry
@limits(calls=100, period=60)  # 100 calls per minute
def call_llm(prompt):
    return client.chat.completions.create(...)
```

### Monitoring

```python
import logging

logger = logging.getLogger("api_usage")

def call_llm(prompt, user_id):
    logger.info(
        "llm_call",
        extra={
            "user_id": user_id,
            "prompt_length": len(prompt),
            "timestamp": datetime.now().isoformat()
        }
    )
    response = client.chat.completions.create(...)
    logger.info(
        "llm_response",
        extra={
            "user_id": user_id,
            "tokens_used": response.usage.total_tokens
        }
    )
    return response
```

## Environment Segregation

```
Development:  OPENAI_API_KEY_DEV   (low limits, test data)
Staging:      OPENAI_API_KEY_STG   (higher limits)
Production:   OPENAI_API_KEY_PROD  (full limits, monitored)
```

## When API Keys Aren't Enough

API keys fail when:
- Multiple services need different permissions
- You need to track usage per user/agent
- Agents need to work across organizations
- You need fine-grained, per-action authorization

This is where agent identity protocols come in.

## Files

- `api_key_manager.py` - Secure key management patterns
- `rate_limiter.py` - Rate limiting implementation

## Key Takeaways

1. Never hardcode API keys
2. Use secret managers in production
3. Rotate keys regularly
4. Monitor and rate limit usage
5. API keys are stepping stones to proper identity

## What's Next?

Step 2: **OAuth 2.0 for AI** — structured authorization.
