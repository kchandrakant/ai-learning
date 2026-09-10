# Step 2: OAuth 2.0 for AI Applications

## OAuth Recap

OAuth 2.0 separates:
- **Resource Owner:** User who owns the data
- **Client:** Application wanting access
- **Authorization Server:** Issues tokens
- **Resource Server:** Holds protected resources

## Common Flows

### Authorization Code (User-facing apps)

```
User ──▶ App ──▶ Auth Server ──▶ User Login ──▶ Auth Code ──▶ App ──▶ Tokens
```

```python
# 1. Redirect user to auth
auth_url = f"{AUTH_SERVER}/authorize?client_id={CLIENT_ID}&response_type=code&redirect_uri={REDIRECT}"

# 2. Exchange code for tokens
response = requests.post(f"{AUTH_SERVER}/token", data={
    "grant_type": "authorization_code",
    "code": auth_code,
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "redirect_uri": REDIRECT
})
tokens = response.json()
```

### Client Credentials (Service-to-service)

```
Service ──▶ Auth Server ──▶ Access Token
```

```python
# Service authenticates directly
response = requests.post(f"{AUTH_SERVER}/token", data={
    "grant_type": "client_credentials",
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "scope": "read:data"
})
access_token = response.json()["access_token"]
```

### Device Flow (CLI/IoT)

```
Device ──▶ Auth Server ──▶ User Code
User ──▶ Browser ──▶ Enter Code ──▶ Approve
Device ──▶ Poll ──▶ Tokens
```

Good for CLI tools and agents without browsers.

## OAuth for LLM Applications

### User Authorization for AI Actions

```python
# User authorizes app to access their Google Calendar
# App then uses AI to manage calendar

class CalendarAgent:
    def __init__(self, oauth_token):
        self.token = oauth_token
    
    async def schedule_meeting(self, request: str):
        # AI parses request
        meeting_details = await self.llm.parse(request)
        
        # Use OAuth token to create event
        response = requests.post(
            "https://calendar.google.com/api/events",
            headers={"Authorization": f"Bearer {self.token}"},
            json=meeting_details
        )
        return response.json()
```

### Scopes for AI Features

```python
# Define scopes for AI capabilities
SCOPES = {
    "ai:read": "AI can read your data",
    "ai:write": "AI can modify your data",
    "ai:send": "AI can send messages on your behalf"
}

# Request only needed scopes
auth_url = f"{AUTH_SERVER}/authorize?scope=ai:read+ai:write"
```

## Token Management

### Refresh Tokens

```python
class TokenManager:
    def __init__(self, access_token, refresh_token, expires_at):
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.expires_at = expires_at
    
    def get_token(self):
        if datetime.now() >= self.expires_at - timedelta(minutes=5):
            self.refresh()
        return self.access_token
    
    def refresh(self):
        response = requests.post(f"{AUTH_SERVER}/token", data={
            "grant_type": "refresh_token",
            "refresh_token": self.refresh_token,
            "client_id": CLIENT_ID
        })
        data = response.json()
        self.access_token = data["access_token"]
        self.expires_at = datetime.now() + timedelta(seconds=data["expires_in"])
```

### Token Storage

```python
# Encrypt tokens at rest
from cryptography.fernet import Fernet

class SecureTokenStore:
    def __init__(self, encryption_key):
        self.fernet = Fernet(encryption_key)
    
    def store(self, user_id, tokens):
        encrypted = self.fernet.encrypt(json.dumps(tokens).encode())
        db.save(user_id, encrypted)
    
    def retrieve(self, user_id):
        encrypted = db.get(user_id)
        decrypted = self.fernet.decrypt(encrypted)
        return json.loads(decrypted)
```

## OAuth Works Well For

- User-authorized access to their data
- Well-defined, static scopes
- Pre-registered applications
- Synchronous consent flows

## OAuth Struggles With

- Dynamic, per-action authorization
- Agent-to-agent communication
- Cross-organization identity
- Long-running autonomous tasks

## Files

- `oauth_flows.py` - OAuth flow implementations
- `token_manager.py` - Token refresh and storage

## Key Takeaways

1. OAuth separates concerns (user, client, server)
2. Choose flow based on use case
3. Always use refresh tokens
4. Encrypt tokens at rest
5. OAuth has limits for agent scenarios

## What's Next?

Step 3: **OAuth Limitations** — where OAuth fails for agents.
