# Step 14: Secure Agent Deployment

## Deployment Security Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Security                      │
│  (Code, dependencies, secrets handling)                     │
├─────────────────────────────────────────────────────────────┤
│                    Container Security                        │
│  (Image scanning, runtime protection)                       │
├─────────────────────────────────────────────────────────────┤
│                    Network Security                          │
│  (Isolation, encryption, firewalls)                         │
├─────────────────────────────────────────────────────────────┤
│                    Infrastructure Security                   │
│  (Cloud IAM, secrets management)                            │
└─────────────────────────────────────────────────────────────┘
```

## Secret Injection

### Environment Variables (Basic)

```yaml
# kubernetes deployment
env:
  - name: OPENAI_API_KEY
    valueFrom:
      secretKeyRef:
        name: llm-secrets
        key: openai-key
```

### Secret Managers (Production)

```python
# AWS Secrets Manager
import boto3

def get_secrets():
    client = boto3.client('secretsmanager')
    secret = client.get_secret_value(SecretId='agent/llm-keys')
    return json.loads(secret['SecretString'])

# HashiCorp Vault with auto-renewal
from hvac import Client

class VaultSecrets:
    def __init__(self, vault_addr, role):
        self.client = Client(url=vault_addr)
        self.client.auth.kubernetes.login(role=role)
    
    def get_secret(self, path):
        return self.client.secrets.kv.read_secret_version(path=path)
```

### Sealed Secrets (GitOps)

```yaml
# Encrypted secret for Git
apiVersion: bitnami.com/v1alpha1
kind: SealedSecret
metadata:
  name: llm-secrets
spec:
  encryptedData:
    openai-key: AgB2k3j4...encrypted...
```

## Agent Sandboxing

### Container Isolation

```dockerfile
# Minimal base image
FROM python:3.11-slim

# Non-root user
RUN useradd -r -s /bin/false agent
USER agent

# Read-only filesystem
# --read-only flag at runtime

# No network by default
# --network none at runtime
```

```yaml
# Kubernetes security context
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
      - ALL
```

### Code Execution Sandboxing

```python
# For agents that execute code
import docker

def run_code_sandboxed(code: str, timeout: int = 30):
    client = docker.from_env()
    
    container = client.containers.run(
        "python:3.11-slim",
        command=["python", "-c", code],
        mem_limit="256m",
        cpu_period=100000,
        cpu_quota=50000,  # 50% CPU
        network_disabled=True,
        read_only=True,
        remove=True,
        detach=True
    )
    
    try:
        result = container.wait(timeout=timeout)
        logs = container.logs()
        return logs.decode()
    except:
        container.kill()
        raise TimeoutError("Code execution timed out")
```

### gVisor/Kata Containers (Deep isolation)

```yaml
# Use gVisor runtime
runtimeClassName: gvisor
```

## Network Security

### Network Policies

```yaml
# Only allow egress to LLM APIs
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: agent-network-policy
spec:
  podSelector:
    matchLabels:
      app: agent
  policyTypes:
    - Egress
  egress:
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - port: 443
          protocol: TCP
    # Allow DNS
    - to:
        - namespaceSelector: {}
          podSelector:
            matchLabels:
              k8s-app: kube-dns
      ports:
        - port: 53
          protocol: UDP
```

### mTLS (Mutual TLS)

```python
# Agent-to-service mTLS
import ssl
import httpx

ssl_context = ssl.create_default_context()
ssl_context.load_cert_chain(
    certfile="/certs/agent.crt",
    keyfile="/certs/agent.key"
)
ssl_context.load_verify_locations("/certs/ca.crt")

client = httpx.AsyncClient(verify=ssl_context)
```

## Supply Chain Security

### Dependency Scanning

```yaml
# GitHub Actions
- name: Run Snyk to check for vulnerabilities
  uses: snyk/actions/python@master
  with:
    command: test
```

### Image Signing

```bash
# Sign image with cosign
cosign sign --key cosign.key myregistry/agent:v1.0

# Verify in cluster
kubectl apply -f - <<EOF
apiVersion: policy.sigstore.dev/v1alpha1
kind: ClusterImagePolicy
metadata:
  name: require-signatures
spec:
  images:
    - glob: "myregistry/**"
  authorities:
    - key:
        data: |
          -----BEGIN PUBLIC KEY-----
          ...
          -----END PUBLIC KEY-----
EOF
```

### SBOM (Software Bill of Materials)

```bash
# Generate SBOM
syft packages dir:. -o spdx-json > sbom.json

# Attach to image
cosign attach sbom --sbom sbom.json myregistry/agent:v1.0
```

## Files

- `secure_dockerfile` - Hardened Dockerfile
- `network_policy.yaml` - Kubernetes network policy
- `secrets_manager.py` - Secret management patterns

## Key Takeaways

1. Inject secrets, never bake them in
2. Run agents with minimal privileges
3. Sandbox code execution
4. Restrict network access
5. Verify supply chain integrity

## What's Next?

Step 15: **Monitoring & Audit** — visibility into agent actions.
