# AI Learning Roadmap

A comprehensive, hands-on learning path for mastering AI/ML — from mathematical foundations to production LLM systems.

## 🎯 Overview

This repository contains structured courses covering the complete AI/ML journey:

```
Foundations → Deep Learning → Transformers → LLM Applications → Production
```

## 📚 Course Structure

### Foundation Track
| # | Course | Modules | Description |
|---|--------|---------|-------------|
| 01 | [ml-foundations](./01-ml-foundations/) | 14 | Linear algebra, calculus, probability, classical ML algorithms |
| 02 | [deep-learning-to-transformers](./02-deep-learning-to-transformers/) | 15 | Neural networks, CNNs, RNNs, attention mechanism |
| 03 | [transformer-architecture](./03-transformer-architecture/) | 8 | Deep dive into transformer implementation |
| 04 | [llm-internals](./04-llm-internals/) | 12 | Tokenization, pre-training, scaling laws, RLHF |

### Application Track
| # | Course | Modules | Description |
|---|--------|---------|-------------|
| 05 | [prompt-engineering-rag](./05-prompt-engineering-rag/) | 12 | Prompting techniques, RAG systems |
| 06 | [multi-agent-systems](./06-multi-agent-systems/) | 12 | Agent architectures, tool use, orchestration |
| 07 | [fine-tuning-adaptation](./07-fine-tuning-adaptation/) | 14 | LoRA, RLHF, domain adaptation |
| 08 | [multimodal-ai](./08-multimodal-ai/) | 12 | Vision-language, diffusion, audio, video |

### Production Track
| # | Course | Modules | Description |
|---|--------|---------|-------------|
| 09 | [mlops-model-serving](./09-mlops-model-serving/) | 16 | Inference optimization, deployment, monitoring |
| 10 | [agent-system-design](./10-agent-system-design/) | 12 | Agent system architecture, evaluation, benchmarks |
| 11 | [ai-security-identity](./11-ai-security-identity/) | 16 | Agent identity, authentication, access control |
| 12 | [agentic-protocols](./12-agentic-protocols/) | 15 | MCP, A2A, ACP, and emerging agent protocols |

## 🚀 Getting Started

1. **Start with foundations** if you're new to ML:
   ```
   01-ml-foundations → 02-deep-learning-to-transformers → 03-transformer-architecture → 04-llm-internals
   ```

2. **Jump to applications** if you know the basics:
   ```
   05-prompt-engineering-rag → 06-multi-agent-systems → 08-multimodal-ai
   ```

3. **Focus on production** if building systems:
   ```
   09-mlops-model-serving → 10-agent-system-design → 12-agentic-protocols
   ```

## 📖 Each Course Contains

- `LEARNING_PATH.md` — Structured learning guide
- `requirements.txt` — Python dependencies
- `verify_setup.py` — Environment verification
- Numbered module folders with `README.md`
- `demo/` — Interactive demonstrations
- `evolutions/` — Future topics and advanced concepts

## 🛠️ Setup

```bash
# Clone the repository
git clone https://github.com/kchandrakant/ai-learning.git
cd ai-learning

# Choose a course and install dependencies
cd 01-ml-foundations
pip install -r requirements.txt
python verify_setup.py
```

## 📋 Full Roadmap

See [AI_LEARNING_ROADMAP.md](./AI_LEARNING_ROADMAP.md) for the complete learning path and course relationships.

## 📄 License

MIT License - feel free to use for learning and teaching.
