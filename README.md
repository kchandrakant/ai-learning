# AI Learning Roadmap

A comprehensive, hands-on learning path for mastering AI/ML — from mathematical foundations to production LLM systems.

## 🎯 Overview

This repository contains structured courses covering the complete AI/ML journey:

```
Foundations → Deep Learning → Transformers → LLM Applications → Production
```

## 📚 Course Structure

### Foundation Track
| Course | Modules | Description |
|--------|---------|-------------|
| [ml-foundations](./ml-foundations/) | 14 | Linear algebra, calculus, probability, classical ML algorithms |
| [deep-learning-to-transformers](./deep-learning-to-transformers/) | 15 | Neural networks, CNNs, RNNs, attention mechanism |
| [transformer-architecture](./transformer-architecture/) | 8 | Deep dive into transformer implementation |

### Application Track
| Course | Modules | Description |
|--------|---------|-------------|
| [prompt-engineering-rag](./prompt-engineering-rag/) | 12 | Prompting techniques, RAG systems |
| [multi-agent-systems](./multi-agent-systems/) | 12 | Agent architectures, tool use, orchestration |
| [fine-tuning-adaptation](./fine-tuning-adaptation/) | 14 | LoRA, RLHF, domain adaptation |

### Production Track
| Course | Modules | Description |
|--------|---------|-------------|
| [mlops-model-serving](./mlops-model-serving/) | 16 | Inference optimization, deployment, monitoring |
| [harness-enginnering](./harness-enginnering/) | 8 | LLM harness design, evaluation, benchmarks |
| [agentic-aauth](./agentic-aauth/) | 8 | Agent authentication, identity, security |

## 🚀 Getting Started

1. **Start with foundations** if you're new to ML:
   ```
   ml-foundations → deep-learning-to-transformers → transformer-architecture
   ```

2. **Jump to applications** if you know the basics:
   ```
   prompt-engineering-rag → multi-agent-systems
   ```

3. **Focus on production** if building systems:
   ```
   mlops-model-serving → harness-enginnering
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
cd ml-foundations
pip install -r requirements.txt
python verify_setup.py
```

## 📋 Full Roadmap

See [AI_LEARNING_ROADMAP.md](./AI_LEARNING_ROADMAP.md) for the complete learning path and course relationships.

## 📄 License

MIT License - feel free to use for learning and teaching.
