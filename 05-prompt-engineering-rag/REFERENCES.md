# Prompt Engineering & RAG: Key References

A curated collection of papers on prompting techniques and retrieval-augmented generation.

---

## 📚 How to Use This Document

- **📖 Essential**: Core papers everyone should read
- **🔧 Hands-on**: Papers with practical implementations
- **🔬 Frontier**: Cutting-edge research

---

## Prompting Foundations

### Language Models are Few-Shot Learners (GPT-3)
**Brown et al., NeurIPS 2020**

Demonstrated in-context learning and few-shot prompting.

- **Key innovations**: Zero-shot, one-shot, few-shot prompting
- **Impact**: Established prompting as a paradigm
- **Link**: [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- **Status**: 📖 Essential

---

### Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
**Wei et al., NeurIPS 2022**

Showed that prompting LLMs to reason step-by-step improves performance.

- **Key innovations**: Chain-of-thought (CoT) prompting
- **Impact**: Foundation for reasoning-based prompting
- **Link**: [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- **Status**: 📖 Essential

---

### Self-Consistency Improves Chain of Thought Reasoning
**Wang et al., ICLR 2023**

Sample multiple reasoning paths and take majority vote.

- **Key innovations**: Self-consistency decoding
- **Impact**: Improved CoT reliability
- **Link**: [arXiv:2203.11171](https://arxiv.org/abs/2203.11171)
- **Status**: 🔧 Hands-on

---

### Tree of Thoughts: Deliberate Problem Solving with Large Language Models
**Yao et al., NeurIPS 2023**

Extends CoT with exploration and backtracking.

- **Key innovations**: Tree-structured reasoning, search algorithms
- **Impact**: Better for complex problem-solving
- **Link**: [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
- **Status**: 🔧 Hands-on

---

### ReAct: Synergizing Reasoning and Acting in Language Models
**Yao et al., ICLR 2023**

Interleaves reasoning traces with actions.

- **Key innovations**: Thought-action-observation loops
- **Impact**: Foundation for tool-using agents
- **Link**: [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)
- **Status**: 🔧 Hands-on

---

## Retrieval-Augmented Generation

### Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
**Lewis et al., NeurIPS 2020**

The original RAG paper combining retrieval with generation.

- **Key innovations**: End-to-end retrieval + generation, parametric + non-parametric memory
- **Impact**: Foundation for all RAG systems
- **Link**: [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- **Status**: 📖 Essential

---

### Dense Passage Retrieval for Open-Domain Question Answering (DPR)
**Karpukhin et al., EMNLP 2020**

Dense retrieval using dual encoders.

- **Key innovations**: Contrastive learning for retrieval, FAISS indexing
- **Impact**: Standard dense retrieval approach
- **Link**: [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)
- **Status**: 🔧 Hands-on

---

### REALM: Retrieval-Augmented Language Model Pre-Training
**Guu et al., ICML 2020**

Pre-training with retrieval for knowledge-intensive tasks.

- **Key innovations**: Joint pre-training of retriever and reader
- **Link**: [arXiv:2002.08909](https://arxiv.org/abs/2002.08909)
- **Status**: 📝 Reference

---

### Improving Language Models by Retrieving from Trillions of Tokens (RETRO)
**Borgeaud et al., DeepMind 2022**

Retrieval-enhanced transformers at scale.

- **Key innovations**: Chunked cross-attention, trillion-token retrieval
- **Impact**: Showed retrieval scales with data
- **Link**: [arXiv:2112.04426](https://arxiv.org/abs/2112.04426)
- **Status**: 📖 Essential

---

## Advanced RAG Techniques

### Self-RAG: Learning to Retrieve, Generate, and Critique
**Asai et al., ICLR 2024**

LLM learns when and what to retrieve.

- **Key innovations**: Reflection tokens, adaptive retrieval
- **Impact**: More efficient RAG
- **Link**: [arXiv:2310.11511](https://arxiv.org/abs/2310.11511)
- **Status**: 🔬 Frontier

---

### Corrective RAG (CRAG)
**Yan et al., 2024**

Evaluates and corrects retrieved documents.

- **Key innovations**: Retrieval evaluator, knowledge refinement
- **Link**: [arXiv:2401.15884](https://arxiv.org/abs/2401.15884)
- **Status**: 🔬 Frontier

---

### HyDE: Hypothetical Document Embeddings
**Gao et al., 2022**

Generate hypothetical answer, then retrieve similar documents.

- **Key innovations**: Query expansion via generation
- **Impact**: Improved retrieval for complex queries
- **Link**: [arXiv:2212.10496](https://arxiv.org/abs/2212.10496)
- **Status**: 🔧 Hands-on

---

## Vector Databases & Embeddings

### Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks
**Reimers & Gurevych, EMNLP 2019**

Efficient sentence embeddings for semantic similarity.

- **Key innovations**: Siamese network training, efficient embeddings
- **Impact**: Foundation for embedding-based retrieval
- **Link**: [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)
- **Status**: 🔧 Hands-on

---

### Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)
**Wang et al., 2022**

State-of-the-art text embeddings.

- **Key innovations**: Weakly-supervised contrastive learning
- **Link**: [arXiv:2212.03533](https://arxiv.org/abs/2212.03533)
- **Status**: 🔧 Hands-on

---

### Matryoshka Representation Learning
**Kusupati et al., NeurIPS 2022**

Embeddings that work at multiple dimensions.

- **Key innovations**: Nested representations, flexible dimensionality
- **Link**: [arXiv:2205.13147](https://arxiv.org/abs/2205.13147)
- **Status**: 🔬 Frontier

---

## Online Resources

| Resource | Description | Link |
|----------|-------------|------|
| Prompt Engineering Guide | Comprehensive prompting techniques | [promptingguide.ai](https://www.promptingguide.ai/) |
| LangChain | RAG framework | [langchain.com](https://www.langchain.com/) |
| LlamaIndex | Data framework for LLMs | [llamaindex.ai](https://www.llamaindex.ai/) |
| MTEB Leaderboard | Embedding benchmarks | [HuggingFace](https://huggingface.co/spaces/mteb/leaderboard) |
| Pinecone Learning | Vector database tutorials | [pinecone.io](https://www.pinecone.io/learn/) |

---

## Implementation Resources

| Resource | Description | Link |
|----------|-------------|------|
| sentence-transformers | Embedding library | [sbert.net](https://www.sbert.net/) |
| FAISS | Vector similarity search | [GitHub](https://github.com/facebookresearch/faiss) |
| ChromaDB | Open-source embedding database | [trychroma.com](https://www.trychroma.com/) |
| Weaviate | Vector search engine | [weaviate.io](https://weaviate.io/) |

---

*Last updated: September 2026*
