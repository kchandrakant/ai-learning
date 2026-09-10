# MLOps & Model Serving: Key References

A curated collection of papers and resources on LLM inference, deployment, and monitoring.

---

## 📚 How to Use This Document

- **📖 Essential**: Core papers everyone should read
- **🔧 Hands-on**: Resources with practical implementations
- **🔬 Frontier**: Cutting-edge research

---

## Inference Optimization

### Efficient Memory Management for Large Language Model Serving with PagedAttention (vLLM)
**Kwon et al., SOSP 2023**

Revolutionary KV cache management for inference.

- **Key innovations**: PagedAttention, virtual memory for KV cache
- **Impact**: Standard for high-throughput serving
- **Link**: [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
- **Status**: 📖 Essential

---

### FlashAttention: Fast and Memory-Efficient Exact Attention
**Dao et al., NeurIPS 2022**

Hardware-aware attention algorithm.

- **Key innovations**: Tiling, kernel fusion, IO-aware algorithms
- **Impact**: 2-4x speedup, used everywhere
- **Link**: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- **Status**: 📖 Essential

---

### FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning
**Dao, 2023**

Improved FlashAttention with better GPU utilization.

- **Key innovations**: Better work partitioning, causal masking
- **Link**: [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- **Status**: 🔧 Hands-on

---

### Orca: A Distributed Serving System for Transformer-Based Generative Models
**Yu et al., OSDI 2022**

Continuous batching for LLM serving.

- **Key innovations**: Iteration-level scheduling, continuous batching
- **Impact**: Foundation for modern serving systems
- **Link**: [USENIX](https://www.usenix.org/conference/osdi22/presentation/yu)
- **Status**: 📖 Essential

---

## Quantization

### GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
**Frantar et al., ICLR 2023**

4-bit quantization with minimal accuracy loss.

- **Key innovations**: Layer-wise quantization, Hessian-based
- **Impact**: Enables running large models on consumer GPUs
- **Link**: [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- **Status**: 🔧 Hands-on

---

### AWQ: Activation-aware Weight Quantization for LLM Compression
**Lin et al., MLSys 2024**

Quantization preserving important weights.

- **Key innovations**: Activation-aware scaling
- **Link**: [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- **Status**: 🔧 Hands-on

---

### LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale
**Dettmers et al., NeurIPS 2022**

Mixed-precision inference for large models.

- **Key innovations**: Vector-wise quantization, outlier handling
- **Link**: [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)
- **Status**: 📖 Essential

---

### SmoothQuant: Accurate and Efficient Post-Training Quantization
**Xiao et al., ICML 2023**

Smooth activations for easier quantization.

- **Key innovations**: Per-channel smoothing, W8A8
- **Link**: [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
- **Status**: 🔧 Hands-on

---

## Speculative Decoding

### Fast Inference from Transformers via Speculative Decoding
**Leviathan et al., ICML 2023**

Use small model to draft, large model to verify.

- **Key innovations**: Draft-then-verify, rejection sampling
- **Impact**: 2-3x speedup without quality loss
- **Link**: [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
- **Status**: 📖 Essential

---

### SpecInfer: Accelerating Generative Large Language Model Serving
**Miao et al., 2023**

Tree-based speculative decoding.

- **Key innovations**: Token tree speculation
- **Link**: [arXiv:2305.09781](https://arxiv.org/abs/2305.09781)
- **Status**: 🔬 Frontier

---

## Serving Systems

### TensorRT-LLM
**NVIDIA, 2023**

NVIDIA's optimized LLM inference library.

- **Key features**: In-flight batching, quantization, tensor parallelism
- **Link**: [GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- **Status**: 🔧 Hands-on

---

### SGLang: Efficient Execution of Structured Language Model Programs
**Zheng et al., 2024**

Programming model for LLM applications.

- **Key innovations**: RadixAttention, constrained decoding
- **Link**: [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
- **Status**: 🔬 Frontier

---

### Text Generation Inference (TGI)
**Hugging Face**

Production-ready inference server.

- **Key features**: Continuous batching, quantization, streaming
- **Link**: [GitHub](https://github.com/huggingface/text-generation-inference)
- **Status**: 🔧 Hands-on

---

## Distributed Inference

### Megatron-LM: Training Multi-Billion Parameter Language Models
**Shoeybi et al., NVIDIA 2019**

Tensor and pipeline parallelism for large models.

- **Key innovations**: Tensor parallelism, pipeline parallelism
- **Impact**: Foundation for distributed training/inference
- **Link**: [arXiv:1909.08053](https://arxiv.org/abs/1909.08053)
- **Status**: 📖 Essential

---

### DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented Scale
**Aminabadi et al., SC 2022**

Distributed inference optimizations.

- **Key innovations**: Inference-optimized parallelism, kernel fusion
- **Link**: [arXiv:2207.00032](https://arxiv.org/abs/2207.00032)
- **Status**: 🔧 Hands-on

---

## Monitoring & Observability

### Monitoring Machine Learning Models in Production
**Google, 2020**

Best practices for ML monitoring.

- **Key topics**: Data drift, model degradation, alerting
- **Link**: [Google Cloud Blog](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- **Status**: 📖 Essential

---

## Online Resources

| Resource | Description | Link |
|----------|-------------|------|
| vLLM | High-throughput serving | [vllm.ai](https://vllm.ai/) |
| TensorRT-LLM | NVIDIA inference | [GitHub](https://github.com/NVIDIA/TensorRT-LLM) |
| llama.cpp | CPU/GPU inference | [GitHub](https://github.com/ggerganov/llama.cpp) |
| Ollama | Local LLM running | [ollama.ai](https://ollama.ai/) |
| LangSmith | LLM observability | [smith.langchain.com](https://smith.langchain.com/) |

---

## Implementation Resources

| Resource | Description | Link |
|----------|-------------|------|
| MLflow | ML lifecycle platform | [mlflow.org](https://mlflow.org/) |
| Weights & Biases | Experiment tracking | [wandb.ai](https://wandb.ai/) |
| BentoML | Model serving framework | [bentoml.com](https://www.bentoml.com/) |
| Ray Serve | Scalable serving | [ray.io](https://docs.ray.io/en/latest/serve/) |

---

*Last updated: September 2026*
