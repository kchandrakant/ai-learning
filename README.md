# AI Learning Roadmap

A comprehensive, hands-on learning path for mastering AI/ML — from mathematical foundations to production LLM systems.

**200+ modules across 16 courses** | **8-14 months** | **Self-paced**

---

## 🎯 What You'll Learn

This curriculum takes you from zero to production-ready AI engineer:

```
Math & ML Basics → Neural Networks → Transformers → LLMs → Agents → Production Systems
```

By the end, you'll be able to:
- Build transformers from scratch
- Design and deploy RAG systems
- Create reliable AI agents
- Fine-tune models for specific domains
- Deploy secure, production-grade AI services

---

## 📚 The 16 Courses

### Pre-Requisite (Course 00)
*Essential skills before diving into ML*

| # | Course | Modules | Duration | Description |
|---|--------|---------|----------|-------------|
| 00 | [foundations-preskills](./00-foundations-preskills/) | 6 | 1-2 weeks | Python patterns, NumPy, visualization, reading ML papers |

### Foundation Track (Courses 01-04)
*Build deep understanding of what powers AI systems*

| # | Course | Modules | Duration | Description |
|---|--------|---------|----------|-------------|
| 01 | [ml-foundations](./01-ml-foundations/) | 14 | 4-6 weeks | Linear algebra, calculus, probability, classical ML |
| 02 | [deep-learning-to-transformers](./02-deep-learning-to-transformers/) | 15 | 4-6 weeks | Neural networks, CNNs, RNNs, attention mechanism |
| 03 | [transformer-architecture](./03-transformer-architecture/) | 8 | 4-6 weeks | Deep dive into transformer implementation |
| 04 | [llm-internals](./04-llm-internals/) | 12 | 4-6 weeks | Tokenization, pre-training, scaling laws, RLHF |

### Application Track (Courses 05-08)
*Learn to build real AI applications*

| # | Course | Modules | Duration | Description |
|---|--------|---------|----------|-------------|
| 05 | [prompt-engineering-rag](./05-prompt-engineering-rag/) | 12 | 4-5 weeks | Prompting techniques, RAG systems |
| 06 | [multi-agent-systems](./06-multi-agent-systems/) | 12 | 4-5 weeks | Agent architectures, tool use, orchestration |
| 07 | [fine-tuning-adaptation](./07-fine-tuning-adaptation/) | 14 | 4-5 weeks | LoRA, QLoRA, RLHF, DPO |
| 08 | [multimodal-ai](./08-multimodal-ai/) | 12 | 4-5 weeks | Vision-language, diffusion, audio, video |

### Production Track (Courses 09-12)
*Deploy and operate AI systems at scale*

| # | Course | Modules | Duration | Description |
|---|--------|---------|----------|-------------|
| 09 | [mlops-model-serving](./09-mlops-model-serving/) | 17 | 5-6 weeks | Inference optimization, deployment, local + cloud |
| 10 | [agent-system-design](./10-agent-system-design/) | 13 | 4-5 weeks | Agent runtime architecture, evaluation |
| 11 | [ai-security-identity](./11-ai-security-identity/) | 16 | 4-5 weeks | Agent identity, auth, access control |
| 12 | [agentic-protocols](./12-agentic-protocols/) | 15 | 4-5 weeks | MCP, A2A, ACP, emerging standards |

### Capstone Track (Courses 13-15)
*Ethics, frontier developments, and integration project*

| # | Course | Modules | Duration | Description |
|---|--------|---------|----------|-------------|
| 13 | [ai-ethics-responsible-ai](./13-ai-ethics-responsible-ai/) | 12 | 4-5 weeks | Alignment, bias, fairness, regulation, governance |
| 14 | [emerging-ai-trends](./14-emerging-ai-trends/) | 14 | 4-5 weeks | System 1/2, Mamba, MoE, tabular, frontier research |
| 15 | [capstone-project](./15-capstone-project/) | 5 options | 6-8 weeks | Build a complete AI system from scratch |

---

## 🚀 How to Use This Repository

### Step 1: Choose Your Path

**Completely new to programming/ML?** Start with pre-skills:
```
00 → 01 → 02 → 03 → 04 → 05 → ... (follow the numbers)
```

**Know Python but new to ML?** Start from the beginning:
```
01 → 02 → 03 → 04 → 05 → 06 → ... (follow the numbers)
```

**Know the basics?** Jump to applications:
```
05-prompt-engineering-rag → 06-multi-agent-systems → 08-multimodal-ai
```

**Building production systems?** Focus on deployment:
```
09-mlops-model-serving → 10-agent-system-design → 12-agentic-protocols
```

**Just want to fine-tune models?**
```
03-transformer-architecture → 04-llm-internals → 07-fine-tuning-adaptation
```

**Ready to prove your skills?** Jump to capstone:
```
15-capstone-project (after completing relevant prerequisites)
```

### Step 2: Set Up a Course

```bash
# Clone the repository
git clone https://github.com/kchandrakant/ai-learning.git
cd ai-learning

# Pick a course
cd 01-ml-foundations

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py
```

### Step 3: Follow the Learning Path

Each course contains:
```
📁 XX-course-name/
├── LEARNING_PATH.md      ← Start here! Your guide through the course
├── requirements.txt      ← Dependencies
├── verify_setup.py       ← Check your environment
├── 01_first_module/
│   └── README.md         ← Module content and exercises
├── 02_second_module/
│   └── README.md
├── ...
├── demo/                 ← Hands-on demonstrations
└── beyond/               ← Advanced topics and future directions
```

### Step 4: Interactive Learning with Kiro

Each course has a `.kiro/steering/` file that enables interactive, pair-programming style learning:

1. Open the course folder in Kiro IDE
2. Say "ready" to start a module
3. The agent will explain concepts, then you'll implement together
4. Ask questions, experiment, iterate

---

## 📅 Suggested Timeline

### Full Curriculum: 12-14 Months

| Phase | Months | Courses | Focus |
|-------|--------|---------|-------|
| **Pre-Skills** | 0.5 | 00 | Python patterns, NumPy, paper reading |
| **Foundations** | 1-3 | 01, 02, 03, 04 | Math, neural nets, transformers, LLM internals |
| **Applications** | 4-6 | 05, 06, 07, 08 | RAG, agents, fine-tuning, multimodal |
| **Production** | 7-9 | 09, 10, 11, 12 | MLOps, system design, security, protocols |
| **Capstone** | 10-11 | 13, 14 | Ethics, governance, emerging trends |
| **Final Project** | 12-14 | 15 | Build complete AI system |

### Accelerated: 6 Months

| Phase | Months | Courses | Notes |
|-------|--------|---------|-------|
| **Core** | 1-2 | 02, 03 | Skip 00-01 if you know Python and basic ML |
| **Applications** | 3-4 | 05, 06 | Focus on RAG and agents |
| **Production** | 5 | 09 | MLOps essentials |
| **Capstone** | 6 | 15 | Build your project |

### Practitioner Fast-Track: 3 Months

| Month | Courses | Notes |
|-------|---------|-------|
| 1 | 05-prompt-engineering-rag | Most immediately practical |
| 2 | 06-multi-agent-systems | Build agent applications |
| 3 | 09-mlops-model-serving | Deploy to production (includes local Ollama+Docker) |

---

## ⏱️ Weekly Time Commitment

### Recommended: 10-15 hours/week

```
Monday      (2 hrs)  — Read theory, watch explanations
Tuesday     (2 hrs)  — Start implementing
Wednesday   (2 hrs)  — Continue implementation
Thursday    (2 hrs)  — Test, debug, experiment
Friday      (2 hrs)  — Review, document learnings
Weekend     (2-4 hrs) — Catch-up, exploration, side projects
```

### Minimum Viable: 5-7 hours/week
- Progress will be slower but still effective
- Focus on one module per week
- Skip some exercises if needed

### Intensive: 20+ hours/week
- Can complete the full curriculum in 4-6 months
- Risk of burnout — take breaks
- Great for career transitions

---

## 🎯 Learning Principles

### 1. Build, Don't Just Read
Every module has implementation exercises. Do them. Theory without code doesn't stick.

### 2. Depth Over Breadth
It's better to deeply understand one topic than superficially know many. Don't rush.

### 3. Work in Public
Document your learning. Share on GitHub, write blog posts, post on social media. This builds your portfolio.

### 4. Ask Questions
Use the Kiro interactive mode. Ask "why?" constantly. The best learning comes from curiosity.

### 5. Embrace Confusion
Feeling confused is normal and means you're learning. Push through, then revisit.

---

## ✅ Progress Checklist

### Pre-Requisite
- [ ] **00-foundations-preskills**: Python patterns, NumPy, visualization, reading ML papers

### Foundation Track
- [ ] **01-ml-foundations**: Linear algebra, calculus, probability, classical ML
- [ ] **02-deep-learning-to-transformers**: Perceptron → attention mechanism
- [ ] **03-transformer-architecture**: Build transformer from scratch
- [ ] **04-llm-internals**: Tokenization, scaling laws, RLHF

### Application Track
- [ ] **05-prompt-engineering-rag**: Prompting patterns, build RAG system
- [ ] **06-multi-agent-systems**: Build multi-agent workflow
- [ ] **07-fine-tuning-adaptation**: Fine-tune a model with LoRA
- [ ] **08-multimodal-ai**: Work with images, audio, video

### Production Track
- [ ] **09-mlops-model-serving**: Deploy an LLM service
- [ ] **10-agent-system-design**: Build production agent harness
- [ ] **11-ai-security-identity**: Implement agent auth system
- [ ] **12-agentic-protocols**: Implement MCP server

### Capstone Track
- [ ] **13-ai-ethics-responsible-ai**: Alignment, bias, fairness, responsible deployment
- [ ] **14-emerging-ai-trends**: System 1/2, Mamba, MoE, frontier research
- [ ] **15-capstone-project**: Build complete AI system (choose from 5 options)

### Portfolio Projects (built during Course 15)
- [ ] Domain-specific RAG assistant
- [ ] Code assistant with repository understanding
- [ ] Research agent with multi-source synthesis
- [ ] Multimodal application
- [ ] Custom project of your choice

---

## 🛠️ Prerequisites

### For Pre-Requisite Course (00)
- Basic programming experience in any language
- High school algebra
- Willingness to practice

### For Foundation Track (01-04)
- Course 00 completed (or equivalent Python/NumPy skills)
- High school math (algebra, basic functions)
- Willingness to work through equations

### For Application Track (05-08)
- Completed Foundation Track (or equivalent knowledge)
- Familiarity with LLM APIs (OpenAI, Anthropic)
- Basic understanding of embeddings

### For Production Track (09-12)
- Completed Application Track (or equivalent)
- Docker basics
- API development experience (REST)

### For Capstone (15)
- Completed at least 5 relevant courses
- Ready to build a complete system independently

### Hardware
- **Minimum**: CPU-only, 8GB RAM (will be slow for some modules)
- **Recommended**: GPU with 8GB+ VRAM, 16GB RAM
- **Cloud alternative**: Google Colab, Lambda Labs, etc.
- **Local LLM**: Ollama works great for Course 09 and Capstone projects

---

## 📖 Supplementary Resources

### Books
- "Mathematics for Machine Learning" — Deisenroth et al. (free PDF)
- "Designing Machine Learning Systems" — Chip Huyen
- "Deep Learning" — Goodfellow, Bengio, Courville

### Video Courses
- 3Blue1Brown: Linear Algebra, Neural Networks series
- Andrej Karpathy: Neural Networks: Zero to Hero
- Stanford CS224N: NLP with Deep Learning

### Blogs
- Lilian Weng (lilianweng.github.io) — Technical deep-dives
- Jay Alammar (jalammar.github.io) — Visual explanations
- Hugging Face Blog — Practical tutorials

### Communities
- r/LocalLLaMA, r/MachineLearning
- Hugging Face Discord
- LangChain Discord

---

## 🤝 Contributing

Found an error? Have a suggestion? Contributions welcome!

1. Fork the repository
2. Create a branch (`git checkout -b fix/module-name`)
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License — feel free to use for learning and teaching.

---

## 🙏 Acknowledgments

This curriculum synthesizes insights from:
- Anthropic, OpenAI, Google DeepMind research
- Hugging Face, LangChain, LlamaIndex documentation
- Andrej Karpathy, Lilian Weng, Jay Alammar
- The open-source AI community

---

**Ready to start?** → Open [00-foundations-preskills/LEARNING_PATH.md](./00-foundations-preskills/LEARNING_PATH.md)

**Already know Python/NumPy?** → Start with [01-ml-foundations/LEARNING_PATH.md](./01-ml-foundations/LEARNING_PATH.md)

**Already know ML?** → Jump to [05-prompt-engineering-rag/LEARNING_PATH.md](./05-prompt-engineering-rag/LEARNING_PATH.md)

---

*Last updated: September 2026*
