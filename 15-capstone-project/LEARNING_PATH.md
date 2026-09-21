# Capstone Project: Bringing It All Together

This final course is where you integrate everything you've learned. You'll build a complete AI system from scratch — designing, implementing, deploying, securing, and documenting it.

---

## 🎯 Purpose

The capstone project demonstrates that you can:
- Apply knowledge from multiple courses simultaneously
- Make architectural decisions and justify them
- Build production-quality systems, not just demos
- Handle the messy realities of real-world AI development

**This is not a tutorial.** You'll choose a project, design it, build it, and present it. We provide guidance and templates, but the work is yours.

---

## 🎯 Prerequisites

You should have completed (or have equivalent knowledge from):

| Course | Key Skills Needed |
|--------|-------------------|
| 01-04 | ML/DL fundamentals, transformers, LLM internals |
| 05 | Prompt engineering, RAG systems |
| 06 | Multi-agent systems |
| 07 | Fine-tuning (if your project needs it) |
| 08 | Multimodal AI (if applicable) |
| 09 | MLOps, deployment, local serving |
| 10 | Agent system design |
| 11-12 | Security, protocols |
| 13 | Ethics, responsible AI |

**Not everything applies to every project.** Choose a project that matches your interests and the courses you've completed.

---

## 📚 Project Options

Choose ONE of the following project tracks, or propose your own.

### Option A: Domain-Specific Assistant

Build an AI assistant specialized for a specific domain.

**Example domains:**
- Legal document analyzer
- Medical information assistant (with appropriate disclaimers)
- Financial report summarizer
- Educational tutor for a subject
- Customer support agent for a product

**Required components:**
- RAG system with domain-specific documents (Course 05)
- Custom prompt engineering for the domain (Course 05)
- Agent capabilities with tools (Course 06)
- Deployed API (Course 09)
- Basic security/rate limiting (Course 11)
- Documentation for compliance (Course 13)

**Stretch goals:**
- Fine-tuned model for domain terminology (Course 07)
- Multi-agent workflow for complex queries (Course 06, 10)
- Evaluation suite with domain-specific metrics (Course 10)

---

### Option B: Code Assistant

Build a coding assistant that helps with a specific programming task.

**Example focuses:**
- Code review assistant
- Documentation generator
- Test case generator
- Bug finder and fixer
- Code migration helper (e.g., Python 2 to 3)

**Required components:**
- Code understanding and generation (Course 05)
- Tool use for code execution/validation (Course 06)
- Context management for large codebases (Course 10)
- Local deployment for privacy (Course 09)
- Safety guardrails (Course 13)

**Stretch goals:**
- Integration with IDE (VS Code extension)
- Multi-file understanding and refactoring
- Learning from user feedback

---

### Option C: Research Agent

Build an agent that helps with research tasks.

**Example focuses:**
- Paper summarizer and Q&A
- Literature review assistant
- Data analysis helper
- Experiment tracker and reporter

**Required components:**
- Document processing and RAG (Course 05)
- Multi-step reasoning (Course 06, 10)
- Citation and source tracking
- Web search integration (if applicable)
- Structured output generation

**Stretch goals:**
- Paper reading with figures/tables (Course 08)
- Automated experiment running
- Knowledge graph construction

---

### Option D: Multimodal Application

Build a system that works with multiple modalities.

**Example projects:**
- Image-based Q&A system
- Document analyzer (PDFs with text + figures)
- Video summarizer
- Accessibility assistant (describe images)

**Required components:**
- Vision-language model integration (Course 08)
- Multimodal RAG (Course 05, 08)
- Appropriate input/output handling
- Performance optimization for large inputs (Course 09)

**Stretch goals:**
- Real-time video processing
- Multi-turn visual dialogue
- Fine-tuned visual understanding

---

### Option E: Custom Project

Propose your own project that:
- Integrates knowledge from at least 5 courses
- Solves a real problem you care about
- Can be completed in 4-8 weeks
- Has clear success criteria

Submit a 1-page proposal for review before starting.

---

## 📋 Project Phases

Every capstone project follows these phases:

### Phase 1: Requirements & Design (Week 1-2)

**Deliverables:**
1. **Requirements Document**
   - Problem statement
   - User stories / use cases
   - Functional requirements
   - Non-functional requirements (performance, security, etc.)
   - Out of scope (what you're NOT building)

2. **Design Document**
   - System architecture diagram
   - Component breakdown
   - Data flow
   - Technology choices with justification
   - API design (if applicable)
   - Security considerations

**Template provided:** `templates/design_doc_template.md`

---

### Phase 2: Implementation (Week 3-6)

**Build iteratively:**
1. **Sprint 1:** Core functionality (MVP)
   - Basic working system
   - Happy path works

2. **Sprint 2:** Robustness
   - Error handling
   - Edge cases
   - Input validation

3. **Sprint 3:** Production readiness
   - Deployment
   - Monitoring
   - Documentation

**Checkpoints:**
- End of each sprint: Working demo
- Code committed to version control
- Progress tracked in project board

---

### Phase 3: Evaluation (Week 6-7)

**Evaluate your system:**
1. **Functional testing**
   - Does it meet requirements?
   - Test cases pass?

2. **Performance testing**
   - Latency benchmarks
   - Throughput under load
   - Resource usage

3. **Quality evaluation**
   - For RAG: Retrieval accuracy, answer quality
   - For agents: Task completion rate
   - For generation: Human evaluation

4. **Security review**
   - Input validation
   - Prompt injection resistance
   - Access controls

**Template provided:** `evaluation/evaluation_template.md`

---

### Phase 4: Documentation & Presentation (Week 7-8)

**Deliverables:**
1. **README.md**
   - What it does
   - How to run it
   - Architecture overview
   - Known limitations

2. **Technical Documentation**
   - API documentation
   - Configuration guide
   - Troubleshooting guide

3. **Project Report**
   - What you built
   - Key decisions and tradeoffs
   - What worked, what didn't
   - What you'd do differently
   - Lessons learned

4. **Demo**
   - 10-minute video or live demo
   - Show the system working
   - Explain key technical decisions

---

## 🏗️ Project Structure Template

```
capstone-project/
├── README.md                 # Project overview
├── docs/
│   ├── requirements.md       # Requirements document
│   ├── design.md             # Design document
│   ├── api.md                # API documentation
│   └── report.md             # Final project report
│
├── src/
│   ├── __init__.py
│   ├── main.py               # Entry point
│   ├── config.py             # Configuration
│   ├── api/                  # API layer
│   ├── core/                 # Core logic
│   ├── agents/               # Agent definitions (if applicable)
│   ├── rag/                  # RAG components (if applicable)
│   └── utils/                # Utilities
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── evaluation/           # Quality evaluations
│
├── data/
│   ├── raw/                  # Raw data (gitignored if large)
│   └── processed/            # Processed data
│
├── deployment/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── kubernetes/           # K8s manifests (if applicable)
│
├── notebooks/                # Exploration notebooks
├── scripts/                  # Utility scripts
├── requirements.txt
├── pyproject.toml
└── .env.example              # Environment variables template
```

---

## ✅ Evaluation Criteria

Your project will be evaluated on:

### Technical Quality (40%)
- Code quality and organization
- Appropriate use of techniques from courses
- Error handling and edge cases
- Performance and scalability considerations

### Functionality (25%)
- Meets stated requirements
- Works reliably
- Handles real-world inputs

### Design & Architecture (15%)
- Clear, justified architecture
- Appropriate technology choices
- Security considerations

### Documentation (10%)
- Clear README
- API documentation
- Design rationale documented

### Presentation (10%)
- Clear explanation
- Effective demo
- Honest assessment of limitations

---

## 📅 Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Requirements | Problem statement, user stories |
| 2 | Design | Architecture, technology choices |
| 3-4 | Implementation Sprint 1 | MVP, core functionality |
| 5 | Implementation Sprint 2 | Robustness, error handling |
| 6 | Implementation Sprint 3 | Deployment, monitoring |
| 7 | Evaluation | Testing, benchmarks, quality eval |
| 8 | Documentation | Report, demo, final polish |

**Total: 8 weeks**

Adjust based on your availability. Part-time: extend to 12-16 weeks.

---

## 🛠️ Tools & Resources

### Recommended Stack

**For most projects:**
- Python 3.10+
- FastAPI or Flask for APIs
- LangChain or LlamaIndex for RAG/agents
- Ollama + Docker for local deployment
- PostgreSQL + pgvector for vector storage
- Prometheus + Grafana for monitoring

**For evaluation:**
- pytest for testing
- RAGAS for RAG evaluation
- Custom metrics for your domain

### Templates Provided

```
templates/
├── domain_assistant/       # Starter for Option A
│   ├── README.md
│   └── ...
├── code_assistant/         # Starter for Option B
│   └── ...
├── research_agent/         # Starter for Option C
│   └── ...
└── design_doc_template.md  # Design document template
```

### External Resources

- Course materials (01-14)
- LangChain documentation
- LlamaIndex documentation
- FastAPI documentation
- Docker documentation

---

## 🚀 Getting Started

### Step 1: Choose Your Project

1. Review the project options above
2. Consider which courses you've completed
3. Think about what interests you
4. Choose one option (or propose custom)

### Step 2: Set Up Your Environment

```bash
# Create project directory
mkdir my-capstone-project
cd my-capstone-project

# Initialize git
git init

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Copy starter template
cp -r ../templates/[your-option]/* .

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Start with Requirements

Before writing any code:
1. Open `docs/requirements.md`
2. Define your problem clearly
3. Write user stories
4. List functional requirements
5. Identify constraints

### Step 4: Design Before Building

1. Sketch your architecture
2. Choose your technologies (justify each choice)
3. Design your APIs
4. Identify risks and mitigation
5. Get feedback if possible

### Step 5: Build Iteratively

1. Start with the simplest possible version
2. Get it working end-to-end
3. Add complexity incrementally
4. Test as you go
5. Commit frequently

---

## 💡 Tips for Success

### Do

- **Start simple** — Get a basic version working first
- **Test early** — Don't wait until the end
- **Document as you go** — Not at the end
- **Use version control** — Commit often with good messages
- **Ask for feedback** — Share early, iterate
- **Timebox** — Don't let perfection block progress

### Don't

- **Over-engineer** — Build what you need, not what you might need
- **Skip the design** — Planning saves time overall
- **Ignore errors** — Handle edge cases properly
- **Copy-paste blindly** — Understand what you're using
- **Hide limitations** — Be honest about what doesn't work

### Common Pitfalls

1. **Scope creep** — Stick to your requirements
2. **Premature optimization** — Make it work, then make it fast
3. **Tutorial addiction** — Build, don't just watch
4. **Perfect is the enemy of done** — Ship something
5. **Going alone** — Get feedback, ask questions

---

## 🗂️ Project Structure

```
15-capstone-project/
├── LEARNING_PATH.md          # This file
├── requirements.txt          # Common dependencies
│
├── project_options/
│   ├── option_a_domain_assistant.md
│   ├── option_b_code_assistant.md
│   ├── option_c_research_agent.md
│   ├── option_d_multimodal.md
│   └── custom_proposal_template.md
│
├── templates/
│   ├── domain_assistant/     # Starter code for Option A
│   ├── code_assistant/       # Starter code for Option B
│   ├── research_agent/       # Starter code for Option C
│   └── design_doc_template.md
│
├── evaluation/
│   ├── evaluation_template.md
│   └── metrics_guide.md
│
└── examples/
    └── example_project/      # Reference implementation
```

---

## 🎓 Completion

When you've finished:

1. ✅ All code committed and documented
2. ✅ README clearly explains the project
3. ✅ Design document captures decisions
4. ✅ Evaluation demonstrates quality
5. ✅ Demo shows the system working
6. ✅ Report reflects on the experience

**Congratulations!** You've completed the AI/ML learning path.

You now have:
- Deep knowledge from foundations to frontiers
- A portfolio project demonstrating your skills
- The ability to build production AI systems

---

## 📖 References

### Project Management
- "The Mythical Man-Month" — Fred Brooks
- "Shape Up" — Basecamp (free online)

### Software Engineering
- "Clean Code" — Robert Martin
- "Designing Data-Intensive Applications" — Martin Kleppmann

### ML Systems
- "Designing Machine Learning Systems" — Chip Huyen
- "Building Machine Learning Powered Applications" — Emmanuel Ameisen

---

*"The best way to learn is to build."*

Good luck with your capstone project!
