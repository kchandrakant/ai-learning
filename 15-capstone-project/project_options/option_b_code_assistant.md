# Option B: Code Assistant

Build a coding assistant that helps with code generation, review, and explanation using RAG over code repositories.

## Project Description

Create an assistant that understands code by indexing repositories and using that context to help with coding tasks. The assistant should handle code generation, review, debugging, and explanation.

## Core Capabilities

1. **Code Generation**: Generate functions/classes from descriptions
2. **Code Review**: Analyze code for issues and improvements
3. **Debugging Help**: Diagnose errors and suggest fixes
4. **Code Explanation**: Explain what code does

## Core Requirements

### Must Have (80% of grade)

1. **Repository Indexing**
   - Parse and index code files
   - Code-aware chunking (function/class level)
   - Multi-language support (at least 2)

2. **Code-Aware RAG**
   - Retrieve relevant code context
   - Handle code structure in prompts
   - Maintain code formatting

3. **Code Generation**
   - Generate from natural language
   - Match project style/patterns
   - Include documentation

4. **Code Review**
   - Identify bugs and issues
   - Style and best practice feedback
   - Actionable suggestions

5. **API Interface**
   - REST API with endpoints for each capability
   - Repository management
   - Session/project context

### Should Have (Additional 15%)

6. **Multi-File Context**
   - Understand cross-file dependencies
   - Import/reference awareness
   - Project-level context

7. **Debugging Assistance**
   - Error message parsing
   - Stack trace analysis
   - Fix suggestions

### Nice to Have (Additional 5%)

8. **Advanced Features**
   - Test generation
   - Refactoring suggestions
   - Code completion

## Technical Stack

```
Required:
- Python 3.10+
- FastAPI
- LangChain
- Chroma or FAISS
- Ollama with CodeLlama/DeepSeek-Coder
- Tree-sitter (for code parsing)

Optional:
- Language-specific linters
- CodeBERT embeddings
- IDE extension
```

## Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Code Retrieval | 20% | Finds relevant code context |
| Generation Quality | 25% | Generated code is correct |
| Review Helpfulness | 20% | Reviews are actionable |
| Multi-Language | 15% | Works across languages |
| Code Quality | 10% | Clean, documented code |
| API Design | 10% | Well-designed interface |

## Milestones

### Week 1-2: Code Parsing
- [ ] Set up tree-sitter for target languages
- [ ] Implement code-aware chunking
- [ ] Basic repository indexing

### Week 3-4: Code RAG
- [ ] Code embedding strategy
- [ ] Retrieval with code context
- [ ] Prompt engineering for code

### Week 5-6: Capabilities
- [ ] Generation endpoint
- [ ] Review endpoint
- [ ] Debugging endpoint

### Week 7-8: Integration
- [ ] Multi-file context
- [ ] Testing
- [ ] Documentation

### Week 9-10: Polish
- [ ] Additional languages
- [ ] Edge cases
- [ ] Demo preparation

## Getting Started

```bash
# Use the provided template
cp -r templates/code_assistant my-code-assistant
cd my-code-assistant

# Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install tree-sitter languages
pip install tree-sitter-python tree-sitter-javascript

# Pull code model
ollama pull codellama:13b

# Run
python -m src.main
```

## Deliverables

1. **Source Code**: Complete application
2. **Design Document**: Architecture and code handling approach
3. **Demo**: Show all capabilities with real repository
4. **Self-Evaluation**: Using the evaluation template

## Challenges to Consider

- **Code Chunking**: How to split code meaningfully?
- **Context Window**: Code is verbose, how to fit context?
- **Correctness**: How to validate generated code?
- **Multi-Language**: Different syntax and patterns

## Resources

- [Tree-sitter Documentation](https://tree-sitter.github.io/)
- [CodeLlama Paper](https://arxiv.org/abs/2308.12950)
- Course 03 (RAG Fundamentals)
- Course 08 (Agentic Patterns)
