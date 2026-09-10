# Prompt Engineering & RAG Demos

Interactive demonstrations that bring together concepts from the course modules.

## Available Demos

### 1. Prompting Playground (`prompting_demo.py`)
Experiment with different prompting techniques side by side:
- Zero-shot vs few-shot comparison
- Chain-of-thought visualization
- Temperature effects
- Structured output formats

```bash
python demo/prompting_demo.py
```

### 2. RAG Pipeline Demo (`rag_demo.py`)
Build and query a complete RAG system:
- Document ingestion and chunking
- Vector store setup
- Query processing
- Answer generation with citations

```bash
python demo/rag_demo.py --docs ./sample_docs
```

### 3. Retrieval Comparison (`retrieval_demo.py`)
Compare retrieval strategies on the same queries:
- Vector search only
- Hybrid search (vector + BM25)
- With and without reranking
- Visualize score distributions

```bash
python demo/retrieval_demo.py
```

### 4. Evaluation Dashboard (`eval_demo.py`)
Run a full evaluation suite on your RAG system:
- Retrieval metrics (Recall, MRR, NDCG)
- Generation metrics (Faithfulness, Relevance)
- Export results to CSV/JSON

```bash
python demo/eval_demo.py --eval-set ./eval_questions.json
```

### 5. Advanced RAG Demo (`advanced_rag_demo.py`)
Explore advanced architectures:
- Self-RAG with reflection
- Agentic RAG with tools
- Side-by-side comparison

```bash
python demo/advanced_rag_demo.py
```

### 6. Cost Analyzer (`cost_demo.py`)
Understand the cost of your RAG pipeline:
- Token counting for each stage
- Model comparison (GPT-3.5 vs GPT-4)
- Cost optimization suggestions

```bash
python demo/cost_demo.py
```

## Sample Documents

The `sample_docs/` folder contains example documents for demos:
- Technical documentation (Markdown)
- FAQ pages (HTML)
- Research papers (PDF)

## Running Demos

1. Ensure you've completed setup:
   ```bash
   pip install -r requirements.txt
   python verify_setup.py
   ```

2. Set your API keys:
   ```bash
   export OPENAI_API_KEY="your-key"
   ```

3. Run any demo:
   ```bash
   python demo/<demo_name>.py
   ```

## Demo Output

Each demo generates:
- Console output with explanations
- Visualizations (saved as PNG)
- Metrics and statistics

---

Ready to experiment? Start with `prompting_demo.py` to explore prompting techniques.
