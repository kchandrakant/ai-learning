# Step 12: Advanced RAG Architectures

## Beyond Standard RAG

Standard RAG: retrieve → generate. Advanced architectures add:
- Self-reflection and correction
- Knowledge graphs
- Multi-step reasoning
- Tool use within RAG

## Self-RAG

Dynamically decide whether to retrieve:

```
┌─────────────────────────────────────────────────────┐
│                     Self-RAG                         │
├─────────────────────────────────────────────────────┤
│                                                      │
│   Query → "Do I need retrieval?" ────┐              │
│                │                      │              │
│                ▼                      ▼              │
│              YES                     NO              │
│                │                      │              │
│                ▼                      ▼              │
│           Retrieve              Generate            │
│                │                directly            │
│                ▼                                     │
│     "Is this relevant?" ───────┐                    │
│                │                │                    │
│                ▼                ▼                    │
│             YES               NO                    │
│                │                │                    │
│                ▼                ▼                    │
│           Generate         Retrieve                 │
│           with context     again                    │
│                │                                     │
│                ▼                                     │
│     "Is response supported?" ──┐                    │
│                │                │                    │
│                ▼                ▼                    │
│             YES               NO                    │
│                │                │                    │
│                ▼                ▼                    │
│            Return          Regenerate               │
│                                                      │
└─────────────────────────────────────────────────────┘
```

```python
class SelfRAG:
    def query(self, question: str) -> str:
        # Step 1: Decide if retrieval is needed
        if self.needs_retrieval(question):
            docs = self.retrieve(question)
            
            # Step 2: Check relevance
            relevant_docs = [d for d in docs if self.is_relevant(question, d)]
            
            if not relevant_docs:
                # Retry with different strategy
                docs = self.retrieve_broader(question)
                relevant_docs = [d for d in docs if self.is_relevant(question, d)]
            
            # Step 3: Generate with context
            response = self.generate(question, relevant_docs)
            
            # Step 4: Verify support
            if not self.is_supported(response, relevant_docs):
                response = self.regenerate(question, relevant_docs)
        else:
            # Generate without retrieval
            response = self.generate_direct(question)
        
        return response
    
    def needs_retrieval(self, question: str) -> bool:
        """Decide if retrieval would help."""
        prompt = f"""
        Does this question require looking up specific information,
        or can it be answered from general knowledge?
        
        Question: {question}
        
        Answer RETRIEVE or DIRECT:
        """
        return "RETRIEVE" in llm(prompt)
    
    def is_supported(self, response: str, docs: list) -> bool:
        """Check if response is grounded in documents."""
        context = "\n".join(d.text for d in docs)
        prompt = f"""
        Is this response fully supported by the given documents?
        
        Documents: {context}
        Response: {response}
        
        Answer YES or NO:
        """
        return "YES" in llm(prompt)
```

## CRAG (Corrective RAG)

Evaluate and correct retrieval before generation:

```python
class CorrectiveRAG:
    def query(self, question: str) -> str:
        docs = self.retrieve(question)
        
        # Evaluate each document
        evaluations = []
        for doc in docs:
            score = self.evaluate_relevance(question, doc)
            evaluations.append((doc, score))
        
        # Classify overall retrieval quality
        avg_score = sum(s for _, s in evaluations) / len(evaluations)
        
        if avg_score > 0.7:
            # Good retrieval - use directly
            return self.generate(question, docs)
        
        elif avg_score > 0.3:
            # Ambiguous - refine and augment
            refined_docs = self.knowledge_refinement(question, docs)
            web_docs = self.web_search(question)
            return self.generate(question, refined_docs + web_docs)
        
        else:
            # Poor retrieval - use web search
            web_docs = self.web_search(question)
            return self.generate(question, web_docs)
    
    def knowledge_refinement(self, question: str, docs: list) -> list:
        """Extract only relevant parts of documents."""
        refined = []
        for doc in docs:
            relevant_parts = self.extract_relevant(question, doc)
            if relevant_parts:
                refined.append(relevant_parts)
        return refined
```

## Graph RAG

Combine knowledge graphs with retrieval:

```
┌─────────────────────────────────────────────────────┐
│                    Graph RAG                         │
├─────────────────────────────────────────────────────┤
│                                                      │
│   Documents ─────▶ Entity Extraction                │
│                           │                         │
│                           ▼                         │
│                    Knowledge Graph                  │
│                    ┌───────────────┐                │
│                    │ Entity─Relation│               │
│                    │    Entity      │               │
│                    └───────────────┘                │
│                           │                         │
│   Query ──────────▶ Graph Traversal                │
│                           │                         │
│                           ▼                         │
│                    Related Entities                 │
│                           │                         │
│                           ▼                         │
│              Retrieve docs for entities             │
│                           │                         │
│                           ▼                         │
│              Generate with graph context            │
│                                                      │
└─────────────────────────────────────────────────────┘
```

```python
class GraphRAG:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entity_to_docs = {}
    
    def index(self, documents: list):
        for doc in documents:
            # Extract entities
            entities = self.extract_entities(doc.text)
            
            # Extract relations
            relations = self.extract_relations(doc.text, entities)
            
            # Add to graph
            for entity in entities:
                self.graph.add_node(entity)
                self.entity_to_docs.setdefault(entity, []).append(doc.id)
            
            for subj, rel, obj in relations:
                self.graph.add_edge(subj, obj, relation=rel)
    
    def query(self, question: str) -> str:
        # Extract entities from question
        query_entities = self.extract_entities(question)
        
        # Find related entities via graph traversal
        related = set()
        for entity in query_entities:
            if entity in self.graph:
                # Get neighbors within 2 hops
                neighbors = nx.single_source_shortest_path_length(
                    self.graph, entity, cutoff=2
                )
                related.update(neighbors.keys())
        
        # Get documents for related entities
        doc_ids = set()
        for entity in related:
            doc_ids.update(self.entity_to_docs.get(entity, []))
        
        docs = self.get_docs(doc_ids)
        
        # Also do vector search
        vector_docs = self.vector_search(question)
        
        # Combine and generate
        all_docs = list(set(docs + vector_docs))
        return self.generate(question, all_docs)
```

## Agentic RAG

RAG with tool use and multi-step reasoning:

```python
class AgenticRAG:
    def __init__(self):
        self.tools = {
            "search_docs": self.search_documents,
            "search_web": self.search_web,
            "calculate": self.calculate,
            "summarize": self.summarize_doc,
        }
    
    def query(self, question: str) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question}
        ]
        
        for _ in range(10):  # Max iterations
            response = llm(
                messages=messages,
                tools=self.tool_definitions,
            )
            
            if response.tool_calls:
                # Execute tools
                for tool_call in response.tool_calls:
                    result = self.execute_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            else:
                # Final answer
                return response.content
        
        return "Could not determine answer within iteration limit."
    
    def search_documents(self, query: str, filters: dict = None) -> str:
        """Search internal documents."""
        docs = self.vector_store.search(query, filters=filters, k=5)
        return self.format_results(docs)
    
    def search_web(self, query: str) -> str:
        """Search the web for current information."""
        # Use web search API
        results = web_search(query)
        return self.format_results(results)
```

## Multi-Modal RAG

Handle images, tables, and other modalities:

```python
class MultiModalRAG:
    def index(self, documents: list):
        for doc in documents:
            if doc.type == "text":
                # Standard text embedding
                embedding = self.text_embedder(doc.content)
                self.text_store.add(doc.id, embedding, doc.content)
            
            elif doc.type == "image":
                # Image embedding (CLIP, etc.)
                embedding = self.image_embedder(doc.content)
                # Also generate description
                description = self.describe_image(doc.content)
                self.image_store.add(doc.id, embedding, description)
            
            elif doc.type == "table":
                # Table embedding
                table_text = self.table_to_text(doc.content)
                embedding = self.text_embedder(table_text)
                self.table_store.add(doc.id, embedding, doc.content)
    
    def query(self, question: str, image: bytes = None) -> str:
        # Handle text + image queries
        if image:
            # Multi-modal understanding
            image_context = self.describe_image(image)
            question = f"{question}\n[Image context: {image_context}]"
        
        # Search across modalities
        text_results = self.text_store.search(question)
        image_results = self.image_store.search(question)
        table_results = self.table_store.search(question)
        
        # Combine and generate
        context = self.combine_modalities(
            text_results, image_results, table_results
        )
        
        return self.generate(question, context)
```

## Architecture Selection Guide

| Use Case | Architecture |
|----------|--------------|
| Simple Q&A | Standard RAG |
| Need to verify accuracy | Self-RAG |
| Low-quality corpus | CRAG |
| Entity-rich domain | Graph RAG |
| Complex multi-step questions | Agentic RAG |
| Images, tables, mixed content | Multi-Modal RAG |

## Files

- `advanced_rag.py` - All advanced architectures

## Key Takeaways

1. Self-RAG: Decide when to retrieve, verify outputs
2. CRAG: Evaluate and correct retrieval quality
3. Graph RAG: Leverage entity relationships
4. Agentic RAG: Use tools for multi-step reasoning
5. Choose architecture based on use case complexity

## Demo Projects

With the fundamentals complete, try building:
1. **Technical Documentation Q&A** — Standard RAG + evaluation
2. **Research Assistant** — Agentic RAG with web search
3. **Enterprise Knowledge Base** — Graph RAG + access control
