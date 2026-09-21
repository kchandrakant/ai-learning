"""
RAG Pipeline Implementation

Core retrieval-augmented generation pipeline for the domain assistant.
This module orchestrates document retrieval and LLM response generation.
"""

import logging
import hashlib
from typing import Optional
from pathlib import Path

import httpx
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..config import settings
from .prompts import PromptManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    RAG Pipeline for domain-specific question answering.
    
    Components:
    1. Document loader & chunker
    2. Embedding model (sentence-transformers)
    3. Vector store (Chroma)
    4. LLM (Ollama)
    5. Prompt manager
    """
    
    def __init__(self):
        """Initialize RAG pipeline components."""
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._vectorstore: Optional[Chroma] = None
        self._text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self._prompt_manager = PromptManager()
        self._http_client = httpx.AsyncClient(timeout=settings.llm_timeout)
        
        # Ensure data directories exist
        Path(settings.vector_store_path).mkdir(parents=True, exist_ok=True)
        Path(settings.docs_path).mkdir(parents=True, exist_ok=True)
    
    # =========================================================================
    # Lazy Initialization (avoid loading heavy models until needed)
    # =========================================================================
    
    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        """Get or create embedding model."""
        if self._embeddings is None:
            logger.info(f"Loading embedding model: {settings.embedding_model}")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=settings.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
        return self._embeddings
    
    @property
    def vectorstore(self) -> Chroma:
        """Get or create vector store."""
        if self._vectorstore is None:
            logger.info(f"Initializing vector store at: {settings.vector_store_path}")
            self._vectorstore = Chroma(
                collection_name=settings.collection_name,
                embedding_function=self.embeddings,
                persist_directory=settings.vector_store_path
            )
        return self._vectorstore
    
    @property
    def text_splitter(self) -> RecursiveCharacterTextSplitter:
        """Get or create text splitter."""
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        return self._text_splitter
    
    # =========================================================================
    # Document Indexing
    # =========================================================================
    
    async def index_document(
        self, 
        content: bytes, 
        filename: str, 
        content_type: str
    ) -> dict:
        """
        Index a document into the vector store.
        
        Args:
            content: Raw file content
            filename: Original filename
            content_type: MIME type
            
        Returns:
            dict with id and chunk_count
        """
        # Generate document ID from content hash
        doc_id = hashlib.sha256(content).hexdigest()[:16]
        
        # Extract text based on file type
        text = self._extract_text(content, filename, content_type)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split {filename} into {len(chunks)} chunks")
        
        # Create metadata for each chunk
        metadatas = [
            {
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            for i in range(len(chunks))
        ]
        
        # Add to vector store
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
        self.vectorstore.add_texts(
            texts=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        # Persist
        self.vectorstore.persist()
        
        return {"id": doc_id, "chunk_count": len(chunks)}
    
    def _extract_text(
        self, 
        content: bytes, 
        filename: str, 
        content_type: str
    ) -> str:
        """
        Extract text from document content.
        
        TODO: Implement proper extraction for different file types.
        Currently handles plain text and markdown.
        """
        extension = "." + filename.split(".")[-1].lower() if "." in filename else ""
        
        if extension in [".txt", ".md"]:
            return content.decode("utf-8", errors="ignore")
        
        elif extension == ".pdf":
            # TODO: Implement PDF extraction with pypdf2 or pdfplumber
            raise NotImplementedError(
                "PDF extraction not implemented. "
                "Install: pip install pypdf2 and implement extraction."
            )
        
        elif extension == ".docx":
            # TODO: Implement DOCX extraction with python-docx
            raise NotImplementedError(
                "DOCX extraction not implemented. "
                "Install: pip install python-docx and implement extraction."
            )
        
        else:
            # Fallback: try UTF-8 decode
            return content.decode("utf-8", errors="ignore")
    
    # =========================================================================
    # Query Pipeline
    # =========================================================================
    
    async def query(
        self,
        question: str,
        conversation_history: list[dict] = None,
        include_sources: bool = True
    ) -> dict:
        """
        Main RAG query pipeline.
        
        Args:
            question: User's question
            conversation_history: Previous conversation turns
            include_sources: Whether to return source documents
            
        Returns:
            dict with response, sources, and metadata
        """
        conversation_history = conversation_history or []
        
        # Step 1: Retrieve relevant documents
        retrieval_results = await self._retrieve(question)
        
        # Step 2: Build prompt with context
        prompt = self._prompt_manager.build_qa_prompt(
            question=question,
            context_docs=retrieval_results,
            conversation_history=conversation_history
        )
        
        # Step 3: Generate response with LLM
        response = await self._generate(prompt)
        
        # Build result
        result = {
            "response": response,
            "sources": []
        }
        
        if include_sources:
            result["sources"] = [
                {
                    "content": doc["content"],
                    "source": doc["metadata"].get("filename", "unknown"),
                    "score": doc["score"],
                    "metadata": doc["metadata"]
                }
                for doc in retrieval_results
            ]
        
        return result
    
    async def _retrieve(self, query: str) -> list[dict]:
        """
        Retrieve relevant documents from vector store.
        
        Args:
            query: Search query
            
        Returns:
            List of documents with content, metadata, and score
        """
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=settings.retrieval_top_k
        )
        
        # Filter by score threshold and format results
        retrieved = []
        for doc, score in results:
            if score >= settings.retrieval_score_threshold:
                retrieved.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
        
        logger.info(f"Retrieved {len(retrieved)} documents for query")
        return retrieved
    
    async def _generate(self, prompt: str) -> str:
        """
        Generate response using Ollama LLM.
        
        Args:
            prompt: Full prompt including context
            
        Returns:
            Generated response text
        """
        url = f"{settings.ollama_base_url}/api/generate"
        
        payload = {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.llm_temperature,
                "num_predict": settings.llm_max_tokens
            }
        }
        
        response = await self._http_client.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        return data.get("response", "")
    
    # =========================================================================
    # Document Management
    # =========================================================================
    
    async def list_documents(self) -> list[dict]:
        """List all indexed documents (unique by doc_id)."""
        # Get all documents from collection
        collection = self.vectorstore._collection
        results = collection.get()
        
        # Extract unique documents
        docs = {}
        for i, metadata in enumerate(results.get("metadatas", [])):
            if metadata:
                doc_id = metadata.get("doc_id")
                if doc_id and doc_id not in docs:
                    docs[doc_id] = {
                        "id": doc_id,
                        "filename": metadata.get("filename", "unknown"),
                        "chunk_count": metadata.get("total_chunks", 0)
                    }
        
        return list(docs.values())
    
    async def delete_document(self, document_id: str) -> dict:
        """Delete a document and all its chunks."""
        collection = self.vectorstore._collection
        
        # Find all chunks for this document
        results = collection.get(where={"doc_id": document_id})
        ids_to_delete = results.get("ids", [])
        
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            self.vectorstore.persist()
        
        return {"deleted_chunks": len(ids_to_delete)}
    
    async def get_document_count(self) -> int:
        """Get total number of unique documents."""
        docs = await self.list_documents()
        return len(docs)
    
    # =========================================================================
    # Health Checks
    # =========================================================================
    
    async def check_ollama_health(self):
        """Check Ollama connection."""
        url = f"{settings.ollama_base_url}/api/tags"
        response = await self._http_client.get(url)
        response.raise_for_status()
        return response.json()
    
    async def close(self):
        """Clean up resources."""
        await self._http_client.aclose()
