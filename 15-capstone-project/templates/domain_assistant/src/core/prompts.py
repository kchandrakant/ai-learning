"""
Prompt Templates and Management

Centralized prompt engineering for the domain assistant.
"""

from typing import Optional


class PromptManager:
    """
    Manages prompt templates for different use cases.
    
    Design principles:
    1. Clear role definition
    2. Structured context injection
    3. Output format guidance
    4. Conversation history handling
    """
    
    # =========================================================================
    # System Prompts
    # =========================================================================
    
    SYSTEM_PROMPT = """You are a helpful domain assistant that answers questions 
based on provided context documents. 

Your responsibilities:
1. Answer questions accurately using ONLY the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite which documents/sources you used when answering
4. Be concise but thorough
5. If asked about something not in the context, acknowledge the limitation

Important guidelines:
- Do not make up information not present in the context
- If multiple sources provide conflicting information, mention this
- Use a professional, helpful tone
- Structure complex answers with bullet points or numbered lists"""

    # =========================================================================
    # QA Prompt Template
    # =========================================================================
    
    QA_PROMPT_TEMPLATE = """{system_prompt}

## Context Documents
The following documents are relevant to the user's question:

{context}

## Conversation History
{conversation_history}

## Current Question
User: {question}

## Instructions
Based on the context documents above, provide a helpful and accurate answer.
If the context doesn't contain relevant information, say "I don't have enough 
information in my knowledge base to answer this question."


Answer:"""

    # =========================================================================
    # Prompt Building Methods
    # =========================================================================
    
    def build_qa_prompt(
        self,
        question: str,
        context_docs: list[dict],
        conversation_history: list[dict] = None
    ) -> str:
        """
        Build a complete QA prompt with context and history.
        
        Args:
            question: User's current question
            context_docs: Retrieved documents with content, source, score
            conversation_history: Previous conversation turns
            
        Returns:
            Formatted prompt string
        """
        # Format context documents
        context = self._format_context(context_docs)
        
        # Format conversation history
        history = self._format_conversation_history(conversation_history or [])
        
        return self.QA_PROMPT_TEMPLATE.format(
            system_prompt=self.SYSTEM_PROMPT,
            context=context,
            conversation_history=history,
            question=question
        )
    
    def _format_context(self, docs: list[dict]) -> str:
        """Format retrieved documents for prompt injection."""
        if not docs:
            return "[No relevant documents found]"
        
        formatted_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.get("metadata", {}).get("filename", "Unknown source")
            score = doc.get("score", 0)
            content = doc.get("content", "")
            
            formatted_parts.append(
                f"### Document {i} (Source: {source}, Relevance: {score:.2f})\n"
                f"{content}"
            )
        
        return "\n\n".join(formatted_parts)
    
    def _format_conversation_history(self, history: list[dict]) -> str:
        """Format conversation history for prompt injection."""
        if not history:
            return "[No previous conversation]"
        
        formatted_parts = []
        for turn in history[-6:]:  # Keep last 6 turns (3 exchanges)
            role = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            formatted_parts.append(f"{role}: {content}")
        
        return "\n".join(formatted_parts)
    
    # =========================================================================
    # Specialized Prompts
    # =========================================================================
    
    def build_summarization_prompt(self, text: str, max_length: int = 200) -> str:
        """Build a prompt for text summarization."""
        return f"""Summarize the following text in approximately {max_length} words.
Focus on the key points and main ideas.

Text to summarize:
{text}

Summary:"""

    def build_extraction_prompt(
        self, 
        text: str, 
        fields: list[str]
    ) -> str:
        """Build a prompt for structured information extraction."""
        fields_list = "\n".join(f"- {field}" for field in fields)
        
        return f"""Extract the following information from the text below.
If a field is not found, write "Not found".

Fields to extract:
{fields_list}

Text:
{text}

Extracted information:"""

    def build_classification_prompt(
        self, 
        text: str, 
        categories: list[str]
    ) -> str:
        """Build a prompt for text classification."""
        categories_list = ", ".join(categories)
        
        return f"""Classify the following text into one of these categories: {categories_list}

Respond with ONLY the category name, nothing else.

Text:
{text}

Category:"""
