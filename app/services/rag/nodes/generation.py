# app/services/rag/nodes/generation.py
from typing import List, Dict, Any
from app.services.rag.nodes.base import RAGNode
from app.services.rag.workflow_state import RAGState
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class GenerationNode(RAGNode):
    """Generates response using retrieved context and conversation memory"""
    
    def __init__(self, ollama_service):
        super().__init__("generation")
        self.ollama = ollama_service
    
    async def execute(self, state: RAGState) -> RAGState:
        """Generate response using RAG context"""
        
        try:
            # Build context prompt
            context_prompt = self._build_context_prompt(state)
            state.context_prompt = context_prompt
            
            # Generate response using configurable model
            model_name = settings.RAG_GENERATION_MODEL
            response = await self.ollama.generate(
                prompt=context_prompt,
                model=model_name,
                temperature=0.3,
                max_tokens=1000
            )
            
            state.response = response.strip()
            state.model_used = model_name
            
            logger.info(f"Generated response with {len(state.response)} characters")
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            state.error_message = f"Generation failed: {e}"
            state.response = self._fallback_response(state)
        
        return state
    
    def _build_context_prompt(self, state: RAGState) -> str:
        """Build the complete context prompt for generation"""
        
        prompt_parts = []
        
        # System prompt
        prompt_parts.append("""You are an expert AI assistant for Immigration, Refugees and Citizenship Canada (IRCC). 
Your role is to provide accurate, helpful information about Canadian immigration policies, procedures, and requirements.

Guidelines:
- Base your answers on the provided context documents
- If information is not in the context, clearly state this limitation
- Provide specific citations when referencing documents
- Be concise but comprehensive
- Use clear, professional language
- If asked about code or technical topics, provide practical examples""")
        
        # Add enhanced conversation memory context
        if state.memory_context:
            # Use the pre-formatted memory context from MemoryRetrievalNode
            prompt_parts.append(f"\n{state.memory_context}")
            
            # Log memory usage for monitoring
            memory_turns = state.memory_metadata.get("turns_retrieved", 0)
            avg_score = state.memory_metadata.get("avg_relevance_score", 0.0)
            logger.debug(f"Using memory context: {memory_turns} turns (avg score: {avg_score:.3f})")
            
        elif state.conversation_id:
            # Indicate memory was attempted but unavailable
            prompt_parts.append("\n## Note: Previous conversation context unavailable")
            logger.debug("Memory retrieval was attempted but no relevant context found")
        
        # Add retrieved documents
        if state.reranked_documents or state.raw_documents:
            documents = state.reranked_documents or state.raw_documents
            prompt_parts.append("\n## Context Documents:")
            
            for i, doc in enumerate(documents[:5], 1):  # Top 5 documents
                title = doc.get('title', 'Untitled Document')
                content = doc.get('content', '')[:1000]  # Limit content length
                source = doc.get('source_type', 'unknown')
                
                prompt_parts.append(f"\n[Document {i}] {title} ({source})")
                prompt_parts.append(f"Content: {content}")
        
        # Add the user query
        prompt_parts.append(f"\n## User Question:\n{state.user_query}")
        
        # Add response instructions
        prompt_parts.append("""
## Instructions:
Please provide a comprehensive answer based on the context documents above. 
Include specific citations in the format [Document X] when referencing information.
If the context doesn't contain sufficient information, clearly state this limitation.""")
        
        return "\n".join(prompt_parts)
    
    def _fallback_response(self, state: RAGState) -> str:
        """Provide fallback response when generation fails"""
        
        return f"""I apologize, but I encountered an issue generating a response to your question: "{state.user_query}"

Please try rephrasing your question or contact support if the issue persists.

Error details: {state.error_message or 'Unknown error occurred'}"""
