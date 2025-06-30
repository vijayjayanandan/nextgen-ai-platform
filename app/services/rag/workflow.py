# app/services/rag/workflow.py
import time
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from app.services.rag.workflow_state import RAGState
from app.services.rag.nodes import (
    QueryAnalysisNode,
    HybridRetrievalNode,
    RerankingNode,
    GenerationNode,
    CitationNode,
    EvaluationNode,
    MemoryUpdateNode
)
from app.services.rag.nodes.memory_retrieval import create_memory_retrieval_node, MemoryConfig
from app.services.retrieval.qdrant_service import QdrantService
from app.services.llm.adapters.ollama import OllamaAdapter
from app.core.logging import get_logger

logger = get_logger(__name__)


class ProductionRAGWorkflow:
    """
    Production-grade RAG workflow using LangGraph for orchestration.
    
    Implements a sophisticated RAG pipeline with:
    - Query analysis and intent detection
    - Hybrid semantic + keyword retrieval
    - Conversation memory management
    - LLM-based reranking
    - Citation extraction
    - Quality evaluation
    """
    
    def __init__(
        self,
        qdrant_service: QdrantService,
        ollama_service: OllamaAdapter
    ):
        self.qdrant = qdrant_service
        self.ollama = ollama_service
        
        # Configure memory retrieval settings
        memory_config = MemoryConfig(
            max_turns=5,
            score_threshold=0.6,
            include_recent_turns=2,
            max_context_length=2000
        )
        
        # Initialize workflow nodes
        self.nodes = {
            "query_analysis": QueryAnalysisNode(ollama_service),
            "memory_retrieval": create_memory_retrieval_node(
                qdrant_service=qdrant_service,
                max_turns=memory_config.max_turns,
                score_threshold=memory_config.score_threshold
            ),
            "hybrid_retrieval": HybridRetrievalNode(qdrant_service),
            "reranking": RerankingNode(ollama_service),
            "generation": GenerationNode(ollama_service),
            "citation": CitationNode(),
            "evaluation": EvaluationNode(),
            "memory_update": MemoryUpdateNode(qdrant_service)
        }
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the state graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        for name, node in self.nodes.items():
            workflow.add_node(name, node)
        
        # Define the workflow edges
        workflow.set_entry_point("query_analysis")
        
        # Sequential flow with conditional branching
        workflow.add_edge("query_analysis", "memory_retrieval")
        workflow.add_edge("memory_retrieval", "hybrid_retrieval")
        
        # Conditional edge: skip reranking if few documents
        workflow.add_conditional_edges(
            "hybrid_retrieval",
            self._should_rerank,
            {
                "rerank": "reranking",
                "skip_rerank": "generation"
            }
        )
        
        workflow.add_edge("reranking", "generation")
        workflow.add_edge("generation", "citation")
        workflow.add_edge("citation", "evaluation")
        
        # Conditional edge: update memory only if conversation_id exists
        workflow.add_conditional_edges(
            "evaluation",
            self._should_update_memory,
            {
                "update_memory": "memory_update",
                "skip_memory": END
            }
        )
        
        workflow.add_edge("memory_update", END)
        
        return workflow.compile()
    
    def _should_rerank(self, state: RAGState) -> str:
        """Decide whether to rerank documents"""
        
        doc_count = len(state.raw_documents)
        
        # Skip reranking if we have few documents or if retrieval failed
        if doc_count <= 5 or state.error_message:
            return "skip_rerank"
        
        return "rerank"
    
    def _should_update_memory(self, state: RAGState) -> str:
        """Decide whether to update conversation memory"""
        
        if state.conversation_id and state.response:
            return "update_memory"
        
        return "skip_memory"
    
    async def process_query(
        self,
        user_query: str,
        conversation_id: Optional[str] = None,
        user_id: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG workflow.
        
        Args:
            user_query: The user's question
            conversation_id: Optional conversation ID for memory
            user_id: User identifier for personalization
            **kwargs: Additional parameters
            
        Returns:
            Complete RAG response with citations and metadata
        """
        
        start_time = time.time()
        
        try:
            # Initialize state
            initial_state = RAGState(
                user_query=user_query,
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            # Execute the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Calculate total processing time
            processing_time = time.time() - start_time
            final_state.processing_time = processing_time
            
            # Format response
            response = self._format_response(final_state)
            
            logger.info(
                f"RAG workflow completed in {processing_time:.3f}s "
                f"(confidence: {final_state.confidence_score:.3f})"
            )
            
            return response
            
        except Exception as e:
            logger.error(f"RAG workflow failed: {e}")
            
            # Return fallback response
            return self._fallback_response(user_query, str(e), time.time() - start_time)
    
    def _format_response(self, state: RAGState) -> Dict[str, Any]:
        """Format the final workflow state into API response"""
        
        return {
            "response": state.response,
            "confidence_score": state.confidence_score,
            "citations": state.citations,
            "source_documents": state.source_documents,
            "metadata": {
                "query_type": state.query_type.value if state.query_type else None,
                "intent": state.intent,
                "entities": state.entities,
                "retrieval_strategy": state.retrieval_strategy.value if state.retrieval_strategy else None,
                "documents_retrieved": len(state.raw_documents),
                "documents_used": len(state.reranked_documents or state.raw_documents),
                "memory_turns_used": len(state.relevant_history),
                "memory_metadata": state.memory_metadata,
                "model_used": state.model_used,
                "processing_time": state.processing_time,
                "retry_count": state.retry_count,
                "fallback_triggered": state.fallback_triggered,
                "error_message": state.error_message
            }
        }
    
    def _fallback_response(self, query: str, error: str, processing_time: float) -> Dict[str, Any]:
        """Generate fallback response when workflow fails"""
        
        return {
            "response": f"I apologize, but I encountered an issue processing your question: '{query}'. Please try rephrasing your question or contact support if the issue persists.",
            "confidence_score": 0.1,
            "citations": [],
            "source_documents": [],
            "metadata": {
                "query_type": None,
                "intent": "error_fallback",
                "entities": [],
                "retrieval_strategy": None,
                "documents_retrieved": 0,
                "documents_used": 0,
                "memory_turns_used": 0,
                "model_used": None,
                "processing_time": processing_time,
                "retry_count": 0,
                "fallback_triggered": True,
                "error_message": error
            }
        }


# Factory function for dependency injection
async def create_rag_workflow(
    qdrant_service: QdrantService,
    ollama_service: OllamaAdapter
) -> ProductionRAGWorkflow:
    """Create and initialize the RAG workflow"""
    
    return ProductionRAGWorkflow(
        qdrant_service=qdrant_service,
        ollama_service=ollama_service
    )


# Alias for backward compatibility with tests
RAGWorkflow = ProductionRAGWorkflow
