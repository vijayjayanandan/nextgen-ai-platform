# app/services/rag/rag_service.py
from typing import Dict, Any, Optional, List
from fastapi import Depends, HTTPException
from app.services.rag.workflow import ProductionRAGWorkflow
from app.services.retrieval.qdrant_service import QdrantService, get_qdrant_service
from app.services.llm.adapters.ollama import OllamaAdapter
from app.core.logging import get_logger

logger = get_logger(__name__)


class ProductionRAGService:
    """
    Production-grade RAG service that integrates with existing FastAPI endpoints.
    
    This service provides a clean interface for the RAG workflow and handles
    integration with your existing chat and retrieval endpoints.
    """
    
    def __init__(
        self,
        qdrant_service: QdrantService,
        ollama_adapter: OllamaAdapter
    ):
        self.qdrant = qdrant_service
        self.ollama = ollama_adapter
        self.workflow = ProductionRAGWorkflow(qdrant_service, ollama_adapter)
    
    async def chat_completion(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        user_id: str = "",
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Process a chat completion request using the RAG workflow.
        
        Args:
            message: User's message/question
            conversation_id: Optional conversation ID for memory
            user_id: User identifier
            stream: Whether to stream the response (not implemented yet)
            **kwargs: Additional parameters
            
        Returns:
            Chat completion response with citations and metadata
        """
        
        try:
            # Process through RAG workflow
            rag_response = await self.workflow.process_query(
                user_query=message,
                conversation_id=conversation_id,
                user_id=user_id,
                **kwargs
            )
            
            # Format as chat completion response
            return {
                "id": f"chatcmpl-{conversation_id or 'single'}",
                "object": "chat.completion",
                "model": rag_response["metadata"]["model_used"] or "rag-workflow",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": rag_response["response"]
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": len(message.split()),
                    "completion_tokens": len(rag_response["response"].split()),
                    "total_tokens": len(message.split()) + len(rag_response["response"].split())
                },
                "rag_metadata": {
                    "confidence_score": rag_response["confidence_score"],
                    "citations": rag_response["citations"],
                    "source_documents": rag_response["source_documents"],
                    "processing_time": rag_response["metadata"]["processing_time"],
                    "retrieval_strategy": rag_response["metadata"]["retrieval_strategy"],
                    "documents_used": rag_response["metadata"]["documents_used"]
                }
            }
            
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Chat completion failed: {str(e)}"
            )
    
    async def retrieve_documents(
        self,
        query: str,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        search_type: str = "hybrid"
    ) -> Dict[str, Any]:
        """
        Retrieve documents using the hybrid search capabilities.
        
        Args:
            query: Search query
            limit: Maximum number of documents to return
            filters: Optional metadata filters
            search_type: Type of search (semantic, keyword, hybrid)
            
        Returns:
            Retrieved documents with scores and metadata
        """
        
        try:
            from app.services.retrieval.qdrant_service import SearchType
            
            # Map string to SearchType enum
            search_type_enum = SearchType.HYBRID
            if search_type == "semantic":
                search_type_enum = SearchType.SEMANTIC
            elif search_type == "keyword":
                search_type_enum = SearchType.KEYWORD
            
            # Perform search
            results = await self.qdrant.search_documents(
                query=query,
                metadata_filter=filters,
                search_type=search_type_enum,
                limit=limit,
                score_threshold=0.3,
                include_vectors=False
            )
            
            # Format results
            documents = []
            for result in results:
                doc = {
                    "id": result.id,
                    "score": result.score,
                    "content": result.content,
                    "metadata": result.metadata,
                    "title": result.metadata.get("title", "Untitled"),
                    "source_type": result.metadata.get("source_type", "unknown"),
                    "document_id": result.metadata.get("document_id", ""),
                    "page_number": result.metadata.get("page_number"),
                    "section_title": result.metadata.get("section_title")
                }
                documents.append(doc)
            
            return {
                "documents": documents,
                "total_count": len(documents),
                "search_type": search_type,
                "query": query,
                "filters_applied": filters or {}
            }
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Document retrieval failed: {str(e)}"
            )
    
    async def ingest_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ingest documents into the Qdrant vector database.
        
        Args:
            documents: List of document chunks to ingest
            
        Returns:
            Ingestion results with success/failure counts
        """
        
        try:
            result = await self.qdrant.upsert_documents(documents)
            
            return {
                "success": result["success"],
                "failed": result["failed"],
                "total_processed": len(documents),
                "errors": result["errors"]
            }
            
        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Document ingestion failed: {str(e)}"
            )
    
    async def get_conversation_memory(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Retrieve conversation memory for a given conversation.
        
        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of turns to return
            
        Returns:
            Conversation memory turns
        """
        
        try:
            # Search for all memory turns for this conversation
            memory_results = await self.qdrant.search_memory(
                conversation_id=conversation_id,
                query="",  # Empty query to get all turns
                limit=limit,
                score_threshold=0.0  # Get all turns regardless of score
            )
            
            # Format memory turns
            turns = []
            for result in memory_results:
                turn = {
                    "id": result.id,
                    "user_message": result.metadata.get("user_message", ""),
                    "assistant_message": result.metadata.get("assistant_message", ""),
                    "timestamp": result.metadata.get("timestamp", 0),
                    "turn_number": result.metadata.get("turn_number", 0),
                    "metadata": result.metadata.get("metadata", {})
                }
                turns.append(turn)
            
            # Sort by turn number
            turns.sort(key=lambda x: x["turn_number"])
            
            return {
                "conversation_id": conversation_id,
                "turns": turns,
                "total_turns": len(turns)
            }
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Memory retrieval failed: {str(e)}"
            )


# Dependency injection function
async def get_rag_service(
    qdrant_service: QdrantService = Depends(get_qdrant_service)
) -> ProductionRAGService:
    """Dependency for ProductionRAGService"""
    
    # Create Ollama adapter
    ollama_adapter = OllamaAdapter()
    
    return ProductionRAGService(
        qdrant_service=qdrant_service,
        ollama_adapter=ollama_adapter
    )
