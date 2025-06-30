# app/services/rag/nodes/memory_update.py
import time
from typing import Dict, Any
from app.services.rag.nodes.base import RAGNode
from app.services.rag.workflow_state import RAGState
from app.services.retrieval.qdrant_service import QdrantService
from app.core.logging import get_logger

logger = get_logger(__name__)


class MemoryUpdateNode(RAGNode):
    """Updates conversation memory with the current turn"""
    
    def __init__(self, qdrant_service: QdrantService):
        super().__init__("memory_update")
        self.qdrant = qdrant_service
    
    async def execute(self, state: RAGState) -> RAGState:
        """Store the current conversation turn in memory"""
        
        if not state.conversation_id or not state.response:
            logger.info("Skipping memory update - missing conversation ID or response")
            return state
        
        try:
            # Prepare conversation turn data
            turn_data = {
                "user_message": state.user_query,
                "assistant_message": state.response,
                "timestamp": time.time(),
                "turn_number": len(state.conversation_memory) + 1,
                "metadata": {
                    "query_type": state.query_type.value if state.query_type else None,
                    "retrieval_strategy": state.retrieval_strategy.value if state.retrieval_strategy else None,
                    "confidence_score": state.confidence_score,
                    "documents_used": len(state.reranked_documents or state.raw_documents),
                    "citations_count": len(state.citations),
                    "model_used": state.model_used
                }
            }
            
            # Store in Qdrant memory collection
            success = await self.qdrant.upsert_memory_turn(
                conversation_id=state.conversation_id,
                turn=turn_data
            )
            
            if success:
                logger.info(f"Memory updated for conversation {state.conversation_id}")
            else:
                logger.warning(f"Failed to update memory for conversation {state.conversation_id}")
            
        except Exception as e:
            logger.error(f"Memory update failed: {e}")
            # Don't fail the entire workflow for memory update issues
        
        return state
