# app/services/rag/nodes/memory_retrieval.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from app.services.rag.nodes.base import RAGNode
from app.services.rag.workflow_state import RAGState
from app.services.retrieval.qdrant_service import QdrantService
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for memory retrieval behavior"""
    max_turns: int = 5
    score_threshold: float = 0.5
    include_recent_turns: int = 2  # Always include N most recent turns regardless of score
    merge_consecutive_turns: bool = True
    max_context_length: int = 2000  # Max characters for memory context


@dataclass
class MemoryTurn:
    """Structured representation of a conversation turn"""
    id: str
    turn_number: int
    user_message: str
    assistant_message: str
    timestamp: float
    relevance_score: float
    context_snippet: str  # Combined user + assistant for context
    
    def to_context_string(self) -> str:
        """Convert turn to formatted context string"""
        return f"[Turn {self.turn_number}] User: {self.user_message}\nAssistant: {self.assistant_message}"


class MemoryRetrievalNode(RAGNode):
    """
    Retrieves semantically relevant conversation memory from Qdrant.
    
    Input Schema (RAGState):
        - user_query: str - Current user query for semantic matching
        - conversation_id: Optional[str] - Conversation identifier
        - relevant_history: List[Dict] - Will be populated with memory turns
    
    Output Schema (RAGState):
        - relevant_history: List[MemoryTurn] - Relevant conversation turns
        - memory_context: str - Formatted memory context for LLM prompt
        - memory_metadata: Dict - Statistics about memory retrieval
    """
    
    def __init__(
        self, 
        qdrant_service: QdrantService,
        config: Optional[MemoryConfig] = None
    ):
        super().__init__("memory_retrieval")
        self.qdrant = qdrant_service
        self.config = config or MemoryConfig()
    
    async def execute(self, state: RAGState) -> RAGState:
        """
        Retrieve and format relevant conversation memory.
        
        Args:
            state: RAGState containing user_query and conversation_id
            
        Returns:
            RAGState with populated memory context and metadata
        """
        
        # Initialize memory fields
        state.relevant_history = []
        state.memory_context = ""
        state.memory_metadata = {
            "turns_retrieved": 0,
            "avg_relevance_score": 0.0,
            "memory_enabled": bool(state.conversation_id),
            "retrieval_error": None
        }
        
        # Skip if no conversation ID
        if not state.conversation_id:
            logger.debug("No conversation ID provided, skipping memory retrieval")
            return state
        
        try:
            # Retrieve relevant memory turns
            memory_turns = await self._retrieve_memory_turns(state)
            
            # Process and format memory context
            if memory_turns:
                state.relevant_history = memory_turns
                state.memory_context = self._format_memory_context(memory_turns)
                state.memory_metadata.update({
                    "turns_retrieved": len(memory_turns),
                    "avg_relevance_score": sum(turn.relevance_score for turn in memory_turns) / len(memory_turns)
                })
                
                logger.info(
                    f"Retrieved {len(memory_turns)} relevant memory turns "
                    f"(avg score: {state.memory_metadata['avg_relevance_score']:.3f})"
                )
            else:
                logger.info("No relevant memory turns found")
            
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            state.memory_metadata["retrieval_error"] = str(e)
            # Graceful fallback - don't fail the entire workflow
        
        return state
    
    async def _retrieve_memory_turns(self, state: RAGState) -> List[MemoryTurn]:
        """Retrieve and rank memory turns from Qdrant"""
        
        # Search for semantically relevant turns
        memory_results = await self.qdrant.search_memory(
            conversation_id=state.conversation_id,
            query=state.user_query,
            limit=self.config.max_turns * 2,  # Get more for filtering
            score_threshold=self.config.score_threshold
        )
        
        # Convert to MemoryTurn objects
        memory_turns = []
        for result in memory_results:
            try:
                turn = MemoryTurn(
                    id=result.id,
                    turn_number=result.metadata.get("turn_number", 0),
                    user_message=result.metadata.get("user_message", ""),
                    assistant_message=result.metadata.get("assistant_message", ""),
                    timestamp=result.metadata.get("timestamp", 0),
                    relevance_score=result.score,
                    context_snippet=result.content  # Full conversation text from Qdrant
                )
                memory_turns.append(turn)
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping malformed memory turn {result.id}: {e}")
                continue
        
        # Sort by relevance score (highest first)
        memory_turns.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Apply intelligent filtering and selection
        selected_turns = self._select_optimal_turns(memory_turns)
        
        return selected_turns[:self.config.max_turns]
    
    def _select_optimal_turns(self, memory_turns: List[MemoryTurn]) -> List[MemoryTurn]:
        """
        Intelligently select the most relevant memory turns.
        
        Strategy:
        1. Always include recent high-scoring turns
        2. Avoid redundant/similar turns
        3. Maintain chronological diversity
        4. Respect context length limits
        """
        
        if not memory_turns:
            return []
        
        selected = []
        total_context_length = 0
        
        # Sort by turn number to get recent turns
        recent_turns = sorted(memory_turns, key=lambda x: x.turn_number, reverse=True)
        
        # Always include most recent high-scoring turns
        for turn in recent_turns[:self.config.include_recent_turns]:
            if turn.relevance_score >= self.config.score_threshold:
                context_length = len(turn.to_context_string())
                if total_context_length + context_length <= self.config.max_context_length:
                    selected.append(turn)
                    total_context_length += context_length
        
        # Add additional relevant turns (avoiding duplicates)
        selected_turn_numbers = {turn.turn_number for turn in selected}
        
        for turn in memory_turns:
            if len(selected) >= self.config.max_turns:
                break
                
            if turn.turn_number not in selected_turn_numbers:
                context_length = len(turn.to_context_string())
                if total_context_length + context_length <= self.config.max_context_length:
                    selected.append(turn)
                    selected_turn_numbers.add(turn.turn_number)
                    total_context_length += context_length
        
        # Sort selected turns chronologically for context
        selected.sort(key=lambda x: x.turn_number)
        
        return selected
    
    def _format_memory_context(self, memory_turns: List[MemoryTurn]) -> str:
        """
        Format memory turns into structured context for LLM prompt.
        
        Returns formatted string ready for injection into LLM prompt.
        """
        
        if not memory_turns:
            return ""
        
        context_parts = ["## Conversation History:"]
        
        for turn in memory_turns:
            # Add turn with relevance indicator
            relevance_indicator = "🔥" if turn.relevance_score > 0.8 else "📝"
            context_parts.append(
                f"{relevance_indicator} {turn.to_context_string()}"
            )
        
        context_parts.append("")  # Empty line for separation
        
        return "\n".join(context_parts)
    
    def get_memory_summary(self, state: RAGState) -> Dict[str, Any]:
        """
        Get summary statistics about retrieved memory.
        
        Useful for debugging and monitoring memory retrieval effectiveness.
        """
        
        if not state.relevant_history:
            return {"status": "no_memory", "turns": 0}
        
        turns = state.relevant_history
        return {
            "status": "success",
            "turns": len(turns),
            "score_range": {
                "min": min(turn.relevance_score for turn in turns),
                "max": max(turn.relevance_score for turn in turns),
                "avg": sum(turn.relevance_score for turn in turns) / len(turns)
            },
            "time_range": {
                "earliest_turn": min(turn.turn_number for turn in turns),
                "latest_turn": max(turn.turn_number for turn in turns)
            },
            "context_length": len(state.memory_context),
            "conversation_id": state.conversation_id
        }


# Factory function for easy instantiation
def create_memory_retrieval_node(
    qdrant_service: QdrantService,
    max_turns: int = 5,
    score_threshold: float = 0.5
) -> MemoryRetrievalNode:
    """Create MemoryRetrievalNode with custom configuration"""
    
    config = MemoryConfig(
        max_turns=max_turns,
        score_threshold=score_threshold
    )
    
    return MemoryRetrievalNode(qdrant_service, config)
