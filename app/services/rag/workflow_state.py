# app/services/rag/workflow_state.py
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
from enum import Enum
import time


class QueryType(str, Enum):
    SIMPLE = "simple"
    CONVERSATIONAL = "conversational"
    COMPLEX = "complex"
    CODE_RELATED = "code_related"


class RetrievalStrategy(str, Enum):
    HYBRID = "hybrid"
    SEMANTIC_ONLY = "semantic_only"
    KEYWORD_ONLY = "keyword_only"
    MEMORY_FIRST = "memory_first"


@dataclass
class RAGState:
    """Central state object passed between all workflow nodes"""
    
    # Input
    user_query: str
    conversation_id: Optional[str] = None
    user_id: str = ""
    
    # Query Analysis
    query_type: Optional[QueryType] = None
    intent: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    
    # Memory Context
    conversation_memory: List[Dict[str, Any]] = field(default_factory=list)
    relevant_history: List[Dict[str, Any]] = field(default_factory=list)
    memory_context: str = ""  # Formatted memory context for LLM prompt
    memory_metadata: Dict[str, Any] = field(default_factory=dict)  # Memory retrieval stats
    
    # Retrieval
    retrieval_strategy: Optional[RetrievalStrategy] = None
    raw_documents: List[Dict[str, Any]] = field(default_factory=list)
    reranked_documents: List[Dict[str, Any]] = field(default_factory=list)
    
    # Generation
    context_prompt: Optional[str] = None
    response: Optional[str] = None
    model_used: Optional[str] = None
    
    # Quality & Citations
    confidence_score: Optional[float] = None
    citations: List[Dict[str, Any]] = field(default_factory=list)
    source_documents: List[Dict[str, Any]] = field(default_factory=list)
    
    # Workflow Control
    retry_count: int = 0
    fallback_triggered: bool = False
    error_message: Optional[str] = None
    
    # Metadata
    processing_time: float = 0.0
    tokens_used: Dict[str, int] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default factory fields if needed"""
        if not hasattr(self, 'entities') or self.entities is None:
            self.entities = []
        if not hasattr(self, 'conversation_memory') or self.conversation_memory is None:
            self.conversation_memory = []
        if not hasattr(self, 'relevant_history') or self.relevant_history is None:
            self.relevant_history = []
        if not hasattr(self, 'raw_documents') or self.raw_documents is None:
            self.raw_documents = []
        if not hasattr(self, 'reranked_documents') or self.reranked_documents is None:
            self.reranked_documents = []
        if not hasattr(self, 'citations') or self.citations is None:
            self.citations = []
        if not hasattr(self, 'source_documents') or self.source_documents is None:
            self.source_documents = []
        if not hasattr(self, 'tokens_used') or self.tokens_used is None:
            self.tokens_used = {}
