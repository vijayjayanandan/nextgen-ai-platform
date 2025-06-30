# app/services/rag/nodes/hybrid_retrieval.py
from typing import Dict, List, Any, Optional
from app.services.rag.nodes.base import RAGNode
from app.services.rag.workflow_state import RAGState, RetrievalStrategy
from app.services.retrieval.qdrant_service import QdrantService, SearchType
from app.core.logging import get_logger

logger = get_logger(__name__)


class HybridRetrievalNode(RAGNode):
    """Performs hybrid semantic + keyword retrieval from Qdrant"""
    
    def __init__(self, qdrant_service: QdrantService):
        super().__init__("hybrid_retrieval")
        self.qdrant = qdrant_service
    
    async def execute(self, state: RAGState) -> RAGState:
        """Execute hybrid retrieval combining semantic and keyword search"""
        
        try:
            # Build search filters based on query analysis
            filters = self._build_filters(state)
            
            # Determine search strategy
            search_type = self._determine_search_type(state)
            state.retrieval_strategy = RetrievalStrategy(search_type.value)
            
            # Perform search
            results = await self.qdrant.search_documents(
                query=state.user_query,
                metadata_filter=filters,
                search_type=search_type,
                limit=15,  # Get more results for reranking
                score_threshold=0.3,
                include_vectors=False
            )
            
            # Convert Qdrant results to state format
            state.raw_documents = self._format_results(results)
            
            logger.info(f"Retrieved {len(state.raw_documents)} documents using {search_type}")
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            state.error_message = f"Retrieval failed: {e}"
            state.raw_documents = []
        
        return state
    
    def _determine_search_type(self, state: RAGState) -> SearchType:
        """Determine the best search type based on query analysis"""
        
        # Use query type to determine search strategy
        if state.query_type and state.query_type.value == "code_related":
            # Code queries benefit from exact keyword matching
            return SearchType.HYBRID
        elif state.query_type and state.query_type.value == "simple":
            # Simple queries work well with semantic search
            return SearchType.SEMANTIC
        else:
            # Default to hybrid for best coverage
            return SearchType.HYBRID
    
    def _build_filters(self, state: RAGState) -> Dict[str, Any]:
        """Build Qdrant filters based on user context and query analysis"""
        
        filters = {}
        
        # Add entity-based filters
        if state.entities:
            # Filter for documents that contain any of the entities
            filters["tags"] = {"any": state.entities}
        
        # Add query type specific filters
        if state.query_type:
            if state.query_type.value == "code_related":
                filters["content_type"] = {"value": "code"}
            elif state.query_type.value == "simple":
                # Prefer policy documents for simple queries
                filters["source_type"] = {"any": ["policy", "guideline", "faq"]}
        
        # TODO: Add user permission filters based on state.user_id
        # TODO: Add recency filters for time-sensitive queries
        # TODO: Add language filters based on user preferences
        
        return filters if filters else None
    
    def _format_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """Format Qdrant search results for workflow state"""
        
        formatted_results = []
        
        for result in results:
            formatted_result = {
                "id": result.id,
                "score": result.score,
                "content": result.content,
                "metadata": result.metadata,
                "document_id": result.metadata.get("document_id", ""),
                "title": result.metadata.get("title", "Untitled"),
                "source_type": result.metadata.get("source_type", "unknown"),
                "chunk_index": result.metadata.get("chunk_index", 0),
                "page_number": result.metadata.get("page_number"),
                "section_title": result.metadata.get("section_title")
            }
            formatted_results.append(formatted_result)
        
        return formatted_results
