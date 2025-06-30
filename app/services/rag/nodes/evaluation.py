# app/services/rag/nodes/evaluation.py
from typing import Dict, Any
from app.services.rag.nodes.base import RAGNode
from app.services.rag.workflow_state import RAGState
from app.core.logging import get_logger

logger = get_logger(__name__)


class EvaluationNode(RAGNode):
    """Evaluates response quality and confidence"""
    
    def __init__(self):
        super().__init__("evaluation")
    
    async def execute(self, state: RAGState) -> RAGState:
        """Evaluate response quality and assign confidence score"""
        
        try:
            confidence_score = self._calculate_confidence(state)
            state.confidence_score = confidence_score
            
            logger.info(f"Response confidence score: {confidence_score:.3f}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            state.confidence_score = 0.5  # Default moderate confidence
        
        return state
    
    def _calculate_confidence(self, state: RAGState) -> float:
        """Calculate confidence score based on multiple factors"""
        
        confidence_factors = []
        
        # Factor 1: Document retrieval quality
        if state.raw_documents:
            avg_retrieval_score = sum(doc.get("score", 0) for doc in state.raw_documents) / len(state.raw_documents)
            confidence_factors.append(min(avg_retrieval_score * 2, 1.0))  # Scale to 0-1
        else:
            confidence_factors.append(0.0)
        
        # Factor 2: Number of relevant documents
        doc_count = len(state.reranked_documents or state.raw_documents)
        doc_confidence = min(doc_count / 5.0, 1.0)  # Optimal around 5 documents
        confidence_factors.append(doc_confidence)
        
        # Factor 3: Citation coverage
        if state.response and state.citations:
            citation_density = len(state.citations) / max(len(state.response.split()), 1)
            citation_confidence = min(citation_density * 100, 1.0)  # Scale appropriately
            confidence_factors.append(citation_confidence)
        else:
            confidence_factors.append(0.3)  # Lower confidence without citations
        
        # Factor 4: Response length appropriateness
        if state.response:
            response_length = len(state.response)
            if 100 <= response_length <= 2000:  # Appropriate length range
                length_confidence = 1.0
            elif response_length < 50:  # Too short
                length_confidence = 0.3
            elif response_length > 3000:  # Too long
                length_confidence = 0.7
            else:
                length_confidence = 0.8
            confidence_factors.append(length_confidence)
        else:
            confidence_factors.append(0.0)
        
        # Factor 5: Query complexity handling
        if state.query_type:
            if state.query_type.value == "simple" and doc_count >= 2:
                complexity_confidence = 0.9
            elif state.query_type.value == "complex" and doc_count >= 4:
                complexity_confidence = 0.8
            elif state.query_type.value == "conversational" and state.relevant_history:
                complexity_confidence = 0.85
            else:
                complexity_confidence = 0.6
            confidence_factors.append(complexity_confidence)
        else:
            confidence_factors.append(0.5)
        
        # Calculate weighted average
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # Prioritize retrieval quality
        weighted_confidence = sum(factor * weight for factor, weight in zip(confidence_factors, weights))
        
        # Apply penalties for errors
        if state.error_message:
            weighted_confidence *= 0.7
        
        if state.fallback_triggered:
            weighted_confidence *= 0.5
        
        return max(0.0, min(1.0, weighted_confidence))
