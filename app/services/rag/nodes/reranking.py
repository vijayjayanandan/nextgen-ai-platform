# app/services/rag/nodes/reranking.py
import json
from typing import List, Dict, Any
from app.services.rag.nodes.base import RAGNode
from app.services.rag.workflow_state import RAGState
from app.utils.llm_response_parser import parse_llm_json, validate_reranking_indices
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class RerankingNode(RAGNode):
    """Reranks retrieved documents based on relevance to the specific query"""
    
    def __init__(self, ollama_service):
        super().__init__("reranking")
        self.ollama = ollama_service
    
    async def execute(self, state: RAGState) -> RAGState:
        """Rerank documents using LLM-based relevance scoring"""
        
        if not state.raw_documents:
            logger.info("No documents to rerank")
            state.reranked_documents = []
            return state
        
        try:
            # If we have few documents, skip reranking
            if len(state.raw_documents) <= 5:
                state.reranked_documents = state.raw_documents
                logger.info("Skipping reranking for small document set")
                return state
            
            # Rerank using LLM
            reranked_docs = await self._llm_rerank(state)
            state.reranked_documents = reranked_docs
            
            logger.info(f"Reranked {len(state.raw_documents)} -> {len(state.reranked_documents)} documents")
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}, using original order")
            state.reranked_documents = state.raw_documents[:10]  # Fallback to top 10
        
        return state
    
    async def _llm_rerank(self, state: RAGState) -> List[Dict[str, Any]]:
        """Use LLM to rerank documents based on relevance"""
        
        # Prepare documents for reranking
        doc_summaries = []
        for i, doc in enumerate(state.raw_documents):
            summary = {
                "index": i,
                "title": doc.get("title", "Untitled"),
                "content_preview": doc.get("content", "")[:200],
                "source_type": doc.get("source_type", "unknown")
            }
            doc_summaries.append(summary)
        
        rerank_prompt = f"""Rank documents by relevance to the user query. Return ONLY a JSON array of indices.

User Query: "{state.user_query}"

Documents:
{json.dumps(doc_summaries, indent=2)}

Return ONLY the JSON array of the top 8 most relevant document indices in order:
[2, 5, 1, 8, 3, 7, 4, 6]

IMPORTANT: Return ONLY the JSON array, no explanations or markdown."""
        
        try:
            # Use configurable model for reranking
            model_name = settings.RAG_RERANKING_MODEL
            response = await self.ollama.generate(
                prompt=rerank_prompt,
                model=model_name,
                temperature=0.1,
                max_tokens=100
            )
            
            # Use robust JSON parsing
            fallback_indices = list(range(min(8, len(state.raw_documents))))
            
            parse_result = parse_llm_json(
                response=response,
                expected_type=list,
                validator=validate_reranking_indices,
                fallback_value=fallback_indices
            )
            
            if parse_result.fallback_used:
                logger.warning(f"Reranking JSON parsing used fallback: {parse_result.error_message}")
            
            ranked_indices = parse_result.data
            
            # Validate indices are within bounds
            valid_indices = [i for i in ranked_indices if isinstance(i, int) and 0 <= i < len(state.raw_documents)]
            
            # Return reranked documents
            reranked_docs = []
            for idx in valid_indices[:8]:  # Top 8 documents
                doc = state.raw_documents[idx].copy()
                doc["rerank_score"] = 1.0 - (len(reranked_docs) * 0.1)  # Decreasing score
                reranked_docs.append(doc)
            
            return reranked_docs
            
        except (KeyError, IndexError) as e:
            logger.warning(f"LLM reranking failed: {e}, using fallback")
            return self._fallback_rerank(state)
    
    def _fallback_rerank(self, state: RAGState) -> List[Dict[str, Any]]:
        """Fallback reranking using simple heuristics"""
        
        # Simple scoring based on query terms
        query_terms = set(state.user_query.lower().split())
        
        scored_docs = []
        for doc in state.raw_documents:
            content = doc.get("content", "").lower()
            title = doc.get("title", "").lower()
            
            # Count query term matches
            content_matches = sum(1 for term in query_terms if term in content)
            title_matches = sum(1 for term in query_terms if term in title) * 2  # Weight title higher
            
            # Combine with original score
            original_score = doc.get("score", 0.0)
            combined_score = original_score + (content_matches + title_matches) * 0.1
            
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = combined_score
            scored_docs.append(doc_copy)
        
        # Sort by combined score and return top 8
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_docs[:8]
