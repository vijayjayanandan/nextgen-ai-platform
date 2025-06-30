"""
Citation Node for RAG workflow.

This node extracts and formats citations from retrieved documents,
appending them to the generated response for transparency and auditability.
"""

from typing import List, Dict, Any
from app.services.rag.nodes.base import RAGNode
from app.services.rag.workflow_state import RAGState
from app.utils.citation_formatter import append_citations_to_response, format_citations
from app.core.logging import get_logger

logger = get_logger(__name__)


class CitationNode(RAGNode):
    """Extracts citations from retrieved documents and appends them to the response"""
    
    def __init__(self):
        super().__init__("citation")
    
    async def execute(self, state: RAGState) -> RAGState:
        """Extract citations and append to response"""
        
        try:
            # Skip citation processing if no response was generated
            if not state.response:
                logger.debug("No response generated, skipping citation processing")
                return state
            
            # Get the documents that were used for generation
            documents_used = state.reranked_documents or state.raw_documents
            
            if not documents_used:
                logger.debug("No documents available for citation")
                return state
            
            # Append citations to the response
            state.response = append_citations_to_response(state.response, documents_used)
            
            # Store citation metadata for API response
            state.citations = self._extract_citation_metadata(documents_used)
            state.source_documents = self._extract_source_documents(documents_used)
            
            logger.info(f"Added {len(state.citations)} citations to response")
            
        except Exception as e:
            logger.error(f"Citation processing failed: {e}")
            # Don't fail the entire workflow if citation processing fails
            # Just log the error and continue with the original response
            state.error_message = f"Citation processing failed: {e}"
        
        return state
    
    def _extract_citation_metadata(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract citation metadata for API response.
        
        Args:
            documents: List of document chunks
            
        Returns:
            List of citation metadata dictionaries
        """
        citations = []
        
        try:
            # Use the same deduplication logic as the formatter
            from app.utils.citation_formatter import _extract_unique_sources, _sort_sources
            
            unique_sources = _extract_unique_sources(documents)
            sorted_sources = _sort_sources(unique_sources)
            
            for i, source in enumerate(sorted_sources, 1):
                citation = {
                    "id": i,
                    "document_title": source.get('document_title', 'Untitled Document'),
                    "document_id": source.get('document_id', ''),
                    "page_number": source.get('page_number'),
                    "section_title": source.get('section_title'),
                    "source_type": source.get('source_type', 'document'),
                    "chunk_index": source.get('chunk_index', 0)
                }
                citations.append(citation)
                
        except Exception as e:
            logger.error(f"Error extracting citation metadata: {e}")
        
        return citations
    
    def _extract_source_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract source document information for API response.
        
        Args:
            documents: List of document chunks
            
        Returns:
            List of source document dictionaries
        """
        source_docs = []
        
        try:
            # Group chunks by document
            doc_groups = {}
            
            for doc in documents:
                doc_id = doc.get('document_id') or doc.get('metadata', {}).get('document_id', 'unknown')
                
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = {
                        "document_id": doc_id,
                        "title": doc.get('title') or doc.get('metadata', {}).get('document_title', 'Untitled'),
                        "source_type": doc.get('source_type') or doc.get('metadata', {}).get('source_type', 'document'),
                        "chunks_used": [],
                        "relevance_score": 0.0
                    }
                
                # Add chunk information
                chunk_info = {
                    "chunk_id": doc.get('id', ''),
                    "chunk_index": doc.get('chunk_index') or doc.get('metadata', {}).get('chunk_index', 0),
                    "page_number": doc.get('page_number') or doc.get('metadata', {}).get('page_number'),
                    "section_title": doc.get('section_title') or doc.get('metadata', {}).get('section_title'),
                    "score": doc.get('score', 0.0),
                    "content_preview": doc.get('content', '')[:200] + "..." if len(doc.get('content', '')) > 200 else doc.get('content', '')
                }
                
                doc_groups[doc_id]["chunks_used"].append(chunk_info)
                
                # Update relevance score (use highest chunk score)
                chunk_score = doc.get('score', 0.0)
                if chunk_score > doc_groups[doc_id]["relevance_score"]:
                    doc_groups[doc_id]["relevance_score"] = chunk_score
            
            # Convert to list and sort by relevance
            source_docs = list(doc_groups.values())
            source_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
            
        except Exception as e:
            logger.error(f"Error extracting source documents: {e}")
        
        return source_docs
