"""
Citation formatting utilities for RAG responses.

This module provides functions to format document citations from retrieved chunks,
ensuring proper attribution and transparency in AI-generated responses.
"""

from typing import List, Dict, Any, Set, Tuple
from app.core.logging import get_logger

logger = get_logger(__name__)


def format_citations(chunks: List[Dict[str, Any]]) -> str:
    """
    Format citations from retrieved document chunks.
    
    Args:
        chunks: List of document chunks with metadata
        
    Returns:
        Formatted citation string with deduplicated sources
    """
    if not chunks:
        return ""
    
    try:
        # Extract and deduplicate sources
        unique_sources = _extract_unique_sources(chunks)
        
        if not unique_sources:
            logger.debug("No valid sources found in chunks for citation")
            return ""
        
        # Sort sources for consistent output
        sorted_sources = _sort_sources(unique_sources)
        
        # Format each source
        formatted_citations = []
        for source in sorted_sources:
            citation = _format_single_source(source)
            if citation:
                formatted_citations.append(citation)
        
        if not formatted_citations:
            return ""
        
        # Join with newlines for readability
        return "\n".join(formatted_citations)
        
    except Exception as e:
        logger.error(f"Error formatting citations: {e}")
        return ""


def _extract_unique_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract unique sources from chunks, deduplicating by document title, page, and section.
    
    Args:
        chunks: List of document chunks
        
    Returns:
        List of unique source dictionaries
    """
    seen_sources: Set[Tuple[str, str, str]] = set()
    unique_sources = []
    
    for chunk in chunks:
        try:
            # Extract metadata - handle both direct fields and nested metadata
            metadata = chunk.get('metadata', {})
            
            # Get document title from multiple possible locations
            document_title = (
                chunk.get('title') or 
                metadata.get('document_title') or 
                metadata.get('title') or
                'Untitled Document'
            )
            
            # Get page number (can be None)
            page_number = (
                chunk.get('page_number') or 
                metadata.get('page_number')
            )
            
            # Get section title (can be None)
            section_title = (
                chunk.get('section_title') or 
                metadata.get('section_title')
            )
            
            # Create deduplication key
            page_str = str(page_number) if page_number is not None else ""
            section_str = section_title or ""
            dedup_key = (document_title, page_str, section_str)
            
            # Skip if we've already seen this exact source
            if dedup_key in seen_sources:
                continue
            
            seen_sources.add(dedup_key)
            
            # Store source information
            source = {
                'document_title': document_title,
                'page_number': page_number,
                'section_title': section_title,
                'source_type': chunk.get('source_type') or metadata.get('source_type', 'document'),
                'document_id': chunk.get('document_id') or metadata.get('document_id', ''),
                'chunk_index': chunk.get('chunk_index') or metadata.get('chunk_index', 0)
            }
            
            unique_sources.append(source)
            
        except Exception as e:
            logger.warning(f"Error processing chunk for citation: {e}")
            continue
    
    return unique_sources


def _sort_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort sources by document title and then by page number.
    
    Args:
        sources: List of source dictionaries
        
    Returns:
        Sorted list of sources
    """
    def sort_key(source):
        document_title = source.get('document_title', 'Untitled Document')
        page_number = source.get('page_number')
        
        # Handle page number sorting - None values go to end
        if page_number is None:
            page_sort_key = float('inf')
        else:
            try:
                page_sort_key = int(page_number)
            except (ValueError, TypeError):
                page_sort_key = float('inf')
        
        return (document_title.lower(), page_sort_key)
    
    return sorted(sources, key=sort_key)


def _format_single_source(source: Dict[str, Any]) -> str:
    """
    Format a single source into a citation string.
    
    Args:
        source: Source dictionary with metadata
        
    Returns:
        Formatted citation string
    """
    try:
        document_title = source.get('document_title', 'Untitled Document')
        page_number = source.get('page_number')
        section_title = source.get('section_title')
        
        # Build citation components
        citation_parts = [f"📄 {document_title}"]
        
        # Add page and section information
        location_parts = []
        
        if page_number is not None:
            try:
                # Ensure page number is valid
                page_int = int(page_number)
                location_parts.append(f"page {page_int}")
            except (ValueError, TypeError):
                # Skip invalid page numbers
                pass
        
        if section_title and section_title.strip():
            location_parts.append(f"section: {section_title.strip()}")
        
        # Combine location information
        if location_parts:
            location_str = ", ".join(location_parts)
            citation_parts.append(f"({location_str})")
        
        return " ".join(citation_parts)
        
    except Exception as e:
        logger.warning(f"Error formatting single source: {e}")
        return ""


def append_citations_to_response(llm_response: str, chunks: List[Dict[str, Any]]) -> str:
    """
    Append formatted citations to an LLM response.
    
    Args:
        llm_response: The original LLM response
        chunks: List of document chunks used for the response
        
    Returns:
        Response with citations appended, or original response if no citations
    """
    if not llm_response:
        return llm_response
    
    try:
        formatted_citations = format_citations(chunks)
        
        if not formatted_citations:
            # No valid citations found, return original response
            return llm_response.strip()
        
        # Append citations section
        final_response = f"{llm_response.strip()}\n\nSources:\n{formatted_citations}"
        
        logger.debug(f"Appended {len(chunks)} chunks as citations to response")
        return final_response
        
    except Exception as e:
        logger.error(f"Error appending citations to response: {e}")
        # Return original response on error
        return llm_response.strip()


# Example usage and testing functions
def _example_usage():
    """
    Example usage of the citation formatting functions.
    """
    
    # Example chunks with various metadata structures
    example_chunks = [
        {
            "content": "Canadian citizens must meet residency requirements...",
            "title": "Canadian Immigration Guide 2024",
            "page_number": 15,
            "section_title": "Citizenship Requirements",
            "source_type": "policy",
            "metadata": {
                "document_id": "doc-123",
                "chunk_index": 5
            }
        },
        {
            "content": "Language proficiency tests include CELPIP...",
            "metadata": {
                "document_title": "Canadian Immigration Guide 2024", 
                "page_number": 16,
                "section_title": "Language Requirements",
                "source_type": "policy",
                "document_id": "doc-123",
                "chunk_index": 6
            }
        },
        {
            "content": "Processing times vary by application type...",
            "title": "IRCC Processing Times",
            "metadata": {
                "section_title": "Current Wait Times",
                "source_type": "guideline",
                "document_id": "doc-456"
            }
        },
        {
            "content": "Duplicate content for testing deduplication...",
            "title": "Canadian Immigration Guide 2024",
            "page_number": 15,
            "section_title": "Citizenship Requirements",
            "source_type": "policy"
        }
    ]
    
    # Format citations
    citations = format_citations(example_chunks)
    print("Formatted Citations:")
    print(citations)
    print()
    
    # Example response with citations
    llm_response = """To become a Canadian citizen, you must meet several key requirements:

1. **Residency**: You must have been physically present in Canada for at least 1,095 days (3 years) during the 5 years immediately before applying.

2. **Language Proficiency**: Demonstrate adequate knowledge of English or French through approved tests like CELPIP or IELTS.

3. **Tax Obligations**: File income tax returns for at least 3 years during the 5-year period if required.

Processing times for citizenship applications vary depending on the type of application and current workload."""
    
    # Append citations
    final_response = append_citations_to_response(llm_response, example_chunks)
    print("Final Response with Citations:")
    print(final_response)


if __name__ == "__main__":
    _example_usage()
