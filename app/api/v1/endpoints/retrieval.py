from typing import Dict, List, Optional, Any
import uuid
from fastapi import APIRouter, Depends, HTTPException, status, Path, Query, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from pydantic import BaseModel

from app.core.config import settings
from app.core.security import get_current_user
from app.core.logging import get_logger, audit_log
from app.schemas.user import UserInDB
from app.db.session import get_db
from app.models.document import Document, DocumentChunk
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.retrieval.vector_db_service import VectorDBService

router = APIRouter()
logger = get_logger(__name__)


class SearchQuery(BaseModel):
    """Model for semantic search queries."""
    query: str
    filters: Optional[Dict[str, Any]] = None
    top_k: int = 5


@router.post("/semantic-search", response_model=Dict[str, Any])
async def semantic_search(
    search_query: SearchQuery,
    include_content: bool = Query(True, description="Include chunk content in results"),
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Perform semantic search on the document corpus.
    """
    try:
        # Initialize services
        embedding_service = EmbeddingService()
        
        # Generate query embedding and search vector DB
        results = await embedding_service.semantic_search(
            query=search_query.query,
            top_k=search_query.top_k,
            filters=search_query.filters
        )
        
        # Extract document IDs from results
        document_ids = set()
        for result in results:
            # Handle both dict and object results
            if hasattr(result, 'metadata') and result.metadata:
                doc_id = result.metadata.get("document_id")
            elif isinstance(result, dict) and "metadata" in result:
                doc_id = result["metadata"].get("document_id")
            elif isinstance(result, dict) and "document_id" in result:
                doc_id = result["document_id"]
            else:
                continue
                
            if doc_id:
                try:
                    # Convert to UUID if it's a string
                    if isinstance(doc_id, str):
                        document_ids.add(uuid.UUID(doc_id))
                    elif isinstance(doc_id, uuid.UUID):
                        document_ids.add(doc_id)
                except (ValueError, TypeError):
                    continue
        
        # Fetch document metadata
        documents = {}
        if document_ids:
            for doc_id in document_ids:
                result = await db.execute(select(Document).filter(Document.id == doc_id))
                document = result.scalars().first()
                
                if document:
                    # Access control check
                    is_admin = "admin" in current_user.roles
                    if not is_admin and not document.is_public:
                        # Check if user has required role
                        user_roles = set(current_user.roles)
                        allowed_roles = set(document.allowed_roles or [])
                        
                        if not user_roles.intersection(allowed_roles):
                            # Skip this document due to access restrictions
                            continue
                    
                    documents[str(doc_id)] = {
                        "id": str(document.id),
                        "title": document.title,
                        "description": document.description,
                        "source_type": document.source_type.value,
                        "content_type": document.content_type,
                        "language": document.language
                    }
        
        # Format response
        formatted_results = []
        for result in results:
            doc_id = str(result.document_id)
            
            # Skip if document not found or access denied
            if doc_id not in documents:
                continue
                
            # Format result
            formatted_result = {
                "chunk_id": str(result.chunk_id),
                "document_id": doc_id,
                "document": documents[doc_id],
                "similarity": result.similarity,
                "metadata": result.metadata or {}
            }
            
            # Include content if requested
            if include_content:
                formatted_result["content"] = result.content
            
            formatted_results.append(formatted_result)
        
        # Log the search
        audit_log(
            user_id=str(current_user.id),
            action="semantic_search",
            resource_type="retrieval",
            resource_id="search",
            details={
                "query": search_query.query,
                "filters": search_query.filters,
                "top_k": search_query.top_k,
                "result_count": len(formatted_results)
            }
        )
        
        return {
            "query": search_query.query,
            "results": formatted_results,
            "total": len(formatted_results)
        }
    except Exception as e:
        logger.error(f"Error performing semantic search: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error performing semantic search: {str(e)}"
        )


@router.get("/document-chunks/{document_id}", response_model=List[Dict[str, Any]])
async def get_document_chunks(
    document_id: str = Path(..., description="ID of the document"),
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all chunks for a specific document.
    """
    try:
        # Get document to check access
        uuid_id = uuid.UUID(document_id)
        result = await db.execute(select(Document).filter(Document.id == uuid_id))
        document = result.scalars().first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Access control
        is_admin = "admin" in current_user.roles
        if not is_admin and not document.is_public:
            # Check if user has required role
            user_roles = set(current_user.roles)
            allowed_roles = set(document.allowed_roles or [])
            
            if not user_roles.intersection(allowed_roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: You don't have permission to view this document"
                )
        
        # Get chunks
        result = await db.execute(
            select(DocumentChunk)
            .filter(DocumentChunk.document_id == uuid_id)
            .order_by(DocumentChunk.chunk_index)
        )
        chunks = result.scalars().all()
        
        # Format response
        return [
            {
                "id": str(chunk.id),
                "document_id": str(chunk.document_id),
                "content": chunk.content,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "section_title": chunk.section_title,
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document chunks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting document chunks: {str(e)}"
        )


@router.get("/chunks/{chunk_id}", response_model=Dict[str, Any])
async def get_chunk_by_id(
    chunk_id: str = Path(..., description="ID of the chunk"),
    include_document_info: bool = Query(True, description="Include document metadata"),
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific document chunk by ID.
    """
    try:
        # Get chunk
        uuid_id = uuid.UUID(chunk_id)
        result = await db.execute(select(DocumentChunk).filter(DocumentChunk.id == uuid_id))
        chunk = result.scalars().first()
        
        if not chunk:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chunk {chunk_id} not found"
            )
        
        # Get document to check access
        result = await db.execute(select(Document).filter(Document.id == chunk.document_id))
        document = result.scalars().first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Parent document for chunk {chunk_id} not found"
            )
        
        # Access control
        is_admin = "admin" in current_user.roles
        if not is_admin and not document.is_public:
            # Check if user has required role
            user_roles = set(current_user.roles)
            allowed_roles = set(document.allowed_roles or [])
            
            if not user_roles.intersection(allowed_roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: You don't have permission to view this chunk"
                )
        
        # Format response
        response = {
            "id": str(chunk.id),
            "document_id": str(chunk.document_id),
            "content": chunk.content,
            "chunk_index": chunk.chunk_index,
            "page_number": chunk.page_number,
            "section_title": chunk.section_title,
            "metadata": chunk.metadata
        }
        
        # Include document info if requested
        if include_document_info:
            response["document"] = {
                "id": str(document.id),
                "title": document.title,
                "description": document.description,
                "source_type": document.source_type.value,
                "content_type": document.content_type,
                "language": document.language
            }
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting chunk: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting chunk: {str(e)}"
        )


class RelatedChunksQuery(BaseModel):
    """Model for finding related document chunks."""
    chunk_id: str
    top_k: int = 5
    exclude_same_document: bool = False


@router.post("/related-chunks", response_model=Dict[str, Any])
async def find_related_chunks(
    query: RelatedChunksQuery,
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Find chunks related to a specific chunk.
    """
    try:
        # Get source chunk
        try:
            uuid_id = uuid.UUID(query.chunk_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid chunk ID: {query.chunk_id}"
            )
            
        result = await db.execute(select(DocumentChunk).filter(DocumentChunk.id == uuid_id))
        chunk = result.scalars().first()
        
        if not chunk:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chunk {query.chunk_id} not found"
            )
        
        # Get document to check access
        result = await db.execute(select(Document).filter(Document.id == chunk.document_id))
        document = result.scalars().first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Parent document for chunk {query.chunk_id} not found"
            )
        
        # Access control
        is_admin = "admin" in current_user.roles
        if not is_admin and not document.is_public:
            # Check if user has required role
            user_roles = set(current_user.roles)
            allowed_roles = set(document.allowed_roles or [])
            
            if not user_roles.intersection(allowed_roles):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied: You don't have permission to view this chunk"
                )
        
        # Initialize services
        embedding_service = EmbeddingService()
        
        # Get embedding for the chunk
        # In a real implementation, we would retrieve the embedding from the DB
        # For now, we'll re-generate it
        chunk_embedding = await embedding_service.generate_embedding(chunk.content)
        
        # Prepare filters
        filters = {}
        if query.exclude_same_document:
            filters["metadata.document_id"] = {"$ne": str(chunk.document_id)}
        
        # Perform semantic search
        results = await embedding_service.vector_db_service.search_vectors({
            "query": chunk_embedding,
            "top_k": query.top_k + 1,  # +1 because source chunk might be included
            "filters": filters,
            "include_metadata": True,
            "include_vectors": False
        })
        
        # Filter out the source chunk
        filtered_results = [
            result for result in results 
            if result.chunk_id != str(chunk.id)
        ][:query.top_k]
        
        # Format results
        formatted_results = []
        for result in filtered_results:
            # Get document info
            doc_id = result.document_id
            try:
                doc_uuid = uuid.UUID(doc_id)
                result_doc = await db.execute(select(Document).filter(Document.id == doc_uuid))
                document_info = result_doc.scalars().first()
            except:
                document_info = None
            
            # Format result
            formatted_result = {
                "chunk_id": result.chunk_id,
                "document_id": result.document_id,
                "content": result.content,
                "similarity": result.similarity,
                "metadata": result.metadata
            }
            
            # Add document info if available
            if document_info:
                formatted_result["document"] = {
                    "id": str(document_info.id),
                    "title": document_info.title,
                    "source_type": document_info.source_type.value,
                    "content_type": document_info.content_type
                }
            
            formatted_results.append(formatted_result)
        
        return {
            "source_chunk_id": query.chunk_id,
            "results": formatted_results,
            "total": len(formatted_results)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding related chunks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error finding related chunks: {str(e)}"
        )
