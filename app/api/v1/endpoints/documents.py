from typing import Dict, List, Optional, Any
import uuid
import os
import shutil
from fastapi import APIRouter, Depends, HTTPException, status, Path, Query, UploadFile, File, Form, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.core.config import settings
from app.core.security import get_current_user, require_role
from app.core.logging import get_logger, audit_log
from app.schemas.user import UserInDB
from app.db.session import get_db
from app.models.document import Document, DocumentChunk, DocumentSourceType, DocumentStatus
from app.services.retrieval.document_processor import DocumentProcessor
from app.services.embeddings.embedding_service import EmbeddingService

router = APIRouter()
logger = get_logger(__name__)


async def process_document_background(document_id: uuid.UUID, db: AsyncSession):
    """
    Background task to process a document after upload.
    """
    # Create a new db session for background task
    async with db:
        # Initialize services
        embedding_service = EmbeddingService()
        document_processor = DocumentProcessor(db, embedding_service)
        
        # Process the document
        result = await document_processor.process_document(document_id)
        
        if result["status"] != "success":
            logger.error(f"Error processing document {document_id}: {result['message']}")


@router.post("/", response_model=Dict[str, Any])
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: str = Form(None),
    description: str = Form(None),
    source_type: DocumentSourceType = Form(DocumentSourceType.UPLOADED),
    source_id: Optional[str] = Form(None),
    source_url: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    is_public: bool = Form(False),
    security_classification: Optional[str] = Form(None),
    allowed_roles: List[str] = Form([]),
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a document for ingestion into the system.
    """
    try:
        # Create storage directory if it doesn't exist
        storage_dir = os.path.join("storage", "documents")
        os.makedirs(storage_dir, exist_ok=True)
        
        # Generate a unique filename
        file_ext = os.path.splitext(file.filename)[1]
        document_uuid = uuid.uuid4()
        filename = f"{document_uuid}{file_ext}"
        file_path = os.path.join(storage_dir, filename)
        
        # Save file to storage
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Determine content type
        content_type = file.content_type or "application/octet-stream"
        
        # If title not provided, use filename
        if not title:
            title = file.filename
        
        # Create document record
        document = Document(
            id=document_uuid,
            title=title,
            description=description,
            source_type=source_type,
            source_id=source_id,
            source_url=source_url,
            content_type=content_type,
            language=language,
            status=DocumentStatus.PENDING,
            is_public=is_public,
            security_classification=security_classification,
            allowed_roles=allowed_roles,
            storage_path=file_path
        )
        
        db.add(document)
        await db.commit()
        await db.refresh(document)
        
        # Log the upload
        audit_log(
            user_id=str(current_user.id),
            action="document_upload",
            resource_type="document",
            resource_id=str(document.id),
            details={
                "title": title,
                "source_type": source_type.value,
                "content_type": content_type,
                "file_size": os.path.getsize(file_path)
            }
        )
        
        # Schedule document processing in the background
        background_tasks.add_task(process_document_background, document.id, db)
        
        return {
            "id": str(document.id),
            "title": document.title,
            "status": document.status.value,
            "message": "Document uploaded successfully and processing started",
            "content_type": document.content_type
        }
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading document: {str(e)}"
        )


@router.get("/", response_model=List[Dict[str, Any]])
async def list_documents(
    status: Optional[DocumentStatus] = Query(None, description="Filter by document status"),
    source_type: Optional[DocumentSourceType] = Query(None, description="Filter by source type"),
    security_classification: Optional[str] = Query(None, description="Filter by security classification"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all documents.
    """
    try:
        # Build query
        query = select(Document)
        
        # Apply filters
        if status:
            query = query.filter(Document.status == status)
        if source_type:
            query = query.filter(Document.source_type == source_type)
        if security_classification:
            query = query.filter(Document.security_classification == security_classification)
        
        # Access control
        # Only show public documents or those where user has the required role
        is_admin = "admin" in current_user.roles
        if not is_admin:
            query = query.filter(
                (Document.is_public == True) | 
                (Document.allowed_roles.overlap(current_user.roles))
            )
        
        # Get total count
        result = await db.execute(query)
        all_documents = result.scalars().all()
        total_count = len(all_documents)
        
        # Apply pagination
        query = query.order_by(Document.created_at.desc()).limit(limit).offset(offset)
        
        # Execute query
        result = await db.execute(query)
        documents = result.scalars().all()
        
        # Format response
        return [
            {
                "id": str(document.id),
                "title": document.title,
                "description": document.description,
                "source_type": document.source_type.value,
                "content_type": document.content_type,
                "language": document.language,
                "status": document.status.value,
                "is_public": document.is_public,
                "security_classification": document.security_classification,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat()
            }
            for document in documents
        ]
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing documents: {str(e)}"
        )


@router.get("/{document_id}", response_model=Dict[str, Any])
async def get_document(
    document_id: str = Path(..., description="ID of the document"),
    include_chunks: bool = Query(False, description="Include document chunks"),
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a document by ID.
    """
    try:
        # Get document
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
        
        # Format response
        response = {
            "id": str(document.id),
            "title": document.title,
            "description": document.description,
            "source_type": document.source_type.value,
            "source_id": document.source_id,
            "source_url": document.source_url,
            "content_type": document.content_type,
            "language": document.language,
            "status": document.status.value,
            "error_message": document.error_message,
            "is_public": document.is_public,
            "security_classification": document.security_classification,
            "allowed_roles": document.allowed_roles,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat()
        }
        
        # Include chunks if requested
        if include_chunks:
            result = await db.execute(
                select(DocumentChunk)
                .filter(DocumentChunk.document_id == uuid_id)
                .order_by(DocumentChunk.chunk_index)
            )
            chunks = result.scalars().all()
            
            response["chunks"] = [
                {
                    "id": str(chunk.id),
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "page_number": chunk.page_number,
                    "section_title": chunk.section_title,
                    "metadata": chunk.metadata
                }
                for chunk in chunks
            ]
            
            response["chunk_count"] = len(response["chunks"])
        
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting document: {str(e)}"
        )


@router.delete("/{document_id}", response_model=Dict[str, Any])
async def delete_document(
    document_id: str = Path(..., description="ID of the document"),
    current_user: UserInDB = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a document (admin only).
    """
    try:
        # Get document
        uuid_id = uuid.UUID(document_id)
        result = await db.execute(select(Document).filter(Document.id == uuid_id))
        document = result.scalars().first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Delete the file if it exists
        if document.storage_path and os.path.exists(document.storage_path):
            os.remove(document.storage_path)
        
        # Delete the document record (cascades to chunks)
        await db.delete(document)
        await db.commit()
        
        # Log the deletion
        audit_log(
            user_id=str(current_user.id),
            action="document_delete",
            resource_type="document",
            resource_id=document_id,
            details={
                "title": document.title,
                "source_type": document.source_type.value,
                "content_type": document.content_type
            }
        )
        
        return {
            "id": document_id,
            "message": "Document deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting document: {str(e)}"
        )


@router.post("/{document_id}/reprocess", response_model=Dict[str, Any])
async def reprocess_document(
    document_id: str = Path(..., description="ID of the document"),
    force_reembed: bool = Query(False, description="Force re-embedding of all chunks"),
    current_user: UserInDB = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db)
):
    """
    Reprocess a document to update chunks and embeddings (admin only).
    """
    try:
        # Get document
        uuid_id = uuid.UUID(document_id)
        result = await db.execute(select(Document).filter(Document.id == uuid_id))
        document = result.scalars().first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Initialize services
        embedding_service = EmbeddingService()
        document_processor = DocumentProcessor(db, embedding_service)
        
        # Reprocess the document
        result = await document_processor.reprocess_document(uuid_id, force_reembed)
        
        if result["status"] != "success":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error reprocessing document: {result['message']}"
            )
        
        # Log the reprocessing
        audit_log(
            user_id=str(current_user.id),
            action="document_reprocess",
            resource_type="document",
            resource_id=document_id,
            details=result
        )
        
        return {
            "id": document_id,
            "message": "Document reprocessed successfully",
            "chunks_added": result.get("chunks_added", 0),
            "chunks_updated": result.get("chunks_updated", 0),
            "chunks_unchanged": result.get("chunks_unchanged", 0),
            "chunks_embedded": result.get("chunks_embedded", 0)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reprocessing document: {str(e)}"
        )


@router.get("/{document_id}/status", response_model=Dict[str, Any])
async def get_document_status(
    document_id: str = Path(..., description="ID of the document"),
    current_user: UserInDB = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get the processing status of a document.
    """
    try:
        # Get document
        uuid_id = uuid.UUID(document_id)
        result = await db.execute(select(Document).filter(Document.id == uuid_id))
        document = result.scalars().first()
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document {document_id} not found"
            )
        
        # Check for chunks if processed
        chunk_count = 0
        if document.status == DocumentStatus.PROCESSED:
            result = await db.execute(
                select(DocumentChunk)
                .filter(DocumentChunk.document_id == uuid_id)
            )
            chunks = result.scalars().all()
            chunk_count = len(chunks)
        
        return {
            "id": str(document.id),
            "status": document.status.value,
            "error_message": document.error_message,
            "updated_at": document.updated_at.isoformat(),
            "chunk_count": chunk_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting document status: {str(e)}"
        )