from typing import Dict, List, Optional, Any, Tuple, Set
import uuid
import re
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update

from app.core.config import settings
from app.core.logging import get_logger
from app.models.document import Document, DocumentChunk, DocumentStatus
from app.schemas.document import DocumentCreate, DocumentChunkCreate
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.retrieval.document_extractors import get_extractor, DocumentExtractionError

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Service for processing documents into chunks for embedding and retrieval.
    """
    
    def __init__(
        self,
        db: AsyncSession,
        embedding_service: EmbeddingService
    ):
        """
        Initialize the document processor.
        
        Args:
            db: Database session
            embedding_service: Service for generating embeddings
        """
        self.db = db
        self.embedding_service = embedding_service
    
    async def process_document(
        self,
        document_id: uuid.UUID
    ) -> Dict[str, Any]:
        """
        Process a document by chunking it and generating embeddings.
        
        Args:
            document_id: ID of the document to process
            
        Returns:
            Dictionary with processing results
        """
        # Retrieve the document
        result = await self.db.execute(select(Document).where(Document.id == document_id))
        document = result.scalars().first()
        
        if not document:
            logger.error(f"Document {document_id} not found")
            return {"status": "error", "message": f"Document {document_id} not found"}
        
        # Update document status to processing
        await self.db.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(status=DocumentStatus.PROCESSING)
        )
        await self.db.commit()
        
        try:
            # Read document content from storage
            # This would depend on where/how the document is stored
            # For simplicity, we'll assume the content is already available
            content = await self._read_document_content(document)
            
            if not content:
                logger.error(f"Failed to read content for document {document_id}")
                await self._update_document_failed(document_id, "Failed to read document content")
                return {"status": "error", "message": "Failed to read document content"}
            
            # Chunk the document
            chunks = self._chunk_document(content, document)
            
            if not chunks:
                logger.error(f"Failed to chunk document {document_id}")
                await self._update_document_failed(document_id, "Failed to chunk document")
                return {"status": "error", "message": "Failed to chunk document"}
            
            # Store the chunks
            for chunk in chunks:
                chunk_model = DocumentChunk(
                    document_id=document_id,
                    content=chunk["content"],
                    chunk_index=chunk["index"],
                    metadata=chunk.get("metadata", {}),
                    page_number=chunk.get("page_number"),
                    section_title=chunk.get("section_title")
                )
                self.db.add(chunk_model)
            
            await self.db.commit()
            
            # Generate embeddings for each chunk
            num_embedded = await self._generate_embeddings_for_chunks(document_id)
            
            # Update document status to processed
            await self.db.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(status=DocumentStatus.PROCESSED)
            )
            await self.db.commit()
            
            return {
                "status": "success",
                "document_id": str(document_id),
                "chunks_created": len(chunks),
                "chunks_embedded": num_embedded
            }
            
        except Exception as e:
            logger.error(f"Error processing document {document_id}: {str(e)}")
            await self._update_document_failed(document_id, str(e))
            return {"status": "error", "message": str(e)}
    
    async def reprocess_document(
        self,
        document_id: uuid.UUID,
        force_reembed: bool = False
    ) -> Dict[str, Any]:
        """
        Reprocess an existing document by re-chunking and optionally re-embedding.
        
        Args:
            document_id: ID of the document to reprocess
            force_reembed: Whether to regenerate embeddings even if chunks haven't changed
            
        Returns:
            Dictionary with processing results
        """
        # Similar flow to process_document, but handles existing chunks
        # First, check if document exists
        result = await self.db.execute(select(Document).where(Document.id == document_id))
        document = result.scalars().first()
        
        if not document:
            logger.error(f"Document {document_id} not found for reprocessing")
            return {"status": "error", "message": f"Document {document_id} not found"}
        
        # Update document status to processing
        await self.db.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(status=DocumentStatus.PROCESSING)
        )
        await self.db.commit()
        
        try:
            # Read document content
            content = await self._read_document_content(document)
            
            if not content:
                logger.error(f"Failed to read content for document {document_id}")
                await self._update_document_failed(document_id, "Failed to read document content")
                return {"status": "error", "message": "Failed to read document content"}
            
            # Get existing chunks
            result = await self.db.execute(
                select(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )
            existing_chunks = result.scalars().all()
            existing_chunk_dict = {chunk.chunk_index: chunk for chunk in existing_chunks}
            
            # Generate new chunks
            new_chunks = self._chunk_document(content, document)
            
            if not new_chunks:
                logger.error(f"Failed to chunk document {document_id}")
                await self._update_document_failed(document_id, "Failed to chunk document")
                return {"status": "error", "message": "Failed to chunk document"}
            
            # Compare and update chunks
            chunks_added = 0
            chunks_updated = 0
            chunks_unchanged = 0
            chunks_to_embed = set()
            
            for chunk in new_chunks:
                chunk_index = chunk["index"]
                
                if chunk_index in existing_chunk_dict:
                    # Chunk exists, check if content changed
                    existing_chunk = existing_chunk_dict[chunk_index]
                    existing_content = existing_chunk.content
                    
                    if existing_content != chunk["content"]:
                        # Content changed, update chunk
                        await self.db.execute(
                            update(DocumentChunk)
                            .where(DocumentChunk.id == existing_chunk.id)
                            .values(
                                content=chunk["content"],
                                metadata=chunk.get("metadata", {}),
                                page_number=chunk.get("page_number"),
                                section_title=chunk.get("section_title")
                            )
                        )
                        chunks_updated += 1
                        chunks_to_embed.add(existing_chunk.id)
                    else:
                        chunks_unchanged += 1
                        if force_reembed:
                            chunks_to_embed.add(existing_chunk.id)
                else:
                    # New chunk, add it
                    chunk_model = DocumentChunk(
                        document_id=document_id,
                        content=chunk["content"],
                        chunk_index=chunk_index,
                        metadata=chunk.get("metadata", {}),
                        page_number=chunk.get("page_number"),
                        section_title=chunk.get("section_title")
                    )
                    self.db.add(chunk_model)
                    chunks_added += 1
                    # We'll need to get the ID after commit
            
            # Commit changes to get IDs for new chunks
            await self.db.commit()
            
            # Get IDs of new chunks for embedding
            if chunks_added > 0:
                result = await self.db.execute(
                    select(DocumentChunk)
                    .where(DocumentChunk.document_id == document_id)
                    .where(DocumentChunk.id.notin_([chunk.id for chunk in existing_chunks]))
                )
                new_chunk_models = result.scalars().all()
                for chunk in new_chunk_models:
                    chunks_to_embed.add(chunk.id)
            
            # Generate embeddings for changed and new chunks
            num_embedded = 0
            if chunks_to_embed:
                num_embedded = await self._generate_embeddings_for_specific_chunks(list(chunks_to_embed))
            
            # Update document status to processed
            await self.db.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(status=DocumentStatus.PROCESSED)
            )
            await self.db.commit()
            
            return {
                "status": "success",
                "document_id": str(document_id),
                "chunks_added": chunks_added,
                "chunks_updated": chunks_updated,
                "chunks_unchanged": chunks_unchanged,
                "chunks_embedded": num_embedded
            }
            
        except Exception as e:
            logger.error(f"Error reprocessing document {document_id}: {str(e)}")
            await self._update_document_failed(document_id, str(e))
            return {"status": "error", "message": str(e)}
    
    async def _extract_content_with_extractor(self, document: Document) -> Optional[str]:
        """
        Extract content from a document using the unified extractor system.
        
        Args:
            document: Document model instance
            
        Returns:
            Extracted text content or None if extraction fails
        """
        try:
            import os
            
            # Enhanced debug logging for file path resolution
            logger.info(f"🔍 DEBUG: Starting extraction for document {document.id}")
            logger.info(f"🔍 DEBUG: Document storage_path: {document.storage_path}")
            logger.info(f"🔍 DEBUG: Document content_type: {document.content_type}")
            logger.info(f"🔍 DEBUG: Current working directory: {os.getcwd()}")
            
            # Check if file exists before attempting to read
            if not os.path.exists(document.storage_path):
                logger.error(f"🔍 DEBUG: File does not exist at storage_path: {document.storage_path}")
                return None
            
            file_size = os.path.getsize(document.storage_path)
            logger.info(f"🔍 DEBUG: File exists, size: {file_size} bytes")
            
            logger.debug(f"Reading document file: {document.storage_path}")
            with open(document.storage_path, "rb") as f:
                file_bytes = f.read()
            
            # Derive filename from storage path for extractor
            filename = os.path.basename(document.storage_path) if document.storage_path else document.title
            logger.info(f"🔍 DEBUG: Read {len(file_bytes)} bytes from {filename}")
            
            # Debug extractor selection
            logger.info(f"🔍 DEBUG: Getting extractor for filename: {filename}")
            extractor = get_extractor(filename, file_bytes)
            logger.info(f"🔍 DEBUG: Using extractor: {extractor.__class__.__name__}")
            
            # Attempt extraction with detailed error handling
            logger.info(f"🔍 DEBUG: Starting content extraction...")
            extracted_content = extractor.extract(file_bytes)
            
            if extracted_content:
                logger.info(f"🔍 DEBUG: Extraction successful, content length: {len(extracted_content)}")
                logger.info(f"🔍 DEBUG: Content preview: {extracted_content[:100]}...")
            else:
                logger.error(f"🔍 DEBUG: Extraction returned None/empty content")
            
            return extracted_content

        except DocumentExtractionError as e:
            logger.error(f"🔍 DEBUG: DocumentExtractionError for document {document.id}: {str(e)}")
            import traceback
            logger.error(f"🔍 DEBUG: Full traceback: {traceback.format_exc()}")
            return None
        except Exception as e:
            logger.error(f"🔍 DEBUG: Unexpected error extracting content from document {document.id}: {str(e)}")
            import traceback
            logger.error(f"🔍 DEBUG: Full traceback: {traceback.format_exc()}")
            return None
    
    async def _read_document_content(self, document: Document) -> Optional[str]:
        """
        Read the content of a document from storage.
        
        Args:
            document: Document model instance
            
        Returns:
            Document content as string or None if not found
        """
        try:
            import os
            from pathlib import Path
            
            # Check if we have a storage path
            if not document.storage_path:
                logger.error(f"Document {document.id} has no storage path")
                return None
            
            # Resolve to absolute path to handle working directory issues
            storage_path = document.storage_path
            if not os.path.isabs(storage_path):
                # Convert relative path to absolute path
                current_dir = os.getcwd()
                storage_path = os.path.join(current_dir, storage_path)
                logger.debug(f"Converted relative path to absolute: {storage_path}")
            
            # Normalize the path for cross-platform compatibility
            storage_path = os.path.normpath(storage_path)
            logger.debug(f"Normalized storage path: {storage_path}")
            
            if not os.path.exists(storage_path):
                logger.error(f"Document file not found: {storage_path} (original: {document.storage_path})")
                logger.debug(f"Current working directory: {os.getcwd()}")
                logger.debug(f"Checking if original path exists: {os.path.exists(document.storage_path)}")
                return None
            
            # Log file info for debugging
            file_size = os.path.getsize(storage_path)
            logger.debug(f"Found document file: {storage_path} ({file_size} bytes)")
            
            # Use unified extractor system for all document types
            # Temporarily update the document storage_path to the resolved path
            original_path = document.storage_path
            document.storage_path = storage_path
            
            try:
                result = await self._extract_content_with_extractor(document)
                return result
            finally:
                # Restore original path
                document.storage_path = original_path
                
        except Exception as e:
            logger.error(f"Error reading document content: {str(e)}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _normalize_text_content(self, text: str) -> str:
        """
        Normalize extracted text content for consistent processing.
        
        Args:
            text: Raw extracted text (should already be a string from extractors)
            
        Returns:
            Normalized text content
        """
        if not text:
            return ""
        
        try:
            # Text should already be a string from extractors, but check just in case
            if isinstance(text, bytes):
                logger.warning("Received bytes instead of string in _normalize_text_content - this indicates an extractor issue")
                text = text.decode('utf-8', errors='replace')
            
            # Ensure we have a string
            if not isinstance(text, str):
                logger.warning(f"Received non-string type {type(text)} in _normalize_text_content")
                text = str(text)
            
            # Normalize unicode characters
            import unicodedata
            text = unicodedata.normalize('NFKC', text)
            
            # Clean up common PDF extraction artifacts
            # Remove excessive whitespace
            import re
            text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Multiple newlines to double
            text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
            text = re.sub(r'[ \t]*\n[ \t]*', '\n', text)  # Clean line breaks
            
            # Remove common PDF artifacts
            text = re.sub(r'\x0c', '\n', text)  # Form feed to newline
            text = re.sub(r'[\x00-\x08\x0b\x0e-\x1f\x7f]', '', text)  # Control characters
            
            # Ensure text ends with newline
            text = text.strip() + '\n'
            
            return text
            
        except Exception as e:
            logger.warning(f"Error normalizing text content: {str(e)}")
            return text  # Return original if normalization fails
    
    def _chunk_document(
        self,
        content: str,
        document: Document
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document into smaller pieces for embedding.
        
        Args:
            content: Document content to chunk
            document: Document model instance
            
        Returns:
            List of chunks, each with content and metadata
        """
        # The chunking strategy depends on the document type
        # For simplicity, we'll use a basic text chunking strategy
        
        # Clean the content
        content = content.strip()
        
        # Different chunking strategies for different content types
        if document.content_type == "application/pdf":
            return self._chunk_pdf(content, document)
        elif document.content_type == "text/plain":
            return self._chunk_text(content, document)
        elif document.content_type in ["text/html", "application/xhtml+xml"]:
            return self._chunk_html(content, document)
        else:
            # Default to simple text chunking
            return self._chunk_text(content, document)
    
    def _chunk_text(
        self,
        content: str,
        document: Document,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Chunk plain text into overlapping chunks.
        
        Args:
            content: Text content to chunk
            document: Document model instance
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Split text into sentences (simplified)
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        current_chunk = ""
        current_size = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence would exceed chunk size, start a new chunk
            if current_size + sentence_len > chunk_size and current_size > 0:
                # Store the current chunk
                chunks.append({
                    "index": chunk_index,
                    "content": current_chunk.strip(),
                    "metadata": {
                        "document_id": str(document.id),
                        "document_title": document.title,
                        "source_type": document.source_type.value,
                        "chunk_index": chunk_index
                    }
                })
                
                # Start a new chunk with overlap
                overlap_point = max(0, len(current_chunk) - chunk_overlap)
                current_chunk = current_chunk[overlap_point:] + " " + sentence
                current_size = len(current_chunk)
                chunk_index += 1
            else:
                # Add to current chunk
                current_chunk += " " + sentence
                current_size += sentence_len + 1  # +1 for the space
        
        # Add the last chunk if there's anything left
        if current_chunk.strip():
            chunks.append({
                "index": chunk_index,
                "content": current_chunk.strip(),
                "metadata": {
                    "document_id": str(document.id),
                    "document_title": document.title,
                    "source_type": document.source_type.value,
                    "chunk_index": chunk_index
                }
            })
        
        return chunks
    
    def _chunk_pdf(
        self,
        content: str,
        document: Document,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Chunk PDF content, preserving page boundaries.
        
        Args:
            content: PDF content (extracted text)
            document: Document model instance
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List of chunks with metadata including page numbers
        """
        # For a real PDF processor, we'd use a library like PyPDF2 or pdfplumber
        # Here we'll assume the content is already extracted and has page markers
        
        # Split by pages (assuming content has page markers like [Page X])
        page_pattern = r'\[Page (\d+)\](.*?)(?=\[Page \d+\]|$)'
        pages = re.findall(page_pattern, content, re.DOTALL)
        
        if not pages:
            # Fallback to regular text chunking if no page markers found
            return self._chunk_text(content, document, chunk_size, chunk_overlap)
        
        chunks = []
        chunk_index = 0
        
        for page_num, page_content in pages:
            # Clean the page content
            page_content = page_content.strip()
            
            # Skip empty pages
            if not page_content:
                continue
            
            # Chunk each page separately, respecting page boundaries
            page_chunks = self._chunk_text(page_content, document, chunk_size, chunk_overlap)
            
            # Add page number to metadata
            for i, chunk in enumerate(page_chunks):
                chunk["metadata"]["page_number"] = int(page_num)
                chunk["index"] = chunk_index
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks
    
    def _chunk_html(
        self,
        content: str,
        document: Document,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Dict[str, Any]]:
        """
        Chunk HTML content, respecting section boundaries.
        
        Args:
            content: HTML content
            document: Document model instance
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            
        Returns:
            List of chunks with metadata including section titles
        """
        # For a real HTML processor, we'd use a library like BeautifulSoup
        # Here we'll do a simplified version
        
        # Strip HTML tags (very simplified)
        cleaned_content = re.sub(r'<[^>]+>', ' ', content)
        cleaned_content = re.sub(r'\s+', ' ', cleaned_content).strip()
        
        # Try to extract section titles (very simplified)
        section_pattern = r'<h[1-6][^>]*>(.*?)</h[1-6]>'
        sections = re.findall(section_pattern, content)
        
        # If no sections found, fall back to text chunking
        if not sections:
            return self._chunk_text(cleaned_content, document, chunk_size, chunk_overlap)
        
        # Split content by headings (simplified)
        # In a real implementation, use proper HTML parsing
        heading_pattern = r'<h[1-6][^>]*>.*?</h[1-6]>'
        content_parts = re.split(heading_pattern, content)
        
        # First part is before any heading
        if content_parts and content_parts[0].strip():
            intro_text = re.sub(r'<[^>]+>', ' ', content_parts[0])
            intro_text = re.sub(r'\s+', ' ', intro_text).strip()
            
            intro_chunks = self._chunk_text(intro_text, document, chunk_size, chunk_overlap)
            chunks = intro_chunks
            chunk_index = len(intro_chunks)
        else:
            chunks = []
            chunk_index = 0
        
        # Process each section
        for i, section_title in enumerate(sections):
            if i + 1 < len(content_parts):
                section_content = content_parts[i + 1]
                section_text = re.sub(r'<[^>]+>', ' ', section_content)
                section_text = re.sub(r'\s+', ' ', section_text).strip()
                
                if not section_text:
                    continue
                
                section_chunks = self._chunk_text(section_text, document, chunk_size, chunk_overlap)
                
                # Add section title to metadata
                for j, chunk in enumerate(section_chunks):
                    chunk["metadata"]["section_title"] = section_title
                    chunk["index"] = chunk_index + j
                    chunks.append(chunk)
                
                chunk_index += len(section_chunks)
        
        return chunks
    
    async def _generate_embeddings_for_chunks(self, document_id: uuid.UUID) -> int:
        """
        Generate embeddings for all chunks of a document.
        
        Args:
            document_id: ID of the document
            
        Returns:
            Number of chunks successfully embedded
        """
        # Retrieve all chunks for the document
        result = await self.db.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document_id)
        )
        chunks = result.scalars().all()
        
        if not chunks:
            logger.warning(f"No chunks found for document {document_id}")
            return 0
        
        # Generate embeddings for each chunk
        return await self._generate_embeddings_for_specific_chunks([chunk.id for chunk in chunks])
    
    async def _generate_embeddings_for_specific_chunks(self, chunk_ids: List[uuid.UUID]) -> int:
        """
        Generate embeddings for specific document chunks.
        
        Args:
            chunk_ids: List of chunk IDs to generate embeddings for
            
        Returns:
            Number of chunks successfully embedded
        """
        if not chunk_ids:
            return 0
        
        # Retrieve the chunks
        result = await self.db.execute(
            select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))
        )
        chunks = result.scalars().all()
        
        if not chunks:
            logger.warning("No chunks found with the provided IDs")
            return 0
        
        # Group chunks by content for batch processing
        contents = [chunk.content for chunk in chunks]
        
        # Generate embeddings
        try:
            # Batch generate embeddings
            embeddings = await self.embedding_service.generate_embeddings(contents)
            
            # Store the embeddings
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    # Create embedding record
                    embedding = {
                        "chunk_id": chunk.id,
                        "model_name": self.embedding_service.model_name,
                        "model_version": self.embedding_service.model_version,
                        "dimensions": len(embeddings[i]),
                        "vector": embeddings[i],
                        # Vector DB reference will be added by the embedding service
                    }
                    
                    # Get document info for metadata
                    result = await self.db.execute(select(Document).where(Document.id == chunk.document_id))
                    document = result.scalars().first()
                    
                    # Create serializable metadata from chunk
                    chunk_metadata = {
                        "chunk_id": str(chunk.id),
                        "document_id": str(chunk.document_id),
                        "document_title": document.title if document else "Untitled",
                        "source_type": document.source_type.value if document else "unknown",
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "page_number": chunk.page_number,
                        "section_title": chunk.section_title
                    }
                    
                    # Store in vector DB and database
                    await self.embedding_service.store_embedding(embedding, chunk_metadata)
            
            return len(chunks)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return 0
    
    async def _update_document_failed(self, document_id: uuid.UUID, error_message: str) -> None:
        """
        Update document status to failed with error message.
        
        Args:
            document_id: ID of the document
            error_message: Error message to store
        """
        try:
            await self.db.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(
                    status=DocumentStatus.FAILED,
                    error_message=error_message
                )
            )
            await self.db.commit()
        except Exception as e:
            logger.error(f"Error updating document status: {str(e)}")
            # Try to roll back if possible
            try:
                await self.db.rollback()
            except:
                pass
