# app/services/retrieval/qdrant_service.py
from typing import Dict, List, Optional, Any, Union, Tuple
import uuid
import asyncio
import time
import re
from dataclasses import dataclass
from enum import Enum

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    Distance, VectorParams, CollectionStatus, PointStruct, 
    Filter, FieldCondition, MatchValue, MatchAny, Range,
    SearchRequest, ScrollRequest, UpdateResult, SparseVector
)
from qdrant_client.http.exceptions import UnexpectedResponse
from fastapi import HTTPException, Depends
import numpy as np

from app.core.config import settings
from app.core.logging import get_logger
# Removed circular import - EmbeddingService will be injected

logger = get_logger(__name__)


class SearchType(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword" 
    HYBRID = "hybrid"


@dataclass
class SearchResult:
    """Standardized search result format"""
    id: str
    score: float
    content: str
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search parameters"""
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    score_threshold: float = 0.3
    max_results: int = 20
    rerank_top_k: int = 100


class QdrantService:
    """
    Production-grade Qdrant service for semantic, keyword, and hybrid search.
    Supports document retrieval and conversation memory management.
    """
    
    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        embedding_service: Optional[Any] = None  # EmbeddingService type
    ):
        """
        Initialize Qdrant service with client and embedding service.
        
        Args:
            client: Qdrant client instance (injected via dependency)
            embedding_service: Service for generating embeddings
        """
        self.client = client or self._create_client()
        self.embedding_service = embedding_service
        
        # Collection names from settings
        self.documents_collection = getattr(settings, 'QDRANT_COLLECTION_NAME', 'documents')
        self.memory_collection = getattr(settings, 'QDRANT_MEMORY_COLLECTION', 'conversation_memory')
        
        # Search configuration
        self.hybrid_config = HybridSearchConfig()
        
        # Initialize collections on startup
        asyncio.create_task(self._ensure_collections_exist())
    
    def _create_client(self) -> QdrantClient:
        """Create Qdrant client with configuration from settings"""
        
        return QdrantClient(
            host=getattr(settings, 'QDRANT_HOST', 'localhost'),
            port=getattr(settings, 'QDRANT_PORT', 6333),
            api_key=getattr(settings, 'QDRANT_API_KEY', None),
            timeout=30.0,
            prefer_grpc=True  # Better performance for production
        )
    
    async def _ensure_collections_exist(self) -> None:
        """Ensure required collections exist with proper configuration"""
        
        try:
            # Check and create documents collection
            await self._create_collection_if_not_exists(
                collection_name=self.documents_collection,
                vector_config={
                    "content": VectorParams(
                        size=384,  # all-MiniLM-L6-v2 dimension
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "keywords": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                        )
                    )
                }
            )
            
            # Check and create memory collection
            await self._create_collection_if_not_exists(
                collection_name=self.memory_collection,
                vector_config={
                    "conversation": VectorParams(
                        size=384,
                        distance=Distance.COSINE
                    )
                }
            )
            
            logger.info("Qdrant collections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant collections: {e}")
            raise
    
    async def _create_collection_if_not_exists(
        self,
        collection_name: str,
        vector_config: Dict[str, VectorParams],
        sparse_vectors_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Create collection if it doesn't exist"""
        
        try:
            collection_info = await asyncio.to_thread(
                self.client.get_collection, collection_name
            )
            
            if collection_info.status == CollectionStatus.GREEN:
                logger.info(f"Collection '{collection_name}' already exists")
                return
                
        except UnexpectedResponse as e:
            if e.status_code == 404:
                # Collection doesn't exist, create it
                logger.info(f"Creating collection '{collection_name}'")
                
                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=collection_name,
                    vectors_config=vector_config,
                    sparse_vectors_config=sparse_vectors_config
                )
                
                logger.info(f"Collection '{collection_name}' created successfully")
            else:
                raise
    
    async def search_documents(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        search_type: SearchType = SearchType.HYBRID,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        include_vectors: bool = False
    ) -> List[SearchResult]:
        """
        Search documents using semantic, keyword, or hybrid search.
        
        Args:
            query: Search query text
            metadata_filter: Optional metadata filters
            search_type: Type of search to perform
            limit: Maximum number of results
            score_threshold: Minimum score threshold
            include_vectors: Whether to include vectors in results
            
        Returns:
            List of search results sorted by relevance score
        """
        
        if not self.embedding_service:
            raise ValueError("EmbeddingService is required for document search")
        
        start_time = time.time()
        
        try:
            if search_type == SearchType.SEMANTIC:
                results = await self._semantic_search(
                    query, metadata_filter, limit, score_threshold, include_vectors
                )
            elif search_type == SearchType.KEYWORD:
                results = await self._keyword_search(
                    query, metadata_filter, limit, score_threshold, include_vectors
                )
            elif search_type == SearchType.HYBRID:
                results = await self._hybrid_search(
                    query, metadata_filter, limit, score_threshold, include_vectors
                )
            else:
                raise ValueError(f"Unsupported search type: {search_type}")
            
            search_time = time.time() - start_time
            logger.info(
                f"Document search completed: {len(results)} results in {search_time:.3f}s "
                f"(type: {search_type}, query: '{query[:50]}...')"
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Document search failed: {str(e)}"
            )
    
    async def search_memory(
        self,
        conversation_id: str,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.5
    ) -> List[SearchResult]:
        """
        Search conversation memory for relevant context.
        
        Args:
            conversation_id: ID of the conversation
            query: Search query for relevant memory
            limit: Maximum number of memory turns to return
            score_threshold: Minimum relevance score
            
        Returns:
            List of relevant conversation turns
        """
        
        if not self.embedding_service:
            raise ValueError("EmbeddingService is required for memory search")
        
        try:
            # Generate query embedding
            query_vector = await self.embedding_service.generate_embedding(query)
            
            # Build filter for conversation
            conversation_filter = Filter(
                must=[
                    FieldCondition(
                        key="conversation_id",
                        match=MatchValue(value=conversation_id)
                    )
                ]
            )
            
            # Search memory
            search_result = await asyncio.to_thread(
                self.client.search,
                collection_name=self.memory_collection,
                query_vector=("conversation", query_vector),
                query_filter=conversation_filter,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=False
            )
            
            # Format results
            results = []
            for point in search_result:
                result = SearchResult(
                    id=str(point.id),
                    score=point.score,
                    content=point.payload.get("content", ""),
                    metadata=point.payload
                )
                results.append(result)
            
            logger.info(f"Memory search found {len(results)} relevant turns")
            return results
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return []  # Graceful fallback for memory search
    
    async def upsert_documents(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Insert or update document chunks in Qdrant.
        
        Args:
            chunks: List of document chunks with content and metadata
            
        Returns:
            Operation result with success/failure counts
        """
        
        if not chunks:
            return {"success": 0, "failed": 0, "errors": []}
        
        if not self.embedding_service:
            raise ValueError("EmbeddingService is required for document upsert")
        
        try:
            # Prepare points for upsert
            points = []
            errors = []
            
            # Extract content for batch embedding
            contents = [chunk.get("content", "") for chunk in chunks]
            
            # Generate embeddings in batch
            embeddings = await self.embedding_service.generate_embeddings(contents)
            
            for i, chunk in enumerate(chunks):
                try:
                    # Get embedding for this chunk
                    if i < len(embeddings):
                        content_embedding = embeddings[i]
                    else:
                        logger.warning(f"No embedding generated for chunk {i}")
                        continue
                    
                    # Generate sparse vector for keywords
                    sparse_vector = self._generate_sparse_vector(chunk.get("content", ""))
                    
                    # Prepare point
                    point = PointStruct(
                        id=chunk.get("id", str(uuid.uuid4())),
                        vector={
                            "content": content_embedding
                        },
                        sparse_vector={
                            "keywords": sparse_vector
                        },
                        payload=self._prepare_document_payload(chunk)
                    )
                    
                    points.append(point)
                    
                except Exception as e:
                    error_msg = f"Failed to prepare chunk {i}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            # Upsert points in batches
            batch_size = 100
            success_count = 0
            
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                
                try:
                    result = await asyncio.to_thread(
                        self.client.upsert,
                        collection_name=self.documents_collection,
                        points=batch
                    )
                    
                    if result.status == models.UpdateStatus.COMPLETED:
                        success_count += len(batch)
                    else:
                        errors.append(f"Batch {i//batch_size + 1} failed: {result.status}")
                        
                except Exception as e:
                    error_msg = f"Batch upsert failed: {str(e)}"
                    errors.append(error_msg)
                    logger.error(error_msg)
            
            logger.info(f"Document upsert completed: {success_count} success, {len(errors)} errors")
            
            return {
                "success": success_count,
                "failed": len(chunks) - success_count,
                "errors": errors
            }
            
        except Exception as e:
            logger.error(f"Document upsert failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Document upsert failed: {str(e)}"
            )
    
    async def upsert_memory_turn(
        self,
        conversation_id: str,
        turn: Dict[str, Any]
    ) -> bool:
        """
        Store a conversation turn in memory.
        
        Args:
            conversation_id: ID of the conversation
            turn: Conversation turn data
            
        Returns:
            True if successful, False otherwise
        """
        
        if not self.embedding_service:
            logger.warning("EmbeddingService not available, skipping memory storage")
            return False
        
        try:
            # Create conversation text for embedding
            user_msg = turn.get("user_message", "")
            assistant_msg = turn.get("assistant_message", "")
            conversation_text = f"User: {user_msg}\nAssistant: {assistant_msg}"
            
            # Generate embedding
            embedding = await self.embedding_service.generate_embedding(conversation_text)
            
            # Prepare point
            point = PointStruct(
                id=turn.get("id", str(uuid.uuid4())),
                vector={
                    "conversation": embedding
                },
                payload={
                    "conversation_id": conversation_id,
                    "user_message": user_msg,
                    "assistant_message": assistant_msg,
                    "timestamp": turn.get("timestamp", time.time()),
                    "turn_number": turn.get("turn_number", 0),
                    "content": conversation_text,
                    "metadata": turn.get("metadata", {})
                }
            )
            
            # Upsert point
            result = await asyncio.to_thread(
                self.client.upsert,
                collection_name=self.memory_collection,
                points=[point]
            )
            
            success = result.status == models.UpdateStatus.COMPLETED
            
            if success:
                logger.debug(f"Memory turn stored for conversation {conversation_id}")
            else:
                logger.error(f"Failed to store memory turn: {result.status}")
            
            return success
            
        except Exception as e:
            logger.error(f"Memory upsert failed: {e}")
            return False
    
    async def _semantic_search(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]],
        limit: int,
        score_threshold: Optional[float],
        include_vectors: bool
    ) -> List[SearchResult]:
        """Perform semantic search using dense vectors"""
        
        # Generate query embedding
        query_vector = await self.embedding_service.generate_embedding(query)
        
        # Build filter
        qdrant_filter = self._build_filter(metadata_filter) if metadata_filter else None
        
        # Perform search
        search_result = await asyncio.to_thread(
            self.client.search,
            collection_name=self.documents_collection,
            query_vector=("content", query_vector),
            query_filter=qdrant_filter,
            limit=limit,
            score_threshold=score_threshold or self.hybrid_config.score_threshold,
            with_payload=True,
            with_vectors=include_vectors
        )
        
        return self._format_search_results(search_result)
    
    async def _keyword_search(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]],
        limit: int,
        score_threshold: Optional[float],
        include_vectors: bool
    ) -> List[SearchResult]:
        """Perform keyword search using sparse vectors"""
        
        # Generate sparse vector for query
        sparse_vector = self._generate_sparse_vector(query)
        
        # Build filter
        qdrant_filter = self._build_filter(metadata_filter) if metadata_filter else None
        
        # Perform search using sparse vectors
        search_result = await asyncio.to_thread(
            self.client.search,
            collection_name=self.documents_collection,
            query_vector=models.NamedSparseVector(
                name="keywords",
                vector=sparse_vector
            ),
            query_filter=qdrant_filter,
            limit=limit,
            score_threshold=score_threshold or self.hybrid_config.score_threshold,
            with_payload=True,
            with_vectors=include_vectors
        )
        
        return self._format_search_results(search_result)
    
    async def _hybrid_search(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]],
        limit: int,
        score_threshold: Optional[float],
        include_vectors: bool
    ) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword search"""
        
        # Generate both dense and sparse vectors
        query_vector = await self.embedding_service.generate_embedding(query)
        sparse_vector = self._generate_sparse_vector(query)
        
        # Build filter
        qdrant_filter = self._build_filter(metadata_filter) if metadata_filter else None
        
        # Perform hybrid search using Qdrant's native hybrid search
        search_result = await asyncio.to_thread(
            self.client.search,
            collection_name=self.documents_collection,
            query_vector=("content", query_vector),
            query_filter=qdrant_filter,
            limit=self.hybrid_config.rerank_top_k,  # Get more results for fusion
            score_threshold=score_threshold or self.hybrid_config.score_threshold,
            with_payload=True,
            with_vectors=include_vectors
        )
        
        # Get keyword results
        keyword_results = await self._keyword_search(
            query, metadata_filter, self.hybrid_config.rerank_top_k, 
            score_threshold, include_vectors
        )
        
        # Fuse results using RRF (Reciprocal Rank Fusion)
        fused_results = self._fuse_results(
            semantic_results=self._format_search_results(search_result),
            keyword_results=keyword_results,
            limit=limit
        )
        
        return fused_results
    
    def _generate_sparse_vector(self, text: str) -> SparseVector:
        """
        Generate sparse vector for keyword search using simple TF-IDF-like approach.
        
        TODO: Replace with proper BM25 implementation or use external library
        """
        
        # Simple tokenization and scoring
        # In production, use proper tokenization and BM25 scoring
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Create word frequency map
        word_freq = {}
        for word in words:
            if len(word) > 2:  # Skip very short words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Convert to sparse vector format
        # This is a simplified implementation
        indices = []
        values = []
        
        for i, (word, freq) in enumerate(word_freq.items()):
            # Use hash of word as index (simplified)
            word_hash = hash(word) % 100000  # Limit vocabulary size
            indices.append(word_hash)
            values.append(float(freq))
        
        return SparseVector(
            indices=indices,
            values=values
        )
    
    def _build_filter(self, metadata_filter: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from metadata filter dictionary"""
        
        if not metadata_filter:
            return None
        
        conditions = []
        
        for key, value in metadata_filter.items():
            if isinstance(value, str):
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            elif isinstance(value, list):
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchAny(any=value)
                    )
                )
            elif isinstance(value, dict):
                # Handle range queries
                if "gte" in value or "lte" in value or "gt" in value or "lt" in value:
                    range_condition = Range()
                    if "gte" in value:
                        range_condition.gte = value["gte"]
                    if "lte" in value:
                        range_condition.lte = value["lte"]
                    if "gt" in value:
                        range_condition.gt = value["gt"]
                    if "lt" in value:
                        range_condition.lt = value["lt"]
                    
                    conditions.append(
                        FieldCondition(
                            key=key,
                            range=range_condition
                        )
                    )
                # Handle nested conditions
                elif "any" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchAny(any=value["any"])
                        )
                    )
                elif "value" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value["value"])
                        )
                    )
        
        if not conditions:
            return None
        
        return Filter(must=conditions)
    
    def _prepare_document_payload(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare document payload for Qdrant storage"""
        
        payload = {
            "content": chunk.get("content", ""),
            "document_id": str(chunk.get("document_id", "")),
            "chunk_id": str(chunk.get("id", "")),
            "chunk_index": chunk.get("chunk_index", 0),
        }
        
        # Add metadata fields
        metadata = chunk.get("metadata", {})
        for key, value in metadata.items():
            # Ensure values are JSON serializable
            if isinstance(value, (str, int, float, bool, list)):
                payload[key] = value
            elif value is not None:
                payload[key] = str(value)
        
        # Add common fields
        if "title" in chunk:
            payload["title"] = chunk["title"]
        if "source_type" in chunk:
            payload["source_type"] = chunk["source_type"]
        if "content_type" in chunk:
            payload["content_type"] = chunk["content_type"]
        if "page_number" in chunk:
            payload["page_number"] = chunk["page_number"]
        if "section_title" in chunk:
            payload["section_title"] = chunk["section_title"]
        
        return payload
    
    def _format_search_results(self, search_result: List[Any]) -> List[SearchResult]:
        """Format Qdrant search results to standardized format"""
        
        results = []
        for point in search_result:
            result = SearchResult(
                id=str(point.id),
                score=point.score,
                content=point.payload.get("content", ""),
                metadata=point.payload,
                vector=point.vector if hasattr(point, 'vector') else None
            )
            results.append(result)
        
        return results
    
    def _fuse_results(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        limit: int
    ) -> List[SearchResult]:
        """
        Fuse semantic and keyword search results using Reciprocal Rank Fusion (RRF).
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            limit: Maximum number of results to return
            
        Returns:
            Fused and ranked results
        """
        
        # RRF constant (typically 60)
        k = 60
        
        # Create score maps
        semantic_scores = {result.id: 1.0 / (k + i + 1) for i, result in enumerate(semantic_results)}
        keyword_scores = {result.id: 1.0 / (k + i + 1) for i, result in enumerate(keyword_results)}
        
        # Combine all unique results
        all_results = {}
        
        # Add semantic results
        for result in semantic_results:
            all_results[result.id] = result
        
        # Add keyword results
        for result in keyword_results:
            if result.id not in all_results:
                all_results[result.id] = result
        
        # Calculate fused scores
        fused_scores = []
        for result_id, result in all_results.items():
            semantic_score = semantic_scores.get(result_id, 0.0)
            keyword_score = keyword_scores.get(result_id, 0.0)
            
            # Weighted combination
            fused_score = (
                self.hybrid_config.semantic_weight * semantic_score +
                self.hybrid_config.keyword_weight * keyword_score
            )
            
            # Update result score
            result.score = fused_score
            fused_scores.append(result)
        
        # Sort by fused score and return top results
        fused_scores.sort(key=lambda x: x.score, reverse=True)
        return fused_scores[:limit]


# Dependency injection functions
async def get_qdrant_client() -> QdrantClient:
    """Dependency for Qdrant client"""
    return QdrantClient(
        host=getattr(settings, 'QDRANT_HOST', 'localhost'),
        port=getattr(settings, 'QDRANT_PORT', 6333),
        api_key=getattr(settings, 'QDRANT_API_KEY', None),
        timeout=30.0,
        prefer_grpc=True
    )


async def get_qdrant_service(
    client: QdrantClient = Depends(get_qdrant_client)
) -> QdrantService:
    """Dependency for QdrantService"""
    # Import here to avoid circular import
    from app.services.embeddings.embedding_service import EmbeddingService
    embedding_service = EmbeddingService()
    
    return QdrantService(
        client=client,
        embedding_service=embedding_service
    )
