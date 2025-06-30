from typing import Dict, List, Optional, Any, Union
import uuid
import httpx
import numpy as np
from fastapi import HTTPException

from app.core.config import settings
from app.core.logging import get_logger
from app.services.retrieval.qdrant_service import QdrantService
from app.services.embeddings.local_embedding_service import get_local_embedding_service
from app.schemas.embedding import VectorSearchQuery

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating and storing embeddings using various embedding models.
    """
    
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        qdrant_service: Optional[QdrantService] = None
    ):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the embedding model to use
            api_key: API key for the embedding provider (defaults to OpenAI key)
            api_base: Base URL for the embedding API (defaults to OpenAI base)
            qdrant_service: Service for storing embeddings in Qdrant vector database
        """
        self.model_name = model_name
        self.model_version = "1"  # Simplified, would be fetched from API in production
        
        # Configure API access based on model type
        if "text-embedding-ada" in model_name or "text-embedding-3" in model_name:
            # OpenAI embedding model
            self.api_key = api_key or settings.OPENAI_API_KEY
            self.api_base = api_base or settings.OPENAI_API_BASE
            self.provider = "openai"
        elif "e5-" in model_name or "instructor" in model_name:
            # On-prem embedding model
            self.api_key = None
            self.api_base = settings.ON_PREM_MODEL_ENDPOINT
            self.provider = "on_prem"
        else:
            # Default to OpenAI
            self.api_key = api_key or settings.OPENAI_API_KEY
            self.api_base = api_base or settings.OPENAI_API_BASE
            self.provider = "openai"
        
        # Initialize Qdrant service if not provided
        self.qdrant_service = qdrant_service
    
    async def generate_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        Prioritizes local embeddings with API fallback.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Try local embeddings first (no API key required)
        try:
            logger.debug(f"Attempting local embedding generation for {len(texts)} texts")
            local_service = get_local_embedding_service()
            embeddings = await local_service.generate_embeddings(texts)
            logger.info(f"Successfully generated {len(embeddings)} local embeddings")
            return embeddings
        except Exception as e:
            logger.warning(f"Local embedding generation failed: {e}")
            logger.info("Falling back to API-based embeddings")
        
        # Fallback to API-based embeddings
        if self.provider == "openai":
            return await self._generate_openai_embeddings(texts)
        elif self.provider == "on_prem":
            return await self._generate_on_prem_embeddings(texts)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")
    
    async def generate_embedding(
        self,
        text: str
    ) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []
    
    async def store_embedding(
        self,
        embedding: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Store an embedding in the vector database.
        
        Args:
            embedding: Embedding data including vector and metadata
            metadata: Additional metadata to store with the embedding
            
        Returns:
            Storage result from the vector database
        """
        # Prepare the vector object for storage
        vector_id = str(uuid.uuid4())
        
        vector_obj = {
            "id": embedding.get("id", vector_id),
            "values": embedding["vector"],
            "metadata": {
                "chunk_id": str(embedding["chunk_id"]),
                "model_name": embedding["model_name"],
                "model_version": embedding["model_version"],
                "dimensions": embedding["dimensions"]
            }
        }
        
        # Add additional metadata if provided, filtering out None values
        if metadata:
            # Filter out None values as Pinecone doesn't accept them
            filtered_metadata = {
                k: v for k, v in metadata.items() 
                if v is not None and v != ""
            }
            vector_obj["metadata"].update(filtered_metadata)
        
        # Store in Qdrant if service is available
        if self.qdrant_service:
            # Convert to Qdrant format
            chunks = [{
                "id": vector_obj["id"],
                "content": metadata.get("content", "") if metadata else "",
                "metadata": vector_obj["metadata"]
            }]
            result = await self.qdrant_service.upsert_documents(chunks)
            embedding["vector_db_id"] = vector_obj["id"]
            return result
        else:
            logger.warning("No Qdrant service available for embedding storage")
            return {"success": 0, "failed": 1, "errors": ["No Qdrant service available"]}
    
    async def _generate_openai_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings using OpenAI's embedding API.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        url = f"{self.api_base}/embeddings"
        
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    url,
                    json={
                        "input": texts,
                        "model": self.model_name
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"OpenAI API error: {response.text}"
                    )
                
                result = response.json()
                
                # Extract the embedding data
                data = result.get("data", [])
                
                # Sort by index to ensure correct order
                data.sort(key=lambda x: x.get("index", 0))
                
                # Extract embedding vectors
                embeddings = [item.get("embedding", []) for item in data]
                
                return embeddings
                
        except httpx.TimeoutException:
            logger.error(f"OpenAI API timeout for model {self.model_name}")
            raise HTTPException(
                status_code=504,
                detail="Request to OpenAI API timed out"
            )
        except Exception as e:
            logger.error(f"Error generating OpenAI embeddings: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating OpenAI embeddings: {str(e)}"
            )
    
    async def _generate_on_prem_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generate embeddings using on-premises embedding service.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        url = f"{self.api_base}/v1/embeddings"
        
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    url,
                    json={
                        "input": texts,
                        "model": self.model_name
                    },
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    logger.error(f"On-prem API error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"On-prem API error: {response.text}"
                    )
                
                result = response.json()
                
                # Extract the embedding data
                data = result.get("data", [])
                
                # Sort by index to ensure correct order
                data.sort(key=lambda x: x.get("index", 0))
                
                # Extract embedding vectors
                embeddings = [item.get("embedding", []) for item in data]
                
                return embeddings
                
        except httpx.TimeoutException:
            logger.error(f"On-prem API timeout for model {self.model_name}")
            raise HTTPException(
                status_code=504,
                detail="Request to on-prem embedding service timed out"
            )
        except Exception as e:
            logger.error(f"Error generating on-prem embeddings: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error generating on-prem embeddings: {str(e)}"
            )
    
    async def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search using the query embedding.
        
        Args:
            query: Query text to search for
            top_k: Number of results to return
            filters: Optional metadata filters
            
        Returns:
            List of search results with similarity scores
        """
        # Generate embedding for the query
        query_embedding = await self.generate_embedding(query)
        
        if not query_embedding:
            logger.error("Failed to generate embedding for query")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate embedding for query"
            )
        
        # Create proper VectorSearchQuery object
        search_query = VectorSearchQuery(
            query=query_embedding,
            filters=filters,
            top_k=top_k,
            include_metadata=True,
            include_vectors=False
        )
        
        try:
            if self.qdrant_service:
                # Use Qdrant for semantic search
                results = await self.qdrant_service.search_documents(
                    query=query,
                    metadata_filter=filters,
                    search_type="semantic",
                    limit=top_k,
                    include_vectors=False
                )
                # Convert to expected format
                return [
                    {
                        "id": result.id,
                        "score": result.score,
                        "content": result.content,
                        "metadata": result.metadata
                    }
                    for result in results
                ]
            else:
                logger.error("No Qdrant service available for semantic search")
                raise HTTPException(
                    status_code=500,
                    detail="No Qdrant service available for semantic search"
                )
        except Exception as e:
            logger.error(f"Error performing semantic search: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error performing semantic search: {str(e)}"
            )
