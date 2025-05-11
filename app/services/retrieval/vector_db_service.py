from typing import Dict, List, Optional, Any, Union, Tuple
import uuid
import httpx
import json
from fastapi import HTTPException

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.embedding import VectorSearchQuery, VectorSearchResult

logger = get_logger(__name__)


class VectorDBService:
    """
    Service for interacting with vector databases for semantic search.
    Currently supports Pinecone and Weaviate.
    """
    
    def __init__(
        self,
        db_type: Optional[str] = None,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        namespace: Optional[str] = None
    ):
        """
        Initialize the vector database service.
        
        Args:
            db_type: Vector database type ("pinecone" or "weaviate")
            api_key: API key for the vector database
            endpoint: Endpoint URL for the vector database
            namespace: Namespace or collection name for vectors
        """
        self.db_type = db_type or settings.VECTOR_DB_TYPE
        self.api_key = api_key or settings.VECTOR_DB_API_KEY
        self.endpoint = endpoint or settings.VECTOR_DB_URI
        self.namespace = namespace or settings.VECTOR_DB_NAMESPACE
        
        if not self.api_key:
            logger.error("Vector database API key not provided")
            raise ValueError("Vector database API key is required")
            
        if not self.endpoint:
            logger.error("Vector database endpoint not provided")
            raise ValueError("Vector database endpoint is required")
    
    async def upsert_vectors(
        self,
        vectors: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Insert or update vectors in the vector database.
        
        Args:
            vectors: List of vector objects to upsert
                Each object should contain:
                - id: Unique identifier
                - values: Vector values
                - metadata: Additional metadata
                
        Returns:
            Response from the vector database
        """
        if self.db_type == "pinecone":
            return await self._pinecone_upsert(vectors)
        elif self.db_type == "weaviate":
            return await self._weaviate_upsert(vectors)
        else:
            raise ValueError(f"Unsupported vector database type: {self.db_type}")
    
    async def search_vectors(
        self,
        query: VectorSearchQuery
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors in the vector database.
        
        Args:
            query: Search query containing:
                - query: Query text or vector
                - filters: Optional metadata filters
                - top_k: Number of results to return
                - include_metadata: Whether to include metadata in results
                - include_vectors: Whether to include vector values in results
                
        Returns:
            List of search results with similarity scores
        """
        if self.db_type == "pinecone":
            return await self._pinecone_search(query)
        elif self.db_type == "weaviate":
            return await self._weaviate_search(query)
        else:
            raise ValueError(f"Unsupported vector database type: {self.db_type}")
    
    async def delete_vectors(
        self,
        ids: List[str],
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delete vectors from the vector database.
        
        Args:
            ids: List of vector IDs to delete
            filter: Optional metadata filter for bulk deletion
                
        Returns:
            Response from the vector database
        """
        if self.db_type == "pinecone":
            return await self._pinecone_delete(ids, filter)
        elif self.db_type == "weaviate":
            return await self._weaviate_delete(ids, filter)
        else:
            raise ValueError(f"Unsupported vector database type: {self.db_type}")
    
    async def _pinecone_upsert(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Insert or update vectors in Pinecone.
        """
        url = f"{self.endpoint}/vectors/upsert"
        
        # Format vectors for Pinecone
        pinecone_vectors = []
        for vec in vectors:
            pinecone_vectors.append({
                "id": vec["id"],
                "values": vec["values"],
                "metadata": vec.get("metadata", {})
            })
        
        payload = {
            "vectors": pinecone_vectors,
            "namespace": self.namespace
        }
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Api-Key": self.api_key,
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Pinecone API error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Pinecone API error: {response.text}"
                    )
                
                return response.json()
                
        except httpx.TimeoutException:
            logger.error("Pinecone API timeout")
            raise HTTPException(
                status_code=504,
                detail="Request to Pinecone API timed out"
            )
        except Exception as e:
            logger.error(f"Error calling Pinecone API: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error calling Pinecone API: {str(e)}"
            )
    
    async def _pinecone_search(self, query: VectorSearchQuery) -> List[VectorSearchResult]:
        """
        Search for similar vectors in Pinecone.
        """
        url = f"{self.endpoint}/query"
        
        # Need to convert query text to vector first
        # This would typically be done using an embedding service
        # For now, we'll assume query.query is already a vector
        
        payload = {
            "vector": query.query if isinstance(query.query, list) else [],
            "namespace": self.namespace,
            "topK": query.top_k,
            "includeMetadata": query.include_metadata,
            "includeValues": query.include_vectors
        }
        
        if query.filters:
            payload["filter"] = query.filters
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Api-Key": self.api_key,
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Pinecone API error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Pinecone API error: {response.text}"
                    )
                
                result = response.json()
                
                # Map Pinecone response to our schema
                search_results = []
                for match in result.get("matches", []):
                    # Extract document_id and content from metadata
                    metadata = match.get("metadata", {})
                    document_id = metadata.get("document_id", str(uuid.uuid4()))
                    content = metadata.get("content", "")
                    
                    search_result = VectorSearchResult(
                        chunk_id=match["id"],
                        document_id=document_id,
                        content=content,
                        similarity=match["score"],
                        metadata=metadata,
                        vector=match.get("values") if query.include_vectors else None
                    )
                    search_results.append(search_result)
                
                return search_results
                
        except httpx.TimeoutException:
            logger.error("Pinecone API timeout")
            raise HTTPException(
                status_code=504,
                detail="Request to Pinecone API timed out"
            )
        except Exception as e:
            logger.error(f"Error calling Pinecone API: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error calling Pinecone API: {str(e)}"
            )
    
    async def _pinecone_delete(
        self,
        ids: List[str],
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delete vectors from Pinecone.
        """
        url = f"{self.endpoint}/vectors/delete"
        
        payload = {
            "namespace": self.namespace
        }
        
        if ids:
            payload["ids"] = ids
            
        if filter:
            payload["filter"] = filter
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Api-Key": self.api_key,
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Pinecone API error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Pinecone API error: {response.text}"
                    )
                
                return response.json()
                
        except httpx.TimeoutException:
            logger.error("Pinecone API timeout")
            raise HTTPException(
                status_code=504,
                detail="Request to Pinecone API timed out"
            )
        except Exception as e:
            logger.error(f"Error calling Pinecone API: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error calling Pinecone API: {str(e)}"
            )
    
    async def _weaviate_upsert(self, vectors: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Insert or update vectors in Weaviate.
        """
        # Weaviate uses a different API structure
        # We'll batch the objects
        responses = {}
        
        for vec in vectors:
            object_uuid = vec["id"]
            url = f"{self.endpoint}/v1/objects/{object_uuid}"
            
            # Prepare the object
            weaviate_obj = {
                "class": self.namespace,
                "properties": vec.get("metadata", {}),
                "vector": vec["values"]
            }
            
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.put(
                        url,
                        json=weaviate_obj,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        }
                    )
                    
                    if response.status_code not in [200, 201]:
                        logger.error(f"Weaviate API error: {response.status_code} - {response.text}")
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Weaviate API error: {response.text}"
                        )
                    
                    responses[object_uuid] = response.json()
                    
            except httpx.TimeoutException:
                logger.error("Weaviate API timeout")
                raise HTTPException(
                    status_code=504,
                    detail="Request to Weaviate API timed out"
                )
            except Exception as e:
                logger.error(f"Error calling Weaviate API: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error calling Weaviate API: {str(e)}"
                )
        
        return {"status": "success", "objects": len(vectors), "responses": responses}
    
    async def _weaviate_search(self, query: VectorSearchQuery) -> List[VectorSearchResult]:
        """
        Search for similar vectors in Weaviate.
        """
        # Assume query.query is already a vector
        # For GraphQL query to Weaviate
        url = f"{self.endpoint}/v1/graphql"
        
        # Build Weaviate GraphQL query
        properties = ["document_id", "content"]
        
        # Add all metadata fields if include_metadata is True
        if query.include_metadata:
            properties.extend(["metadata"])
        
        # GraphQL query
        graphql_query = {
            "query": f"""
            {{
                Get {{
                    {self.namespace}(
                        nearVector: {{
                            vector: {json.dumps(query.query if isinstance(query.query, list) else [])}
                            certainty: 0.7
                        }}
                        limit: {query.top_k}
                    ) {{
                        _additional {{
                            id
                            certainty
                            vector
                        }}
                        {" ".join(properties)}
                    }}
                }}
            }}
            """
        }
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    url,
                    json=graphql_query,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code != 200:
                    logger.error(f"Weaviate API error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Weaviate API error: {response.text}"
                    )
                
                result = response.json()
                
                # Map Weaviate response to our schema
                search_results = []
                
                if (
                    "data" in result 
                    and "Get" in result["data"] 
                    and self.namespace in result["data"]["Get"]
                ):
                    matches = result["data"]["Get"][self.namespace]
                    
                    for match in matches:
                        additional = match.get("_additional", {})
                        
                        search_result = VectorSearchResult(
                            chunk_id=additional.get("id", str(uuid.uuid4())),
                            document_id=match.get("document_id", str(uuid.uuid4())),
                            content=match.get("content", ""),
                            similarity=additional.get("certainty", 0),
                            metadata=match.get("metadata", {}),
                            vector=additional.get("vector") if query.include_vectors else None
                        )
                        search_results.append(search_result)
                
                return search_results
                
        except httpx.TimeoutException:
            logger.error("Weaviate API timeout")
            raise HTTPException(
                status_code=504,
                detail="Request to Weaviate API timed out"
            )
        except Exception as e:
            logger.error(f"Error calling Weaviate API: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error calling Weaviate API: {str(e)}"
            )
    
    async def _weaviate_delete(
        self,
        ids: List[str],
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Delete vectors from Weaviate.
        """
        responses = {}
        
        # Delete by IDs
        if ids:
            for object_id in ids:
                url = f"{self.endpoint}/v1/objects/{object_id}"
                
                try:
                    async with httpx.AsyncClient(timeout=30) as client:
                        response = await client.delete(
                            url,
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            }
                        )
                        
                        if response.status_code != 204:
                            logger.error(f"Weaviate API error: {response.status_code} - {response.text}")
                            raise HTTPException(
                                status_code=response.status_code,
                                detail=f"Weaviate API error: {response.text}"
                            )
                        
                        responses[object_id] = "deleted"
                        
                except httpx.TimeoutException:
                    logger.error("Weaviate API timeout")
                    raise HTTPException(
                        status_code=504,
                        detail="Request to Weaviate API timed out"
                    )
                except Exception as e:
                    logger.error(f"Error calling Weaviate API: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error calling Weaviate API: {str(e)}"
                    )
        
        # Delete by filter (batch delete)
        if filter:
            # For Weaviate, we need to construct a WHERE clause
            where_filter = self._build_weaviate_where_filter(filter)
            
            # Use the batch delete endpoint
            url = f"{self.endpoint}/v1/batch/objects"
            
            payload = {
                "class": self.namespace,
                "where": where_filter
            }
            
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.delete(
                        url,
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        }
                    )
                    
                    if response.status_code not in [200, 204]:
                        logger.error(f"Weaviate API error: {response.status_code} - {response.text}")
                        raise HTTPException(
                            status_code=response.status_code,
                            detail=f"Weaviate API error: {response.text}"
                        )
                    
                    # Try to parse response, but it might be empty for 204
                    try:
                        batch_response = response.json()
                        responses["batch"] = batch_response
                    except:
                        responses["batch"] = {"status": "success"}
                    
            except httpx.TimeoutException:
                logger.error("Weaviate API timeout")
                raise HTTPException(
                    status_code=504,
                    detail="Request to Weaviate API timed out"
                )
            except Exception as e:
                logger.error(f"Error calling Weaviate API: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error calling Weaviate API: {str(e)}"
                )
        
        return {"status": "success", "deleted": len(responses), "responses": responses}
    
    def _build_weaviate_where_filter(self, filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a generic filter to Weaviate's filter format.
        This is a simplified implementation.
        """
        operands = []
        
        for key, value in filter.items():
            if isinstance(value, (str, int, float, bool)):
                # Simple equality filter
                operands.append({
                    "path": [key],
                    "operator": "Equal",
                    "valueString": str(value)
                })
            elif isinstance(value, dict):
                # Handle operators like $gt, $lt, etc.
                for op, op_value in value.items():
                    weaviate_op = {
                        "$eq": "Equal",
                        "$ne": "NotEqual",
                        "$gt": "GreaterThan",
                        "$gte": "GreaterThanEqual",
                        "$lt": "LessThan",
                        "$lte": "LessThanEqual",
                    }.get(op, "Equal")
                    
                    # Determine value type
                    value_field = "valueString"
                    if isinstance(op_value, int):
                        value_field = "valueInt"
                    elif isinstance(op_value, float):
                        value_field = "valueNumber"
                    elif isinstance(op_value, bool):
                        value_field = "valueBoolean"
                    
                    operands.append({
                        "path": [key],
                        "operator": weaviate_op,
                        value_field: op_value
                    })
        
        if len(operands) == 1:
            return operands[0]
        else:
            return {
                "operator": "And",
                "operands": operands
            }