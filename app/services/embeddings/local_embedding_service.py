"""
Local Embedding Service using Sentence Transformers
Provides self-hosted embeddings without requiring external API keys.
"""

import asyncio
import time
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class LocalEmbeddingService:
    """
    Self-hosted embedding service using Sentence Transformers.
    Provides enterprise-grade embeddings without external dependencies.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_folder: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the local embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            cache_folder: Directory to cache downloaded models
            device: Device to run model on ('cpu', 'cuda', 'auto')
        """
        self.model_name = model_name
        self.model = None
        self.device = device or "cpu"  # Default to CPU for laptop compatibility
        self.cache_folder = cache_folder
        self.is_initialized = False
        
        # Model specifications
        self.model_specs = {
            "all-MiniLM-L6-v2": {
                "dimensions": 384,
                "max_seq_length": 256,
                "size_mb": 22,
                "description": "Fast and efficient, good for most use cases"
            },
            "all-MiniLM-L12-v2": {
                "dimensions": 384,
                "max_seq_length": 256,
                "size_mb": 33,
                "description": "Slightly better quality than L6"
            },
            "all-mpnet-base-v2": {
                "dimensions": 768,
                "max_seq_length": 384,
                "size_mb": 420,
                "description": "High quality, larger model"
            }
        }
        
        # Performance tracking
        self.stats = {
            "total_embeddings_generated": 0,
            "total_processing_time": 0.0,
            "average_time_per_embedding": 0.0,
            "model_load_time": 0.0
        }
    
    async def initialize(self) -> bool:
        """
        Initialize the embedding model.
        Downloads model on first use (~22MB for all-MiniLM-L6-v2).
        
        Returns:
            True if initialization successful, False otherwise
        """
        if self.is_initialized:
            return True
        
        try:
            logger.info(f"Initializing local embedding model: {self.model_name}")
            start_time = time.time()
            
            # Import here to avoid startup delays if not used
            from sentence_transformers import SentenceTransformer
            
            # Initialize model with caching
            model_kwargs = {}
            if self.cache_folder:
                model_kwargs["cache_folder"] = self.cache_folder
            
            # Load model (downloads on first use)
            logger.info(f"Loading model {self.model_name} (may download ~{self.get_model_size()}MB on first use)")
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                **model_kwargs
            )
            
            load_time = time.time() - start_time
            self.stats["model_load_time"] = load_time
            
            logger.info(f"Local embedding model loaded successfully in {load_time:.2f}s")
            logger.info(f"Model dimensions: {self.get_dimensions()}")
            logger.info(f"Max sequence length: {self.get_max_seq_length()}")
            
            self.is_initialized = True
            return True
            
        except ImportError as e:
            logger.error(f"sentence-transformers not installed: {e}")
            logger.error("Install with: pip install sentence-transformers")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize local embedding model: {e}")
            return False
    
    async def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            show_progress: Whether to show progress bar for large batches
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Initialize model if not already done
        if not await self.initialize():
            raise RuntimeError("Failed to initialize local embedding model")
        
        try:
            start_time = time.time()
            
            # Generate embeddings
            logger.debug(f"Generating embeddings for {len(texts)} texts")
            
            # Use asyncio to run in thread pool to avoid blocking
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None,
                self._generate_embeddings_sync,
                texts,
                batch_size,
                show_progress
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["total_embeddings_generated"] += len(texts)
            self.stats["total_processing_time"] += processing_time
            self.stats["average_time_per_embedding"] = (
                self.stats["total_processing_time"] / 
                max(self.stats["total_embeddings_generated"], 1)
            )
            
            logger.debug(
                f"Generated {len(embeddings)} embeddings in {processing_time:.3f}s "
                f"({len(texts)/processing_time:.1f} embeddings/sec)"
            )
            
            return embeddings.tolist()
            
        except Exception as e:
            logger.error(f"Error generating local embeddings: {e}")
            raise RuntimeError(f"Local embedding generation failed: {e}")
    
    def _generate_embeddings_sync(
        self,
        texts: List[str],
        batch_size: int,
        show_progress: bool
    ):
        """
        Synchronous embedding generation (runs in thread pool).
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=False,
            normalize_embeddings=True  # Normalize for better similarity search
        )
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector
        """
        embeddings = await self.generate_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def get_dimensions(self) -> int:
        """Get the embedding dimensions for this model."""
        return self.model_specs.get(self.model_name, {}).get("dimensions", 384)
    
    def get_max_seq_length(self) -> int:
        """Get the maximum sequence length for this model."""
        return self.model_specs.get(self.model_name, {}).get("max_seq_length", 256)
    
    def get_model_size(self) -> int:
        """Get the approximate model size in MB."""
        return self.model_specs.get(self.model_name, {}).get("size_mb", 22)
    
    def get_model_description(self) -> str:
        """Get a description of the model."""
        return self.model_specs.get(self.model_name, {}).get(
            "description", 
            "Sentence transformer model"
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the embedding service.
        
        Returns:
            Dictionary with performance metrics
        """
        return {
            "model_name": self.model_name,
            "model_dimensions": self.get_dimensions(),
            "model_size_mb": self.get_model_size(),
            "is_initialized": self.is_initialized,
            "device": self.device,
            "total_embeddings_generated": self.stats["total_embeddings_generated"],
            "total_processing_time_seconds": self.stats["total_processing_time"],
            "average_time_per_embedding_ms": self.stats["average_time_per_embedding"] * 1000,
            "model_load_time_seconds": self.stats["model_load_time"],
            "embeddings_per_second": (
                self.stats["total_embeddings_generated"] / 
                max(self.stats["total_processing_time"], 0.001)
            )
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the embedding service.
        
        Returns:
            Health check results
        """
        try:
            # Test embedding generation
            test_text = "This is a test sentence for health check."
            start_time = time.time()
            
            embedding = await self.generate_embedding(test_text)
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "model_name": self.model_name,
                "model_loaded": self.is_initialized,
                "embedding_dimensions": len(embedding) if embedding else 0,
                "test_response_time_ms": response_time * 1000,
                "performance_stats": self.get_performance_stats()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_name": self.model_name,
                "model_loaded": self.is_initialized
            }


# Global instance for reuse
_local_embedding_service: Optional[LocalEmbeddingService] = None


def get_local_embedding_service() -> LocalEmbeddingService:
    """
    Get or create the global local embedding service instance.
    
    Returns:
        LocalEmbeddingService instance
    """
    global _local_embedding_service
    
    if _local_embedding_service is None:
        model_name = getattr(settings, 'LOCAL_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        _local_embedding_service = LocalEmbeddingService(model_name=model_name)
    
    return _local_embedding_service


async def test_local_embeddings():
    """
    Test function for local embeddings.
    """
    service = get_local_embedding_service()
    
    # Test texts
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown fox leaps over a sleepy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers."
    ]
    
    print(f"Testing local embeddings with model: {service.model_name}")
    print(f"Expected dimensions: {service.get_dimensions()}")
    
    # Generate embeddings
    embeddings = await service.generate_embeddings(texts)
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimensions: {len(embeddings[0]) if embeddings else 0}")
    
    # Show performance stats
    stats = service.get_performance_stats()
    print(f"Performance: {stats['embeddings_per_second']:.1f} embeddings/sec")
    
    return embeddings


if __name__ == "__main__":
    # Test the local embedding service
    import asyncio
    asyncio.run(test_local_embeddings())
