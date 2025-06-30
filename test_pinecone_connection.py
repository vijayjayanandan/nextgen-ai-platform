"""
Test Pinecone connection and basic operations
"""

import asyncio
import sys
from app.services.embeddings.embedding_service import EmbeddingService
from app.services.retrieval.vector_db_service import VectorDBService
from app.services.embeddings.local_embedding_service import get_local_embedding_service
from app.schemas.embedding import VectorSearchQuery

async def test_pinecone_connection():
    """Test Pinecone connection and operations."""
    print("🔍 Testing Pinecone Connection...")
    
    try:
        # Initialize services
        embedding_service = EmbeddingService()
        vector_db_service = VectorDBService()
        local_embedding_service = get_local_embedding_service()
        
        print("✅ Services initialized successfully")
        
        # Test 1: Generate local embeddings
        print("\n📊 Testing Local Embedding Generation...")
        test_texts = [
            "This is a test document about artificial intelligence.",
            "Machine learning is a subset of AI that focuses on algorithms.",
            "Deep learning uses neural networks with multiple layers."
        ]
        
        embeddings = await local_embedding_service.generate_embeddings(test_texts)
        print(f"✅ Generated {len(embeddings)} embeddings with {len(embeddings[0])} dimensions")
        
        # Test 2: Prepare vectors for Pinecone
        print("\n🗄️ Preparing vectors for Pinecone...")
        vectors = []
        for i, (text, embedding) in enumerate(zip(test_texts, embeddings)):
            vector = {
                "id": f"test_doc_{i}",
                "values": embedding,
                "metadata": {
                    "content": text,
                    "document_id": f"doc_{i}",
                    "chunk_index": 0,
                    "test": True
                }
            }
            vectors.append(vector)
        
        print(f"✅ Prepared {len(vectors)} vectors for upsert")
        
        # Test 3: Upsert vectors to Pinecone
        print("\n⬆️ Upserting vectors to Pinecone...")
        try:
            upsert_result = await vector_db_service.upsert_vectors(vectors)
            print(f"✅ Upsert successful: {upsert_result}")
        except Exception as e:
            print(f"❌ Upsert failed: {str(e)}")
            return False
        
        # Test 4: Search vectors in Pinecone
        print("\n🔍 Testing vector search...")
        try:
            # Generate query embedding
            query_text = "What is artificial intelligence?"
            query_embedding = await local_embedding_service.generate_embedding(query_text)
            
            # Create search query
            search_query = VectorSearchQuery(
                query=query_embedding,
                top_k=3,
                include_metadata=True,
                include_vectors=False
            )
            
            # Perform search
            search_results = await vector_db_service.search_vectors(search_query)
            print(f"✅ Search successful: Found {len(search_results)} results")
            
            for i, result in enumerate(search_results):
                print(f"  {i+1}. Similarity: {result.similarity:.3f}, Content: {result.content[:50]}...")
            
        except Exception as e:
            print(f"❌ Search failed: {str(e)}")
            return False
        
        # Test 5: Test embedding service semantic search
        print("\n🧠 Testing EmbeddingService semantic search...")
        try:
            semantic_results = await embedding_service.semantic_search(
                query="artificial intelligence",
                top_k=2
            )
            print(f"✅ Semantic search successful: Found {len(semantic_results)} results")
            
            for i, result in enumerate(semantic_results):
                print(f"  {i+1}. Similarity: {result.get('similarity', 0):.3f}")
                
        except Exception as e:
            print(f"❌ Semantic search failed: {str(e)}")
            print(f"Error details: {type(e).__name__}: {str(e)}")
            return False
        
        print("\n🎉 All Pinecone tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def cleanup_test_data():
    """Clean up test data from Pinecone."""
    print("\n🧹 Cleaning up test data...")
    try:
        vector_db_service = VectorDBService()
        
        # Delete test vectors
        test_ids = [f"test_doc_{i}" for i in range(3)]
        delete_result = await vector_db_service.delete_vectors(test_ids)
        print(f"✅ Cleanup successful: {delete_result}")
        
    except Exception as e:
        print(f"⚠️ Cleanup failed: {str(e)}")

async def main():
    """Main test function."""
    print("🚀 Pinecone Connection Test")
    print("=" * 50)
    
    success = await test_pinecone_connection()
    
    if success:
        print("\n✅ Pinecone is working correctly!")
        
        # Ask if user wants to keep test data
        print("\nTest data has been added to Pinecone.")
        print("This will help with semantic search testing.")
        
        # For now, keep the test data to help with semantic search
        print("✅ Keeping test data for semantic search validation")
    else:
        print("\n❌ Pinecone connection test failed!")
        await cleanup_test_data()
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
