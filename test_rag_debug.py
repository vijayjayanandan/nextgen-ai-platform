import asyncio
import httpx
from app.services.embeddings.embedding_service import EmbeddingService

async def debug_rag():
    """Debug RAG retrieval to see what's actually being returned."""
    
    print("🔍 Debugging RAG Retrieval")
    print("=" * 50)
    
    # Initialize embedding service
    embedding_service = EmbeddingService()
    
    # Test query
    query = "What are the immigration policy objectives?"
    
    print(f"📝 Query: {query}")
    print()
    
    try:
        # Perform semantic search
        print("🔍 Performing semantic search...")
        results = await embedding_service.semantic_search(query, top_k=3)
        
        print(f"✅ Retrieved {len(results)} results")
        print()
        
        # Examine each result
        for i, result in enumerate(results, 1):
            print(f"📄 Result {i}:")
            print(f"   Type: {type(result)}")
            print(f"   Chunk ID: {result.chunk_id}")
            print(f"   Document ID: {result.document_id}")
            print(f"   Similarity: {result.similarity}")
            print(f"   Content Length: {len(result.content) if result.content else 0}")
            print(f"   Content Preview: {result.content[:100] if result.content else 'EMPTY'}...")
            print(f"   Metadata: {result.metadata}")
            print()
            
            # Test context building
            if result.metadata:
                title = result.metadata.get('document_title', 'Untitled')
                print(f"   Title from metadata: {title}")
            else:
                print("   ❌ No metadata found")
            print()
        
        # Test context building like in the chat endpoint
        print("🔧 Testing context building...")
        context_text = "\n\n".join([
            f"Document: {chunk.metadata.get('document_title', 'Untitled') if chunk.metadata else 'Untitled'}\n"
            f"Content: {chunk.content}"
            for chunk in results
        ])
        
        print(f"📝 Generated context length: {len(context_text)}")
        print(f"📝 Context preview:\n{context_text[:500]}...")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_rag())
