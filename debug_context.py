"""
Debug what context is actually being sent to the LLM
"""

import asyncio
from app.services.embeddings.embedding_service import EmbeddingService

async def debug_context():
    """Debug the context building process."""
    
    print("🔍 Debugging Context Building")
    print("=" * 40)
    
    # Initialize embedding service
    embedding_service = EmbeddingService()
    
    # Test query
    query = "What are the main objectives of Canada's immigration policy?"
    
    print(f"📝 Query: {query}")
    print()
    
    try:
        # Perform semantic search
        print("🔍 Performing semantic search...")
        results = await embedding_service.semantic_search(query, top_k=3)
        
        print(f"✅ Retrieved {len(results)} results")
        print()
        
        # Build context exactly like the orchestrator does
        print("🔧 Building context like orchestrator...")
        context_text = "\n\n".join([
            f"Document: {chunk.metadata.get('document_title', 'Untitled') if chunk.metadata else 'Untitled'}\n"
            f"Content: {chunk.content}"
            for chunk in results
        ])
        
        print(f"📝 Generated context ({len(context_text)} chars):")
        print("=" * 60)
        print(context_text)
        print("=" * 60)
        
        # Show what the final prompt would look like
        user_query = query
        
        # Check if there's a system message being added
        print("\n🤖 Final prompt structure:")
        print("SYSTEM MESSAGE:")
        print(f"Context information:\n{context_text}\n\nPlease use this context to answer the user's questions.")
        print("\nUSER MESSAGE:")
        print(user_query)
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_context())
