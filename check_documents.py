"""
Check document status and processing
"""

import asyncio
import httpx
import json

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

# Test credentials
TEST_USER = {
    "username": "admin@example.com",
    "password": "adminpassword"
}

async def check_documents():
    """Check document status and processing."""
    print("🔍 Checking Document Status")
    print("=" * 40)
    
    # Authenticate
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE}/token",
            data={
                "username": TEST_USER["username"],
                "password": TEST_USER["password"]
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code != 200:
            print(f"❌ Authentication failed: {response.status_code}")
            return
        
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        print("✅ Authentication successful")
    
    # List documents
    print("\n📄 Listing Documents...")
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{API_BASE}/documents/", headers=headers)
        
        if response.status_code == 200:
            documents = response.json()
            print(f"   Found {len(documents)} documents")
            
            for i, doc in enumerate(documents, 1):
                print(f"\n   Document {i}:")
                print(f"      ID: {doc.get('id')}")
                print(f"      Title: {doc.get('title')}")
                print(f"      Status: {doc.get('status', 'unknown')}")
                print(f"      Created: {doc.get('created_at')}")
                
                # Check document chunks
                doc_id = doc.get('id')
                if doc_id:
                    chunk_response = await client.get(
                        f"{API_BASE}/retrieval/document-chunks/{doc_id}",
                        headers=headers
                    )
                    
                    if chunk_response.status_code == 200:
                        chunks = chunk_response.json()
                        print(f"      Chunks: {len(chunks)}")
                        
                        if chunks:
                            print(f"      Sample chunk: {chunks[0].get('content', '')[:100]}...")
                    else:
                        print(f"      Chunks: Error {chunk_response.status_code}")
        else:
            print(f"   ❌ Failed to list documents: {response.status_code}")
    
    # Test with a simple query
    print("\n🔍 Testing Simple Search...")
    search_data = {
        "query": "immigration",
        "top_k": 5,
        "filters": None
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_BASE}/retrieval/semantic-search",
            json=search_data,
            headers=headers
        )
        
        if response.status_code == 200:
            results = response.json()
            print(f"   ✅ Search successful: {results['total']} results")
            
            if results['total'] > 0:
                print("   📄 Sample results:")
                for i, result in enumerate(results['results'][:2], 1):
                    content = result.get('content', '')
                    similarity = result.get('similarity', 0)
                    print(f"      Result {i}: Similarity {similarity:.3f}")
                    print(f"      Content: {content[:150]}...")
        else:
            print(f"   ❌ Search failed: {response.status_code} - {response.text}")

if __name__ == "__main__":
    asyncio.run(check_documents())
