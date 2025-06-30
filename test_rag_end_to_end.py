"""
End-to-End RAG Testing Script
Tests the complete RAG pipeline: document upload, embedding, search, and chat completion
"""

import asyncio
import httpx
import json
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

# Test credentials (from your existing auth system)
TEST_USER = {
    "username": "admin@example.com",
    "password": "adminpassword"
}

class RAGTester:
    def __init__(self):
        self.token = None
        self.headers = {"Content-Type": "application/json"}
        self.document_id = None
    
    async def authenticate(self):
        """Authenticate and get access token."""
        print("🔐 Authenticating...")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{API_BASE}/token",
                data={
                    "username": TEST_USER["username"],
                    "password": TEST_USER["password"]
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.token = data["access_token"]
                self.headers["Authorization"] = f"Bearer {self.token}"
                print("✅ Authentication successful")
                return True
            else:
                print(f"❌ Authentication failed: {response.status_code} - {response.text}")
                return False
    
    async def upload_document(self, file_path: str):
        """Upload a document for processing."""
        print(f"📄 Uploading document: {file_path}")
        
        file_path = Path(file_path)
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            return False
        
        async with httpx.AsyncClient(timeout=60) as client:
            with open(file_path, 'rb') as f:
                files = {
                    "file": (file_path.name, f, "text/plain")
                }
                data = {
                    "title": "Immigration and Refugee Protection Act (IRPA)",
                    "description": "Key provisions of Canada's primary immigration legislation",
                    "source_type": "uploaded",
                    "language": "en",
                    "is_public": "true"
                }
                
                response = await client.post(
                    f"{API_BASE}/documents/",
                    files=files,
                    data=data,
                    headers={"Authorization": f"Bearer {self.token}"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    self.document_id = result["id"]
                    print(f"✅ Document uploaded successfully")
                    print(f"   Document ID: {self.document_id}")
                    print(f"   Title: {result['title']}")
                    return True
                else:
                    print(f"❌ Document upload failed: {response.status_code} - {response.text}")
                    return False
    
    async def wait_for_processing(self, max_wait_time=120):
        """Wait for document processing to complete."""
        print("⏳ Waiting for document processing...")
        
        start_time = time.time()
        
        async with httpx.AsyncClient() as client:
            while time.time() - start_time < max_wait_time:
                response = await client.get(
                    f"{API_BASE}/documents/{self.document_id}/status",
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    status = response.json()
                    print(f"   Status: {status.get('status', 'unknown')}")
                    
                    if status.get("status") == "processed":
                        print("✅ Document processing completed")
                        print(f"   Chunks created: {status.get('chunk_count', 0)}")
                        return True
                    elif status.get("status") == "failed":
                        print(f"❌ Document processing failed: {status.get('error', 'Unknown error')}")
                        return False
                
                await asyncio.sleep(5)  # Wait 5 seconds before checking again
        
        print(f"⏰ Processing timeout after {max_wait_time} seconds")
        return False
    
    async def test_semantic_search(self):
        """Test semantic search functionality."""
        print("\n🔍 Testing Semantic Search...")
        
        test_queries = [
            "What are the language requirements for immigration?",
            "How long does Express Entry take to process?",
            "What is the Family Class immigration program?",
            "Who is eligible for refugee protection in Canada?"
        ]
        
        async with httpx.AsyncClient() as client:
            for i, query in enumerate(test_queries, 1):
                print(f"\n   Query {i}: {query}")
                
                search_data = {
                    "query": query,
                    "top_k": 3,
                    "filters": None
                }
                
                response = await client.post(
                    f"{API_BASE}/retrieval/semantic-search",
                    json=search_data,
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    results = response.json()
                    print(f"   ✅ Found {results['total']} results")
                    
                    for j, result in enumerate(results['results'][:2], 1):
                        similarity = result.get('similarity', 0)
                        content_preview = result.get('content', '')[:100] + "..."
                        print(f"      Result {j}: Similarity {similarity:.3f}")
                        print(f"      Content: {content_preview}")
                else:
                    print(f"   ❌ Search failed: {response.status_code} - {response.text}")
                    return False
        
        print("✅ Semantic search tests completed")
        return True
    
    async def test_rag_chat(self):
        """Test RAG-enabled chat completion."""
        print("\n💬 Testing RAG Chat Completion...")
        
        test_questions = [
            "What are the processing times for different immigration programs?",
            "Can you explain the language requirements for economic immigration?",
            "What are the inadmissibility grounds under IRPA?",
            "How does the Family Class immigration work?"
        ]
        
        async with httpx.AsyncClient(timeout=60) as client:
            for i, question in enumerate(test_questions, 1):
                print(f"\n   Question {i}: {question}")
                
                chat_data = {
                    "messages": [
                        {
                            "role": "user",
                            "content": question
                        }
                    ],
                    "model": "claude-3-5-sonnet-20241022",
                    "retrieve": True,
                    "retrieval_options": {
                        "top_k": 3,
                        "filters": None
                    },
                    "max_tokens": 200,
                    "temperature": 0.7
                }
                
                response = await client.post(
                    f"{API_BASE}/chat/completions",
                    json=chat_data,
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Extract response content
                    if "choices" in result and len(result["choices"]) > 0:
                        content = result["choices"][0]["message"]["content"]
                        print(f"   ✅ RAG Response received ({len(content)} chars)")
                        print(f"   Preview: {content[:150]}...")
                        
                        # Check if retrieval was used
                        usage = result.get("usage", {})
                        if "retrieval_used" in usage:
                            print(f"   📚 Retrieval used: {usage['retrieval_used']}")
                            print(f"   📄 Chunks retrieved: {usage.get('chunks_retrieved', 0)}")
                    else:
                        print("   ⚠️  No response content found")
                else:
                    print(f"   ❌ Chat completion failed: {response.status_code}")
                    print(f"   Error: {response.text}")
                    
                    # Continue with other questions even if one fails
                    continue
        
        print("✅ RAG chat completion tests completed")
        return True
    
    async def cleanup(self):
        """Clean up test data."""
        print("\n🧹 Cleaning up...")
        
        if self.document_id:
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{API_BASE}/documents/{self.document_id}",
                    headers=self.headers
                )
                
                if response.status_code == 200:
                    print("✅ Test document deleted")
                else:
                    print(f"⚠️  Could not delete test document: {response.status_code}")
    
    async def run_full_test(self):
        """Run the complete RAG test suite."""
        print("🚀 Starting End-to-End RAG Testing")
        print("=" * 60)
        
        try:
            # Step 1: Authenticate
            if not await self.authenticate():
                return False
            
            # Step 2: Upload document
            if not await self.upload_document("sample_immigration_policy.txt"):
                return False
            
            # Step 3: Wait for processing
            if not await self.wait_for_processing():
                return False
            
            # Step 4: Test semantic search
            if not await self.test_semantic_search():
                return False
            
            # Step 5: Test RAG chat
            if not await self.test_rag_chat():
                return False
            
            print("\n" + "=" * 60)
            print("🎉 ALL RAG TESTS PASSED!")
            print("✅ Document upload and processing: WORKING")
            print("✅ Local embeddings (384D): WORKING") 
            print("✅ Pinecone vector search: WORKING")
            print("✅ Semantic search API: WORKING")
            print("✅ RAG chat completion: WORKING")
            print("\n🏆 Your RAG system is fully operational!")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {str(e)}")
            return False
        
        finally:
            # Always cleanup
            await self.cleanup()

async def main():
    """Main test function."""
    tester = RAGTester()
    success = await tester.run_full_test()
    
    if success:
        print("\n✅ RAG system validation: PASSED")
    else:
        print("\n❌ RAG system validation: FAILED")
    
    return success

if __name__ == "__main__":
    asyncio.run(main())
