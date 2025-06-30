"""
Simple RAG test to debug the actual issue
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

async def test_single_rag_question():
    """Test a single RAG question to see exactly what happens."""
    print("🔍 Testing Single RAG Question")
    print("=" * 40)
    
    # Step 1: Authenticate
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
        
        if response.status_code != 200:
            print(f"❌ Authentication failed: {response.status_code}")
            return
        
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Authentication successful")
    
    # Step 2: Test one simple question
    print("\n💬 Testing RAG Chat Completion...")
    
    question = "What are the main objectives of Canada's immigration policy?"
    print(f"📝 Question: {question}")
    
    async with httpx.AsyncClient(timeout=60) as client:
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
                "top_k": 3
            },
            "max_tokens": 400,
            "temperature": 0.7
        }
        
        try:
            response = await client.post(
                f"{API_BASE}/chat/completions",
                json=chat_data,
                headers=headers
            )
            
            print(f"📊 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Chat completion successful!")
                
                if "choices" in result and len(result["choices"]) > 0:
                    content = result["choices"][0]["message"]["content"]
                    print(f"📝 Response Content ({len(content)} chars):")
                    print(f"   {content}")
                    
                    # Check usage info
                    usage = result.get("usage", {})
                    print(f"📊 Usage: {usage}")
                    
                    # Check for source documents
                    if "source_documents" in result:
                        print(f"📚 Source Documents: {len(result['source_documents'])}")
                        for i, doc in enumerate(result['source_documents'][:2]):
                            print(f"   Doc {i+1}: {doc.get('title', 'Untitled')}")
                else:
                    print("❌ No response content found in result")
                    print(f"📄 Full result: {json.dumps(result, indent=2)}")
            else:
                print(f"❌ Chat completion failed: {response.status_code}")
                print(f"📄 Error response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception occurred: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_single_rag_question())
