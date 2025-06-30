import asyncio
import httpx

async def test_endpoints():
    """Test specific endpoints used by the RAG test."""
    print("Testing specific API endpoints...")
    
    endpoints_to_test = [
        "http://localhost:8000/",
        "http://localhost:8000/api/v1/token",
        "http://localhost:8000/api/v1/documents/",
        "http://localhost:8000/api/v1/retrieval/semantic-search",
    ]
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        for endpoint in endpoints_to_test:
            try:
                print(f"\nTesting: {endpoint}")
                response = await client.get(endpoint)
                print(f"  Status: {response.status_code}")
                if response.status_code != 200:
                    print(f"  Response: {response.text[:200]}...")
            except Exception as e:
                print(f"  Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_endpoints())
