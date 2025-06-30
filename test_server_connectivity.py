import asyncio
import httpx

async def test_server():
    """Test if the server is responding."""
    print("Testing server connectivity...")
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Test basic connectivity
            response = await client.get("http://localhost:8000/docs")
            print(f"✅ Server responding: {response.status_code}")
            
            # Test API health
            response = await client.get("http://localhost:8000/api/v1/")
            print(f"✅ API responding: {response.status_code}")
            
    except httpx.ConnectError:
        print("❌ Cannot connect to server")
    except httpx.TimeoutException:
        print("❌ Server timeout - server is running but not responding")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_server())
