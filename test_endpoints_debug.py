#!/usr/bin/env python3
"""
Debug endpoint connectivity
"""

import asyncio
import httpx
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_URL = "http://localhost:8000"

async def test_endpoints():
    """Test various endpoints to debug connectivity"""
    print("🔍 Testing Endpoint Connectivity")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        
        # Test 1: Root endpoint
        print("1. Testing root endpoint...")
        try:
            response = await client.get(f"{BASE_URL}/")
            print(f"   ✅ Root: {response.status_code} - {response.json()}")
        except Exception as e:
            print(f"   ❌ Root failed: {e}")
        
        # Test 2: API docs
        print("\n2. Testing API docs...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/docs")
            print(f"   ✅ Docs: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Docs failed: {e}")
        
        # Test 3: OpenAPI spec
        print("\n3. Testing OpenAPI spec...")
        try:
            response = await client.get(f"{BASE_URL}/api/v1/openapi.json")
            if response.status_code == 200:
                spec = response.json()
                print(f"   ✅ OpenAPI: {response.status_code}")
                print(f"   📋 Available paths:")
                for path in sorted(spec.get("paths", {}).keys()):
                    print(f"      {path}")
            else:
                print(f"   ❌ OpenAPI: {response.status_code}")
        except Exception as e:
            print(f"   ❌ OpenAPI failed: {e}")
        
        # Test 4: Auth endpoint
        print("\n4. Testing auth endpoint...")
        try:
            response = await client.post(
                f"{BASE_URL}/api/v1/auth/login",
                data={
                    "username": "test@example.com",
                    "password": "testpassword123"
                }
            )
            print(f"   ✅ Auth: {response.status_code}")
            if response.status_code == 200:
                token = response.json().get("access_token")
                print(f"   🔑 Got token: {token[:20]}...")
                
                # Test 5: Chat endpoint with auth
                print("\n5. Testing chat endpoint with auth...")
                headers = {"Authorization": f"Bearer {token}"}
                chat_request = {
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello, this is a test message."
                        }
                    ],
                    "model": "claude-3-5-sonnet-20241022",
                    "retrieve": False  # Disable retrieval for simple test
                }
                
                try:
                    chat_response = await client.post(
                        f"{BASE_URL}/api/v1/chat/completions",
                        json=chat_request,
                        headers=headers
                    )
                    print(f"   ✅ Chat: {chat_response.status_code}")
                    if chat_response.status_code == 200:
                        result = chat_response.json()
                        message = result["choices"][0]["message"]["content"]
                        print(f"   💬 Response: {message[:100]}...")
                    else:
                        print(f"   ❌ Chat error: {chat_response.text}")
                except Exception as e:
                    print(f"   ❌ Chat failed: {e}")
            else:
                print(f"   ❌ Auth error: {response.text}")
        except Exception as e:
            print(f"   ❌ Auth failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_endpoints())
