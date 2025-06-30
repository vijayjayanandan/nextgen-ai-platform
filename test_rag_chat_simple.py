#!/usr/bin/env python3
"""
Simple RAG Chat Test - Tests chat completion with retrieval
"""

import asyncio
import httpx
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

async def test_rag_chat_simple():
    """Test RAG chat completion directly"""
    print("🚀 Testing RAG Chat Completion (Simple)")
    print("=" * 50)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Step 1: Authenticate
        print("🔐 Authenticating...")
        auth_response = await client.post(
            f"{API_BASE}/auth/login",
            data={
                "username": "test@example.com",
                "password": "testpassword123"
            }
        )
        
        if auth_response.status_code != 200:
            print(f"❌ Authentication failed: {auth_response.status_code}")
            print(auth_response.text)
            return
        
        token = auth_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Authentication successful")
        
        # Step 2: Test chat completion with retrieval
        print("\n💬 Testing chat completion with retrieval...")
        
        chat_request = {
            "messages": [
                {
                    "role": "user",
                    "content": "What are the main objectives of the immigration policy?"
                }
            ],
            "model": "claude-3-5-sonnet-20241022",
            "retrieve": True,
            "retrieval_options": {
                "top_k": 3
            }
        }
        
        try:
            response = await client.post(
                f"{API_BASE}/chat/completions",
                json=chat_request,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                assistant_message = result["choices"][0]["message"]["content"]
                
                print("✅ Chat completion successful!")
                print(f"\n📝 Assistant Response:")
                print("-" * 40)
                print(assistant_message)
                print("-" * 40)
                
                # Check if response contains actual document content
                if any(keyword in assistant_message.lower() for keyword in [
                    "according to", "the document", "the policy", "immigration", 
                    "specific", "states", "outlines", "provides"
                ]):
                    print("✅ Response appears to contain document context!")
                    
                    # Check for source documents
                    if "source_documents" in result and result["source_documents"]:
                        print(f"✅ Found {len(result['source_documents'])} source documents")
                        for i, doc in enumerate(result["source_documents"][:2]):
                            print(f"   📄 Document {i+1}: {doc.get('title', 'Untitled')}")
                            content_preview = doc.get('content', '')[:100]
                            print(f"      Content preview: {content_preview}...")
                    else:
                        print("⚠️  No source documents returned")
                        
                else:
                    print("❌ Response appears generic - may not contain document context")
                    print("   Looking for keywords like 'according to', 'the document', etc.")
                
            else:
                print(f"❌ Chat completion failed: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"❌ Error during chat completion: {e}")
        
        # Step 3: Test another question
        print("\n💬 Testing second question...")
        
        chat_request2 = {
            "messages": [
                {
                    "role": "user", 
                    "content": "What are the language requirements mentioned in the policy?"
                }
            ],
            "model": "claude-3-5-sonnet-20241022",
            "retrieve": True,
            "retrieval_options": {
                "top_k": 3
            }
        }
        
        try:
            response2 = await client.post(
                f"{API_BASE}/chat/completions",
                json=chat_request2,
                headers=headers
            )
            
            if response2.status_code == 200:
                result2 = response2.json()
                assistant_message2 = result2["choices"][0]["message"]["content"]
                
                print("✅ Second chat completion successful!")
                print(f"\n📝 Assistant Response:")
                print("-" * 40)
                print(assistant_message2)
                print("-" * 40)
                
                # Check if response contains actual document content
                if any(keyword in assistant_message2.lower() for keyword in [
                    "according to", "the document", "the policy", "language", 
                    "specific", "states", "outlines", "provides", "requirement"
                ]):
                    print("✅ Second response also contains document context!")
                else:
                    print("❌ Second response appears generic")
                    
            else:
                print(f"❌ Second chat completion failed: {response2.status_code}")
                print(response2.text)
                
        except Exception as e:
            print(f"❌ Error during second chat completion: {e}")

if __name__ == "__main__":
    asyncio.run(test_rag_chat_simple())
