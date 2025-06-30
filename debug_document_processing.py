"""
Debug document processing step by step
"""

import asyncio
import httpx
import json
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

# Test credentials
TEST_USER = {
    "username": "admin@example.com",
    "password": "adminpassword"
}

async def debug_document_processing():
    """Debug document processing step by step."""
    print("🔍 Debug Document Processing")
    print("=" * 50)
    
    # Step 1: Authenticate
    print("1. Authenticating...")
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
            print(f"❌ Authentication failed: {response.status_code} - {response.text}")
            return
        
        token = response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        print("✅ Authentication successful")
    
    # Step 2: Check if sample file exists
    sample_file = "sample_immigration_policy.txt"
    if not Path(sample_file).exists():
        print(f"❌ Sample file {sample_file} not found")
        return
    
    print(f"✅ Sample file {sample_file} found")
    
    # Step 3: Upload document
    print("\n2. Uploading document...")
    async with httpx.AsyncClient(timeout=30) as client:
        with open(sample_file, 'rb') as f:
            files = {
                "file": (sample_file, f, "text/plain")
            }
            data = {
                "title": "Test Immigration Policy",
                "description": "Test document for debugging",
                "source_type": "uploaded",
                "language": "en",
                "is_public": "true"
            }
            
            response = await client.post(
                f"{API_BASE}/documents/",
                files=files,
                data=data,
                headers={"Authorization": f"Bearer {token}"}
            )
            
            if response.status_code != 200:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                return
            
            result = response.json()
            document_id = result["id"]
            print(f"✅ Document uploaded: {document_id}")
            print(f"   Status: {result.get('status')}")
    
    # Step 4: Monitor processing with detailed status checks
    print(f"\n3. Monitoring processing for document {document_id}...")
    
    async with httpx.AsyncClient(timeout=10) as client:
        for i in range(12):  # Check for 60 seconds (12 * 5 seconds)
            try:
                print(f"   Check {i+1}/12...")
                
                # Get document status
                response = await client.get(
                    f"{API_BASE}/documents/{document_id}/status",
                    headers=headers
                )
                
                if response.status_code == 200:
                    status = response.json()
                    print(f"   Status: {status.get('status', 'unknown')}")
                    print(f"   Updated: {status.get('updated_at', 'unknown')}")
                    print(f"   Chunks: {status.get('chunk_count', 0)}")
                    
                    if status.get("error_message"):
                        print(f"   Error: {status.get('error_message')}")
                    
                    if status.get("status") == "completed":
                        print("✅ Processing completed!")
                        break
                    elif status.get("status") == "failed":
                        print("❌ Processing failed!")
                        break
                else:
                    print(f"   ❌ Status check failed: {response.status_code}")
                    print(f"   Response: {response.text}")
                
                if i < 11:  # Don't wait after the last check
                    await asyncio.sleep(5)
                    
            except Exception as e:
                print(f"   ❌ Error during status check: {e}")
                break
    
    # Step 5: Check final document state
    print(f"\n4. Final document check...")
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(
                f"{API_BASE}/documents/{document_id}?include_chunks=true",
                headers=headers
            )
            
            if response.status_code == 200:
                doc = response.json()
                print(f"✅ Final status: {doc.get('status')}")
                print(f"   Chunks: {doc.get('chunk_count', 0)}")
                print(f"   Title: {doc.get('title')}")
                print(f"   Storage path: {doc.get('storage_path', 'Not set')}")
                
                if doc.get('chunks'):
                    print(f"   First chunk preview: {doc['chunks'][0]['content'][:100]}...")
            else:
                print(f"❌ Final check failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error during final check: {e}")
    
    print("\n" + "=" * 50)
    print("Debug completed!")

if __name__ == "__main__":
    asyncio.run(debug_document_processing())
