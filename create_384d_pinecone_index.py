"""
Create a new Pinecone index with 384 dimensions for all-MiniLM-L6-v2 model
"""

import asyncio
import httpx
from app.core.config import settings

async def create_384d_index():
    """Create a new Pinecone index with 384 dimensions."""
    print("🚀 Creating New 384D Pinecone Index...")
    
    try:
        # Extract environment from current URL
        # https://ircc-documents-izi1icy.svc.aped-4627-b74a.pinecone.io
        index_url = settings.VECTOR_DB_URI
        parts = index_url.split('.')
        if len(parts) >= 4:
            environment = parts[-3]  # aped-4627-b74a
            controller_url = f"https://controller.{environment}.pinecone.io"
        else:
            print("❌ Could not parse Pinecone URL to find controller")
            return False
        
        # New index configuration with generic name
        new_index_name = "documents-384d"
        
        index_config = {
            "name": new_index_name,
            "dimension": 384,
            "metric": "cosine",
            "pods": 1,
            "replicas": 1,
            "pod_type": "p1.x1"
        }
        
        print(f"📋 Index Configuration:")
        print(f"   Name: {new_index_name}")
        print(f"   Dimensions: 384 (matches all-MiniLM-L6-v2)")
        print(f"   Metric: cosine")
        print(f"   Environment: {environment}")
        
        # Create the index
        url = f"{controller_url}/databases"
        
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                url,
                json=index_config,
                headers={
                    "Api-Key": settings.VECTOR_DB_API_KEY,
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code in [200, 201]:
                print("✅ Index creation request submitted successfully!")
                print("⏳ Index is being created... This may take 1-2 minutes.")
                
                # Generate the new index URL
                new_index_url = f"https://{new_index_name}-{index_url.split('-', 2)[2]}"
                
                print(f"\n🔗 New Index URL: {new_index_url}")
                print(f"\n📝 Update your .env file:")
                print(f"VECTOR_DB_URI={new_index_url}")
                print(f"VECTOR_DB_NAMESPACE=documents")
                
                return new_index_url
                
            elif response.status_code == 409:
                print("⚠️  Index already exists with this name!")
                
                # Try to get the existing index URL
                new_index_url = f"https://{new_index_name}-{index_url.split('-', 2)[2]}"
                print(f"🔗 Existing Index URL: {new_index_url}")
                
                return new_index_url
                
            else:
                print(f"❌ Failed to create index: {response.status_code}")
                print(f"Response: {response.text}")
                
                # Show manual creation instructions
                print(f"\n📋 Manual Creation Instructions:")
                print(f"1. Go to: https://app.pinecone.io/")
                print(f"2. Click 'Create Index'")
                print(f"3. Name: {new_index_name}")
                print(f"4. Dimensions: 384")
                print(f"5. Metric: cosine")
                print(f"6. Environment: {environment}")
                
                return False
                
    except Exception as e:
        print(f"❌ Error creating index: {str(e)}")
        
        # Show manual creation instructions
        print(f"\n📋 Manual Creation Instructions:")
        print(f"1. Go to: https://app.pinecone.io/")
        print(f"2. Click 'Create Index'")
        print(f"3. Name: documents-384d")
        print(f"4. Dimensions: 384")
        print(f"5. Metric: cosine")
        
        return False

async def wait_for_index_ready(index_url: str, max_wait_minutes: int = 5):
    """Wait for the index to be ready."""
    print(f"\n⏳ Waiting for index to be ready (max {max_wait_minutes} minutes)...")
    
    import time
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while time.time() - start_time < max_wait_seconds:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{index_url}/describe_index_stats",
                    json={},
                    headers={
                        "Api-Key": settings.VECTOR_DB_API_KEY,
                        "Content-Type": "application/json"
                    }
                )
                
                if response.status_code == 200:
                    stats = response.json()
                    dimension = stats.get("dimension")
                    
                    if dimension == 384:
                        print("✅ Index is ready and configured correctly!")
                        print(f"📊 Dimension: {dimension}")
                        print(f"📈 Vector Count: {stats.get('totalVectorCount', 0)}")
                        return True
                    else:
                        print(f"⚠️  Index ready but wrong dimension: {dimension}")
                        return False
                        
                elif response.status_code == 404:
                    print("⏳ Index still being created...")
                    await asyncio.sleep(10)
                    continue
                else:
                    print(f"❌ Unexpected response: {response.status_code}")
                    await asyncio.sleep(10)
                    continue
                    
        except Exception as e:
            print(f"⏳ Waiting... ({str(e)})")
            await asyncio.sleep(10)
            continue
    
    print(f"⏰ Timeout waiting for index to be ready")
    return False

async def update_env_file(new_index_url: str):
    """Update the .env file with the new index URL."""
    print(f"\n📝 Updating .env file...")
    
    try:
        # Read current .env file
        with open('.env', 'r') as f:
            lines = f.readlines()
        
        # Update the VECTOR_DB_URI line and namespace
        updated_lines = []
        uri_updated = False
        namespace_updated = False
        
        for line in lines:
            if line.startswith('VECTOR_DB_URI='):
                updated_lines.append(f'VECTOR_DB_URI={new_index_url}\n')
                uri_updated = True
            elif line.startswith('VECTOR_DB_NAMESPACE='):
                updated_lines.append(f'VECTOR_DB_NAMESPACE=documents\n')
                namespace_updated = True
            else:
                updated_lines.append(line)
        
        # If VECTOR_DB_URI wasn't found, add it
        if not uri_updated:
            updated_lines.append(f'VECTOR_DB_URI={new_index_url}\n')
        
        # If VECTOR_DB_NAMESPACE wasn't found, add it
        if not namespace_updated:
            updated_lines.append(f'VECTOR_DB_NAMESPACE=documents\n')
        
        # Write updated .env file
        with open('.env', 'w') as f:
            f.writelines(updated_lines)
        
        print("✅ .env file updated successfully!")
        print("🔄 Restart your FastAPI application to use the new index")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to update .env file: {str(e)}")
        print(f"\n📝 Please manually update your .env file:")
        print(f"VECTOR_DB_URI={new_index_url}")
        print(f"VECTOR_DB_NAMESPACE=documents")
        return False

async def main():
    """Main function."""
    print("🚀 Pinecone 384D Index Creation")
    print("=" * 50)
    print("This will create a new Pinecone index optimized for your local embedding model")
    print()
    
    # Create the index
    new_index_url = await create_384d_index()
    
    if new_index_url:
        # Wait for index to be ready
        if await wait_for_index_ready(new_index_url):
            # Update .env file
            if await update_env_file(new_index_url):
                print("\n🎉 SUCCESS! Your 384D Pinecone index is ready!")
                print("\n📋 Next Steps:")
                print("1. ✅ Index created and ready")
                print("2. ✅ .env file updated")
                print("3. 🔄 Restart FastAPI app: Ctrl+C then restart")
                print("4. 🧪 Run: python enhanced_baseline_validation_test.py")
                print("5. 🎯 Expected: 100% test success rate!")
                
                return True
            else:
                print("\n⚠️  Index created but .env update failed")
                print("Please manually update your .env file and restart the app")
                return False
        else:
            print("\n⏰ Index creation timed out")
            print("Please check Pinecone console and try again in a few minutes")
            return False
    else:
        print("\n❌ Index creation failed")
        print("Please create the index manually in Pinecone console")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
