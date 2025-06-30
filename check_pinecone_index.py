"""
Check Pinecone index configuration and create a new index if needed
"""

import asyncio
import httpx
from app.core.config import settings

async def check_pinecone_index():
    """Check the current Pinecone index configuration."""
    print("🔍 Checking Pinecone Index Configuration...")
    
    try:
        # Get index stats
        url = f"{settings.VECTOR_DB_URI}/describe_index_stats"
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                url,
                json={},
                headers={
                    "Api-Key": settings.VECTOR_DB_API_KEY,
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code == 200:
                stats = response.json()
                print(f"✅ Index Stats: {stats}")
                
                # Check dimension
                dimension = stats.get("dimension")
                if dimension:
                    print(f"📏 Index Dimension: {dimension}")
                    
                    if dimension == 384:
                        print("✅ Perfect! Index is configured for 384 dimensions (matches all-MiniLM-L6-v2)")
                        return True
                    elif dimension == 1536:
                        print("⚠️  Index is configured for 1536 dimensions (OpenAI ada-002)")
                        print("   Your local model generates 384 dimensions")
                        print("   Options:")
                        print("   1. Create a new Pinecone index with 384 dimensions")
                        print("   2. Switch to a model that generates 1536 dimensions")
                        return False
                    else:
                        print(f"⚠️  Index has {dimension} dimensions - unexpected configuration")
                        return False
                else:
                    print("❌ Could not determine index dimension")
                    return False
            else:
                print(f"❌ Failed to get index stats: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Error checking index: {str(e)}")
        return False

async def list_pinecone_indexes():
    """List all Pinecone indexes in the project."""
    print("\n📋 Listing Pinecone Indexes...")
    
    try:
        # Extract the base URL from the index URL
        # https://ircc-documents-izi1icy.svc.aped-4627-b74a.pinecone.io
        # becomes https://controller.aped-4627-b74a.pinecone.io
        
        index_url = settings.VECTOR_DB_URI
        parts = index_url.split('.')
        if len(parts) >= 4:
            environment = parts[-3]  # aped-4627-b74a
            controller_url = f"https://controller.{environment}.pinecone.io"
        else:
            print("❌ Could not parse Pinecone URL to find controller")
            return False
        
        url = f"{controller_url}/databases"
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                url,
                headers={
                    "Api-Key": settings.VECTOR_DB_API_KEY,
                    "Content-Type": "application/json"
                }
            )
            
            if response.status_code == 200:
                indexes = response.json()
                print(f"✅ Found {len(indexes)} indexes:")
                
                for index in indexes:
                    name = index.get("name", "unknown")
                    dimension = index.get("dimension", "unknown")
                    metric = index.get("metric", "unknown")
                    status = index.get("status", {}).get("ready", False)
                    
                    print(f"  📊 {name}: {dimension}D, {metric} metric, {'✅ Ready' if status else '⏳ Not Ready'}")
                
                return True
            else:
                print(f"❌ Failed to list indexes: {response.status_code} - {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Error listing indexes: {str(e)}")
        return False

async def main():
    """Main function."""
    print("🚀 Pinecone Index Configuration Check")
    print("=" * 50)
    
    # Check current index
    index_ok = await check_pinecone_index()
    
    # List all indexes
    await list_pinecone_indexes()
    
    if not index_ok:
        print("\n💡 Recommendations:")
        print("1. **Easiest**: Create a new Pinecone index with 384 dimensions")
        print("   - This matches your local all-MiniLM-L6-v2 model perfectly")
        print("   - No need to download larger models")
        print("   - Optimal performance")
        print()
        print("2. **Alternative**: Switch to a larger local model")
        print("   - Would need to download ~420MB model")
        print("   - Higher memory usage")
        print("   - Slower inference")
        print()
        print("🎯 **Recommended**: Create a new 384D index for optimal local embedding performance")
    
    return index_ok

if __name__ == "__main__":
    asyncio.run(main())
