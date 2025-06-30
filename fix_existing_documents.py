import asyncio
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.db.session import get_db
from app.models.document import Document
from app.services.retrieval.document_processor import DocumentProcessor
from app.services.embeddings.embedding_service import EmbeddingService

async def fix_existing_documents():
    """Reprocess existing documents to fix metadata."""
    
    print("🔧 Fixing Existing Document Metadata")
    print("=" * 50)
    
    # Get database session
    async for db in get_db():
        try:
            # Get all processed documents
            result = await db.execute(select(Document))
            documents = result.scalars().all()
            
            print(f"📄 Found {len(documents)} documents")
            
            # Initialize services
            embedding_service = EmbeddingService()
            processor = DocumentProcessor(db, embedding_service)
            
            # Reprocess each document
            for doc in documents:
                print(f"\n🔄 Reprocessing: {doc.title} (ID: {doc.id})")
                
                try:
                    result = await processor.reprocess_document(doc.id, force_reembed=True)
                    
                    if result["status"] == "success":
                        print(f"   ✅ Success: {result['chunks_embedded']} chunks re-embedded")
                    else:
                        print(f"   ❌ Failed: {result['message']}")
                        
                except Exception as e:
                    print(f"   ❌ Error: {str(e)}")
            
            print(f"\n🎉 Document reprocessing complete!")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            await db.close()
        break

if __name__ == "__main__":
    asyncio.run(fix_existing_documents())
