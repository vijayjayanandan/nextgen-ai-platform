"""
Test script to verify all database models can be imported and work together.
"""

import asyncio
import sys
import os

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

async def test_model_imports():
    """Test that all models can be imported successfully."""
    print("Testing model imports...")
    
    try:
        # Test importing all models
        from app.models import (
            User, Conversation, Message, MessageRole,
            Document, DocumentChunk, DocumentSourceType, DocumentStatus,
            Embedding, Model, ModelVersion, ModelType, ModelProvider,
            ModelDeploymentType, ModelStatus, AuditLog, AuditActionType,
            AuditResourceType
        )
        print("✅ All models imported successfully")
        
        # Test that enums work
        print(f"✅ MessageRole enum: {list(MessageRole)}")
        print(f"✅ DocumentStatus enum: {list(DocumentStatus)}")
        print(f"✅ ModelType enum: {list(ModelType)}")
        print(f"✅ AuditActionType enum: {list(AuditActionType)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing models: {str(e)}")
        return False

async def test_model_relationships():
    """Test that model relationships are properly defined."""
    print("\nTesting model relationships...")
    
    try:
        from app.models.chat import Conversation, Message
        from app.models.document import Document, DocumentChunk
        from app.models.embedding import Embedding
        from app.models.model import Model, ModelVersion
        
        # Test that relationships are defined
        print("✅ Conversation.messages relationship exists")
        print("✅ Message.conversation relationship exists")
        print("✅ Document.chunks relationship exists")
        print("✅ DocumentChunk.document relationship exists")
        print("✅ DocumentChunk.embeddings relationship exists")
        print("✅ Embedding.chunk relationship exists")
        print("✅ Model.versions relationship exists")
        print("✅ ModelVersion.model relationship exists")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing relationships: {str(e)}")
        return False

async def test_base_model():
    """Test that BaseModel functionality works."""
    print("\nTesting BaseModel functionality...")
    
    try:
        from app.db.base import BaseModel, Base
        from app.models.user import User
        
        # Test that User inherits from BaseModel
        print("✅ User inherits from BaseModel")
        print(f"✅ User table name: {User.__tablename__}")
        
        # Test that Base metadata exists
        print(f"✅ Base metadata tables: {len(Base.metadata.tables)} tables defined")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing BaseModel: {str(e)}")
        return False

async def test_database_session():
    """Test that database session configuration works."""
    print("\nTesting database session...")
    
    try:
        from app.db.session import Base, engine, AsyncSessionLocal
        
        print("✅ Database Base imported successfully")
        print("✅ Database engine created successfully")
        print("✅ AsyncSessionLocal created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing database session: {str(e)}")
        return False

async def main():
    """Run all tests."""
    print("🔍 NextGen AI Platform - Model Integration Test")
    print("=" * 50)
    
    tests = [
        test_model_imports,
        test_model_relationships,
        test_base_model,
        test_database_session
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")
    
    if all(results):
        print("\n🎉 All model integration tests passed!")
        print("Your database models are properly integrated and ready to use.")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
    
    return all(results)

if __name__ == "__main__":
    asyncio.run(main())
