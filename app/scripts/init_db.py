# Create file: scripts/init_db.py

import asyncio
import sys
from sqlalchemy.ext.asyncio import AsyncSession
import uuid

sys.path.append('.')

from app.db.session import get_db, create_db_and_tables
from app.models.model import Model, ModelProvider, ModelType, ModelStatus, ModelDeploymentType
from app.models.user import User
from app.core.security import get_password_hash

async def init_db():
    # Create tables
    await create_db_and_tables()
    
    # Get DB session
    async for db in get_db():
        # Create admin user
        admin_id = uuid.uuid4()
        admin = User(
            id=admin_id,
            email="admin@example.com",
            hashed_password=get_password_hash("adminpassword"),
            full_name="Admin User",
            is_active=True,
            is_superuser=True,
            roles=["admin"]
        )
        db.add(admin)
        
        # Add models
        models_to_add = [
            {
                "name": "claude-3-opus-20240229",
                "display_name": "Claude 3 Opus",
                "description": "Anthropic's most powerful model",
                "type": ModelType.LLM,
                "provider": ModelProvider.ANTHROPIC,
                "deployment_type": ModelDeploymentType.API,
                "max_tokens": 100000,
                "supports_functions": True,
                "status": ModelStatus.ACTIVE
            },
            {
                "name": "claude-3-sonnet-20240229",
                "display_name": "Claude 3 Sonnet",
                "description": "Anthropic's balanced model",
                "type": ModelType.LLM,
                "provider": ModelProvider.ANTHROPIC,
                "deployment_type": ModelDeploymentType.API,
                "max_tokens": 100000,
                "supports_functions": True,
                "status": ModelStatus.ACTIVE
            }
        ]
        
        for model_data in models_to_add:
            model = Model(**model_data)
            db.add(model)
        
        await db.commit()
        print("Database initialized successfully!")
        break

if __name__ == "__main__":
    asyncio.run(init_db())