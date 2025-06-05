import asyncio
import sys
import uuid
sys.path.append('.')

from app.db.session import get_db
from app.models.model import Model, ModelProvider, ModelType, ModelStatus, ModelDeploymentType
from sqlalchemy import select

async def add_ollama_models():
    async for db in get_db():
        # Define Ollama models
        ollama_models = [
            {
                "id": uuid.uuid4(),
                "name": "deepseek-coder:1.3b",
                "display_name": "DeepSeek Coder 1.3B (Local)",
                "description": "DeepSeek Coder 1.3B model for code generation, running locally via Ollama",
                "type": ModelType.LLM,
                "provider": ModelProvider.ON_PREM,
                "base_model": "deepseek-coder",
                "deployment_type": ModelDeploymentType.ON_PREM,
                "endpoint_url": "http://localhost:11434/api/generate",
                "api_key_variable": None,  # Ollama doesn't need API key
                "max_tokens": 4096,
                "supports_functions": False,
                "supported_languages": ["en", "code"],
                "security_classification": "PROTECTED_B",
                "allowed_for_protected_b": True,
                "allowed_roles": ["user", "admin", "developer"],
                "status": ModelStatus.ACTIVE,
                "default_parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 2048
                }
            },
            {
                "id": uuid.uuid4(),
                "name": "mistral:7b-instruct",
                "display_name": "Mistral 7B Instruct (Local)",
                "description": "Mistral 7B Instruct model for general tasks, running locally via Ollama",
                "type": ModelType.LLM,
                "provider": ModelProvider.ON_PREM,
                "base_model": "mistral",
                "deployment_type": ModelDeploymentType.ON_PREM,
                "endpoint_url": "http://localhost:11434/api/generate",
                "api_key_variable": None,
                "max_tokens": 8192,
                "supports_functions": False,
                "supported_languages": ["en", "fr", "es", "de", "it"],
                "security_classification": "PROTECTED_B",
                "allowed_for_protected_b": True,
                "allowed_roles": ["user", "admin"],
                "status": ModelStatus.ACTIVE,
                "default_parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 4096
                }
            }
        ]
        
        # Add each model
        for model_data in ollama_models:
            # Check if model already exists
            result = await db.execute(
                select(Model).where(Model.name == model_data["name"])
            )
            existing_model = result.scalar_one_or_none()
            
            if not existing_model:
                model = Model(**model_data)
                db.add(model)
                print(f"Added model: {model_data['display_name']}")
            else:
                print(f"Model already exists: {model_data['display_name']}")
        
        await db.commit()
        print("\nOllama models added successfully!")
        break

async def list_all_models():
    """List all models in the database"""
    async for db in get_db():
        result = await db.execute(select(Model))
        models = result.scalars().all()
        
        print("\nCurrent models in database:")
        print("-" * 80)
        for model in models:
            print(f"Name: {model.name}")
            print(f"  Display: {model.display_name}")
            print(f"  Provider: {model.provider}")
            print(f"  Type: {model.deployment_type}")
            print(f"  Status: {model.status}")
            print(f"  Endpoint: {model.endpoint_url}")
            print(f"  Protected B: {model.allowed_for_protected_b}")
            print()
        break

if __name__ == "__main__":
    # Add the models
    asyncio.run(add_ollama_models())
    
    # List all models
    asyncio.run(list_all_models())
