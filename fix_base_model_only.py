import asyncio
import sys
sys.path.append('.')

from app.db.session import get_db
from app.models.model import Model
from sqlalchemy import update

async def fix_base_model():
    async for db in get_db():
        # Update base_model to "ollama" for both models
        for model_name in ["deepseek-coder:1.3b", "mistral:7b-instruct"]:
            await db.execute(
                update(Model)
                .where(Model.name == model_name)
                .values(base_model="ollama")
            )
            print(f"Updated base_model for {model_name} to 'ollama'")
        
        await db.commit()
        print("\nBase model field updated successfully!")
        break

if __name__ == "__main__":
    asyncio.run(fix_base_model())
