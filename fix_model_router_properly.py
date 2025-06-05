import sys

with open('app/services/model_router.py', 'r') as f:
    content = f.read()

# Fix 1: Update the initialization in __init__ to not create a generic service
old_init = '''        if settings.ON_PREM_MODEL_ENABLED:
            self.llm_services["on_prem"] = OnPremLLMService()'''

new_init = '''        if settings.ON_PREM_MODEL_ENABLED:
            # Don't create a generic service - we'll create model-specific ones
            self.llm_services["on_prem"] = True  # Just mark as available'''

content = content.replace(old_init, new_init)

# Fix 2: Update the on-prem service creation to pass endpoint_url
old_creation = '''            # Create a specific instance of the OnPremLLMService for this model
            on_prem_service = OnPremLLMService(model_name=model_name)
            return None, on_prem_service'''

new_creation = '''            # Create a specific instance of the OnPremLLMService for this model
            # First try to get model info from database
            result = await self.db.execute(select(Model).filter(Model.name == model_name))
            model = result.scalars().first()
            
            if model:
                on_prem_service = OnPremLLMService(
                    model_name=model_name,
                    endpoint_url=model.endpoint_url
                )
            else:
                on_prem_service = OnPremLLMService(model_name=model_name)
            
            return model, on_prem_service'''

content = content.replace(old_creation, new_creation)

# Fix 3: Update the fallback cases for llama and deepseek
old_llama = '''                if "on_prem" in self.llm_services:
                    return None, OnPremLLMService(model_name="llama-7b")'''

new_llama = '''                if "on_prem" in self.llm_services:
                    # Try to get from database first
                    result = await self.db.execute(select(Model).filter(Model.name == model_name))
                    model = result.scalars().first()
                    if model:
                        return model, OnPremLLMService(model_name=model_name, endpoint_url=model.endpoint_url)
                    else:
                        return None, OnPremLLMService(model_name="llama-7b")'''

content = content.replace(old_llama, new_llama)

old_deepseek = '''                if "on_prem" in self.llm_services:
                    return None, OnPremLLMService(model_name="deepseek-7b")'''

new_deepseek = '''                if "on_prem" in self.llm_services:
                    # Try to get from database first
                    result = await self.db.execute(select(Model).filter(Model.name == model_name))
                    model = result.scalars().first()
                    if model:
                        return model, OnPremLLMService(model_name=model_name, endpoint_url=model.endpoint_url)
                    else:
                        return None, OnPremLLMService(model_name="deepseek-7b")'''

content = content.replace(old_deepseek, new_deepseek)

# Write back
with open('app/services/model_router.py', 'w') as f:
    f.write(content)

print("Updated model_router.py to pass endpoint_url from database to OnPremLLMService")
