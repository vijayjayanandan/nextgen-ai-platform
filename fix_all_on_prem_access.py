import sys
import re

with open('app/services/model_router.py', 'r') as f:
    content = f.read()

# First, let's properly initialize on_prem in __init__
init_pattern = r'self\.llm_services\["on_prem"\] = .*'
init_replacement = 'self.llm_services["on_prem"] = None  # Will create instances as needed'
content = re.sub(init_pattern, init_replacement, content)

# Now we need to fix ALL places where llm_services["on_prem"] is accessed
# Let's create a helper method first
helper_method = '''
    def _get_on_prem_service(self, model_info=None):
        """Get or create an on-prem service instance."""
        if model_info and model_info.provider == ModelProvider.ON_PREM:
            service = OnPremLLMService(
                model_name=model_info.name,
                endpoint_url=model_info.endpoint_url
            )
            if hasattr(service, 'update_from_model'):
                service.update_from_model(model_info)
            return service
        else:
            # Return a default on-prem service
            return OnPremLLMService()
'''

# Insert the helper method after the __init__ method
init_end = content.find('self.db = None')
if init_end != -1:
    # Find the end of __init__ method
    next_method_start = content.find('\n    async def', init_end)
    if next_method_start != -1:
        # Insert helper method
        content = content[:next_method_start] + helper_method + content[next_method_start:]

# Now replace all direct accesses to llm_services["on_prem"]
# Pattern 1: Simple assignment
pattern1 = r'llm_service = self\.llm_services\["on_prem"\]'
replacement1 = 'llm_service = self._get_on_prem_service(model_info)'
content = re.sub(pattern1, replacement1, content)

# Also update the get_model_info method to handle on_prem properly
# Find the section where it returns on_prem service
old_on_prem_return = '''if "on_prem" not in self.llm_services:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"On-premises service not configured for model {model_name}"
                )
            # Create a specific instance of the OnPremLLMService for this model
            # First try to get model info from database
            result = await self.db.execute(select(Model).filter(Model.name == model_name))
            model = result.scalars().first()
            
            if model:
                on_prem_service = OnPremLLMService(model_name=model_name)
                on_prem_service.update_from_model(model)
            else:
                on_prem_service = OnPremLLMService(model_name=model_name)
            
            return model, on_prem_service'''

new_on_prem_return = '''if not settings.ON_PREM_MODEL_ENABLED:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"On-premises service not configured for model {model_name}"
                )
            # Get model info from database
            result = await self.db.execute(select(Model).filter(Model.name == model_name))
            model = result.scalars().first()
            
            # Create service instance
            on_prem_service = self._get_on_prem_service(model)
            
            return model, on_prem_service'''

content = content.replace(old_on_prem_return, new_on_prem_return)

# Update the check for on_prem availability
content = content.replace('"on_prem" not in self.llm_services', 'not settings.ON_PREM_MODEL_ENABLED')
content = content.replace('"on_prem" in self.llm_services', 'settings.ON_PREM_MODEL_ENABLED')

with open('app/services/model_router.py', 'w') as f:
    f.write(content)

print("Fixed all on_prem service access points")
