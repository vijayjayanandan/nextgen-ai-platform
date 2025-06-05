import re

with open('app/services/model_router.py', 'r') as f:
    content = f.read()

# 1. First, add the missing _get_on_prem_service method if it doesn't exist
if 'def _get_on_prem_service' not in content:
    # Find where to insert it (after __init__ method)
    init_end = content.find('self.llm_services["on_prem"] = None')
    if init_end != -1:
        # Find the next method
        next_method_pos = content.find('\n    async def', init_end)
        if next_method_pos != -1:
            # Insert the method
            method_to_add = '''
    
    def _get_on_prem_service(self, model_info=None):
        """Get or create an on-prem service instance."""
        if model_info:
            service = OnPremLLMService(
                model_name=model_info.name,
                endpoint_url=model_info.endpoint_url
            )
            # Update with model config if method exists
            if hasattr(service, 'update_from_model'):
                service.update_from_model(model_info)
            return service
        else:
            # Return a default on-prem service
            return OnPremLLMService()
'''
            content = content[:next_method_pos] + method_to_add + content[next_method_pos:]
            print("Added _get_on_prem_service method")

# 2. Fix the get_model_info return logic for on_prem
old_return = '''        provider_key = model.provider.value
        if provider_key not in self.llm_services:
            raise HTTPException(
                status_code=400,
                detail=f"Service for provider {provider_key} not configured"
            )
        return model, self.llm_services[provider_key]'''

new_return = '''        provider_key = model.provider.value
        
        # Special handling for on_prem models
        if provider_key == "on_prem":
            if not settings.ON_PREM_MODEL_ENABLED:
                raise HTTPException(
                    status_code=400,
                    detail=f"On-premises service not configured"
                )
            return model, self._get_on_prem_service(model)
        
        # For other providers, use the pre-initialized services
        if provider_key not in self.llm_services:
            raise HTTPException(
                status_code=400,
                detail=f"Service for provider {provider_key} not configured"
            )
        return model, self.llm_services[provider_key]'''

if old_return in content:
    content = content.replace(old_return, new_return)
    print("Fixed get_model_info return logic")

# Write back
with open('app/services/model_router.py', 'w') as f:
    f.write(content)

print("Completed fixing model_router.py")
