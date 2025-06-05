import sys

with open('app/services/model_router.py', 'r') as f:
    content = f.read()

# 1. Add the missing _get_on_prem_service method after __init__
init_end = content.find('self.llm_services["on_prem"] = None  # Will create instances as needed')
next_method = content.find('\n    async def get_model_info', init_end)

if next_method != -1 and '_get_on_prem_service' not in content:
    helper_method = '''
    
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
    content = content[:next_method] + helper_method + content[next_method:]

# 2. Fix the get_model_info method for on_prem models
# The current logic returns self.llm_services[provider_key] which is None for on_prem
old_logic = '''        # Get the appropriate service based on the model provider
        provider_key = model.provider.value
        if provider_key not in self.llm_services:
            raise HTTPException(
                status_code=400,
                detail=f"Service for provider {provider_key} not configured"
            )
        return model, self.llm_services[provider_key]'''

new_logic = '''        # Get the appropriate service based on the model provider
        provider_key = model.provider.value
        
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

content = content.replace(old_logic, new_logic)

# Write the fixed content
with open('app/services/model_router.py', 'w') as f:
    f.write(content)

print("Fixed model_router.py:")
print("1. Added missing _get_on_prem_service method")
print("2. Fixed get_model_info to properly handle on_prem models")
