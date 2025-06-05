with open('app/services/model_router.py', 'r') as f:
    content = f.read()

# Find and replace the return logic
old_section = '''        provider_key = model.provider.value
        if provider_key not in self.llm_services:
            raise HTTPException(
                status_code=400,
                detail=f"Service for provider {provider_key} not configured"
            )
        return model, self.llm_services[provider_key]'''

new_section = '''        provider_key = model.provider.value
        
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

if old_section in content:
    content = content.replace(old_section, new_section)
    print("Fixed the on_prem return logic in get_model_info")
    
    with open('app/services/model_router.py', 'w') as f:
        f.write(content)
else:
    print("Could not find the exact pattern. Let me check the actual content...")
    # Show the area around provider_key assignment
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'provider_key = model.provider.value' in line:
            print(f"Found at line {i+1}. Context:")
            for j in range(max(0, i-5), min(len(lines), i+15)):
                print(f"{j+1}: {lines[j]}")
            break
