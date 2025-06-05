import sys

with open('app/services/model_router.py', 'r') as f:
    content = f.read()

# Update to use the new method
old_pattern = '''            if model:
                on_prem_service = OnPremLLMService(
                    model_name=model_name,
                    endpoint_url=model.endpoint_url
                )
            else:
                on_prem_service = OnPremLLMService(model_name=model_name)'''

new_pattern = '''            if model:
                on_prem_service = OnPremLLMService(model_name=model_name)
                on_prem_service.update_from_model(model)
            else:
                on_prem_service = OnPremLLMService(model_name=model_name)'''

if old_pattern in content:
    content = content.replace(old_pattern, new_pattern)
    with open('app/services/model_router.py', 'w') as f:
        f.write(content)
    print("Updated model_router to use update_from_model method")
else:
    print("Pattern not found - model_router may have different structure")
