import sys

with open('app/services/llm/on_prem_service.py', 'r') as f:
    content = f.read()

# Add a method to update config from model
method_to_add = '''
    def update_from_model(self, model):
        """Update configuration from database model."""
        if model:
            self.endpoint_url = model.endpoint_url
            if model.base_model == "ollama":
                self.model_type = "ollama"
            self.model_config = model.default_parameters or {}
'''

# Find where to insert it (after __init__ method)
import re
# Find the end of __init__ method
init_end = content.find('self.adapter = self._get_adapter(self.model_type)')
if init_end != -1:
    # Find the next method definition
    next_method = content.find('\n    def ', init_end)
    if next_method != -1:
        # Insert our new method before the next method
        content = content[:next_method] + method_to_add + content[next_method:]
        
        with open('app/services/llm/on_prem_service.py', 'w') as f:
            f.write(content)
        print("Added update_from_model method to on_prem_service.py")
else:
    print("Could not find insertion point")
