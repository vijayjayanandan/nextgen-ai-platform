with open('app/services/llm/on_prem_service.py', 'r') as f:
    content = f.read()

# Fix update_from_model to update adapter
old_method = '''    def update_from_model(self, model):
        """Update configuration from database model."""
        if model:
            self.endpoint_url = model.endpoint_url
            if model.base_model == "ollama":
                self.model_type = "ollama"
            self.model_config = model.default_parameters or {}'''

new_method = '''    def update_from_model(self, model):
        """Update configuration from database model."""
        if model:
            self.endpoint_url = model.endpoint_url
            if model.base_model == "ollama":
                self.model_type = "ollama"
                # IMPORTANT: Update the adapter when model type changes
                self.adapter = self._get_adapter("ollama")
            self.model_config = model.default_parameters or {}'''

if old_method in content:
    content = content.replace(old_method, new_method)
    with open('app/services/llm/on_prem_service.py', 'w') as f:
        f.write(content)
    print("Fixed update_from_model to update adapter")
else:
    print("Method not found or already fixed")
