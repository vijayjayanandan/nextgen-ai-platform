import sys
sys.path.append('.')

from app.core.config import settings
from app.services.llm.on_prem_service import OnPremLLMService

# Check settings
print("ON_PREM_MODEL_ENABLED:", settings.ON_PREM_MODEL_ENABLED)
print("ON_PREM_MODEL_ENDPOINT:", settings.ON_PREM_MODEL_ENDPOINT)

# Try to initialize the service
try:
    service = OnPremLLMService(model_name="deepseek-coder:1.3b")
    print(f"Service initialized with endpoint: {service.endpoint_url}")
    print(f"Model type: {service.model_type}")
except Exception as e:
    print(f"Error initializing service: {e}")
