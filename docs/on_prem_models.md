# On-Premises Models Integration Guide

This guide explains how to use open source LLM models like Llama and Deepseek that are hosted on Azure VMs within the IRCC AI Platform.

## Supported Models

The IRCC AI Platform now explicitly supports the following open source models:

1. **Llama Models**:
   - `llama-7b` - Llama 3 7B model for general-purpose use
   - `llama-70b` - Llama 3 70B model for advanced reasoning tasks

2. **Deepseek Models**:
   - `deepseek-7b` - Deepseek 7B base model for general tasks
   - `deepseek-coder` - Deepseek Coder model optimized for code generation

## Configuration

The models are configured in `app/core/config.py` under the `ON_PREM_MODELS` dictionary. Each model entry includes:

- `endpoint`: The URL of the VM hosting the model
- `model_type`: The type of model ("llama", "deepseek", or "default")  
- `max_tokens`: Maximum context length the model supports
- `description`: Human-readable description of the model
- `supported_features`: List of capabilities the model supports

Example configuration:

```python
ON_PREM_MODELS: Dict[str, Dict[str, Any]] = {
    "llama-7b": {
        "endpoint": "http://llama-vm.canadacentral.azure.com:8000/v1",
        "model_type": "llama",
        "max_tokens": 8192,
        "description": "Llama 3 7B model for general-purpose use",
        "supported_features": ["chat", "completion", "function_calling"]
    },
    # Additional models...
}
```

## Using On-Premises Models

### 1. Via Model Name in API Requests

To use an on-premises model, simply specify its name in your API request:

```json
{
  "model": "llama-7b",
  "messages": [
    {"role": "user", "content": "How can I assist Canadian immigrants with finding housing resources?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024
}
```

The system will automatically:
1. Recognize the model name
2. Route the request to the appropriate VM endpoint
3. Apply the model-specific adapter for parameter and response handling

### 2. Via Model Registry

For a more controlled approach, register the models in the database:

```python
model = Model(
    name="llama-7b",
    display_name="Llama 3 (7B)",
    description="Llama 3 7B open source model hosted on-premises",
    type=ModelType.LLM,
    provider=ModelProvider.ON_PREM,
    deployment_type=ModelDeploymentType.ON_PREM,
    max_tokens=8192,
    supports_functions=True,
    allowed_for_protected_b=True,
    status=ModelStatus.ACTIVE
)
```

## VM Requirements

Each model requires a VM with specific hardware configurations:

1. **Llama Models**:
   - llama-7b: Standard Azure NC4as_T4_v3 (4 vCPU, 28GB RAM, 1 NVIDIA T4 GPU)
   - llama-70b: Standard Azure NC24ads_A100_v4 (24 vCPU, 220GB RAM, 1 NVIDIA A100 GPU)

2. **Deepseek Models**:
   - deepseek-7b: Standard Azure NC4as_T4_v3 (4 vCPU, 28GB RAM, 1 NVIDIA T4 GPU)
   - deepseek-coder: Standard Azure NC8as_T4_v3 (8 vCPU, 56GB RAM, 1 NVIDIA T4 GPU)

All VMs should be deployed in the Canada Central region to meet data residency requirements.

## Model Server Setup

The models should be deployed with a server that exposes an OpenAI-compatible API:

1. **vLLM**: Recommended for high-performance serving
   ```bash
   pip install vllm
   python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-7b-hf --host 0.0.0.0 --port 8000
   ```

2. **Text Generation Inference (TGI)**: For more complex deployments
   ```bash
   docker run --gpus all -p 8000:80 ghcr.io/huggingface/text-generation-inference:latest --model-id meta-llama/Llama-3-7b-hf
   ```

3. **LiteLLM**: For standardizing API interfaces
   ```bash
   pip install litellm
   litellm --model meta-llama/Llama-3-7b-hf --port 8000
   ```

## Routing Logic

The platform decides which model to use based on:

1. **Explicit request**: When the model is specified by name
2. **Data sensitivity**: Protected B data is routed to on-premises models
3. **Availability**: Falls back to other models if specified model is unavailable

## Model-Specific Adapters

The system uses model-specific adapters to handle differences in API parameters and responses:

- `LlamaAdapter`: Handles Llama-specific parameters like `repetition_penalty` and tool calling format
- `DeepseekAdapter`: Optimizes system prompts for Deepseek models, especially for code generation
- `OpenAICompatibleAdapter`: Default adapter for models with standard OpenAI-compatible APIs

## Monitoring and Maintenance

- VM health is monitored through Azure Monitor
- Model performance metrics are logged for each request
- Usage patterns determine scaling and optimization needs