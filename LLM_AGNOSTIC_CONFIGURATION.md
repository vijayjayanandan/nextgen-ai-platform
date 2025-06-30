# LLM-Agnostic Configuration Guide

## Overview

The IRCC AI Platform has been designed with full LLM-agnostic capabilities, allowing you to seamlessly switch between different language model providers without code changes. This guide explains how to configure and use different LLM providers.

## 🎯 Key Benefits

- **Provider Independence**: Switch between OpenAI, Anthropic, and local models without code changes
- **Cost Optimization**: Use cheaper models for simple tasks, premium models for complex generation
- **Compliance**: Route sensitive data to on-premises models automatically
- **Performance**: Mix fast local models with high-quality cloud models
- **Resilience**: Automatic fallbacks if a provider is unavailable

## 🔧 Configuration Architecture

### Model Assignment by RAG Component

Each RAG workflow component can use a different model, optimized for its specific task:

| Component | Purpose | Recommended Model Type | Config Variable |
|-----------|---------|----------------------|-----------------|
| **Query Analysis** | Parse user intent, extract entities | Fast, lightweight | `RAG_QUERY_ANALYSIS_MODEL` |
| **Generation** | Generate final response | High-quality, context-aware | `RAG_GENERATION_MODEL` |
| **Reranking** | Score document relevance | Good reasoning, efficient | `RAG_RERANKING_MODEL` |
| **Memory Retrieval** | Process conversation history | Fast, good comprehension | `RAG_MEMORY_RETRIEVAL_MODEL` |
| **Citation** | Extract and format citations | Pattern recognition | `RAG_CITATION_MODEL` |

## 🚀 Quick Start Configurations

### Option 1: Local Models Only (Ollama)

**Best for**: Development, privacy-sensitive data, cost control

```bash
# .env configuration
ON_PREM_MODEL_ENABLED=true
ON_PREM_MODEL_ENDPOINT=http://localhost:11434

RAG_QUERY_ANALYSIS_MODEL=mistral:7b
RAG_GENERATION_MODEL=deepseek-coder:6.7b
RAG_RERANKING_MODEL=mistral:7b
RAG_MEMORY_RETRIEVAL_MODEL=mistral:7b
RAG_CITATION_MODEL=mistral:7b
```

**Prerequisites**:
```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull required models
ollama pull mistral:7b
ollama pull deepseek-coder:6.7b
```

### Option 2: OpenAI Models Only

**Best for**: High-quality responses, function calling, production

```bash
# .env configuration
OPENAI_API_KEY=your-openai-api-key

RAG_QUERY_ANALYSIS_MODEL=gpt-3.5-turbo
RAG_GENERATION_MODEL=gpt-4
RAG_RERANKING_MODEL=gpt-3.5-turbo
RAG_MEMORY_RETRIEVAL_MODEL=gpt-3.5-turbo
RAG_CITATION_MODEL=gpt-3.5-turbo
```

### Option 3: Anthropic Models Only

**Best for**: Long context, reasoning, safety

```bash
# .env configuration
ANTHROPIC_API_KEY=your-anthropic-api-key

RAG_QUERY_ANALYSIS_MODEL=claude-3-haiku-20240307
RAG_GENERATION_MODEL=claude-3-sonnet-20240229
RAG_RERANKING_MODEL=claude-3-haiku-20240307
RAG_MEMORY_RETRIEVAL_MODEL=claude-3-haiku-20240307
RAG_CITATION_MODEL=claude-3-haiku-20240307
```

### Option 4: Hybrid Setup (Recommended for Production)

**Best for**: Cost optimization, performance, resilience

```bash
# .env configuration
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
ON_PREM_MODEL_ENABLED=true

# Fast local models for analysis tasks
RAG_QUERY_ANALYSIS_MODEL=mistral:7b
RAG_MEMORY_RETRIEVAL_MODEL=mistral:7b
RAG_CITATION_MODEL=mistral:7b

# High-quality cloud models for generation
RAG_GENERATION_MODEL=gpt-4
RAG_RERANKING_MODEL=claude-3-haiku-20240307
```

## 🔄 Switching Between Providers

### Runtime Switching

You can switch providers by simply updating environment variables and restarting the application:

```bash
# Switch from local to OpenAI
export RAG_GENERATION_MODEL=gpt-4
export RAG_QUERY_ANALYSIS_MODEL=gpt-3.5-turbo

# Restart the application
docker-compose restart
```

### A/B Testing Configuration

Test different models by environment:

```bash
# Development environment
RAG_GENERATION_MODEL=mistral:7b

# Staging environment  
RAG_GENERATION_MODEL=claude-3-sonnet-20240229

# Production environment
RAG_GENERATION_MODEL=gpt-4
```

## 📊 Model Recommendations by Use Case

### Cost-Optimized Setup
```bash
RAG_QUERY_ANALYSIS_MODEL=mistral:7b          # Free local
RAG_GENERATION_MODEL=gpt-3.5-turbo          # $0.002/1K tokens
RAG_RERANKING_MODEL=mistral:7b               # Free local
RAG_MEMORY_RETRIEVAL_MODEL=mistral:7b        # Free local
RAG_CITATION_MODEL=mistral:7b                # Free local
```

### Quality-Optimized Setup
```bash
RAG_QUERY_ANALYSIS_MODEL=claude-3-haiku-20240307    # Fast, accurate
RAG_GENERATION_MODEL=gpt-4                          # Best quality
RAG_RERANKING_MODEL=claude-3-sonnet-20240229        # Excellent reasoning
RAG_MEMORY_RETRIEVAL_MODEL=claude-3-haiku-20240307  # Good context
RAG_CITATION_MODEL=gpt-3.5-turbo                    # Pattern recognition
```

### Privacy-First Setup
```bash
RAG_QUERY_ANALYSIS_MODEL=mistral:7b          # Local only
RAG_GENERATION_MODEL=llama2:13b              # Local only
RAG_RERANKING_MODEL=mistral:7b               # Local only
RAG_MEMORY_RETRIEVAL_MODEL=mistral:7b        # Local only
RAG_CITATION_MODEL=mistral:7b                # Local only
```

## 🛠 Advanced Configuration

### Custom Model Endpoints

For custom model deployments:

```bash
# Custom OpenAI-compatible endpoint
OPENAI_API_BASE=https://your-custom-endpoint.com/v1
RAG_GENERATION_MODEL=your-custom-model

# Custom Anthropic-compatible endpoint
ANTHROPIC_API_BASE=https://your-anthropic-proxy.com
RAG_GENERATION_MODEL=claude-3-custom
```

### Model-Specific Parameters

Different models may require different parameters. The system automatically adapts:

- **Temperature**: Automatically adjusted per model type
- **Max Tokens**: Optimized for each task
- **Context Length**: Managed based on model capabilities
- **Prompt Format**: Adapted for each provider's preferences

### Fallback Configuration

Configure automatic fallbacks:

```bash
# Primary models
RAG_GENERATION_MODEL=gpt-4

# Automatic fallback to local if API fails
ON_PREM_MODEL_ENABLED=true
RAG_FALLBACK_MODEL=mistral:7b
```

## 🔍 Monitoring and Observability

### Model Usage Tracking

The system automatically logs which models are used:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "component": "generation",
  "model_used": "gpt-4",
  "provider": "openai",
  "tokens_used": 1250,
  "response_time_ms": 2300,
  "cost_estimate": 0.025
}
```

### Performance Metrics

Monitor model performance by component:

- **Query Analysis**: Speed, accuracy of intent detection
- **Generation**: Quality scores, user satisfaction
- **Reranking**: Relevance improvement metrics
- **Memory**: Context retrieval effectiveness

## 🚨 Troubleshooting

### Common Issues

1. **Model Not Found**
   ```bash
   # For Ollama models
   ollama pull mistral:7b
   
   # Check model availability
   ollama list
   ```

2. **API Key Issues**
   ```bash
   # Verify API keys
   echo $OPENAI_API_KEY
   echo $ANTHROPIC_API_KEY
   ```

3. **Provider Unavailable**
   - System automatically falls back to available providers
   - Check logs for fallback notifications

### Debug Mode

Enable detailed logging:

```bash
LOG_LEVEL=DEBUG
ENABLE_AUDIT_LOGGING=true
```

## 📈 Performance Optimization

### Model Selection Guidelines

| Task Complexity | Recommended Model | Reasoning |
|-----------------|-------------------|-----------|
| Simple queries | Local models (Mistral 7B) | Fast, cost-effective |
| Complex analysis | Claude 3 Sonnet | Superior reasoning |
| Code generation | GPT-4 or DeepSeek Coder | Specialized capabilities |
| Long context | Claude 3 (100K+ tokens) | Large context window |
| Function calling | GPT-4 | Best function calling support |

### Caching Strategy

The system implements intelligent caching:

- **Query Analysis**: Cache common patterns
- **Embeddings**: Cache document embeddings
- **Model Responses**: Cache for identical queries

## 🔐 Security Considerations

### Data Sensitivity Routing

The system automatically routes sensitive data:

```python
# Automatic routing based on content
if contains_pii(query):
    route_to_local_model()
else:
    route_to_cloud_model()
```

### Compliance Settings

```bash
# Force all processing to local models
FORCE_LOCAL_PROCESSING=true

# Enable audit logging for compliance
ENABLE_PII_AUDIT_LOGGING=true
PII_RISK_THRESHOLD=0.7
```

## 🎯 Best Practices

1. **Start Simple**: Begin with a single provider, then optimize
2. **Monitor Costs**: Track token usage and costs per component
3. **Test Thoroughly**: Validate quality across different models
4. **Plan Fallbacks**: Always have backup models configured
5. **Security First**: Route sensitive data to appropriate models
6. **Performance Tune**: Optimize model selection based on metrics

## 📚 Additional Resources

- [Model Router Documentation](app/services/model_router.py)
- [LLM Service Architecture](app/services/llm/)
- [RAG Node Implementation](app/services/rag/nodes/)
- [Configuration Schema](app/core/config.py)

---

**Need Help?** Check the logs, review the configuration, or consult the troubleshooting section above.
