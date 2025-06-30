# MemoryRetrievalNode Integration Example

## Overview

This document demonstrates how the MemoryRetrievalNode has been successfully integrated into the LangGraph RAG workflow, showing the complete data flow and example outputs.

## Updated Workflow Sequence

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ QueryAnalysis   │───▶│ MemoryRetrieval  │───▶│ HybridRetrieval │
│ - Intent detect │    │ - Semantic search│    │ - Doc retrieval │
│ - Entity extract│    │ - Context format │    │ - Hybrid search │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Memory Context   │    │ Reranking       │
                       │ - memory_context │    │ - LLM rerank    │
                       │ - memory_metadata│    │ - Score fusion  │
                       └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ MemoryUpdate    │◀───│ Evaluation       │◀───│ Generation      │
│ - Store turn    │    │ - Confidence     │    │ - Use memory    │
│ - Persist conv  │    │ - Quality score  │    │ - Use documents │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Final Response   │    │ Citation        │
                       │ - With citations │    │ - Extract refs  │
                       │ - With metadata  │    │ - Source docs   │
                       └──────────────────┘    └─────────────────┘
```

## Data Flow Example

### 1. Input State
```python
initial_state = RAGState(
    user_query="What documents do I need for work authorization?",
    conversation_id="conv_123",
    user_id="user_456"
)
```

### 2. After MemoryRetrievalNode
```python
state.memory_context = """## Conversation History:
🔥 [Turn 5] User: Can I work while my application is being processed?
Assistant: Work authorization depends on your current status and visa type. Visitor visa holders generally cannot work, but some permit holders may be eligible.

📝 [Turn 1] User: What are the visa requirements for Canada?
Assistant: For Canadian visas, you need to meet eligibility criteria including financial support, clean criminal record, and medical exams.
"""

state.memory_metadata = {
    "turns_retrieved": 2,
    "avg_relevance_score": 0.825,
    "memory_enabled": True,
    "retrieval_error": None
}

state.relevant_history = [
    MemoryTurn(
        id="turn_5",
        turn_number=5,
        user_message="Can I work while my application is being processed?",
        assistant_message="Work authorization depends on your current status...",
        timestamp=1641000000.0,
        relevance_score=0.85,
        context_snippet="User: Can I work while...\nAssistant: Work authorization..."
    ),
    MemoryTurn(
        id="turn_1", 
        turn_number=1,
        user_message="What are the visa requirements for Canada?",
        assistant_message="For Canadian visas, you need to meet eligibility...",
        timestamp=1640995200.0,
        relevance_score=0.8,
        context_snippet="User: What are the visa...\nAssistant: For Canadian visas..."
    )
]
```

### 3. Generated LLM Prompt
```
You are an expert AI assistant for Immigration, Refugees and Citizenship Canada (IRCC). 
Your role is to provide accurate, helpful information about Canadian immigration policies, procedures, and requirements.

Guidelines:
- Base your answers on the provided context documents
- If information is not in the context, clearly state this limitation
- Provide specific citations when referencing documents
- Be concise but comprehensive
- Use clear, professional language
- If asked about code or technical topics, provide practical examples

## Conversation History:
🔥 [Turn 5] User: Can I work while my application is being processed?
Assistant: Work authorization depends on your current status and visa type. Visitor visa holders generally cannot work, but some permit holders may be eligible.

📝 [Turn 1] User: What are the visa requirements for Canada?
Assistant: For Canadian visas, you need to meet eligibility criteria including financial support, clean criminal record, and medical exams.

## Context Documents:
[Document 1] Work Permit Application Guide (policy)
Content: To apply for a work permit, you must provide proof of job offer, Labour Market Impact Assessment (LMIA), educational credentials, and identity documents...

[Document 2] Immigration Forms and Documents (guideline)
Content: Required documents include passport, photographs, medical exam results, police certificates, and proof of funds...

## User Question:
What documents do I need for work authorization?

## Instructions:
Please provide a comprehensive answer based on the context documents above. 
Include specific citations in the format [Document X] when referencing information.
If the context doesn't contain sufficient information, clearly state this limitation.
```

### 4. Final Response with Memory Integration
```json
{
  "response": "Based on your previous questions about work authorization and the available documentation, here are the key documents you'll need:\n\n**Primary Documents [Document 1]:**\n- Valid passport\n- Job offer letter from Canadian employer\n- Labour Market Impact Assessment (LMIA) if required\n- Educational credential assessment\n\n**Supporting Documents [Document 2]:**\n- Medical examination results\n- Police certificates from countries where you've lived\n- Proof of funds to support yourself\n- Passport-style photographs\n\nSince you previously asked about working while your application is being processed, note that work authorization is separate from your main immigration application and requires its own documentation process.",
  
  "confidence_score": 0.89,
  
  "citations": [
    {
      "document_number": 1,
      "citation_text": "[Document 1]",
      "start_position": 156,
      "end_position": 167
    },
    {
      "document_number": 2,
      "citation_text": "[Document 2]", 
      "start_position": 298,
      "end_position": 309
    }
  ],
  
  "source_documents": [
    {
      "document_number": 1,
      "title": "Work Permit Application Guide",
      "source_type": "policy",
      "relevance_score": 0.92,
      "content_preview": "To apply for a work permit, you must provide proof of job offer..."
    },
    {
      "document_number": 2,
      "title": "Immigration Forms and Documents",
      "source_type": "guideline", 
      "relevance_score": 0.87,
      "content_preview": "Required documents include passport, photographs..."
    }
  ],
  
  "metadata": {
    "query_type": "conversational",
    "intent": "document_requirements",
    "entities": ["work authorization", "documents"],
    "retrieval_strategy": "hybrid",
    "documents_retrieved": 8,
    "documents_used": 2,
    "memory_turns_used": 2,
    "memory_metadata": {
      "turns_retrieved": 2,
      "avg_relevance_score": 0.825,
      "memory_enabled": true,
      "retrieval_error": null
    },
    "model_used": "deepseek-coder:6.7b",
    "processing_time": 2.341,
    "retry_count": 0,
    "fallback_triggered": false,
    "error_message": null
  }
}
```

## Fallback Behavior Examples

### 1. No Conversation ID
```python
# Input
state = RAGState(
    user_query="What are the requirements?",
    conversation_id=None,  # No conversation ID
    user_id="user_456"
)

# MemoryRetrievalNode Output
state.memory_context = ""  # Empty
state.memory_metadata = {
    "turns_retrieved": 0,
    "avg_relevance_score": 0.0,
    "memory_enabled": False,
    "retrieval_error": None
}

# Generated Prompt (no memory section)
"""
You are an expert AI assistant for Immigration, Refugees and Citizenship Canada...

## Context Documents:
[Document 1] Immigration Policy Guide...

## User Question:
What are the requirements?
"""
```

### 2. Memory Retrieval Failed
```python
# MemoryRetrievalNode Output (on error)
state.memory_context = ""
state.memory_metadata = {
    "turns_retrieved": 0,
    "avg_relevance_score": 0.0,
    "memory_enabled": True,
    "retrieval_error": "Qdrant connection timeout"
}

# Generated Prompt
"""
You are an expert AI assistant for Immigration, Refugees and Citizenship Canada...

## Note: Previous conversation context unavailable

## Context Documents:
[Document 1] Immigration Policy Guide...
"""
```

### 3. No Relevant Memory Found
```python
# MemoryRetrievalNode Output (no matches above threshold)
state.memory_context = ""
state.memory_metadata = {
    "turns_retrieved": 0,
    "avg_relevance_score": 0.0,
    "memory_enabled": True,
    "retrieval_error": None
}

# Generated Prompt (same as case 2)
"""
You are an expert AI assistant for Immigration, Refugees and Citizenship Canada...

## Note: Previous conversation context unavailable

## Context Documents:
[Document 1] Immigration Policy Guide...
"""
```

## Configuration Options

### Memory Configuration
```python
memory_config = MemoryConfig(
    max_turns=5,                    # Maximum memory turns to retrieve
    score_threshold=0.6,            # Minimum relevance score (0.0-1.0)
    include_recent_turns=2,         # Always include N recent turns
    merge_consecutive_turns=True,   # Merge adjacent turns
    max_context_length=2000         # Max characters for context
)
```

### Workflow Integration
```python
# Factory function usage
memory_node = create_memory_retrieval_node(
    qdrant_service=qdrant_service,
    max_turns=memory_config.max_turns,
    score_threshold=memory_config.score_threshold
)

# Workflow sequence
workflow.add_edge("query_analysis", "memory_retrieval")
workflow.add_edge("memory_retrieval", "hybrid_retrieval")
```

## Key Benefits

1. **Contextual Continuity**: Previous conversation turns inform current responses
2. **Semantic Relevance**: Only retrieves memory turns relevant to current query
3. **Graceful Fallback**: Workflow continues even if memory is unavailable
4. **Rich Metadata**: Detailed statistics for monitoring and debugging
5. **Configurable Behavior**: Tunable parameters for different use cases
6. **Performance Optimized**: Intelligent filtering and context length management

## Monitoring and Debugging

### Memory Usage Logs
```
[INFO] Retrieved 2 relevant memory turns (avg score: 0.825)
[DEBUG] Using memory context: 2 turns (avg score: 0.825)
[DEBUG] Memory retrieval was attempted but no relevant context found
```

### Response Metadata
```json
{
  "memory_metadata": {
    "turns_retrieved": 2,
    "avg_relevance_score": 0.825,
    "memory_enabled": true,
    "retrieval_error": null
  }
}
```

This integration provides a production-ready conversation memory system that enhances the RAG workflow with contextual awareness while maintaining robustness and performance.
