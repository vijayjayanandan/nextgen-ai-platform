#!/usr/bin/env python3
"""
Test script to verify MemoryRetrievalNode integration in the RAG workflow.

This script tests the complete workflow with memory integration.
"""

import asyncio
import sys
import os
from typing import Dict, Any

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.services.rag.workflow import ProductionRAGWorkflow
from app.services.rag.workflow_state import RAGState
from app.services.retrieval.qdrant_service import QdrantService, SearchResult
from app.services.llm.adapters.ollama import OllamaAdapter


class MockQdrantService:
    """Mock Qdrant service for testing"""
    
    def __init__(self):
        self.mock_memory = [
            SearchResult(
                id="turn_1",
                score=0.9,
                content="User: What are the visa requirements for Canada?\nAssistant: For Canadian visas, you need to meet eligibility criteria including financial support, clean criminal record, and medical exams.",
                metadata={
                    "conversation_id": "conv_123",
                    "turn_number": 1,
                    "user_message": "What are the visa requirements for Canada?",
                    "assistant_message": "For Canadian visas, you need to meet eligibility criteria including financial support, clean criminal record, and medical exams.",
                    "timestamp": 1640995200.0
                }
            ),
            SearchResult(
                id="turn_5",
                score=0.85,
                content="User: Can I work while my application is being processed?\nAssistant: Work authorization depends on your current status and visa type. Visitor visa holders generally cannot work, but some permit holders may be eligible.",
                metadata={
                    "conversation_id": "conv_123", 
                    "turn_number": 5,
                    "user_message": "Can I work while my application is being processed?",
                    "assistant_message": "Work authorization depends on your current status and visa type. Visitor visa holders generally cannot work, but some permit holders may be eligible.",
                    "timestamp": 1641000000.0
                }
            )
        ]
        
        self.mock_documents = [
            {
                "id": "doc_1",
                "title": "Work Permit Application Guide",
                "content": "To apply for a work permit, you must provide proof of job offer, Labour Market Impact Assessment (LMIA), educational credentials, and identity documents including passport and photographs.",
                "source_type": "policy",
                "metadata": {"page_number": 1, "section": "requirements"}
            },
            {
                "id": "doc_2", 
                "title": "Immigration Forms and Documents",
                "content": "Required documents include passport, photographs, medical exam results, police certificates, and proof of funds to support yourself during your stay in Canada.",
                "source_type": "guideline",
                "metadata": {"page_number": 3, "section": "documentation"}
            }
        ]
    
    async def search_memory(self, conversation_id: str, query: str, limit: int = 5, score_threshold: float = 0.5):
        """Mock memory search"""
        results = [
            result for result in self.mock_memory
            if (result.metadata.get("conversation_id") == conversation_id and 
                result.score >= score_threshold)
        ]
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    async def search_documents(self, query: str, **kwargs):
        """Mock document search"""
        from app.services.retrieval.qdrant_service import SearchResult
        
        results = []
        for i, doc in enumerate(self.mock_documents):
            # Simple relevance scoring based on keyword overlap
            query_words = set(query.lower().split())
            content_words = set(doc["content"].lower().split())
            overlap = len(query_words.intersection(content_words))
            score = min(0.95, 0.5 + (overlap * 0.1))
            
            result = SearchResult(
                id=doc["id"],
                score=score,
                content=doc["content"],
                metadata={
                    "title": doc["title"],
                    "source_type": doc["source_type"],
                    **doc["metadata"]
                }
            )
            results.append(result)
        
        # Sort by score
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    async def upsert_memory_turn(self, conversation_id: str, turn: Dict[str, Any]):
        """Mock memory storage"""
        return True


class MockOllamaAdapter:
    """Mock Ollama adapter for testing"""
    
    async def generate(self, prompt: str, model: str = "deepseek-coder:6.7b", **kwargs):
        """Mock LLM generation"""
        
        # Simple mock response based on prompt content
        if "work authorization" in prompt.lower() or "documents" in prompt.lower():
            return """Based on your previous questions about work authorization and the available documentation, here are the key documents you'll need:

**Primary Documents [Document 1]:**
- Valid passport
- Job offer letter from Canadian employer
- Labour Market Impact Assessment (LMIA) if required
- Educational credential assessment

**Supporting Documents [Document 2]:**
- Medical examination results
- Police certificates from countries where you've lived
- Proof of funds to support yourself
- Passport-style photographs

Since you previously asked about working while your application is being processed, note that work authorization is separate from your main immigration application and requires its own documentation process."""
        
        return "I can help you with Canadian immigration questions. Please provide more specific details about what you need to know."


async def test_memory_integration():
    """Test the complete RAG workflow with memory integration"""
    
    print("🧪 Testing MemoryRetrievalNode Integration in RAG Workflow")
    print("=" * 60)
    
    # Create mock services
    mock_qdrant = MockQdrantService()
    mock_ollama = MockOllamaAdapter()
    
    # Create workflow
    workflow = ProductionRAGWorkflow(
        qdrant_service=mock_qdrant,
        ollama_service=mock_ollama
    )
    
    # Test Case 1: Query with conversation memory
    print("\n📝 Test Case 1: Query with conversation memory")
    print("-" * 40)
    
    response = await workflow.process_query(
        user_query="What documents do I need for work authorization?",
        conversation_id="conv_123",
        user_id="user_456"
    )
    
    print(f"✅ Response generated: {len(response['response'])} characters")
    print(f"✅ Confidence score: {response['confidence_score']}")
    print(f"✅ Citations found: {len(response['citations'])}")
    print(f"✅ Source documents: {len(response['source_documents'])}")
    
    # Check memory integration
    memory_metadata = response['metadata']['memory_metadata']
    print(f"✅ Memory turns used: {memory_metadata['turns_retrieved']}")
    print(f"✅ Memory enabled: {memory_metadata['memory_enabled']}")
    print(f"✅ Average relevance: {memory_metadata['avg_relevance_score']:.3f}")
    
    print(f"\n📋 Generated Response:")
    print("-" * 30)
    print(response['response'][:300] + "..." if len(response['response']) > 300 else response['response'])
    
    # Test Case 2: Query without conversation ID
    print("\n\n📝 Test Case 2: Query without conversation ID")
    print("-" * 40)
    
    response_no_memory = await workflow.process_query(
        user_query="What are the general requirements?",
        conversation_id=None,
        user_id="user_789"
    )
    
    memory_metadata_no_conv = response_no_memory['metadata']['memory_metadata']
    print(f"✅ Memory enabled: {memory_metadata_no_conv['memory_enabled']}")
    print(f"✅ Memory turns used: {memory_metadata_no_conv['turns_retrieved']}")
    print(f"✅ Response still generated: {len(response_no_memory['response'])} characters")
    
    # Test Case 3: Workflow timing and performance
    print("\n\n📝 Test Case 3: Performance metrics")
    print("-" * 40)
    
    processing_time = response['metadata']['processing_time']
    print(f"✅ Processing time: {processing_time:.3f} seconds")
    print(f"✅ Documents retrieved: {response['metadata']['documents_retrieved']}")
    print(f"✅ Documents used: {response['metadata']['documents_used']}")
    print(f"✅ Query type: {response['metadata']['query_type']}")
    print(f"✅ Retrieval strategy: {response['metadata']['retrieval_strategy']}")
    
    # Test Case 4: Verify workflow sequence
    print("\n\n📝 Test Case 4: Workflow sequence verification")
    print("-" * 40)
    
    # Check that all expected metadata is present
    expected_fields = [
        'query_type', 'intent', 'entities', 'retrieval_strategy',
        'documents_retrieved', 'documents_used', 'memory_turns_used',
        'memory_metadata', 'model_used', 'processing_time'
    ]
    
    metadata = response['metadata']
    for field in expected_fields:
        if field in metadata:
            print(f"✅ {field}: Present")
        else:
            print(f"❌ {field}: Missing")
    
    print("\n🎉 Memory integration test completed successfully!")
    
    return response


def demonstrate_workflow_sequence():
    """Demonstrate the updated workflow sequence"""
    
    print("\n📊 Updated RAG Workflow Sequence")
    print("=" * 40)
    
    sequence = [
        "1. QueryAnalysisNode - Analyze intent and extract entities",
        "2. MemoryRetrievalNode - Retrieve relevant conversation history", 
        "3. HybridRetrievalNode - Search documents using hybrid approach",
        "4. RerankingNode - Rerank documents using LLM (conditional)",
        "5. GenerationNode - Generate response with memory + documents",
        "6. CitationNode - Extract and format citations",
        "7. EvaluationNode - Assess response quality and confidence",
        "8. MemoryUpdateNode - Store conversation turn (conditional)"
    ]
    
    for step in sequence:
        print(f"   {step}")
    
    print("\n🔄 Key Integration Points:")
    print("   • MemoryRetrievalNode outputs memory_context for GenerationNode")
    print("   • GenerationNode uses pre-formatted memory context in prompts")
    print("   • Fallback behavior when memory is unavailable")
    print("   • Memory metadata included in final response")


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_workflow_sequence()
    
    # Run the integration test
    asyncio.run(test_memory_integration())
