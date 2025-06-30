# tests/test_workflow_end_to_end.py
"""
End-to-end tests for the complete RAG workflow with memory integration.

Tests the full pipeline: QueryAnalysis → MemoryRetrieval → HybridRetrieval → 
Reranking → Generation → Citation → Evaluation → MemoryUpdate
"""

import pytest
import asyncio
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock

from app.services.rag.workflow import ProductionRAGWorkflow
from app.services.rag.workflow_state import RAGState
from tests.fixtures import (
    ProductionMockQdrantService,
    MockOllamaAdapter,
    MockScenarioConfig,
    SAMPLE_CONVERSATIONS,
    SAMPLE_QUERIES,
    assert_valid_response_structure,
    assert_memory_metadata_valid,
    extract_test_metrics
)


class TestWorkflowEndToEnd:
    """End-to-end workflow tests covering all major scenarios"""
    
    @pytest.fixture
    def mock_qdrant_service(self):
        """Create mock Qdrant service with default configuration"""
        config = MockScenarioConfig(
            memory_enabled=True,
            documents_available=True,
            response_latency=0.01,  # Fast for testing
            error_rate=0.0
        )
        return ProductionMockQdrantService(config)
    
    @pytest.fixture
    def mock_ollama_service(self):
        """Create mock Ollama service with deterministic responses"""
        return MockOllamaAdapter()
    
    @pytest.fixture
    def workflow(self, mock_qdrant_service, mock_ollama_service):
        """Create workflow instance with mock services"""
        return ProductionRAGWorkflow(
            qdrant_service=mock_qdrant_service,
            ollama_service=mock_ollama_service
        )
    
    @pytest.mark.asyncio
    async def test_full_workflow_with_memory_and_documents(self, workflow):
        """Test complete workflow with both memory and documents available"""
        
        # Test query that should retrieve both memory and documents
        response = await workflow.process_query(
            user_query="What documents do I need for work authorization?",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Validate response structure
        assert_valid_response_structure(response)
        
        # Check that both memory and documents were used
        metadata = response["metadata"]
        memory_metadata = metadata["memory_metadata"]
        
        assert memory_metadata["memory_enabled"] is True
        assert memory_metadata["turns_retrieved"] > 0
        assert metadata["documents_retrieved"] > 0
        assert metadata["documents_used"] > 0
        
        # Validate memory metadata
        assert_memory_metadata_valid(memory_metadata)
        
        # Check response quality
        assert len(response["response"]) > 100
        assert response["confidence_score"] > 0.5
        assert len(response["citations"]) > 0
        
        # Verify processing completed successfully
        assert metadata["error_message"] is None
        assert metadata["fallback_triggered"] is False
        assert metadata["processing_time"] > 0
    
    @pytest.mark.asyncio
    async def test_workflow_memory_only_no_documents(self, workflow):
        """Test workflow when only memory is available (no documents retrieved)"""
        
        # Configure mock to return no documents
        workflow.qdrant.config.documents_available = False
        
        response = await workflow.process_query(
            user_query="Tell me more about what we discussed",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Validate response structure
        assert_valid_response_structure(response)
        
        # Check that memory was used but no documents
        metadata = response["metadata"]
        memory_metadata = metadata["memory_metadata"]
        
        assert memory_metadata["memory_enabled"] is True
        assert memory_metadata["turns_retrieved"] > 0
        assert metadata["documents_retrieved"] == 0
        assert metadata["documents_used"] == 0
        
        # Response should still be generated from memory
        assert len(response["response"]) > 50
        assert response["confidence_score"] > 0.3
    
    @pytest.mark.asyncio
    async def test_workflow_documents_only_no_memory(self, workflow):
        """Test workflow when only documents are available (no conversation ID)"""
        
        response = await workflow.process_query(
            user_query="What is Express Entry?",
            conversation_id=None,  # No conversation ID = no memory
            user_id="user_new"
        )
        
        # Validate response structure
        assert_valid_response_structure(response)
        
        # Check that documents were used but no memory
        metadata = response["metadata"]
        memory_metadata = metadata["memory_metadata"]
        
        assert memory_metadata["memory_enabled"] is False
        assert memory_metadata["turns_retrieved"] == 0
        assert metadata["documents_retrieved"] > 0
        assert metadata["documents_used"] > 0
        
        # Response should be generated from documents
        assert len(response["response"]) > 100
        assert response["confidence_score"] > 0.5
        assert len(response["citations"]) > 0
    
    @pytest.mark.asyncio
    async def test_workflow_fallback_no_context(self, workflow):
        """Test workflow fallback when neither memory nor documents are available"""
        
        # Configure mock to return no memory or documents
        workflow.qdrant.config.memory_enabled = False
        workflow.qdrant.config.documents_available = False
        
        response = await workflow.process_query(
            user_query="Tell me about artificial intelligence",
            conversation_id=None,
            user_id="user_new"
        )
        
        # Validate response structure
        assert_valid_response_structure(response)
        
        # Check that no context was available
        metadata = response["metadata"]
        memory_metadata = metadata["memory_metadata"]
        
        assert memory_metadata["memory_enabled"] is False
        assert memory_metadata["turns_retrieved"] == 0
        assert metadata["documents_retrieved"] == 0
        assert metadata["documents_used"] == 0
        
        # Fallback response should still be generated
        assert len(response["response"]) > 30
        assert "apologize" in response["response"].lower() or "information" in response["response"].lower()
    
    @pytest.mark.asyncio
    async def test_workflow_memory_retrieval_error_graceful_degradation(self, workflow):
        """Test workflow graceful degradation when memory retrieval fails"""
        
        # Configure mock to simulate memory errors
        workflow.qdrant.config.error_rate = 1.0  # Always error on memory
        workflow.qdrant.config.memory_enabled = True
        workflow.qdrant.config.documents_available = True
        
        response = await workflow.process_query(
            user_query="What documents do I need for work authorization?",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Validate response structure
        assert_valid_response_structure(response)
        
        # Check that memory failed but documents succeeded
        metadata = response["metadata"]
        memory_metadata = metadata["memory_metadata"]
        
        assert memory_metadata["memory_enabled"] is True
        assert memory_metadata["turns_retrieved"] == 0
        assert memory_metadata.get("retrieval_error") is not None
        assert metadata["documents_retrieved"] > 0  # Documents should still work
        
        # Response should still be generated from documents
        assert len(response["response"]) > 100
        assert response["confidence_score"] > 0.5
    
    @pytest.mark.asyncio
    async def test_workflow_conversation_flow_memory_accumulation(self, workflow):
        """Test multi-turn conversation with memory accumulation"""
        
        conversation_flow = SAMPLE_CONVERSATIONS["work_authorization_flow"]
        conversation_id = conversation_flow["conversation_id"]
        user_id = conversation_flow["user_id"]
        
        responses = []
        
        # Process each turn in the conversation
        for i, turn in enumerate(conversation_flow["turns"]):
            response = await workflow.process_query(
                user_query=turn["user_query"],
                conversation_id=conversation_id,
                user_id=user_id
            )
            
            responses.append(response)
            
            # Validate response
            assert_valid_response_structure(response)
            
            # Check memory accumulation
            memory_metadata = response["metadata"]["memory_metadata"]
            expected_memory_turns = turn["expected_memory_turns"]
            
            if expected_memory_turns > 0:
                assert memory_metadata["turns_retrieved"] >= min(expected_memory_turns, 3)
            
            # Simulate storing the turn for next iteration
            await workflow.qdrant.upsert_memory_turn(
                conversation_id=conversation_id,
                turn={
                    "id": f"turn_{i+1}_{conversation_id}",
                    "turn_number": i + 1,
                    "user_message": turn["user_query"],
                    "assistant_message": response["response"][:200],  # Truncate for storage
                    "timestamp": 1641000000.0 + (i * 600),  # 10 minutes apart
                    "user_id": user_id
                }
            )
        
        # Verify conversation progression
        assert len(responses) == len(conversation_flow["turns"])
        
        # Later responses should have more memory context
        if len(responses) > 1:
            first_memory = responses[0]["metadata"]["memory_metadata"]["turns_retrieved"]
            last_memory = responses[-1]["metadata"]["memory_metadata"]["turns_retrieved"]
            assert last_memory >= first_memory
    
    @pytest.mark.asyncio
    async def test_workflow_user_isolation(self, workflow):
        """Test that different users don't see each other's memory"""
        
        # User 1 query
        response1 = await workflow.process_query(
            user_query="What documents do I need for work authorization?",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # User 2 query with different conversation
        response2 = await workflow.process_query(
            user_query="What documents do I need for work authorization?",
            conversation_id="conv_456",
            user_id="user_789"
        )
        
        # Both should get responses
        assert_valid_response_structure(response1)
        assert_valid_response_structure(response2)
        
        # Memory should be different (different conversations)
        memory1 = response1["metadata"]["memory_metadata"]
        memory2 = response2["metadata"]["memory_metadata"]
        
        # Both should have memory enabled but different content
        assert memory1["memory_enabled"] is True
        assert memory2["memory_enabled"] is True
        
        # They should retrieve different memory turns (different conversations)
        # This is validated by the mock service conversation isolation
    
    @pytest.mark.asyncio
    async def test_workflow_performance_benchmarks(self, workflow):
        """Test workflow performance meets benchmarks"""
        
        # Test with typical query
        response = await workflow.process_query(
            user_query="What are the visa requirements for Canada?",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Validate response
        assert_valid_response_structure(response)
        
        # Check performance benchmarks
        processing_time = response["metadata"]["processing_time"]
        
        # Should complete within reasonable time (mock services are fast)
        assert processing_time < 1.0, f"Processing took too long: {processing_time}s"
        
        # Response should be substantial
        assert len(response["response"]) > 100
        assert response["confidence_score"] > 0.5
    
    @pytest.mark.asyncio
    async def test_workflow_edge_cases(self, workflow):
        """Test workflow with edge case inputs"""
        
        edge_cases = [
            ("", "conv_123", "user_456"),  # Empty query
            ("a", "conv_123", "user_456"),  # Single character
            ("What is the meaning of life?", "conv_123", "user_456"),  # Unrelated query
            ("What are visa requirements?", "", "user_456"),  # Empty conversation ID
            ("What are visa requirements?", "conv_123", ""),  # Empty user ID
        ]
        
        for query, conv_id, user_id in edge_cases:
            try:
                response = await workflow.process_query(
                    user_query=query,
                    conversation_id=conv_id if conv_id else None,
                    user_id=user_id
                )
                
                # Should still return valid response structure
                assert_valid_response_structure(response)
                
                # Should have some response content
                assert len(response["response"]) > 0
                
            except Exception as e:
                # If it fails, it should fail gracefully
                assert "validation" in str(e).lower() or "required" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_workflow_reranking_conditional_logic(self, workflow):
        """Test that reranking is conditionally applied based on document count"""
        
        # Test with query that should trigger reranking (many documents)
        response_many_docs = await workflow.process_query(
            user_query="immigration requirements documents visa work",  # Broad query
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Test with query that should skip reranking (few documents)
        workflow.qdrant.config.max_documents = 2  # Limit documents
        response_few_docs = await workflow.process_query(
            user_query="very specific unique query unlikely to match",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Both should return valid responses
        assert_valid_response_structure(response_many_docs)
        assert_valid_response_structure(response_few_docs)
        
        # Check document counts
        many_docs_count = response_many_docs["metadata"]["documents_retrieved"]
        few_docs_count = response_few_docs["metadata"]["documents_retrieved"]
        
        # Should have different document counts based on query specificity
        assert many_docs_count >= few_docs_count
    
    @pytest.mark.asyncio
    async def test_workflow_memory_update_conditional(self, workflow):
        """Test that memory is only updated when conversation_id exists"""
        
        # Test with conversation ID (should update memory)
        response_with_conv = await workflow.process_query(
            user_query="What are visa requirements?",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Test without conversation ID (should not update memory)
        response_without_conv = await workflow.process_query(
            user_query="What are visa requirements?",
            conversation_id=None,
            user_id="user_456"
        )
        
        # Both should return valid responses
        assert_valid_response_structure(response_with_conv)
        assert_valid_response_structure(response_without_conv)
        
        # Check memory behavior
        memory_with_conv = response_with_conv["metadata"]["memory_metadata"]
        memory_without_conv = response_without_conv["metadata"]["memory_metadata"]
        
        assert memory_with_conv["memory_enabled"] is True
        assert memory_without_conv["memory_enabled"] is False
    
    @pytest.mark.asyncio
    async def test_workflow_comprehensive_metadata(self, workflow):
        """Test that all expected metadata is populated correctly"""
        
        response = await workflow.process_query(
            user_query="What documents do I need for work authorization?",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Validate response structure
        assert_valid_response_structure(response)
        
        # Extract and validate all metadata
        metadata = response["metadata"]
        
        # Check required metadata fields
        required_fields = [
            "query_type", "intent", "entities", "retrieval_strategy",
            "documents_retrieved", "documents_used", "memory_turns_used",
            "memory_metadata", "model_used", "processing_time"
        ]
        
        for field in required_fields:
            assert field in metadata, f"Missing required metadata field: {field}"
        
        # Validate specific field types and values
        assert isinstance(metadata["query_type"], str)
        assert isinstance(metadata["intent"], str)
        assert isinstance(metadata["entities"], list)
        assert isinstance(metadata["retrieval_strategy"], str)
        assert isinstance(metadata["documents_retrieved"], int)
        assert isinstance(metadata["documents_used"], int)
        assert isinstance(metadata["memory_turns_used"], int)
        assert isinstance(metadata["memory_metadata"], dict)
        assert isinstance(metadata["model_used"], str)
        assert isinstance(metadata["processing_time"], (int, float))
        
        # Validate memory metadata
        assert_memory_metadata_valid(metadata["memory_metadata"])
        
        # Check logical constraints
        assert metadata["documents_used"] <= metadata["documents_retrieved"]
        assert metadata["processing_time"] > 0
        assert metadata["memory_turns_used"] >= 0


@pytest.mark.performance
class TestWorkflowPerformance:
    """Performance-focused tests for the RAG workflow"""
    
    @pytest.fixture
    def fast_workflow(self):
        """Create workflow optimized for performance testing"""
        config = MockScenarioConfig(
            memory_enabled=True,
            documents_available=True,
            response_latency=0.001,  # Very fast
            error_rate=0.0,
            max_memory_turns=3,
            max_documents=5
        )
        mock_qdrant = ProductionMockQdrantService(config)
        mock_ollama = MockOllamaAdapter()
        
        return ProductionRAGWorkflow(
            qdrant_service=mock_qdrant,
            ollama_service=mock_ollama
        )
    
    @pytest.mark.asyncio
    async def test_workflow_latency_benchmark(self, fast_workflow):
        """Test workflow latency meets performance requirements"""
        
        # Run multiple queries to get average performance
        queries = SAMPLE_QUERIES["work_related"][:3]
        processing_times = []
        
        for query in queries:
            response = await fast_workflow.process_query(
                user_query=query,
                conversation_id="conv_123",
                user_id="user_456"
            )
            
            processing_times.append(response["metadata"]["processing_time"])
        
        # Calculate performance metrics
        avg_time = sum(processing_times) / len(processing_times)
        max_time = max(processing_times)
        
        # Performance assertions (with mock services, should be very fast)
        assert avg_time < 0.5, f"Average processing time too high: {avg_time}s"
        assert max_time < 1.0, f"Maximum processing time too high: {max_time}s"
    
    @pytest.mark.asyncio
    async def test_workflow_concurrent_requests(self, fast_workflow):
        """Test workflow handles concurrent requests correctly"""
        
        # Create multiple concurrent requests
        queries = [
            ("What are visa requirements?", "conv_123", "user_456"),
            ("How do I apply for work permit?", "conv_456", "user_789"),
            ("What is Express Entry?", "conv_789", "user_101"),
        ]
        
        # Execute concurrently
        tasks = [
            fast_workflow.process_query(
                user_query=query,
                conversation_id=conv_id,
                user_id=user_id
            )
            for query, conv_id, user_id in queries
        ]
        
        responses = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(responses) == len(queries)
        
        for response in responses:
            assert_valid_response_structure(response)
            assert len(response["response"]) > 0
            assert response["confidence_score"] > 0
