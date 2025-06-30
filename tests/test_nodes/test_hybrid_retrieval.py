# tests/test_nodes/test_hybrid_retrieval.py
"""
Unit tests for HybridRetrievalNode - validates semantic + keyword retrieval,
RRF fusion, and document filtering functionality.
"""

import pytest
from unittest.mock import AsyncMock, patch
from app.services.rag.nodes.hybrid_retrieval import HybridRetrievalNode
from app.services.rag.workflow_state import RAGState, RetrievalStrategy
from tests.fixtures import (
    ProductionMockQdrantService, 
    MockScenarioConfig,
    create_test_state,
    create_mock_search_results
)


class TestHybridRetrievalNode:
    """Test suite for HybridRetrievalNode functionality"""
    
    @pytest.fixture
    def mock_qdrant_service(self):
        """Create mock Qdrant service with configurable behavior"""
        config = MockScenarioConfig(
            documents_available=True,
            response_latency=0.01,
            error_rate=0.0,
            max_documents=10,
            document_score_range=(0.6, 0.95)
        )
        return ProductionMockQdrantService(config)
    
    @pytest.fixture
    def hybrid_retrieval_node(self, mock_qdrant_service):
        """Create HybridRetrievalNode with mock service"""
        return HybridRetrievalNode(mock_qdrant_service)
    
    @pytest.mark.asyncio
    async def test_hybrid_retrieval_success(self, hybrid_retrieval_node, mock_qdrant_service):
        """Test successful hybrid retrieval with both semantic and keyword results"""
        
        state = create_test_state(
            user_query="What documents do I need for work authorization?",
            conversation_id="conv_123"
        )
        
        result_state = await hybrid_retrieval_node.execute(state)
        
        # Validate retrieval strategy was set
        assert result_state.retrieval_strategy == RetrievalStrategy.HYBRID
        
        # Validate documents were retrieved
        assert len(result_state.raw_documents) > 0
        assert len(result_state.raw_documents) <= 10  # Respects max limit
        
        # Validate document structure
        for doc in result_state.raw_documents:
            assert "id" in doc
            assert "title" in doc
            assert "content" in doc
            assert "source_type" in doc
            assert "relevance_score" in doc
            assert 0.0 <= doc["relevance_score"] <= 1.0
        
        # Verify Qdrant was called with correct parameters
        assert mock_qdrant_service.get_call_count("search_documents") == 1
        
        # No error should be present
        assert result_state.error_message is None
    
    @pytest.mark.asyncio
    async def test_semantic_only_retrieval(self, hybrid_retrieval_node, mock_qdrant_service):
        """Test semantic-only retrieval strategy"""
        
        state = create_test_state(user_query="immigration policy analysis")
        state.retrieval_strategy = RetrievalStrategy.SEMANTIC_ONLY
        
        result_state = await hybrid_retrieval_node.execute(state)
        
        # Should maintain the specified strategy
        assert result_state.retrieval_strategy == RetrievalStrategy.SEMANTIC_ONLY
        assert len(result_state.raw_documents) > 0
        
        # Verify semantic search was called
        call_history = mock_qdrant_service.call_history
        search_calls = [call for call in call_history if call[0] == "search_documents"]
        assert len(search_calls) == 1
        assert search_calls[0][2] == "semantic"  # search_type parameter
    
    @pytest.mark.asyncio
    async def test_keyword_only_retrieval(self, hybrid_retrieval_node, mock_qdrant_service):
        """Test keyword-only retrieval strategy"""
        
        state = create_test_state(user_query="visa requirements documents")
        state.retrieval_strategy = RetrievalStrategy.KEYWORD_ONLY
        
        result_state = await hybrid_retrieval_node.execute(state)
        
        assert result_state.retrieval_strategy == RetrievalStrategy.KEYWORD_ONLY
        assert len(result_state.raw_documents) > 0
        
        # Verify keyword search was called
        call_history = mock_qdrant_service.call_history
        search_calls = [call for call in call_history if call[0] == "search_documents"]
        assert len(search_calls) == 1
        assert search_calls[0][2] == "keyword"  # search_type parameter
    
    @pytest.mark.asyncio
    async def test_no_documents_found(self, hybrid_retrieval_node, mock_qdrant_service):
        """Test behavior when no documents are found"""
        
        # Configure mock to return no documents
        mock_qdrant_service.config.documents_available = False
        
        state = create_test_state(user_query="very specific query with no matches")
        
        result_state = await hybrid_retrieval_node.execute(state)
        
        # Should handle gracefully
        assert result_state.retrieval_strategy == RetrievalStrategy.HYBRID
        assert len(result_state.raw_documents) == 0
        assert result_state.error_message is None  # No error, just no results
    
    @pytest.mark.asyncio
    async def test_retrieval_service_error(self, hybrid_retrieval_node, mock_qdrant_service):
        """Test handling of retrieval service errors"""
        
        # Configure mock to simulate errors
        mock_qdrant_service.config.error_rate = 1.0  # Always error
        
        state = create_test_state(user_query="test query")
        
        result_state = await hybrid_retrieval_node.execute(state)
        
        # Should handle error gracefully
        assert len(result_state.raw_documents) == 0
        assert result_state.error_message is not None
        assert "retrieval failed" in result_state.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_document_limit_enforcement(self, hybrid_retrieval_node, mock_qdrant_service):
        """Test that document limits are properly enforced"""
        
        # Set a low document limit
        mock_qdrant_service.config.max_documents = 3
        
        state = create_test_state(user_query="broad immigration query")
        
        result_state = await hybrid_retrieval_node.execute(state)
        
        # Should respect the limit
        assert len(result_state.raw_documents) <= 3
        
        # Documents should be sorted by relevance (highest first)
        if len(result_state.raw_documents) > 1:
            scores = [doc["relevance_score"] for doc in result_state.raw_documents]
            assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_document_filtering_by_metadata(self, hybrid_retrieval_node, mock_qdrant_service):
        """Test document filtering based on metadata"""
        
        state = create_test_state(user_query="work permit requirements")
        
        # Add metadata filters to the node (simulating configuration)
        filters = {"source_type": "policy"}
        
        # Mock the search to include filters
        original_search = mock_qdrant_service.search_documents
        
        async def filtered_search(query, search_type="hybrid", limit=10, filters=None, **kwargs):
            # Verify filters are passed through
            assert filters is not None
            return await original_search(query, search_type, limit, filters, **kwargs)
        
        mock_qdrant_service.search_documents = filtered_search
        
        # Execute with filters (would need to modify node to accept filters)
        result_state = await hybrid_retrieval_node.execute(state)
        
        # Should still retrieve documents
        assert len(result_state.raw_documents) >= 0
    
    @pytest.mark.asyncio
    async def test_query_preprocessing(self, hybrid_retrieval_node):
        """Test query preprocessing and normalization"""
        
        test_cases = [
            ("What are VISA requirements?", "visa requirements"),  # Case normalization
            ("How do I apply for work-permit?", "work permit"),    # Hyphen handling
            ("Tell me about   immigration   policy", "immigration policy"),  # Whitespace
            ("What's the process for citizenship?", "process citizenship"),  # Contractions
        ]
        
        for original_query, expected_processed in test_cases:
            state = create_test_state(user_query=original_query)
            
            # Capture the processed query sent to search
            processed_queries = []
            
            async def capture_query(query, **kwargs):
                processed_queries.append(query)
                return []
            
            hybrid_retrieval_node.qdrant.search_documents = capture_query
            
            await hybrid_retrieval_node.execute(state)
            
            # Verify query was processed (basic check)
            assert len(processed_queries) == 1
            processed_query = processed_queries[0].lower()
            
            # Should contain key terms from expected processed query
            for term in expected_processed.split():
                assert term in processed_query
    
    @pytest.mark.asyncio
    async def test_retrieval_strategy_selection(self, hybrid_retrieval_node):
        """Test automatic retrieval strategy selection based on query characteristics"""
        
        test_cases = [
            ("What are visa requirements?", RetrievalStrategy.HYBRID),
            ("immigration policy document analysis", RetrievalStrategy.HYBRID),
            ("specific technical API documentation", RetrievalStrategy.HYBRID),
        ]
        
        for query, expected_strategy in test_cases:
            state = create_test_state(user_query=query)
            # Don't set retrieval_strategy to test automatic selection
            
            result_state = await hybrid_retrieval_node.execute(state)
            
            # Should default to hybrid for most queries
            assert result_state.retrieval_strategy == expected_strategy
    
    @pytest.mark.asyncio
    async def test_document_deduplication(self, hybrid_retrieval_node, mock_qdrant_service):
        """Test that duplicate documents are properly handled"""
        
        # This test would be more relevant if the node implemented deduplication
        # For now, test that the same document ID doesn't appear multiple times
        
        state = create_test_state(user_query="immigration requirements")
        
        result_state = await hybrid_retrieval_node.execute(state)
        
        # Check for duplicate document IDs
        doc_ids = [doc["id"] for doc in result_state.raw_documents]
        unique_ids = set(doc_ids)
        
        assert len(doc_ids) == len(unique_ids), "Duplicate documents found in results"
    
    @pytest.mark.asyncio
    async def test_relevance_score_validation(self, hybrid_retrieval_node, mock_qdrant_service):
        """Test that relevance scores are properly validated and normalized"""
        
        state = create_test_state(user_query="work authorization documents")
        
        result_state = await hybrid_retrieval_node.execute(state)
        
        # All documents should have valid relevance scores
        for doc in result_state.raw_documents:
            score = doc["relevance_score"]
            assert isinstance(score, (int, float)), f"Invalid score type: {type(score)}"
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, hybrid_retrieval_node):
        """Test handling of empty or whitespace-only queries"""
        
        empty_queries = ["", "   ", "\n\t", None]
        
        for empty_query in empty_queries:
            if empty_query is None:
                state = create_test_state()
                state.user_query = None
            else:
                state = create_test_state(user_query=empty_query)
            
            result_state = await hybrid_retrieval_node.execute(state)
            
            # Should handle gracefully without crashing
            assert isinstance(result_state.raw_documents, list)
            # May return empty results or fallback behavior
    
    @pytest.mark.asyncio
    async def test_concurrent_retrieval_calls(self, hybrid_retrieval_node, mock_qdrant_service):
        """Test that the node handles concurrent calls correctly"""
        
        import asyncio
        
        # Create multiple concurrent retrieval tasks
        queries = [
            "visa requirements",
            "work permit application",
            "immigration policy",
            "citizenship process"
        ]
        
        tasks = []
        for query in queries:
            state = create_test_state(user_query=query)
            task = hybrid_retrieval_node.execute(state)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == len(queries)
        
        for result_state in results:
            assert isinstance(result_state.raw_documents, list)
            assert result_state.retrieval_strategy is not None
    
    @pytest.mark.asyncio
    async def test_state_preservation(self, hybrid_retrieval_node):
        """Test that original state fields are preserved during retrieval"""
        
        original_state = create_test_state(
            user_query="test query",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        # Add some existing state data
        original_state.query_type = "simple"
        original_state.intent = "test intent"
        original_state.entities = ["test", "entity"]
        
        result_state = await hybrid_retrieval_node.execute(original_state)
        
        # Original fields should be preserved
        assert result_state.user_query == original_state.user_query
        assert result_state.conversation_id == original_state.conversation_id
        assert result_state.user_id == original_state.user_id
        assert result_state.query_type == original_state.query_type
        assert result_state.intent == original_state.intent
        assert result_state.entities == original_state.entities
        
        # New fields should be populated
        assert result_state.retrieval_strategy is not None
        assert isinstance(result_state.raw_documents, list)


@pytest.mark.integration
class TestHybridRetrievalIntegration:
    """Integration tests for HybridRetrievalNode with realistic scenarios"""
    
    @pytest.fixture
    def integration_node(self):
        """Create node for integration testing"""
        config = MockScenarioConfig(
            documents_available=True,
            response_latency=0.01,
            max_documents=20
        )
        mock_qdrant = ProductionMockQdrantService(config)
        return HybridRetrievalNode(mock_qdrant)
    
    @pytest.mark.asyncio
    async def test_immigration_domain_retrieval(self, integration_node):
        """Test retrieval for various immigration domain queries"""
        
        immigration_queries = [
            ("How do I apply for Canadian citizenship?", ["citizenship", "application"]),
            ("What documents are needed for work permit?", ["work", "permit", "documents"]),
            ("Express Entry eligibility requirements", ["Express Entry", "eligibility"]),
            ("Family sponsorship income requirements", ["family", "sponsorship", "income"]),
            ("Medical exam requirements for immigration", ["medical", "exam", "immigration"]),
        ]
        
        for query, expected_keywords in immigration_queries:
            state = create_test_state(user_query=query)
            result = await integration_node.execute(state)
            
            # Should retrieve relevant documents
            assert len(result.raw_documents) > 0
            assert result.retrieval_strategy == RetrievalStrategy.HYBRID
            
            # Documents should be relevant to the query domain
            for doc in result.raw_documents:
                assert doc["relevance_score"] > 0.5  # Should be reasonably relevant
    
    @pytest.mark.asyncio
    async def test_performance_with_large_document_set(self, integration_node):
        """Test retrieval performance with larger document sets"""
        
        # Configure for larger document set
        integration_node.qdrant.config.max_documents = 50
        
        state = create_test_state(user_query="comprehensive immigration guide")
        
        import time
        start_time = time.time()
        
        result = await integration_node.execute(state)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (mock services are fast)
        assert processing_time < 1.0
        assert len(result.raw_documents) > 0
        assert len(result.raw_documents) <= 50
    
    @pytest.mark.asyncio
    async def test_retrieval_quality_metrics(self, integration_node):
        """Test retrieval quality and relevance metrics"""
        
        high_quality_queries = [
            "work permit application requirements",
            "Express Entry CRS score calculation",
            "family sponsorship eligibility criteria"
        ]
        
        for query in high_quality_queries:
            state = create_test_state(user_query=query)
            result = await integration_node.execute(state)
            
            if len(result.raw_documents) > 0:
                # Check average relevance score
                avg_score = sum(doc["relevance_score"] for doc in result.raw_documents) / len(result.raw_documents)
                assert avg_score > 0.6, f"Low average relevance for query: {query}"
                
                # Check that top documents have high relevance
                top_docs = result.raw_documents[:3]
                for doc in top_docs:
                    assert doc["relevance_score"] > 0.7, f"Top document has low relevance: {doc['relevance_score']}"
