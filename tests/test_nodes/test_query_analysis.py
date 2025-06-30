# tests/test_nodes/test_query_analysis.py
"""
Unit tests for QueryAnalysisNode - validates query classification, intent extraction,
and entity recognition with various input scenarios.
"""

import pytest
import json
from unittest.mock import AsyncMock, patch
from app.services.rag.nodes.query_analysis import QueryAnalysisNode
from app.services.rag.workflow_state import RAGState, QueryType
from tests.fixtures import MockOllamaAdapter, create_test_state


class TestQueryAnalysisNode:
    """Test suite for QueryAnalysisNode functionality"""
    
    @pytest.fixture
    def mock_ollama(self):
        """Create mock Ollama service with deterministic responses"""
        return MockOllamaAdapter({
            "simple_query": '{"query_type": "simple", "intent": "Get basic information", "entities": ["visa", "requirements"]}',
            "conversational_query": '{"query_type": "conversational", "intent": "Continue previous discussion", "entities": ["work", "permit"]}',
            "complex_query": '{"query_type": "complex", "intent": "Compare multiple options", "entities": ["Express Entry", "PNP", "immigration"]}',
            "code_query": '{"query_type": "code_related", "intent": "Technical implementation", "entities": ["API", "JSON", "integration"]}',
            "invalid_json": 'This is not valid JSON response',
            "empty_response": ''
        })
    
    @pytest.fixture
    def query_analysis_node(self, mock_ollama):
        """Create QueryAnalysisNode with mock service"""
        return QueryAnalysisNode(mock_ollama)
    
    @pytest.mark.asyncio
    async def test_simple_query_analysis(self, query_analysis_node, mock_ollama):
        """Test analysis of simple informational queries"""
        
        # Setup mock response
        mock_ollama.response_templates["default"] = '{"query_type": "simple", "intent": "Get visa requirements", "entities": ["visa", "Canada", "requirements"]}'
        
        state = create_test_state(user_query="What are the visa requirements for Canada?")
        
        result_state = await query_analysis_node.execute(state)
        
        # Validate query type classification
        assert result_state.query_type == QueryType.SIMPLE
        assert result_state.intent == "Get visa requirements"
        assert "visa" in result_state.entities
        assert "Canada" in result_state.entities
        assert "requirements" in result_state.entities
        
        # Verify LLM was called
        assert mock_ollama.get_generation_count() == 1
    
    @pytest.mark.asyncio
    async def test_conversational_query_analysis(self, query_analysis_node, mock_ollama):
        """Test analysis of conversational queries referencing previous context"""
        
        mock_ollama.response_templates["default"] = '{"query_type": "conversational", "intent": "Continue previous discussion about work permits", "entities": ["work", "permit", "processing"]}'
        
        state = create_test_state(
            user_query="How long does processing take for what we discussed earlier?",
            conversation_id="conv_123"
        )
        
        result_state = await query_analysis_node.execute(state)
        
        assert result_state.query_type == QueryType.CONVERSATIONAL
        assert "previous discussion" in result_state.intent.lower()
        assert "work" in result_state.entities
        assert "permit" in result_state.entities
    
    @pytest.mark.asyncio
    async def test_complex_query_analysis(self, query_analysis_node, mock_ollama):
        """Test analysis of complex comparative queries"""
        
        mock_ollama.response_templates["default"] = '{"query_type": "complex", "intent": "Compare immigration pathways", "entities": ["Express Entry", "PNP", "CEC", "comparison"]}'
        
        state = create_test_state(
            user_query="Compare Express Entry vs Provincial Nominee Program vs Canadian Experience Class"
        )
        
        result_state = await query_analysis_node.execute(state)
        
        assert result_state.query_type == QueryType.COMPLEX
        assert "compare" in result_state.intent.lower()
        assert "Express Entry" in result_state.entities
        assert "PNP" in result_state.entities or "Provincial Nominee Program" in result_state.entities
    
    @pytest.mark.asyncio
    async def test_code_related_query_analysis(self, query_analysis_node, mock_ollama):
        """Test analysis of technical/code-related queries"""
        
        mock_ollama.response_templates["default"] = '{"query_type": "code_related", "intent": "API integration help", "entities": ["API", "JSON", "integration", "Python"]}'
        
        state = create_test_state(
            user_query="How do I integrate with the immigration API using Python and JSON?"
        )
        
        result_state = await query_analysis_node.execute(state)
        
        assert result_state.query_type == QueryType.CODE_RELATED
        assert "api" in result_state.intent.lower()
        assert "API" in result_state.entities
        assert "JSON" in result_state.entities
    
    @pytest.mark.asyncio
    async def test_json_parsing_error_fallback(self, query_analysis_node, mock_ollama):
        """Test fallback behavior when LLM returns invalid JSON"""
        
        # Mock invalid JSON response
        mock_ollama.response_templates["default"] = "This is not valid JSON at all"
        
        state = create_test_state(user_query="What are visa requirements?")
        
        result_state = await query_analysis_node.execute(state)
        
        # Should fallback to simple classification
        assert result_state.query_type == QueryType.SIMPLE
        assert result_state.intent == "General information request"
        assert isinstance(result_state.entities, list)
        assert result_state.error_message is not None
        assert "Query analysis failed" in result_state.error_message
    
    @pytest.mark.asyncio
    async def test_empty_response_fallback(self, query_analysis_node, mock_ollama):
        """Test fallback behavior when LLM returns empty response"""
        
        mock_ollama.response_templates["default"] = ""
        
        state = create_test_state(user_query="Tell me about immigration")
        
        result_state = await query_analysis_node.execute(state)
        
        # Should use fallback classification
        assert result_state.query_type in [QueryType.SIMPLE, QueryType.CONVERSATIONAL, QueryType.COMPLEX, QueryType.CODE_RELATED]
        assert result_state.intent == "General information request"
        assert isinstance(result_state.entities, list)
    
    @pytest.mark.asyncio
    async def test_fallback_classification_logic(self, query_analysis_node):
        """Test the fallback classification logic for different query types"""
        
        test_cases = [
            ("What is Python code for API?", QueryType.CODE_RELATED),
            ("Remember what we discussed about visas?", QueryType.CONVERSATIONAL),
            ("Compare and analyze Express Entry vs PNP", QueryType.COMPLEX),
            ("What are visa requirements?", QueryType.SIMPLE),
            ("How do I write a SQL query?", QueryType.CODE_RELATED),
            ("Earlier you mentioned work permits", QueryType.CONVERSATIONAL),
            ("Explain the pros and cons of different programs", QueryType.COMPLEX)
        ]
        
        for query, expected_type in test_cases:
            node = query_analysis_node
            result = node._fallback_classification(query)
            assert result == expected_type, f"Query '{query}' should be classified as {expected_type}, got {result}"
    
    @pytest.mark.asyncio
    async def test_simple_entity_extraction(self, query_analysis_node):
        """Test the simple entity extraction fallback"""
        
        test_cases = [
            ("What are visa requirements for Canada?", ["Canada", "visa"]),
            ("How do I apply for work permit?", ["permit"]),
            ("Tell me about Express Entry immigration", ["Express", "Entry", "immigration"]),
            ("Processing times for citizenship application", ["Processing", "citizenship", "application"])
        ]
        
        for query, expected_entities in test_cases:
            node = query_analysis_node
            entities = node._extract_simple_entities(query)
            
            for expected in expected_entities:
                assert expected in entities, f"Expected entity '{expected}' not found in {entities} for query '{query}'"
    
    @pytest.mark.asyncio
    async def test_llm_service_error_handling(self, query_analysis_node, mock_ollama):
        """Test handling of LLM service errors"""
        
        # Mock LLM service to raise exception
        mock_ollama.generate = AsyncMock(side_effect=Exception("LLM service unavailable"))
        
        state = create_test_state(user_query="What are visa requirements?")
        
        result_state = await query_analysis_node.execute(state)
        
        # Should fallback gracefully
        assert result_state.query_type in [QueryType.SIMPLE, QueryType.CONVERSATIONAL, QueryType.COMPLEX, QueryType.CODE_RELATED]
        assert result_state.intent == "General information request"
        assert isinstance(result_state.entities, list)
        assert result_state.error_message is not None
    
    @pytest.mark.asyncio
    async def test_edge_case_queries(self, query_analysis_node, mock_ollama):
        """Test analysis of edge case queries"""
        
        edge_cases = [
            ("", QueryType.SIMPLE),  # Empty query
            ("a", QueryType.SIMPLE),  # Single character
            ("???", QueryType.SIMPLE),  # Only punctuation
            ("What is the meaning of life?", QueryType.SIMPLE),  # Unrelated query
            ("🤔 visa requirements 🇨🇦", QueryType.SIMPLE),  # With emojis
        ]
        
        for query, expected_fallback_type in edge_cases:
            # Mock to return invalid JSON to trigger fallback
            mock_ollama.response_templates["default"] = "invalid json"
            
            state = create_test_state(user_query=query)
            result_state = await query_analysis_node.execute(state)
            
            # Should handle gracefully with fallback
            assert result_state.query_type == expected_fallback_type
            assert result_state.intent == "General information request"
            assert isinstance(result_state.entities, list)
    
    @pytest.mark.asyncio
    async def test_prompt_construction(self, query_analysis_node, mock_ollama):
        """Test that the analysis prompt is properly constructed"""
        
        state = create_test_state(user_query="What are visa requirements?")
        
        # Capture the prompt sent to LLM
        original_generate = mock_ollama.generate
        captured_prompt = None
        
        async def capture_prompt(prompt, **kwargs):
            nonlocal captured_prompt
            captured_prompt = prompt
            return '{"query_type": "simple", "intent": "test", "entities": []}'
        
        mock_ollama.generate = capture_prompt
        
        await query_analysis_node.execute(state)
        
        # Validate prompt structure
        assert captured_prompt is not None
        assert "Analyze this user query" in captured_prompt
        assert state.user_query in captured_prompt
        assert "JSON format" in captured_prompt
        assert "query_type" in captured_prompt
        assert "intent" in captured_prompt
        assert "entities" in captured_prompt
    
    @pytest.mark.asyncio
    async def test_model_parameters(self, query_analysis_node, mock_ollama):
        """Test that correct model parameters are used"""
        
        state = create_test_state(user_query="Test query")
        
        # Capture model parameters
        captured_params = None
        
        async def capture_params(**kwargs):
            nonlocal captured_params
            captured_params = kwargs
            return '{"query_type": "simple", "intent": "test", "entities": []}'
        
        mock_ollama.generate = capture_params
        
        await query_analysis_node.execute(state)
        
        # Validate model parameters
        assert captured_params is not None
        assert captured_params.get("model") == "mistral:7b"
        assert captured_params.get("temperature") == 0.1
        assert captured_params.get("max_tokens") == 200
    
    @pytest.mark.asyncio
    async def test_state_preservation(self, query_analysis_node, mock_ollama):
        """Test that original state fields are preserved"""
        
        original_state = create_test_state(
            user_query="Test query",
            conversation_id="conv_123",
            user_id="user_456"
        )
        
        mock_ollama.response_templates["default"] = '{"query_type": "simple", "intent": "test intent", "entities": ["test"]}'
        
        result_state = await query_analysis_node.execute(original_state)
        
        # Original fields should be preserved
        assert result_state.user_query == original_state.user_query
        assert result_state.conversation_id == original_state.conversation_id
        assert result_state.user_id == original_state.user_id
        
        # New fields should be populated
        assert result_state.query_type == QueryType.SIMPLE
        assert result_state.intent == "test intent"
        assert result_state.entities == ["test"]


@pytest.mark.integration
class TestQueryAnalysisIntegration:
    """Integration tests for QueryAnalysisNode with real-world scenarios"""
    
    @pytest.fixture
    def integration_node(self):
        """Create node for integration testing"""
        mock_ollama = MockOllamaAdapter()
        return QueryAnalysisNode(mock_ollama)
    
    @pytest.mark.asyncio
    async def test_immigration_domain_queries(self, integration_node):
        """Test analysis of various immigration domain queries"""
        
        immigration_queries = [
            ("How do I apply for Canadian citizenship?", QueryType.SIMPLE, ["citizenship", "Canadian"]),
            ("What's the difference between Express Entry and PNP?", QueryType.COMPLEX, ["Express Entry", "PNP"]),
            ("Remember when we talked about work permits?", QueryType.CONVERSATIONAL, ["work", "permit"]),
            ("Show me the API documentation for IRCC", QueryType.CODE_RELATED, ["API", "IRCC"]),
        ]
        
        for query, expected_type, expected_entities in immigration_queries:
            # Configure mock response based on expected type
            mock_response = {
                "query_type": expected_type.value,
                "intent": f"Handle {expected_type.value} query",
                "entities": expected_entities
            }
            
            integration_node.ollama.response_templates["default"] = json.dumps(mock_response)
            
            state = create_test_state(user_query=query)
            result = await integration_node.execute(state)
            
            assert result.query_type == expected_type
            for entity in expected_entities:
                assert entity in result.entities
    
    @pytest.mark.asyncio
    async def test_multilingual_query_handling(self, integration_node):
        """Test handling of queries with mixed languages or special characters"""
        
        multilingual_queries = [
            "What are visa requirements for français speakers?",
            "How to apply for 工作许可 in Canada?",
            "Visa requirements for España citizens",
        ]
        
        for query in multilingual_queries:
            # Should fallback gracefully for non-English content
            integration_node.ollama.response_templates["default"] = "invalid json"  # Force fallback
            
            state = create_test_state(user_query=query)
            result = await integration_node.execute(state)
            
            # Should handle gracefully with fallback classification
            assert result.query_type in [QueryType.SIMPLE, QueryType.CONVERSATIONAL, QueryType.COMPLEX, QueryType.CODE_RELATED]
            assert result.intent == "General information request"
            assert isinstance(result.entities, list)
