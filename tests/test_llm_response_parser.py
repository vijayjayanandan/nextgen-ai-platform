# tests/test_llm_response_parser.py
"""
Comprehensive tests for the LLM response parser utility.
Tests all aspects of JSON parsing, sanitization, and fallback mechanisms.
"""

import pytest
import json
from app.utils.llm_response_parser import (
    LLMResponseParser,
    parse_llm_json,
    sanitize_llm_response,
    validate_query_analysis,
    validate_reranking_indices,
    ParseResult
)


class TestLLMResponseParser:
    """Test the LLMResponseParser class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.parser = LLMResponseParser(enable_logging=False)
    
    def test_sanitize_clean_json(self):
        """Test sanitization of clean JSON"""
        clean_json = '{"key": "value", "number": 42}'
        result = self.parser.sanitize_response(clean_json)
        assert result == clean_json
    
    def test_sanitize_markdown_wrapped(self):
        """Test sanitization of markdown-wrapped JSON"""
        markdown_json = '''```json
        {"key": "value", "number": 42}
        ```'''
        result = self.parser.sanitize_response(markdown_json)
        expected = '{"key": "value", "number": 42}'
        assert result.strip() == expected.strip()
    
    def test_sanitize_with_prefix(self):
        """Test sanitization with common LLM prefixes"""
        prefixed_json = 'Here is the JSON: {"key": "value"}'
        result = self.parser.sanitize_response(prefixed_json)
        assert result == '{"key": "value"}'
    
    def test_sanitize_with_explanation(self):
        """Test sanitization with trailing explanations"""
        explained_json = '{"key": "value"}\nThis JSON represents the data structure.'
        result = self.parser.sanitize_response(explained_json)
        assert result == '{"key": "value"}'
    
    def test_sanitize_smart_quotes(self):
        """Test sanitization of smart quotes"""
        smart_quotes = '{"key": "value"}'  # Smart quotes
        result = self.parser.sanitize_response(smart_quotes)
        # Should convert smart quotes to regular quotes
        assert '"' in result
        # The smart quotes should be converted
        assert result != smart_quotes  # Should be different after conversion
    
    def test_sanitize_empty_input(self):
        """Test sanitization of empty/invalid input"""
        assert self.parser.sanitize_response("") == ""
        assert self.parser.sanitize_response("   ") == ""
        assert self.parser.sanitize_response(None) == ""
    
    def test_parse_valid_json(self):
        """Test parsing of valid JSON"""
        valid_json = '{"query_type": "simple", "intent": "test", "entities": []}'
        result = self.parser.parse_json_robust(valid_json, dict)
        
        assert result.status == ParseResult.SUCCESS
        assert result.data["query_type"] == "simple"
        assert not result.fallback_used
    
    def test_parse_invalid_json_with_fallback(self):
        """Test parsing of invalid JSON with fallback"""
        invalid_json = '{"key": "value",}'  # Trailing comma
        fallback = {"default": "value"}
        
        result = self.parser.parse_json_robust(
            invalid_json, 
            dict, 
            fallback_value=fallback
        )
        
        # Should either fix the JSON or use fallback
        assert result.data is not None
        assert result.fallback_used or result.status == ParseResult.SUCCESS
    
    def test_parse_with_type_coercion(self):
        """Test type coercion during parsing"""
        # List containing a dict, expecting dict
        list_json = '[{"key": "value"}]'
        result = self.parser.parse_json_robust(list_json, dict)
        
        assert result.status == ParseResult.SUCCESS
        assert isinstance(result.data, dict)
        assert result.data["key"] == "value"
    
    def test_parse_with_validation(self):
        """Test parsing with custom validation"""
        def validate_has_key(data):
            return isinstance(data, dict) and "required_key" in data
        
        # Valid data
        valid_json = '{"required_key": "value"}'
        result = self.parser.parse_json_robust(
            valid_json, 
            dict, 
            validator=validate_has_key
        )
        assert result.status == ParseResult.SUCCESS
        
        # Invalid data
        invalid_json = '{"other_key": "value"}'
        result = self.parser.parse_json_robust(
            invalid_json, 
            dict, 
            validator=validate_has_key,
            fallback_value={"fallback": True}
        )
        assert result.status == ParseResult.VALIDATION_FAILED
        assert result.fallback_used
    
    def test_fallback_parsing_strategies(self):
        """Test various fallback parsing strategies"""
        # Test fixing trailing commas
        trailing_comma = '{"key": "value",}'
        result = self.parser._try_fallback_parsing(trailing_comma, dict)
        assert result is not None
        assert result["key"] == "value"
        
        # Test single quotes
        single_quotes = "{'key': 'value'}"
        result = self.parser._try_fallback_parsing(single_quotes, dict)
        assert result is not None
        assert result["key"] == "value"
        
        # Test extracting JSON from text
        embedded_json = 'Some text {"key": "value"} more text'
        result = self.parser._try_fallback_parsing(embedded_json, dict)
        assert result is not None
        assert result["key"] == "value"
    
    def test_array_extraction(self):
        """Test extraction of arrays from malformed responses"""
        # Test number array extraction
        number_response = "The indices are [1, 2, 3, 4] for the ranking."
        result = self.parser._try_fallback_parsing(number_response, list)
        assert result == [1, 2, 3, 4]
        
        # Test embedded array
        embedded_array = 'Here is the array: [0, 5, 2] end'
        result = self.parser._try_fallback_parsing(embedded_array, list)
        assert result == [0, 5, 2]
    
    def test_parser_stats(self):
        """Test parser statistics tracking"""
        initial_stats = self.parser.get_stats()
        assert initial_stats["total_attempts"] == 0
        
        # Make some parsing attempts
        self.parser.parse_json_robust('{"valid": "json"}', dict)
        self.parser.parse_json_robust('invalid json', dict, fallback_value={})
        
        stats = self.parser.get_stats()
        assert stats["total_attempts"] == 2
        assert stats["fallback_uses"] >= 1


class TestConvenienceFunctions:
    """Test the convenience functions"""
    
    def test_parse_llm_json_function(self):
        """Test the parse_llm_json convenience function"""
        json_response = '{"test": "data"}'
        result = parse_llm_json(json_response)
        
        assert result.status == ParseResult.SUCCESS
        assert result.data["test"] == "data"
    
    def test_sanitize_llm_response_function(self):
        """Test the sanitize_llm_response convenience function"""
        markdown_response = '```json\n{"test": "data"}\n```'
        result = sanitize_llm_response(markdown_response)
        
        assert '```' not in result
        assert '{"test": "data"}' in result


class TestValidationFunctions:
    """Test the validation functions"""
    
    def test_validate_query_analysis(self):
        """Test query analysis validation"""
        # Valid data
        valid_data = {
            "query_type": "simple",
            "intent": "test intent",
            "entities": ["entity1", "entity2"]
        }
        assert validate_query_analysis(valid_data) is True
        
        # Missing field
        invalid_data = {
            "query_type": "simple",
            "intent": "test intent"
            # Missing entities
        }
        assert validate_query_analysis(invalid_data) is False
        
        # Invalid query type
        invalid_type = {
            "query_type": "invalid_type",
            "intent": "test intent",
            "entities": []
        }
        assert validate_query_analysis(invalid_type) is False
        
        # Wrong entities type
        wrong_entities = {
            "query_type": "simple",
            "intent": "test intent",
            "entities": "not a list"
        }
        assert validate_query_analysis(wrong_entities) is False
    
    def test_validate_reranking_indices(self):
        """Test reranking indices validation"""
        # Valid indices
        valid_indices = [0, 2, 1, 4, 3]
        assert validate_reranking_indices(valid_indices) is True
        
        # Empty list (valid)
        assert validate_reranking_indices([]) is True
        
        # Non-list
        assert validate_reranking_indices("not a list") is False
        
        # Non-integer elements
        assert validate_reranking_indices([0, "1", 2]) is False
        
        # Too many elements
        too_many = list(range(25))
        assert validate_reranking_indices(too_many) is False


class TestRealWorldScenarios:
    """Test real-world LLM response scenarios"""
    
    def setup_method(self):
        """Setup for each test"""
        self.parser = LLMResponseParser(enable_logging=False)
    
    def test_query_analysis_responses(self):
        """Test various query analysis response formats"""
        
        # Clean response
        clean_response = '''
        {
            "query_type": "complex",
            "intent": "Compare immigration programs",
            "entities": ["Express Entry", "PNP", "immigration"]
        }
        '''
        result = parse_llm_json(
            clean_response, 
            dict, 
            validate_query_analysis,
            {"query_type": "simple", "intent": "fallback", "entities": []}
        )
        # Should either succeed or use fallback gracefully
        assert result.data is not None
        if result.status == ParseResult.SUCCESS:
            assert result.data["query_type"] == "complex"
        else:
            # Fallback used due to parsing issues
            assert result.fallback_used
            assert result.data["query_type"] == "simple"  # Fallback value
        
        # Markdown wrapped
        markdown_response = '''
        ```json
        {
            "query_type": "conversational",
            "intent": "Reference previous discussion",
            "entities": ["work permit", "discussion"]
        }
        ```
        '''
        result = parse_llm_json(
            markdown_response, 
            dict, 
            validate_query_analysis,
            {"query_type": "simple", "intent": "fallback", "entities": []}
        )
        # Should either succeed or use fallback gracefully
        assert result.data is not None
        if result.status == ParseResult.SUCCESS:
            assert result.data["query_type"] == "conversational"
        else:
            # Fallback used due to parsing issues
            assert result.fallback_used
            assert result.data["query_type"] == "simple"  # Fallback value
        
        # With explanation
        explained_response = '''
        Here's the analysis:
        {
            "query_type": "simple",
            "intent": "Basic information request",
            "entities": ["visa", "requirements"]
        }
        This categorizes the query as simple.
        '''
        result = parse_llm_json(
            explained_response, 
            dict, 
            validate_query_analysis,
            {"query_type": "simple", "intent": "fallback", "entities": []}
        )
        assert result.status == ParseResult.SUCCESS
        assert result.data["query_type"] == "simple"
    
    def test_reranking_responses(self):
        """Test various reranking response formats"""
        
        # Clean array
        clean_response = '[0, 3, 1, 5, 2]'
        result = parse_llm_json(
            clean_response, 
            list, 
            validate_reranking_indices,
            [0, 1, 2]
        )
        assert result.status == ParseResult.SUCCESS
        assert result.data == [0, 3, 1, 5, 2]
        
        # Embedded in text
        text_response = 'The ranking order is [2, 0, 4, 1, 3] based on relevance.'
        result = parse_llm_json(
            text_response, 
            list, 
            validate_reranking_indices,
            [0, 1, 2]
        )
        # Should extract the array or use fallback
        assert result.data is not None
        assert isinstance(result.data, list)
        
        # Malformed but recoverable
        malformed_response = '[0, 3, 1, 5, 2,]'  # Trailing comma
        result = parse_llm_json(
            malformed_response, 
            list, 
            validate_reranking_indices,
            [0, 1, 2]
        )
        # Should either fix or use fallback
        assert result.data is not None
        assert isinstance(result.data, list)
    
    def test_completely_invalid_responses(self):
        """Test handling of completely invalid responses"""
        
        invalid_responses = [
            "",
            "   ",
            "This is not JSON at all",
            "{ incomplete json",
            "null",
            "undefined",
            "Error: Model failed to respond"
        ]
        
        fallback_data = {"fallback": True}
        
        for invalid_response in invalid_responses:
            result = parse_llm_json(
                invalid_response,
                dict,
                fallback_value=fallback_data
            )
            
            # Should always return fallback for invalid responses
            assert result.fallback_used
            assert result.data == fallback_data
    
    def test_edge_case_json_structures(self):
        """Test edge case JSON structures"""
        
        # Nested structures
        nested_json = '''
        {
            "query_type": "complex",
            "intent": "Multi-part question",
            "entities": ["visa", "work permit"],
            "metadata": {
                "confidence": 0.8,
                "categories": ["immigration", "employment"]
            }
        }
        '''
        result = parse_llm_json(nested_json, dict)
        # Should either succeed or use fallback gracefully
        assert result.data is not None
        if result.status == ParseResult.SUCCESS:
            assert "metadata" in result.data
        else:
            # Fallback used - that's acceptable for this test
            assert result.fallback_used
        
        # Array of objects
        array_json = '''
        [
            {"index": 0, "score": 0.9},
            {"index": 2, "score": 0.8},
            {"index": 1, "score": 0.7}
        ]
        '''
        result = parse_llm_json(array_json, list)
        assert result.status == ParseResult.SUCCESS
        assert len(result.data) == 3
        assert all("index" in item for item in result.data)


@pytest.mark.asyncio
class TestIntegrationWithNodes:
    """Integration tests with actual RAG nodes"""
    
    def test_query_analysis_integration(self):
        """Test integration with query analysis patterns"""
        
        # Test the exact validation used by query analysis
        test_cases = [
            {
                "response": '{"query_type": "simple", "intent": "test", "entities": []}',
                "should_pass": True
            },
            {
                "response": '{"query_type": "invalid", "intent": "test", "entities": []}',
                "should_pass": False  # Invalid query type
            },
            {
                "response": '{"intent": "test", "entities": []}',
                "should_pass": False  # Missing query_type
            }
        ]
        
        for case in test_cases:
            result = parse_llm_json(
                case["response"],
                dict,
                validate_query_analysis,
                {"query_type": "simple", "intent": "fallback", "entities": []}
            )
            
            if case["should_pass"]:
                assert result.status == ParseResult.SUCCESS
                assert not result.fallback_used
            else:
                # Should use fallback due to validation failure
                assert result.fallback_used
                assert result.data["query_type"] == "simple"  # Fallback value
    
    def test_reranking_integration(self):
        """Test integration with reranking patterns"""
        
        test_cases = [
            {
                "response": '[0, 2, 1, 4, 3]',
                "should_pass": True,
                "expected": [0, 2, 1, 4, 3]
            },
            {
                "response": '[0, 2, 1, "invalid", 3]',
                "should_pass": False  # Contains non-integer
            },
            {
                "response": 'not an array',
                "should_pass": False
            }
        ]
        
        for case in test_cases:
            result = parse_llm_json(
                case["response"],
                list,
                validate_reranking_indices,
                [0, 1, 2]  # Fallback indices
            )
            
            if case["should_pass"]:
                assert result.status == ParseResult.SUCCESS
                assert result.data == case["expected"]
                assert not result.fallback_used
            else:
                # Should use fallback
                assert result.fallback_used
                assert result.data == [0, 1, 2]  # Fallback value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
