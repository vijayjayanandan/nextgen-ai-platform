# tests/fixtures/test_helpers.py
"""
Test helper functions and utilities for RAG system testing.
Provides common validation functions and test data generators.
"""

import re
from typing import List, Dict, Any, Optional
from app.services.rag.workflow_state import RAGState, QueryType


def create_test_state(
    user_query: str = "Test query",
    conversation_id: Optional[str] = None,
    user_id: str = "test_user",
    memory_context: str = "",
    raw_documents: List[Dict[str, Any]] = None
) -> RAGState:
    """Create a test RAGState with default values"""
    
    state = RAGState(
        user_query=user_query,
        conversation_id=conversation_id,
        user_id=user_id
    )
    
    if memory_context:
        state.memory_context = memory_context
    
    if raw_documents:
        state.raw_documents = raw_documents
    
    return state


def assert_prompt_structure_valid(prompt: str, expected_sections: List[str]):
    """Validate that a prompt contains expected structural sections"""
    
    prompt_lower = prompt.lower()
    
    for section in expected_sections:
        section_lower = section.lower()
        assert section_lower in prompt_lower, f"Expected section '{section}' not found in prompt"


def assert_citation_format_valid(citations: List[Dict[str, Any]]):
    """Validate that citations follow the expected format"""
    
    for citation in citations:
        # Required fields
        assert "citation_text" in citation, "Citation missing citation_text field"
        assert "document_number" in citation, "Citation missing document_number field"
        
        # Validate citation text format
        citation_text = citation["citation_text"]
        assert re.match(r"\[Document \d+\]", citation_text), f"Invalid citation format: {citation_text}"
        
        # Validate document number
        doc_number = citation["document_number"]
        assert isinstance(doc_number, int), f"Document number should be int, got {type(doc_number)}"
        assert doc_number > 0, f"Document number should be positive, got {doc_number}"


def assert_document_metadata_valid(documents: List[Dict[str, Any]]):
    """Validate that document metadata contains required fields"""
    
    for doc in documents:
        # Required fields
        required_fields = ["document_number", "title", "source_type"]
        for field in required_fields:
            assert field in doc, f"Document missing required field: {field}"
        
        # Validate field types
        assert isinstance(doc["document_number"], int), "document_number should be int"
        assert isinstance(doc["title"], str), "title should be string"
        assert isinstance(doc["source_type"], str), "source_type should be string"
        
        # Optional fields validation
        if "relevance_score" in doc:
            score = doc["relevance_score"]
            assert isinstance(score, (int, float)), "relevance_score should be numeric"
            assert 0.0 <= score <= 1.0, f"relevance_score should be 0-1, got {score}"


def assert_valid_response_structure(response: Dict[str, Any]):
    """Validate that a response has the expected structure"""
    
    # Required fields for a valid response
    required_fields = ["response", "citations", "source_documents"]
    
    for field in required_fields:
        assert field in response, f"Response missing required field: {field}"
    
    # Validate field types
    assert isinstance(response["response"], str), "Response should be string"
    assert isinstance(response["citations"], list), "Citations should be list"
    assert isinstance(response["source_documents"], list), "Source documents should be list"
    
    # Validate response content
    assert len(response["response"]) > 0, "Response should not be empty"


def assert_memory_metadata_valid(memory_turns: List[Dict[str, Any]]):
    """Validate that memory turns have proper metadata"""
    
    for turn in memory_turns:
        # Required fields
        required_fields = ["turn_id", "user_query", "assistant_response", "timestamp"]
        for field in required_fields:
            assert field in turn, f"Memory turn missing required field: {field}"
        
        # Validate field types
        assert isinstance(turn["turn_id"], str), "turn_id should be string"
        assert isinstance(turn["user_query"], str), "user_query should be string"
        assert isinstance(turn["assistant_response"], str), "assistant_response should be string"
        assert isinstance(turn["timestamp"], str), "timestamp should be string"
        
        # Validate content
        assert len(turn["user_query"]) > 0, "user_query should not be empty"
        assert len(turn["assistant_response"]) > 0, "assistant_response should not be empty"


def create_mock_search_results(
    query: str,
    num_results: int = 5,
    score_range: tuple = (0.6, 0.95)
) -> List[Dict[str, Any]]:
    """Create mock search results for testing"""
    
    import random
    
    results = []
    for i in range(num_results):
        score = random.uniform(score_range[0], score_range[1])
        
        result = {
            "id": f"doc_{i+1}_{hash(query) % 1000}",
            "title": f"Document {i+1} about {query[:20]}...",
            "content": f"This document contains information about {query}. " * 10,
            "source_type": random.choice(["policy", "guideline", "reference", "faq"]),
            "relevance_score": round(score, 3),
            "metadata": {
                "page_number": i + 1,
                "section": f"section_{i+1}",
                "last_updated": "2024-01-15"
            }
        }
        results.append(result)
    
    # Sort by relevance score (highest first)
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    return results


def validate_rag_state_consistency(state: RAGState) -> bool:
    """Validate that a RAG state is internally consistent"""
    
    # Basic field validation
    if not state.user_query:
        return False
    
    # If citations exist, should have source documents
    if hasattr(state, 'citations') and state.citations:
        if not hasattr(state, 'source_documents') or not state.source_documents:
            return False
        
        # Citation numbers should match available source documents
        cited_numbers = {c["document_number"] for c in state.citations}
        source_numbers = {d["document_number"] for d in state.source_documents}
        
        if not cited_numbers.issubset(source_numbers):
            return False
    
    # If response exists, should have context prompt
    if hasattr(state, 'response') and state.response:
        if not hasattr(state, 'context_prompt') or not state.context_prompt:
            return False
    
    # Memory context should be present for conversational queries
    if hasattr(state, 'query_type') and state.query_type == QueryType.CONVERSATIONAL:
        if state.conversation_id and not state.memory_context:
            # This might be acceptable if memory retrieval failed
            pass
    
    return True


def generate_test_scenarios(scenario_type: str = "immigration") -> List[Dict[str, Any]]:
    """Generate test scenarios for different domains"""
    
    if scenario_type == "immigration":
        return [
            {
                "query": "What documents do I need for a work permit?",
                "expected_type": QueryType.SIMPLE,
                "expected_entities": ["work", "permit", "documents"],
                "context_documents": [
                    {
                        "id": "work_permit_guide",
                        "title": "Work Permit Application Guide",
                        "content": "To apply for a work permit in Canada, you need several documents...",
                        "source_type": "official_guide"
                    }
                ]
            },
            {
                "query": "Remember when we talked about Express Entry?",
                "expected_type": QueryType.CONVERSATIONAL,
                "expected_entities": ["Express Entry"],
                "memory_context": "Previous discussion about Express Entry eligibility...",
                "context_documents": []
            },
            {
                "query": "Compare Express Entry vs Provincial Nominee Program",
                "expected_type": QueryType.COMPLEX,
                "expected_entities": ["Express Entry", "Provincial Nominee Program"],
                "context_documents": [
                    {
                        "id": "express_entry_guide",
                        "title": "Express Entry System Guide",
                        "content": "Express Entry is a system for managing applications...",
                        "source_type": "official_guide"
                    },
                    {
                        "id": "pnp_guide",
                        "title": "Provincial Nominee Program Guide", 
                        "content": "The Provincial Nominee Program allows provinces...",
                        "source_type": "official_guide"
                    }
                ]
            }
        ]
    
    return []


def measure_test_performance(func):
    """Decorator to measure test performance"""
    
    import time
    import functools
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Add performance metadata to result if it's a RAGState
        if hasattr(result, '__dict__'):
            result._test_execution_time = execution_time
        
        return result
    
    return wrapper


def create_large_test_dataset(size: int = 100) -> List[Dict[str, Any]]:
    """Create a large test dataset for performance testing"""
    
    import random
    
    topics = [
        "work permits", "student visas", "citizenship", "family sponsorship",
        "Express Entry", "Provincial Nominee Program", "refugee protection",
        "temporary residence", "permanent residence", "immigration appeals"
    ]
    
    source_types = ["policy", "guideline", "reference", "faq", "form", "checklist"]
    
    documents = []
    for i in range(size):
        topic = random.choice(topics)
        source_type = random.choice(source_types)
        
        doc = {
            "id": f"large_test_doc_{i}",
            "title": f"{topic.title()} - {source_type.title()} {i}",
            "content": f"This document provides detailed information about {topic}. " * 50,
            "source_type": source_type,
            "relevance_score": random.uniform(0.3, 0.95),
            "metadata": {
                "page_number": random.randint(1, 100),
                "section": f"section_{random.randint(1, 10)}",
                "topic": topic,
                "complexity": random.choice(["basic", "intermediate", "advanced"])
            }
        }
        documents.append(doc)
    
    return documents


def validate_test_environment() -> Dict[str, bool]:
    """Validate that the test environment is properly configured"""
    
    checks = {}
    
    # Check if required modules can be imported
    try:
        from app.services.rag.workflow_state import RAGState
        checks["workflow_state_import"] = True
    except ImportError:
        checks["workflow_state_import"] = False
    
    try:
        from app.services.rag.nodes.query_analysis import QueryAnalysisNode
        checks["query_analysis_import"] = True
    except ImportError:
        checks["query_analysis_import"] = False
    
    try:
        from tests.fixtures.mock_services import MockOllamaAdapter
        checks["mock_services_import"] = True
    except ImportError:
        checks["mock_services_import"] = False
    
    # Check if pytest is available
    try:
        import pytest
        checks["pytest_available"] = True
    except ImportError:
        checks["pytest_available"] = False
    
    # Check if asyncio is working
    try:
        import asyncio
        checks["asyncio_available"] = True
    except ImportError:
        checks["asyncio_available"] = False
    
    return checks


def generate_test_report_summary(test_results: Dict[str, Any]) -> str:
    """Generate a summary report from test results"""
    
    total_tests = test_results.get("total", 0)
    passed_tests = test_results.get("passed", 0)
    failed_tests = test_results.get("failed", 0)
    
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    summary = f"""
Test Summary:
=============
Total Tests: {total_tests}
Passed: {passed_tests}
Failed: {failed_tests}
Pass Rate: {pass_rate:.1f}%

Status: {"✅ PASSED" if failed_tests == 0 else "❌ FAILED"}
"""
    
    if "errors" in test_results and test_results["errors"]:
        summary += "\nErrors:\n"
        for error in test_results["errors"]:
            summary += f"  - {error}\n"
    
    return summary
