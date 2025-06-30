"""
Citation Functionality Test Script

This script validates the citation formatting and integration functionality
for the RAG system, ensuring proper attribution and transparency.
"""

import asyncio
import sys
from typing import List, Dict, Any

def test_citation_formatter():
    """Test the citation formatting functions"""
    
    print("🔍 Testing Citation Formatter")
    print("=" * 50)
    
    try:
        from app.utils.citation_formatter import format_citations, append_citations_to_response
        print("✅ Citation formatter imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test data with various metadata structures
    test_chunks = [
        {
            "content": "Canadian citizens must meet residency requirements...",
            "title": "Canadian Immigration Guide 2024",
            "page_number": 15,
            "section_title": "Citizenship Requirements",
            "source_type": "policy",
            "metadata": {
                "document_id": "doc-123",
                "chunk_index": 5
            }
        },
        {
            "content": "Language proficiency tests include CELPIP...",
            "metadata": {
                "document_title": "Canadian Immigration Guide 2024", 
                "page_number": 16,
                "section_title": "Language Requirements",
                "source_type": "policy",
                "document_id": "doc-123",
                "chunk_index": 6
            }
        },
        {
            "content": "Processing times vary by application type...",
            "title": "IRCC Processing Times",
            "metadata": {
                "section_title": "Current Wait Times",
                "source_type": "guideline",
                "document_id": "doc-456"
            }
        },
        {
            "content": "Duplicate content for testing deduplication...",
            "title": "Canadian Immigration Guide 2024",
            "page_number": 15,
            "section_title": "Citizenship Requirements",
            "source_type": "policy"
        }
    ]
    
    # Test 1: Basic citation formatting
    print("\n📋 Test 1: Basic Citation Formatting")
    print("-" * 40)
    
    citations = format_citations(test_chunks)
    print("Formatted Citations:")
    print(citations)
    
    # Validate expected format
    expected_patterns = [
        "📄 Canadian Immigration Guide 2024 (page 15, section: Citizenship Requirements)",
        "📄 Canadian Immigration Guide 2024 (page 16, section: Language Requirements)", 
        "📄 IRCC Processing Times (section: Current Wait Times)"
    ]
    
    success = True
    for pattern in expected_patterns:
        if pattern not in citations:
            print(f"❌ Missing expected citation: {pattern}")
            success = False
        else:
            print(f"✅ Found expected citation: {pattern}")
    
    if not success:
        return False
    
    # Test 2: Response with citations
    print("\n📝 Test 2: Response with Citations")
    print("-" * 35)
    
    llm_response = """To become a Canadian citizen, you must meet several key requirements:

1. **Residency**: You must have been physically present in Canada for at least 1,095 days (3 years) during the 5 years immediately before applying.

2. **Language Proficiency**: Demonstrate adequate knowledge of English or French through approved tests like CELPIP or IELTS.

Processing times for citizenship applications vary depending on the type of application and current workload."""
    
    final_response = append_citations_to_response(llm_response, test_chunks)
    
    print("Final Response with Citations:")
    print("-" * 40)
    print(final_response)
    
    # Validate structure
    if "Sources:" not in final_response:
        print("❌ Sources section not found in response")
        return False
    
    if "📄" not in final_response:
        print("❌ Citation markers not found in response")
        return False
    
    print("✅ Response with citations formatted correctly")
    
    # Test 3: Edge cases
    print("\n⚠️  Test 3: Edge Cases")
    print("-" * 25)
    
    # Empty chunks
    empty_citations = format_citations([])
    if empty_citations != "":
        print("❌ Empty chunks should return empty string")
        return False
    print("✅ Empty chunks handled correctly")
    
    # No metadata
    no_metadata_chunks = [{"content": "Some content"}]
    no_meta_citations = format_citations(no_metadata_chunks)
    if "Untitled Document" not in no_meta_citations:
        print("❌ Missing metadata should use default title")
        return False
    print("✅ Missing metadata handled correctly")
    
    # Empty response
    empty_response = append_citations_to_response("", test_chunks)
    if empty_response != "":
        print("❌ Empty response should remain empty")
        return False
    print("✅ Empty response handled correctly")
    
    return True


async def test_citation_node():
    """Test the CitationNode integration"""
    
    print("\n🔧 Testing Citation Node Integration")
    print("=" * 50)
    
    try:
        from app.services.rag.nodes.citation import CitationNode
        from app.services.rag.workflow_state import RAGState
        print("✅ Citation node imports successful")
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Create test state
    state = RAGState(
        user_query="What are the citizenship requirements?",
        response="You must meet residency and language requirements for Canadian citizenship."
    )
    
    # Add test documents
    state.raw_documents = [
        {
            "id": "chunk-1",
            "content": "Canadian citizens must meet residency requirements...",
            "title": "Immigration Guide",
            "page_number": 15,
            "section_title": "Citizenship",
            "source_type": "policy",
            "score": 0.85,
            "metadata": {
                "document_id": "doc-123",
                "chunk_index": 5
            }
        },
        {
            "id": "chunk-2", 
            "content": "Language proficiency is required...",
            "title": "Immigration Guide",
            "page_number": 16,
            "source_type": "policy",
            "score": 0.78,
            "metadata": {
                "document_id": "doc-123",
                "chunk_index": 6
            }
        }
    ]
    
    # Test citation node execution
    citation_node = CitationNode()
    result_state = await citation_node.execute(state)
    
    # Validate results
    if not result_state.response:
        print("❌ Response was lost during citation processing")
        return False
    
    if "Sources:" not in result_state.response:
        print("❌ Citations not appended to response")
        return False
    
    if not result_state.citations:
        print("❌ Citation metadata not extracted")
        return False
    
    if not result_state.source_documents:
        print("❌ Source documents not extracted")
        return False
    
    print("✅ Citation node executed successfully")
    print(f"✅ Generated {len(result_state.citations)} citations")
    print(f"✅ Extracted {len(result_state.source_documents)} source documents")
    
    print("\nFinal Response:")
    print("-" * 20)
    print(result_state.response)
    
    return True


def test_import_integration():
    """Test that citation functionality integrates with existing workflow"""
    
    print("\n🔌 Testing Workflow Integration")
    print("=" * 50)
    
    try:
        # Test workflow imports
        from app.services.rag.workflow import ProductionRAGWorkflow
        from app.services.rag.nodes import CitationNode
        print("✅ Workflow integration imports successful")
        
        # Test that CitationNode is available in workflow
        citation_node = CitationNode()
        print("✅ CitationNode instantiation successful")
        
        # Test citation formatter availability
        from app.utils.citation_formatter import format_citations, append_citations_to_response
        print("✅ Citation formatter functions available")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


async def main():
    """Main test execution"""
    
    print("🎯 Citation Functionality Validation")
    print("=" * 60)
    
    tests = [
        ("Citation Formatter", test_citation_formatter),
        ("Citation Node", test_citation_node),
        ("Workflow Integration", test_import_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} Test...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All Citation Tests Passed!")
        print("Citation functionality is ready for production use.")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed!")
        print("Please review and fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
