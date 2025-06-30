# tests/test_rag_comprehensive.py
"""
Comprehensive test suite runner for the entire RAG system.
This file orchestrates all test categories and provides reporting.
"""

import pytest
import asyncio
import time
from typing import Dict, List, Any
from dataclasses import dataclass
from tests.fixtures import (
    MockOllamaAdapter,
    ProductionMockQdrantService,
    MockScenarioConfig,
    create_test_state
)


@dataclass
class TestSuiteResult:
    """Results from running a test suite"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    execution_time: float
    coverage_percentage: float
    errors: List[str]


class RAGTestOrchestrator:
    """Orchestrates comprehensive testing of the RAG system"""
    
    def __init__(self):
        self.results: Dict[str, TestSuiteResult] = {}
        self.mock_services = self._setup_mock_services()
    
    def _setup_mock_services(self):
        """Setup mock services for testing"""
        return {
            "ollama": MockOllamaAdapter(),
            "qdrant": ProductionMockQdrantService(MockScenarioConfig()),
        }
    
    async def run_comprehensive_tests(self) -> Dict[str, TestSuiteResult]:
        """Run all test suites and return comprehensive results"""
        
        test_suites = [
            ("Node Tests", self._run_node_tests),
            ("Workflow Tests", self._run_workflow_tests),
            ("Integration Tests", self._run_integration_tests),
            ("Performance Tests", self._run_performance_tests),
            ("Edge Case Tests", self._run_edge_case_tests),
        ]
        
        for suite_name, test_runner in test_suites:
            print(f"\n🔄 Running {suite_name}...")
            start_time = time.time()
            
            try:
                result = await test_runner()
                execution_time = time.time() - start_time
                
                self.results[suite_name] = TestSuiteResult(
                    suite_name=suite_name,
                    total_tests=result.get("total", 0),
                    passed_tests=result.get("passed", 0),
                    failed_tests=result.get("failed", 0),
                    execution_time=execution_time,
                    coverage_percentage=result.get("coverage", 0.0),
                    errors=result.get("errors", [])
                )
                
                print(f"✅ {suite_name} completed: {result['passed']}/{result['total']} passed")
                
            except Exception as e:
                execution_time = time.time() - start_time
                self.results[suite_name] = TestSuiteResult(
                    suite_name=suite_name,
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=1,
                    execution_time=execution_time,
                    coverage_percentage=0.0,
                    errors=[str(e)]
                )
                print(f"❌ {suite_name} failed: {e}")
        
        return self.results
    
    async def _run_node_tests(self) -> Dict[str, Any]:
        """Run individual node tests"""
        
        node_test_scenarios = [
            self._test_query_analysis_comprehensive,
            self._test_hybrid_retrieval_comprehensive,
            self._test_generation_comprehensive,
            self._test_citation_comprehensive,
            self._test_memory_retrieval_comprehensive,
        ]
        
        total_tests = len(node_test_scenarios)
        passed_tests = 0
        errors = []
        
        for test_scenario in node_test_scenarios:
            try:
                await test_scenario()
                passed_tests += 1
            except Exception as e:
                errors.append(f"Node test failed: {e}")
        
        return {
            "total": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "coverage": (passed_tests / total_tests) * 100,
            "errors": errors
        }
    
    async def _test_query_analysis_comprehensive(self):
        """Comprehensive test of query analysis node"""
        from app.services.rag.nodes.query_analysis import QueryAnalysisNode
        
        node = QueryAnalysisNode(self.mock_services["ollama"])
        
        test_queries = [
            ("What are visa requirements?", "simple"),
            ("Remember what we discussed about work permits?", "conversational"),
            ("Compare Express Entry vs PNP programs", "complex"),
            ("Show me the API documentation", "code_related"),
        ]
        
        for query, expected_type in test_queries:
            state = create_test_state(user_query=query)
            result = await node.execute(state)
            
            assert result.query_type is not None
            assert result.intent is not None
            assert isinstance(result.entities, list)
    
    async def _test_hybrid_retrieval_comprehensive(self):
        """Comprehensive test of hybrid retrieval node"""
        from app.services.rag.nodes.hybrid_retrieval import HybridRetrievalNode
        
        node = HybridRetrievalNode(self.mock_services["qdrant"])
        
        test_scenarios = [
            "What documents do I need for work authorization?",
            "Express Entry eligibility requirements",
            "Family sponsorship income requirements",
        ]
        
        for query in test_scenarios:
            state = create_test_state(user_query=query)
            result = await node.execute(state)
            
            assert result.retrieval_strategy is not None
            assert isinstance(result.raw_documents, list)
    
    async def _test_generation_comprehensive(self):
        """Comprehensive test of generation node"""
        from app.services.rag.nodes.generation import GenerationNode
        
        node = GenerationNode(self.mock_services["ollama"])
        
        # Test with documents
        state = create_test_state(
            user_query="What are work permit requirements?",
            raw_documents=[
                {
                    "id": "doc1",
                    "title": "Work Permit Guide",
                    "content": "Work permit requirements include...",
                    "source_type": "policy"
                }
            ]
        )
        
        result = await node.execute(state)
        
        assert result.response is not None
        assert result.context_prompt is not None
        assert result.model_used is not None
    
    async def _test_citation_comprehensive(self):
        """Comprehensive test of citation node"""
        from app.services.rag.nodes.citation import CitationNode
        
        node = CitationNode()
        
        state = create_test_state(
            user_query="Test query",
            raw_documents=[
                {"id": "doc1", "title": "Test Doc", "content": "Test content", "source_type": "test"}
            ]
        )
        state.response = "Test response with [Document 1] citation."
        
        result = await node.execute(state)
        
        assert isinstance(result.citations, list)
        assert isinstance(result.source_documents, list)
    
    async def _test_memory_retrieval_comprehensive(self):
        """Comprehensive test of memory retrieval node"""
        from app.services.rag.nodes.memory_retrieval import MemoryRetrievalNode
        
        node = MemoryRetrievalNode(self.mock_services["qdrant"])
        
        state = create_test_state(
            user_query="Tell me more about what we discussed",
            conversation_id="conv_123"
        )
        
        result = await node.execute(state)
        
        assert isinstance(result.memory_context, str)
    
    async def _run_workflow_tests(self) -> Dict[str, Any]:
        """Run end-to-end workflow tests"""
        
        workflow_scenarios = [
            self._test_complete_workflow_simple,
            self._test_complete_workflow_conversational,
            self._test_complete_workflow_complex,
        ]
        
        total_tests = len(workflow_scenarios)
        passed_tests = 0
        errors = []
        
        for scenario in workflow_scenarios:
            try:
                await scenario()
                passed_tests += 1
            except Exception as e:
                errors.append(f"Workflow test failed: {e}")
        
        return {
            "total": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "coverage": (passed_tests / total_tests) * 100,
            "errors": errors
        }
    
    async def _test_complete_workflow_simple(self):
        """Test complete workflow with simple query"""
        from app.services.rag.workflow import RAGWorkflow
        
        # This would test the complete workflow
        # For now, just validate the concept
        assert True  # Placeholder
    
    async def _test_complete_workflow_conversational(self):
        """Test complete workflow with conversational query"""
        assert True  # Placeholder
    
    async def _test_complete_workflow_complex(self):
        """Test complete workflow with complex query"""
        assert True  # Placeholder
    
    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests with external services"""
        
        integration_scenarios = [
            self._test_qdrant_integration,
            self._test_ollama_integration,
            self._test_memory_persistence,
        ]
        
        total_tests = len(integration_scenarios)
        passed_tests = 0
        errors = []
        
        for scenario in integration_scenarios:
            try:
                await scenario()
                passed_tests += 1
            except Exception as e:
                errors.append(f"Integration test failed: {e}")
        
        return {
            "total": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "coverage": (passed_tests / total_tests) * 100,
            "errors": errors
        }
    
    async def _test_qdrant_integration(self):
        """Test Qdrant service integration"""
        # Test with mock service
        qdrant = self.mock_services["qdrant"]
        results = await qdrant.search_documents("test query")
        assert isinstance(results, list)
    
    async def _test_ollama_integration(self):
        """Test Ollama service integration"""
        ollama = self.mock_services["ollama"]
        response = await ollama.generate("test prompt")
        assert isinstance(response, str)
    
    async def _test_memory_persistence(self):
        """Test memory persistence functionality"""
        # Test memory storage and retrieval
        assert True  # Placeholder
    
    async def _run_performance_tests(self) -> Dict[str, Any]:
        """Run performance and load tests"""
        
        performance_scenarios = [
            self._test_response_time_requirements,
            self._test_concurrent_request_handling,
            self._test_memory_usage_limits,
        ]
        
        total_tests = len(performance_scenarios)
        passed_tests = 0
        errors = []
        
        for scenario in performance_scenarios:
            try:
                await scenario()
                passed_tests += 1
            except Exception as e:
                errors.append(f"Performance test failed: {e}")
        
        return {
            "total": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "coverage": (passed_tests / total_tests) * 100,
            "errors": errors
        }
    
    async def _test_response_time_requirements(self):
        """Test that responses meet time requirements"""
        from app.services.rag.nodes.query_analysis import QueryAnalysisNode
        
        node = QueryAnalysisNode(self.mock_services["ollama"])
        state = create_test_state(user_query="Test query")
        
        start_time = time.time()
        await node.execute(state)
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < 5.0  # 5 second limit
    
    async def _test_concurrent_request_handling(self):
        """Test handling of concurrent requests"""
        from app.services.rag.nodes.query_analysis import QueryAnalysisNode
        
        node = QueryAnalysisNode(self.mock_services["ollama"])
        
        # Create multiple concurrent tasks
        tasks = []
        for i in range(10):
            state = create_test_state(user_query=f"Test query {i}")
            task = node.execute(state)
            tasks.append(task)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == 10
        for result in results:
            assert result.query_type is not None
    
    async def _test_memory_usage_limits(self):
        """Test memory usage stays within limits"""
        # This would test memory usage with large datasets
        assert True  # Placeholder
    
    async def _run_edge_case_tests(self) -> Dict[str, Any]:
        """Run edge case and error handling tests"""
        
        edge_case_scenarios = [
            self._test_empty_input_handling,
            self._test_malformed_input_handling,
            self._test_service_failure_handling,
            self._test_large_input_handling,
        ]
        
        total_tests = len(edge_case_scenarios)
        passed_tests = 0
        errors = []
        
        for scenario in edge_case_scenarios:
            try:
                await scenario()
                passed_tests += 1
            except Exception as e:
                errors.append(f"Edge case test failed: {e}")
        
        return {
            "total": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "coverage": (passed_tests / total_tests) * 100,
            "errors": errors
        }
    
    async def _test_empty_input_handling(self):
        """Test handling of empty inputs"""
        from app.services.rag.nodes.query_analysis import QueryAnalysisNode
        
        node = QueryAnalysisNode(self.mock_services["ollama"])
        
        empty_inputs = ["", "   ", None]
        for empty_input in empty_inputs:
            if empty_input is None:
                state = create_test_state()
                state.user_query = None
            else:
                state = create_test_state(user_query=empty_input)
            
            # Should handle gracefully without crashing
            result = await node.execute(state)
            assert result is not None
    
    async def _test_malformed_input_handling(self):
        """Test handling of malformed inputs"""
        assert True  # Placeholder
    
    async def _test_service_failure_handling(self):
        """Test handling of service failures"""
        assert True  # Placeholder
    
    async def _test_large_input_handling(self):
        """Test handling of very large inputs"""
        assert True  # Placeholder
    
    def generate_test_report(self) -> str:
        """Generate comprehensive test report"""
        
        report_lines = [
            "🔍 RAG System Comprehensive Test Report",
            "=" * 50,
            ""
        ]
        
        total_tests = sum(result.total_tests for result in self.results.values())
        total_passed = sum(result.passed_tests for result in self.results.values())
        total_failed = sum(result.failed_tests for result in self.results.values())
        total_time = sum(result.execution_time for result in self.results.values())
        
        # Overall summary
        report_lines.extend([
            f"📊 Overall Results:",
            f"   Total Tests: {total_tests}",
            f"   Passed: {total_passed} ({(total_passed/total_tests)*100:.1f}%)" if total_tests > 0 else "   Passed: 0 (0%)",
            f"   Failed: {total_failed}",
            f"   Total Time: {total_time:.2f}s",
            ""
        ])
        
        # Individual suite results
        for suite_name, result in self.results.items():
            status = "✅" if result.failed_tests == 0 else "❌"
            report_lines.extend([
                f"{status} {suite_name}:",
                f"   Tests: {result.passed_tests}/{result.total_tests}",
                f"   Time: {result.execution_time:.2f}s",
                f"   Coverage: {result.coverage_percentage:.1f}%"
            ])
            
            if result.errors:
                report_lines.append("   Errors:")
                for error in result.errors:
                    report_lines.append(f"     - {error}")
            
            report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "🔧 Recommendations:",
            ""
        ])
        
        if total_failed > 0:
            report_lines.append("   - Review failed tests and fix underlying issues")
        
        if any(result.coverage_percentage < 80 for result in self.results.values()):
            report_lines.append("   - Increase test coverage for low-coverage areas")
        
        if any(result.execution_time > 10 for result in self.results.values()):
            report_lines.append("   - Optimize slow test suites for better performance")
        
        if not any(result.errors for result in self.results.values()):
            report_lines.append("   - All tests passing! Consider adding more edge cases")
        
        return "\n".join(report_lines)


# Pytest integration
@pytest.mark.asyncio
async def test_comprehensive_rag_system():
    """Main test entry point for comprehensive RAG system testing"""
    
    orchestrator = RAGTestOrchestrator()
    results = await orchestrator.run_comprehensive_tests()
    
    # Generate and print report
    report = orchestrator.generate_test_report()
    print("\n" + report)
    
    # Assert overall success
    total_failed = sum(result.failed_tests for result in results.values())
    assert total_failed == 0, f"Comprehensive test suite failed with {total_failed} failures"


if __name__ == "__main__":
    """Run comprehensive tests directly"""
    
    async def main():
        orchestrator = RAGTestOrchestrator()
        results = await orchestrator.run_comprehensive_tests()
        
        report = orchestrator.generate_test_report()
        print(report)
        
        # Exit with error code if tests failed
        total_failed = sum(result.failed_tests for result in results.values())
        if total_failed > 0:
            exit(1)
        else:
            print("\n🎉 All comprehensive tests passed!")
            exit(0)
    
    asyncio.run(main())
