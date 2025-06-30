#!/usr/bin/env python3
"""
Production-grade test runner for the RAG system.
Executes comprehensive test suite and generates detailed reports.
"""

import asyncio
import sys
import argparse
from pathlib import Path
from tests.test_rag_comprehensive import RAGTestOrchestrator


def main():
    """Main entry point for test execution"""
    
    parser = argparse.ArgumentParser(description="Run comprehensive RAG system tests")
    parser.add_argument(
        "--suite", 
        choices=["all", "nodes", "workflow", "integration", "performance", "edge-cases"],
        default="all",
        help="Test suite to run (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--report-file",
        type=str,
        help="Save test report to file"
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first failure"
    )
    
    args = parser.parse_args()
    
    # Run tests
    asyncio.run(run_tests(args))


async def run_tests(args):
    """Execute the test suite with given arguments"""
    
    print("🚀 Starting RAG System Comprehensive Test Suite")
    print("=" * 60)
    
    orchestrator = RAGTestOrchestrator()
    
    try:
        if args.suite == "all":
            results = await orchestrator.run_comprehensive_tests()
        else:
            # Run specific test suite
            results = await run_specific_suite(orchestrator, args.suite)
        
        # Generate report
        report = orchestrator.generate_test_report()
        
        if args.verbose:
            print("\n" + report)
        else:
            # Print summary only
            total_tests = sum(result.total_tests for result in results.values())
            total_passed = sum(result.passed_tests for result in results.values())
            total_failed = sum(result.failed_tests for result in results.values())
            
            print(f"\n📊 Test Results Summary:")
            print(f"   Total: {total_tests}, Passed: {total_passed}, Failed: {total_failed}")
            
            if total_failed == 0:
                print("✅ All tests passed!")
            else:
                print(f"❌ {total_failed} tests failed")
        
        # Save report to file if requested
        if args.report_file:
            with open(args.report_file, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to {args.report_file}")
        
        # Exit with appropriate code
        total_failed = sum(result.failed_tests for result in results.values())
        sys.exit(1 if total_failed > 0 else 0)
        
    except KeyboardInterrupt:
        print("\n⚠️ Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        sys.exit(1)


async def run_specific_suite(orchestrator, suite_name):
    """Run a specific test suite"""
    
    suite_map = {
        "nodes": orchestrator._run_node_tests,
        "workflow": orchestrator._run_workflow_tests,
        "integration": orchestrator._run_integration_tests,
        "performance": orchestrator._run_performance_tests,
        "edge-cases": orchestrator._run_edge_case_tests,
    }
    
    if suite_name not in suite_map:
        raise ValueError(f"Unknown test suite: {suite_name}")
    
    print(f"🔄 Running {suite_name} tests...")
    result = await suite_map[suite_name]()
    
    return {suite_name: result}


if __name__ == "__main__":
    main()
