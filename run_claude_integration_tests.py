#!/usr/bin/env python3
"""
Claude API Integration Test Runner
Executes the complete test suite and generates comprehensive reports
"""

import asyncio
import subprocess
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any

def check_prerequisites():
    """Check if all prerequisites are met for running the tests"""
    print("🔍 Checking prerequisites...")
    
    # Check if .env file exists
    env_file = Path(".env")
    if not env_file.exists():
        print("❌ .env file not found. Please create one based on .env.example")
        return False
    
    # Check if ANTHROPIC_API_KEY is set
    try:
        from app.core.config import settings
        if not settings.ANTHROPIC_API_KEY:
            print("❌ ANTHROPIC_API_KEY not configured in .env file")
            return False
        print(f"✅ Claude API Key configured: {settings.ANTHROPIC_API_KEY[:10]}...")
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False
    
    # Check if required packages are installed
    try:
        import pytest
        import httpx
        print("✅ Required packages available")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        return False
    
    print("✅ All prerequisites met")
    return True

def run_pytest_tests():
    """Run the pytest test suite with detailed output"""
    print("\n🧪 Running Claude API Integration Tests...")
    print("="*60)
    
    # Pytest command with verbose output and JSON report
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_real_claude_api_integration.py",
        "-v",
        "--tb=short",
        "--json-report",
        "--json-report-file=claude_test_report.json",
        "--log-cli-level=INFO",
        "-s"  # Don't capture output so we can see real-time logs
    ]
    
    try:
        # Run pytest
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running pytest: {e}")
        return False

def analyze_test_results():
    """Analyze test results and generate summary"""
    print("\n📊 Analyzing Test Results...")
    
    # Check for pytest JSON report
    json_report_file = Path("claude_test_report.json")
    results_file = Path("claude_real_api_integration_results.json")
    
    summary = {
        "test_execution": "unknown",
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "success_rate": 0.0,
        "total_duration": 0.0,
        "grounding_accuracy": "unknown",
        "claude_api_performance": "unknown"
    }
    
    # Analyze pytest JSON report if available
    if json_report_file.exists():
        try:
            with open(json_report_file, 'r') as f:
                pytest_data = json.load(f)
            
            summary["test_execution"] = "completed"
            summary["total_tests"] = pytest_data.get("summary", {}).get("total", 0)
            summary["passed_tests"] = pytest_data.get("summary", {}).get("passed", 0)
            summary["failed_tests"] = pytest_data.get("summary", {}).get("failed", 0)
            summary["total_duration"] = pytest_data.get("duration", 0.0)
            
            if summary["total_tests"] > 0:
                summary["success_rate"] = (summary["passed_tests"] / summary["total_tests"]) * 100
            
            print(f"✅ Pytest execution completed")
            print(f"   Tests: {summary['passed_tests']}/{summary['total_tests']} passed")
            print(f"   Success rate: {summary['success_rate']:.1f}%")
            print(f"   Duration: {summary['total_duration']:.2f}s")
            
        except Exception as e:
            print(f"⚠️ Error reading pytest report: {e}")
    
    # Analyze detailed test results if available
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                detailed_results = json.load(f)
            
            test_results = detailed_results.get("test_results", [])
            
            # Analyze grounding accuracy
            grounding_tests = [t for t in test_results if "grounding_keywords_found" in t]
            if grounding_tests:
                avg_grounding = sum(t["grounding_keywords_found"] for t in grounding_tests) / len(grounding_tests)
                summary["grounding_accuracy"] = f"{avg_grounding:.1f} keywords avg"
            
            # Analyze Claude API performance
            performance_tests = [t for t in test_results if t.get("test") == "performance_quality_metrics"]
            if performance_tests:
                perf_data = performance_tests[0]
                avg_time = perf_data.get("avg_sequential_time", 0)
                summary["claude_api_performance"] = f"{avg_time:.2f}s avg response"
            
            print(f"✅ Detailed analysis completed")
            print(f"   Grounding accuracy: {summary['grounding_accuracy']}")
            print(f"   Claude API performance: {summary['claude_api_performance']}")
            
        except Exception as e:
            print(f"⚠️ Error reading detailed results: {e}")
    
    return summary

def generate_final_report(summary: Dict[str, Any]):
    """Generate final comprehensive report"""
    print("\n" + "="*80)
    print("🎯 CLAUDE API INTEGRATION TEST FINAL REPORT")
    print("="*80)
    
    # Test Execution Summary
    print(f"\n📋 Test Execution Summary:")
    print(f"   Status: {summary['test_execution']}")
    print(f"   Total Tests: {summary['total_tests']}")
    print(f"   Passed: {summary['passed_tests']}")
    print(f"   Failed: {summary['failed_tests']}")
    print(f"   Success Rate: {summary['success_rate']:.1f}%")
    print(f"   Total Duration: {summary['total_duration']:.2f}s")
    
    # Quality Metrics
    print(f"\n🎯 Quality Metrics:")
    print(f"   Document Grounding: {summary['grounding_accuracy']}")
    print(f"   Claude API Performance: {summary['claude_api_performance']}")
    
    # Overall Assessment
    print(f"\n🏆 Overall Assessment:")
    if summary['success_rate'] >= 90:
        print(f"   🎉 EXCELLENT - Claude integration is production-ready")
        assessment = "excellent"
    elif summary['success_rate'] >= 70:
        print(f"   ✅ GOOD - Minor issues to address before production")
        assessment = "good"
    elif summary['success_rate'] >= 50:
        print(f"   ⚠️ FAIR - Significant issues need resolution")
        assessment = "fair"
    else:
        print(f"   ❌ POOR - Major compatibility problems detected")
        assessment = "poor"
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if assessment == "excellent":
        print(f"   • Deploy to production with confidence")
        print(f"   • Monitor Claude API usage and costs")
        print(f"   • Set up automated testing pipeline")
    elif assessment == "good":
        print(f"   • Review failed tests and fix issues")
        print(f"   • Optimize slow-performing queries")
        print(f"   • Consider additional error handling")
    else:
        print(f"   • Review Claude API configuration")
        print(f"   • Check document processing pipeline")
        print(f"   • Validate RAG workflow implementation")
        print(f"   • Consider fallback mechanisms")
    
    # File Locations
    print(f"\n📁 Generated Files:")
    files_created = []
    
    if Path("claude_test_report.json").exists():
        files_created.append("claude_test_report.json (pytest report)")
    if Path("claude_real_api_integration_results.json").exists():
        files_created.append("claude_real_api_integration_results.json (detailed results)")
    
    for file_info in files_created:
        print(f"   • {file_info}")
    
    # Save final summary
    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "assessment": assessment,
        "files_generated": files_created
    }
    
    with open("claude_integration_final_report.json", 'w') as f:
        json.dump(final_report, f, indent=2)
    
    print(f"   • claude_integration_final_report.json (this summary)")
    
    print(f"\n🎯 Test execution complete!")
    return assessment

def main():
    """Main execution function"""
    print("🚀 Claude API Integration Test Suite")
    print("="*50)
    print("Production-grade testing for LangGraph RAG platform")
    print()
    
    start_time = time.time()
    
    # Step 1: Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above.")
        return 1
    
    # Step 2: Run tests
    print("\n⏳ Starting test execution...")
    test_success = run_pytest_tests()
    
    # Step 3: Analyze results
    summary = analyze_test_results()
    
    # Step 4: Generate final report
    assessment = generate_final_report(summary)
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total execution time: {total_time:.2f}s")
    
    # Return appropriate exit code
    if assessment in ["excellent", "good"]:
        return 0
    else:
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
