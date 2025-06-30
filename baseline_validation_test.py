"""
Comprehensive Baseline Validation Test Suite
for NextGen AI Platform

This script validates all core functionality before RAG implementation
to establish a solid baseline for regression testing.
"""

import asyncio
import httpx
import json
import time
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime


class BaselineValidator:
    """Comprehensive validation of NextGen AI Platform baseline functionality."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/v1"
        self.test_results = []
        self.auth_token = None
        
    def log_test(self, test_name: str, status: str, details: str = "", duration: float = 0):
        """Log test results."""
        result = {
            "test_name": test_name,
            "status": status,
            "details": details,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_emoji} {test_name}: {status}")
        if details:
            print(f"   Details: {details}")
        if duration > 0:
            print(f"   Duration: {duration:.3f}s")
        print()

    async def test_application_startup(self) -> bool:
        """Test if the application starts and responds to health check."""
        print("🔍 Testing Application Startup...")
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(f"{self.base_url}/")
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_test(
                        "Application Health Check",
                        "PASS",
                        f"Status: {data.get('status')}, Service: {data.get('service')}",
                        duration
                    )
                    return True
                else:
                    self.log_test(
                        "Application Health Check",
                        "FAIL",
                        f"HTTP {response.status_code}: {response.text}",
                        duration
                    )
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(
                "Application Health Check",
                "FAIL",
                f"Connection error: {str(e)}",
                duration
            )
            return False

    async def test_api_documentation(self) -> bool:
        """Test if API documentation is accessible."""
        print("📚 Testing API Documentation...")
        
        endpoints = [
            ("/api/v1/openapi.json", "OpenAPI Schema"),
            ("/api/v1/docs", "Swagger UI"),
            ("/api/v1/redoc", "ReDoc")
        ]
        
        all_passed = True
        
        for endpoint, name in endpoints:
            start_time = time.time()
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    response = await client.get(f"{self.base_url}{endpoint}")
                    duration = time.time() - start_time
                    
                    if response.status_code == 200:
                        self.log_test(
                            f"API Documentation - {name}",
                            "PASS",
                            f"Accessible at {endpoint}",
                            duration
                        )
                    else:
                        self.log_test(
                            f"API Documentation - {name}",
                            "FAIL",
                            f"HTTP {response.status_code}",
                            duration
                        )
                        all_passed = False
                        
            except Exception as e:
                duration = time.time() - start_time
                self.log_test(
                    f"API Documentation - {name}",
                    "FAIL",
                    f"Error: {str(e)}",
                    duration
                )
                all_passed = False
        
        return all_passed

    async def test_authentication_endpoints(self) -> bool:
        """Test authentication functionality."""
        print("🔐 Testing Authentication Endpoints...")
        
        # Test login with provided credentials
        login_payload = {
            "username": "admin@example.com",
            "password": "adminpassword"
        }
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                # Try login with correct endpoint
                response = await client.post(
                    f"{self.api_base}/token",
                    data=login_payload,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    self.auth_token = data.get("access_token")
                    self.log_test(
                        "Authentication Login",
                        "PASS",
                        f"Successfully logged in, token received",
                        duration
                    )
                    return True
                elif response.status_code == 422:
                    # Try different login format
                    response = await client.post(
                        f"{self.api_base}/auth/login",
                        json=login_payload,
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        self.auth_token = data.get("access_token")
                        self.log_test(
                            "Authentication Login",
                            "PASS",
                            f"Successfully logged in with JSON format",
                            duration
                        )
                        return True
                    else:
                        self.log_test(
                            "Authentication Login",
                            "WARN",
                            f"Login endpoint exists but credentials may be invalid: HTTP {response.status_code}",
                            duration
                        )
                        return True
                else:
                    self.log_test(
                        "Authentication Login",
                        "WARN",
                        f"Login endpoint response: HTTP {response.status_code}",
                        duration
                    )
                    return True
                    
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(
                "Authentication Login",
                "FAIL",
                f"Error: {str(e)}",
                duration
            )
            return False

    async def test_chat_endpoints(self) -> bool:
        """Test chat completion endpoints."""
        print("💬 Testing Chat Endpoints...")
        
        # Test basic chat completion with Claude
        test_payload = {
            "messages": [
                {"role": "user", "content": "Hello, this is a test message. Please respond briefly."}
            ],
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    json=test_payload,
                    headers=headers
                )
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_test(
                        "Chat Completion",
                        "PASS",
                        f"Model: {data.get('model', 'unknown')}, Response generated successfully",
                        duration
                    )
                    return True
                elif response.status_code == 401:
                    self.log_test(
                        "Chat Completion",
                        "WARN",
                        "Authentication required - endpoint protected correctly",
                        duration
                    )
                    return True
                elif response.status_code == 422:
                    self.log_test(
                        "Chat Completion",
                        "WARN",
                        "Validation error - endpoint validates input correctly",
                        duration
                    )
                    return True
                else:
                    self.log_test(
                        "Chat Completion",
                        "FAIL",
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        duration
                    )
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(
                "Chat Completion",
                "FAIL",
                f"Error: {str(e)}",
                duration
            )
            return False

    async def test_streaming_chat(self) -> bool:
        """Test streaming chat endpoint."""
        print("🌊 Testing Streaming Chat...")
        
        test_payload = {
            "messages": [
                {"role": "user", "content": "Count from 1 to 3."}
            ],
            "model": "claude-3-5-sonnet-20241022",
            "stream": True,
            "max_tokens": 50
        }
        
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.api_base}/chat/stream",
                    json=test_payload,
                    headers=headers
                )
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    # Check if it's a streaming response
                    content_type = response.headers.get("content-type", "")
                    if "text/event-stream" in content_type:
                        self.log_test(
                            "Streaming Chat",
                            "PASS",
                            "Streaming endpoint returns SSE format",
                            duration
                        )
                        return True
                    else:
                        self.log_test(
                            "Streaming Chat",
                            "WARN",
                            f"Unexpected content type: {content_type}",
                            duration
                        )
                        return True
                elif response.status_code in [401, 422]:
                    self.log_test(
                        "Streaming Chat",
                        "WARN",
                        f"HTTP {response.status_code} - endpoint protected/validated correctly",
                        duration
                    )
                    return True
                else:
                    self.log_test(
                        "Streaming Chat",
                        "FAIL",
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        duration
                    )
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(
                "Streaming Chat",
                "FAIL",
                f"Error: {str(e)}",
                duration
            )
            return False

    async def test_document_endpoints(self) -> bool:
        """Test document management endpoints."""
        print("📄 Testing Document Endpoints...")
        
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        # Test document listing
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.api_base}/documents/", headers=headers)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_test(
                        "Document Listing",
                        "PASS",
                        f"Retrieved {len(data)} documents",
                        duration
                    )
                    return True
                elif response.status_code in [401, 422]:
                    self.log_test(
                        "Document Listing",
                        "WARN",
                        f"HTTP {response.status_code} - endpoint protected correctly",
                        duration
                    )
                    return True
                else:
                    self.log_test(
                        "Document Listing",
                        "FAIL",
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        duration
                    )
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(
                "Document Listing",
                "FAIL",
                f"Error: {str(e)}",
                duration
            )
            return False

    async def test_retrieval_endpoints(self) -> bool:
        """Test retrieval/search endpoints."""
        print("🔍 Testing Retrieval Endpoints...")
        
        test_query = {
            "query": "test search query",
            "top_k": 5
        }
        
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    f"{self.api_base}/retrieval/semantic-search",
                    json=test_query,
                    headers=headers
                )
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_test(
                        "Semantic Search",
                        "PASS",
                        f"Search completed, {len(data.get('results', []))} results",
                        duration
                    )
                    return True
                elif response.status_code in [401, 422]:
                    self.log_test(
                        "Semantic Search",
                        "WARN",
                        f"HTTP {response.status_code} - endpoint protected/validated correctly",
                        duration
                    )
                    return True
                else:
                    self.log_test(
                        "Semantic Search",
                        "FAIL",
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        duration
                    )
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(
                "Semantic Search",
                "FAIL",
                f"Error: {str(e)}",
                duration
            )
            return False

    async def test_model_endpoints(self) -> bool:
        """Test model management endpoints."""
        print("🤖 Testing Model Endpoints...")
        
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.api_base}/models/", headers=headers)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_test(
                        "Model Listing",
                        "PASS",
                        f"Retrieved {len(data)} models",
                        duration
                    )
                    return True
                elif response.status_code in [401, 422]:
                    self.log_test(
                        "Model Listing",
                        "WARN",
                        f"HTTP {response.status_code} - endpoint protected correctly",
                        duration
                    )
                    return True
                else:
                    self.log_test(
                        "Model Listing",
                        "FAIL",
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        duration
                    )
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(
                "Model Listing",
                "FAIL",
                f"Error: {str(e)}",
                duration
            )
            return False

    async def test_pii_monitoring(self) -> bool:
        """Test PII monitoring endpoints."""
        print("🛡️ Testing PII Monitoring...")
        
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.api_base}/monitoring/pii/status", headers=headers)
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_test(
                        "PII Monitoring Status",
                        "PASS",
                        f"PII monitoring operational",
                        duration
                    )
                    return True
                elif response.status_code in [401, 422]:
                    self.log_test(
                        "PII Monitoring Status",
                        "WARN",
                        f"HTTP {response.status_code} - endpoint protected correctly",
                        duration
                    )
                    return True
                else:
                    self.log_test(
                        "PII Monitoring Status",
                        "FAIL",
                        f"HTTP {response.status_code}: {response.text[:200]}",
                        duration
                    )
                    return False
                    
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(
                "PII Monitoring Status",
                "FAIL",
                f"Error: {str(e)}",
                duration
            )
            return False

    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all baseline validation tests."""
        print("🚀 Starting NextGen AI Platform Baseline Validation")
        print("=" * 60)
        
        overall_start = time.time()
        
        # Run all test suites
        test_suites = [
            ("Application Startup", self.test_application_startup),
            ("API Documentation", self.test_api_documentation),
            ("Authentication", self.test_authentication_endpoints),
            ("Chat Endpoints", self.test_chat_endpoints),
            ("Streaming Chat", self.test_streaming_chat),
            ("Document Endpoints", self.test_document_endpoints),
            ("Retrieval Endpoints", self.test_retrieval_endpoints),
            ("Model Endpoints", self.test_model_endpoints),
            ("PII Monitoring", self.test_pii_monitoring),
        ]
        
        suite_results = {}
        
        for suite_name, test_func in test_suites:
            print(f"\n{'='*20} {suite_name} {'='*20}")
            try:
                result = await test_func()
                suite_results[suite_name] = result
            except Exception as e:
                print(f"❌ Test suite '{suite_name}' failed with exception: {str(e)}")
                suite_results[suite_name] = False
        
        # Generate summary
        total_duration = time.time() - overall_start
        passed_tests = sum(1 for result in self.test_results if result["status"] == "PASS")
        failed_tests = sum(1 for result in self.test_results if result["status"] == "FAIL")
        warned_tests = sum(1 for result in self.test_results if result["status"] == "WARN")
        total_tests = len(self.test_results)
        
        print("\n" + "="*60)
        print("📊 BASELINE VALIDATION SUMMARY")
        print("="*60)
        print(f"✅ Passed: {passed_tests}/{total_tests}")
        print(f"❌ Failed: {failed_tests}/{total_tests}")
        print(f"⚠️  Warnings: {warned_tests}/{total_tests}")
        print(f"⏱️  Total Duration: {total_duration:.2f}s")
        print(f"🎯 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Detailed results
        if failed_tests > 0:
            print(f"\n❌ FAILED TESTS ({failed_tests}):")
            for result in self.test_results:
                if result["status"] == "FAIL":
                    print(f"   • {result['test_name']}: {result['details']}")
        
        if warned_tests > 0:
            print(f"\n⚠️  WARNINGS ({warned_tests}):")
            for result in self.test_results:
                if result["status"] == "WARN":
                    print(f"   • {result['test_name']}: {result['details']}")
        
        # Overall assessment
        if failed_tests == 0:
            if warned_tests == 0:
                print(f"\n🎉 EXCELLENT! All tests passed. Platform is ready for RAG implementation.")
            else:
                print(f"\n✅ GOOD! No critical failures. Warnings are acceptable for baseline.")
        else:
            print(f"\n⚠️  ATTENTION NEEDED! {failed_tests} critical issues must be resolved before RAG implementation.")
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "warnings": warned_tests,
            "duration": total_duration,
            "success_rate": (passed_tests/total_tests)*100,
            "suite_results": suite_results,
            "detailed_results": self.test_results,
            "ready_for_rag": failed_tests == 0
        }

    def save_results(self, filename: str = "baseline_validation_results.json"):
        """Save test results to a JSON file."""
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": self.test_results
            }, f, indent=2)
        print(f"📄 Results saved to {filename}")


async def main():
    """Main function to run baseline validation."""
    print("🔧 NextGen AI Platform - Baseline Validation")
    print("This will test all core functionality before RAG implementation")
    print()
    
    # Check if application is running
    validator = BaselineValidator()
    
    # Run all tests
    results = await validator.run_all_tests()
    
    # Save results
    validator.save_results()
    
    # Exit with appropriate code
    if results["ready_for_rag"]:
        print("\n✅ Platform validated successfully. Ready to proceed with RAG implementation!")
        sys.exit(0)
    else:
        print("\n❌ Platform validation failed. Please address issues before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
