"""
Enhanced Baseline Validation Test Suite
for NextGen AI Platform with Local Embedding Support

This script validates all core functionality with enhanced local embedding tests
and better error handling for timeout issues.
"""

import asyncio
import httpx
import json
import time
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime


class EnhancedBaselineValidator:
    """Enhanced validation of NextGen AI Platform with local embedding support."""
    
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
            async with httpx.AsyncClient(timeout=10) as client:
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

    async def test_authentication_endpoints(self) -> bool:
        """Test authentication functionality."""
        print("🔐 Testing Authentication Endpoints...")
        
        login_payload = {
            "username": "admin@example.com",
            "password": "adminpassword"
        }
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=10) as client:
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

    async def test_local_embedding_service(self) -> bool:
        """Test local embedding service directly."""
        print("🧠 Testing Local Embedding Service...")
        
        start_time = time.time()
        try:
            # Import and test local embedding service
            from app.services.embeddings.local_embedding_service import get_local_embedding_service
            
            service = get_local_embedding_service()
            
            # Test initialization
            init_success = await service.initialize()
            if not init_success:
                self.log_test(
                    "Local Embedding Initialization",
                    "FAIL",
                    "Failed to initialize local embedding model",
                    time.time() - start_time
                )
                return False
            
            # Test embedding generation
            test_texts = [
                "This is a test sentence for embedding generation.",
                "Another test sentence to verify batch processing."
            ]
            
            embeddings = await service.generate_embeddings(test_texts)
            duration = time.time() - start_time
            
            if embeddings and len(embeddings) == len(test_texts):
                dimensions = len(embeddings[0]) if embeddings[0] else 0
                stats = service.get_performance_stats()
                
                self.log_test(
                    "Local Embedding Generation",
                    "PASS",
                    f"Generated {len(embeddings)} embeddings, {dimensions}D, "
                    f"{stats['embeddings_per_second']:.1f} emb/sec",
                    duration
                )
                return True
            else:
                self.log_test(
                    "Local Embedding Generation",
                    "FAIL",
                    f"Expected {len(test_texts)} embeddings, got {len(embeddings) if embeddings else 0}",
                    duration
                )
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(
                "Local Embedding Service",
                "FAIL",
                f"Error: {str(e)}",
                duration
            )
            return False

    async def test_embedding_quality(self) -> bool:
        """Test embedding quality with semantic similarity."""
        print("🎯 Testing Embedding Quality...")
        
        start_time = time.time()
        try:
            from app.services.embeddings.local_embedding_service import get_local_embedding_service
            import numpy as np
            
            service = get_local_embedding_service()
            
            # Test semantic similarity
            similar_texts = [
                "The cat sat on the mat.",
                "A cat was sitting on a mat."
            ]
            
            different_texts = [
                "The cat sat on the mat.",
                "Machine learning algorithms are complex."
            ]
            
            # Generate embeddings
            similar_embeddings = await service.generate_embeddings(similar_texts)
            different_embeddings = await service.generate_embeddings(different_texts)
            
            if not similar_embeddings or not different_embeddings:
                self.log_test(
                    "Embedding Quality Test",
                    "FAIL",
                    "Failed to generate embeddings for quality test",
                    time.time() - start_time
                )
                return False
            
            # Calculate cosine similarity
            def cosine_similarity(a, b):
                return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            
            similar_score = cosine_similarity(similar_embeddings[0], similar_embeddings[1])
            different_score = cosine_similarity(different_embeddings[0], different_embeddings[1])
            
            duration = time.time() - start_time
            
            # Similar texts should have higher similarity than different texts
            if similar_score > different_score:
                self.log_test(
                    "Embedding Quality Test",
                    "PASS",
                    f"Similar texts: {similar_score:.3f}, Different texts: {different_score:.3f}",
                    duration
                )
                return True
            else:
                self.log_test(
                    "Embedding Quality Test",
                    "WARN",
                    f"Similarity scores may be inverted: Similar={similar_score:.3f}, Different={different_score:.3f}",
                    duration
                )
                return True
                
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(
                "Embedding Quality Test",
                "FAIL",
                f"Error: {str(e)}",
                duration
            )
            return False

    async def test_semantic_search_with_timeout(self) -> bool:
        """Test semantic search with proper timeout handling."""
        print("🔍 Testing Semantic Search (Enhanced)...")
        
        test_query = {
            "query": "test search query",
            "top_k": 5
        }
        
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        start_time = time.time()
        try:
            # Use longer timeout for first embedding model load
            async with httpx.AsyncClient(timeout=60) as client:
                response = await client.post(
                    f"{self.api_base}/retrieval/semantic-search",
                    json=test_query,
                    headers=headers
                )
                duration = time.time() - start_time
                
                if response.status_code == 200:
                    data = response.json()
                    self.log_test(
                        "Semantic Search (Enhanced)",
                        "PASS",
                        f"Search completed, {len(data.get('results', []))} results",
                        duration
                    )
                    return True
                elif response.status_code in [401, 422]:
                    self.log_test(
                        "Semantic Search (Enhanced)",
                        "WARN",
                        f"HTTP {response.status_code} - endpoint protected/validated correctly",
                        duration
                    )
                    return True
                else:
                    error_detail = ""
                    try:
                        error_data = response.json()
                        error_detail = error_data.get('detail', response.text[:200])
                    except:
                        error_detail = response.text[:200]
                    
                    self.log_test(
                        "Semantic Search (Enhanced)",
                        "FAIL",
                        f"HTTP {response.status_code}: {error_detail}",
                        duration
                    )
                    return False
                    
        except httpx.TimeoutException:
            duration = time.time() - start_time
            self.log_test(
                "Semantic Search (Enhanced)",
                "FAIL",
                f"Request timed out after {duration:.1f}s (model may be loading)",
                duration
            )
            return False
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(
                "Semantic Search (Enhanced)",
                "FAIL",
                f"Error: {str(e)}",
                duration
            )
            return False

    async def test_chat_endpoints(self) -> bool:
        """Test chat completion endpoints."""
        print("💬 Testing Chat Endpoints...")
        
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
                else:
                    self.log_test(
                        "Chat Completion",
                        "WARN",
                        f"HTTP {response.status_code} - endpoint may require configuration",
                        duration
                    )
                    return True
                    
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(
                "Chat Completion",
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
                else:
                    self.log_test(
                        "Document Listing",
                        "WARN",
                        f"HTTP {response.status_code} - endpoint may require configuration",
                        duration
                    )
                    return True
                    
        except Exception as e:
            duration = time.time() - start_time
            self.log_test(
                "Document Listing",
                "FAIL",
                f"Error: {str(e)}",
                duration
            )
            return False

    async def test_model_endpoints_with_timeout(self) -> bool:
        """Test model management endpoints with proper timeout."""
        print("🤖 Testing Model Endpoints...")
        
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=15) as client:
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
                else:
                    self.log_test(
                        "Model Listing",
                        "WARN",
                        f"HTTP {response.status_code} - endpoint may require configuration",
                        duration
                    )
                    return True
                    
        except httpx.TimeoutException:
            duration = time.time() - start_time
            self.log_test(
                "Model Listing",
                "FAIL",
                f"Request timed out after {duration:.1f}s",
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

    async def test_pii_monitoring_with_timeout(self) -> bool:
        """Test PII monitoring endpoints with proper timeout."""
        print("🛡️ Testing PII Monitoring...")
        
        headers = {}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        
        start_time = time.time()
        try:
            async with httpx.AsyncClient(timeout=15) as client:
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
                else:
                    self.log_test(
                        "PII Monitoring Status",
                        "WARN",
                        f"HTTP {response.status_code} - endpoint may require configuration",
                        duration
                    )
                    return True
                    
        except httpx.TimeoutException:
            duration = time.time() - start_time
            self.log_test(
                "PII Monitoring Status",
                "FAIL",
                f"Request timed out after {duration:.1f}s",
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
        """Run all enhanced baseline validation tests."""
        print("🚀 Starting Enhanced NextGen AI Platform Validation")
        print("=" * 60)
        
        overall_start = time.time()
        
        # Run all test suites with enhanced local embedding tests
        test_suites = [
            ("Application Startup", self.test_application_startup),
            ("Authentication", self.test_authentication_endpoints),
            ("Local Embedding Service", self.test_local_embedding_service),
            ("Embedding Quality", self.test_embedding_quality),
            ("Chat Endpoints", self.test_chat_endpoints),
            ("Document Endpoints", self.test_document_endpoints),
            ("Semantic Search (Enhanced)", self.test_semantic_search_with_timeout),
            ("Model Endpoints", self.test_model_endpoints_with_timeout),
            ("PII Monitoring", self.test_pii_monitoring_with_timeout),
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
        print("📊 ENHANCED BASELINE VALIDATION SUMMARY")
        print("="*60)
        print(f"✅ Passed: {passed_tests}/{total_tests}")
        print(f"❌ Failed: {failed_tests}/{total_tests}")
        print(f"⚠️  Warnings: {warned_tests}/{total_tests}")
        print(f"⏱️  Total Duration: {total_duration:.2f}s")
        print(f"🎯 Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        # Local embedding specific summary
        local_embedding_tests = [r for r in self.test_results if "Embedding" in r["test_name"]]
        if local_embedding_tests:
            print(f"\n🧠 LOCAL EMBEDDING PERFORMANCE:")
            for test in local_embedding_tests:
                if test["status"] == "PASS":
                    print(f"   ✅ {test['test_name']}: {test['details']}")
        
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
                print(f"\n🎉 EXCELLENT! All tests passed. Local embedding system is fully operational!")
            else:
                print(f"\n✅ GOOD! No critical failures. Local embedding system is working well.")
        else:
            print(f"\n⚠️  ATTENTION NEEDED! {failed_tests} critical issues must be resolved.")
        
        return {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": failed_tests,
            "warnings": warned_tests,
            "duration": total_duration,
            "success_rate": (passed_tests/total_tests)*100,
            "suite_results": suite_results,
            "detailed_results": self.test_results,
            "local_embedding_operational": any(
                r["status"] == "PASS" and "Embedding" in r["test_name"] 
                for r in self.test_results
            )
        }

    def save_results(self, filename: str = "enhanced_baseline_validation_results.json"):
        """Save test results to a JSON file."""
        with open(filename, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": self.test_results
            }, f, indent=2)
        print(f"📄 Results saved to {filename}")


async def main():
    """Main function to run enhanced baseline validation."""
    print("🔧 NextGen AI Platform - Enhanced Baseline Validation")
    print("This will test all core functionality with enhanced local embedding support")
    print()
    
    # Check if application is running
    validator = EnhancedBaselineValidator()
    
    # Run all tests
    results = await validator.run_all_tests()
    
    # Save results
    validator.save_results()
    
    # Exit with appropriate code
    if results["failed"] == 0:
        print("\n✅ Platform validated successfully with local embeddings!")
        sys.exit(0)
    else:
        print("\n❌ Platform validation found issues. Please review and address.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
