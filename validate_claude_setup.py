#!/usr/bin/env python3
"""
Quick validation script to check Claude API integration setup
"""

import sys
import os
from pathlib import Path

def validate_environment():
    """Validate the environment setup for Claude integration tests"""
    print("🔍 Validating Claude API Integration Setup")
    print("="*50)
    
    issues = []
    
    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file found")
    else:
        print("❌ .env file not found")
        issues.append("Create .env file based on .env.example")
    
    # Check Claude API key
    try:
        sys.path.append(str(Path(__file__).parent / "app"))
        from app.core.config import settings
        
        if settings.ANTHROPIC_API_KEY:
            print(f"✅ Claude API key configured: {settings.ANTHROPIC_API_KEY[:10]}...")
        else:
            print("❌ ANTHROPIC_API_KEY not configured")
            issues.append("Add ANTHROPIC_API_KEY to .env file")
            
        # Check RAG model configuration
        print(f"✅ RAG Models configured:")
        print(f"   Query Analysis: {settings.RAG_QUERY_ANALYSIS_MODEL}")
        print(f"   Generation: {settings.RAG_GENERATION_MODEL}")
        print(f"   Reranking: {settings.RAG_RERANKING_MODEL}")
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        issues.append("Fix configuration loading issues")
    
    # Check required packages
    required_packages = ["pytest", "httpx", "anthropic", "fastapi"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} available")
        except ImportError:
            print(f"❌ {package} not found")
            missing_packages.append(package)
    
    if missing_packages:
        issues.append(f"Install missing packages: {', '.join(missing_packages)}")
    
    # Check test files
    test_file = Path("tests/test_real_claude_api_integration.py")
    if test_file.exists():
        print("✅ Test suite file found")
    else:
        print("❌ Test suite file not found")
        issues.append("Ensure test_real_claude_api_integration.py exists")
    
    runner_file = Path("run_claude_integration_tests.py")
    if runner_file.exists():
        print("✅ Test runner script found")
    else:
        print("❌ Test runner script not found")
        issues.append("Ensure run_claude_integration_tests.py exists")
    
    # Summary
    print("\n" + "="*50)
    if not issues:
        print("🎉 Setup validation PASSED!")
        print("✅ Ready to run Claude API integration tests")
        print("\nNext steps:")
        print("1. Run: python run_claude_integration_tests.py")
        print("2. Or run: pytest tests/test_real_claude_api_integration.py -v")
        return True
    else:
        print("⚠️ Setup validation FAILED!")
        print("❌ Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print("\nFix these issues before running tests.")
        return False

if __name__ == "__main__":
    success = validate_environment()
    sys.exit(0 if success else 1)
