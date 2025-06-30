"""
Demo script for Document Format Validation Harness

This script demonstrates how to run the document format validation
and provides guidance for interpreting results.

USAGE:
    python run_format_validation_demo.py

REQUIREMENTS:
    - FastAPI server running on localhost:8000
    - Test documents in ./test_documents/ directory
"""

import os
import sys
from pathlib import Path

def check_prerequisites():
    """Check if all prerequisites are met"""
    
    print("🔍 Checking Prerequisites...")
    print("=" * 50)
    
    # Check if test documents exist
    test_docs_dir = Path("test_documents")
    if not test_docs_dir.exists():
        print("❌ test_documents directory not found")
        return False
    
    required_files = [
        "citizenship_guide.txt",
        "eligibility_criteria.md", 
        "refund_policy.html",
        "citizenship_requirements.pdf",
        "eligibility_criteria.docx"
    ]
    
    missing_files = []
    for file in required_files:
        file_path = test_docs_dir / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing test documents: {', '.join(missing_files)}")
        print("Please run the document creation scripts first:")
        print("  python create_test_docx.py")
        print("  python create_simple_pdf.py")
        return False
    
    # Check if validation script exists
    if not Path("test_document_format_validation.py").exists():
        print("❌ test_document_format_validation.py not found")
        return False
    
    print("\n✅ All prerequisites met!")
    return True

def run_validation():
    """Run the format validation"""
    
    print("\n🚀 Running Document Format Validation...")
    print("=" * 50)
    
    # Import and run the validator
    try:
        from test_document_format_validation import DocumentFormatValidator
        
        # Create validator instance
        validator = DocumentFormatValidator()
        
        # Run all tests
        summary = validator.run_all_tests()
        
        return summary
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure test_document_format_validation.py is in the current directory")
        return None
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return None

def interpret_results(summary):
    """Provide guidance on interpreting results"""
    
    if not summary:
        print("\n❌ No results to interpret")
        return
    
    print("\n📋 Results Interpretation Guide")
    print("=" * 50)
    
    if summary.get("overall_success", False):
        print("🎉 SUCCESS: All format validation tests passed!")
        print("\nWhat this means:")
        print("✅ Your RAG system can handle all supported document formats")
        print("✅ Document ingestion pipeline is working correctly")
        print("✅ Citation functionality is operational")
        print("✅ Content retrieval is accurate")
        print("✅ System is ready for production use")
        
    else:
        print("❌ FAILURE: Some format validation tests failed")
        print("\nTroubleshooting steps:")
        
        if not summary.get("server_accessible", True):
            print("🔧 Server Issue:")
            print("  - Ensure FastAPI server is running on localhost:8000")
            print("  - Check server logs for errors")
            print("  - Verify API endpoints are accessible")
        
        failed_formats = []
        for format_name, result in summary.get("results", {}).items():
            if not result.get("overall_success", False):
                failed_formats.append(format_name)
        
        if failed_formats:
            print(f"\n🔧 Failed Formats: {', '.join(failed_formats)}")
            print("  - Check document processor configuration")
            print("  - Verify document extractor implementations")
            print("  - Review server logs for processing errors")
            print("  - Ensure all required dependencies are installed")
    
    # Show detailed breakdown
    formats_tested = summary.get("formats_tested", 0)
    formats_passed = summary.get("formats_passed", 0)
    
    print(f"\n📊 Summary: {formats_passed}/{formats_tested} formats passed")
    
    if "results" in summary:
        print("\nDetailed Results:")
        for format_name, result in summary["results"].items():
            status = "✅ PASSED" if result.get("overall_success", False) else "❌ FAILED"
            print(f"  {status} {format_name}")

def main():
    """Main demo function"""
    
    print("🎯 Document Format Validation Demo")
    print("=" * 70)
    print("This demo validates document format support across the RAG pipeline")
    print()
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Please fix the issues above and try again.")
        return 1
    
    # Run validation
    summary = run_validation()
    
    # Interpret results
    interpret_results(summary)
    
    # Show next steps
    print("\n🔄 Next Steps")
    print("=" * 50)
    
    if summary and summary.get("overall_success", False):
        print("✅ Format validation complete!")
        print("You can now:")
        print("  - Deploy the system to production")
        print("  - Run additional integration tests")
        print("  - Test with real government documents")
    else:
        print("🔧 Fix the identified issues and re-run validation:")
        print("  python run_format_validation_demo.py")
    
    # Show results file location
    results_file = "format_validation_results.json"
    if Path(results_file).exists():
        print(f"\n📄 Detailed results saved to: {results_file}")
    
    return 0 if summary and summary.get("overall_success", False) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
