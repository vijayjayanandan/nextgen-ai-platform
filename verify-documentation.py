#!/usr/bin/env python3
"""
Documentation Verification Script
Verifies all documentation files are properly created and accessible.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and return status."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"✅ {description}: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"❌ {description}: {filepath} (MISSING)")
        return False

def verify_html_content(filepath):
    """Verify HTML file has key components."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ('Mermaid.js included', 'mermaid' in content.lower()),
            ('Interactive navigation', 'showSection' in content),
            ('Search functionality', 'searchInput' in content),
            ('Component cards', 'component-card' in content),
            ('Architecture layers', 'layer-btn' in content),
            ('Responsive design', '@media' in content),
        ]
        
        print(f"\n📋 HTML Content Verification:")
        for check_name, result in checks:
            status = "✅" if result else "❌"
            print(f"  {status} {check_name}")
        
        return all(result for _, result in checks)
    except Exception as e:
        print(f"❌ Error reading HTML file: {e}")
        return False

def verify_markdown_content(filepath):
    """Verify Markdown file has key sections."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ('Table of Contents', '## 📋 Table of Contents' in content),
            ('Mermaid diagrams', '```mermaid' in content),
            ('Architecture layers', '## 🏗️ Architecture Layers' in content),
            ('PII Detection section', '## 🔒 PII Detection' in content),
            ('RAG Pipeline section', '## 🧠 RAG Pipeline' in content),
            ('Database schema', '## 🗄️ Database Schema' in content),
            ('API endpoints', '## 🌐 API Endpoints' in content),
            ('Maintenance guides', '## 🔧 Maintenance Guides' in content),
            ('Developer onboarding', '## 🎓 Developer Onboarding' in content),
        ]
        
        print(f"\n📋 Markdown Content Verification:")
        for check_name, result in checks:
            status = "✅" if result else "❌"
            print(f"  {status} {check_name}")
        
        return all(result for _, result in checks)
    except Exception as e:
        print(f"❌ Error reading Markdown file: {e}")
        return False

def main():
    """Main verification function."""
    print("🔍 FastAPI + Next.js AI Platform - Documentation Verification")
    print("=" * 70)
    
    # Check all documentation files
    files_to_check = [
        ("architecture-documentation.html", "Interactive HTML Documentation"),
        ("README-architecture.md", "GitHub README Documentation"),
        ("quick-reference.txt", "ASCII Quick Reference"),
        ("architecture-summary.md", "Architecture Summary"),
        ("verify-documentation.py", "Verification Script"),
    ]
    
    print("\n📁 File Existence Check:")
    all_files_exist = True
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_files_exist = False
    
    if not all_files_exist:
        print("\n❌ Some files are missing. Please ensure all documentation files are created.")
        return False
    
    # Verify HTML content
    html_valid = verify_html_content("architecture-documentation.html")
    
    # Verify Markdown content
    markdown_valid = verify_markdown_content("README-architecture.md")
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 70)
    
    if all_files_exist and html_valid and markdown_valid:
        print("🎉 SUCCESS: All documentation files are properly created and verified!")
        print("\n🚀 Next Steps:")
        print("1. Open 'architecture-documentation.html' in your browser")
        print("2. Share 'README-architecture.md' with your team")
        print("3. Use 'quick-reference.txt' for daily development")
        print("4. Review 'architecture-summary.md' for insights")
        return True
    else:
        print("⚠️  WARNING: Some verification checks failed.")
        print("Please review the issues above and fix any problems.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
