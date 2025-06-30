# tests/fixtures/sample_data.py
"""Sample test data for RAG workflow testing"""

from typing import Dict, List, Any
import time

# Sample memory turns for different conversations
SAMPLE_MEMORY_TURNS = [
    # Conversation 1: Work authorization flow
    {
        "id": "turn_1_conv123",
        "conversation_id": "conv_123",
        "turn_number": 1,
        "user_message": "What are the visa requirements for Canada?",
        "assistant_message": "For Canadian visas, you need to meet eligibility criteria including financial support, clean criminal record, and medical exams. The specific requirements vary by visa type.",
        "timestamp": 1640995200.0,
        "relevance_score": 0.9,
        "user_id": "user_456"
    },
    {
        "id": "turn_2_conv123",
        "conversation_id": "conv_123", 
        "turn_number": 2,
        "user_message": "How long does processing take?",
        "assistant_message": "Processing times vary by visa type, typically 2-12 weeks for visitor visas and 6-12 months for permanent residence applications.",
        "timestamp": 1640995800.0,
        "relevance_score": 0.7,
        "user_id": "user_456"
    },
    {
        "id": "turn_3_conv123",
        "conversation_id": "conv_123",
        "turn_number": 3,
        "user_message": "Can I work while my application is being processed?",
        "assistant_message": "Work authorization depends on your current status and visa type. Visitor visa holders generally cannot work, but some permit holders may be eligible.",
        "timestamp": 1641000000.0,
        "relevance_score": 0.85,
        "user_id": "user_456"
    },
    {
        "id": "turn_4_conv123",
        "conversation_id": "conv_123",
        "turn_number": 4,
        "user_message": "What documents do I need for work permit?",
        "assistant_message": "For work permits, you need a job offer, LMIA (if required), passport, educational credentials, and proof of funds.",
        "timestamp": 1641001200.0,
        "relevance_score": 0.95,
        "user_id": "user_456"
    },
    
    # Conversation 2: Different user, family sponsorship
    {
        "id": "turn_1_conv456",
        "conversation_id": "conv_456",
        "turn_number": 1,
        "user_message": "How do I sponsor my spouse for immigration?",
        "assistant_message": "To sponsor your spouse, you must be a Canadian citizen or permanent resident, meet income requirements, and submit Form IMM 1344.",
        "timestamp": 1641002400.0,
        "relevance_score": 0.9,
        "user_id": "user_789"
    },
    {
        "id": "turn_2_conv456",
        "conversation_id": "conv_456",
        "turn_number": 2,
        "user_message": "What are the income requirements?",
        "assistant_message": "Income requirements vary by family size. You must meet the Low Income Cut-Off (LICO) for the past 3 years.",
        "timestamp": 1641003000.0,
        "relevance_score": 0.8,
        "user_id": "user_789"
    },
    
    # Conversation 3: Student visa queries
    {
        "id": "turn_1_conv789",
        "conversation_id": "conv_789",
        "turn_number": 1,
        "user_message": "What do I need for a study permit?",
        "assistant_message": "For a study permit, you need acceptance from a designated learning institution, proof of funds, and may need a medical exam.",
        "timestamp": 1641004800.0,
        "relevance_score": 0.85,
        "user_id": "user_101"
    }
]

# Sample documents covering various immigration topics
SAMPLE_DOCUMENTS = [
    {
        "id": "doc_work_permit_guide",
        "title": "Work Permit Application Guide",
        "content": """To apply for a work permit in Canada, you must provide several key documents:

**Required Documents:**
- Valid passport or travel document
- Job offer letter from Canadian employer
- Labour Market Impact Assessment (LMIA) if required
- Educational credential assessment
- Proof of work experience
- Language test results (if applicable)

**Supporting Documents:**
- Medical examination results (if required)
- Police certificates from countries where you've lived for 6+ months
- Proof of funds to support yourself and family
- Passport-style photographs
- Application fees

**Processing Information:**
Work permit applications are typically processed within 2-12 weeks, depending on your country of residence and the type of work permit. Some applications may require additional processing time for medical exams or security checks.

**Work Authorization:**
You cannot work in Canada until you receive your work permit. Working without proper authorization is illegal and can result in removal from Canada.""",
        "source_type": "policy",
        "metadata": {
            "page_number": 1,
            "section": "requirements",
            "document_type": "guide",
            "last_updated": "2024-01-15"
        }
    },
    
    {
        "id": "doc_visa_requirements",
        "title": "Canadian Visa Requirements Overview",
        "content": """Canadian visa requirements vary depending on your nationality and the purpose of your visit:

**Visitor Visa Requirements:**
- Valid passport
- Completed application form (IMM 5257)
- Photographs meeting specifications
- Proof of financial support
- Letter of invitation (if applicable)
- Travel itinerary
- Medical exam (if required)

**Study Permit Requirements:**
- Acceptance letter from designated learning institution
- Proof of financial support for tuition and living expenses
- Letter of explanation
- Medical exam and police certificates (if required)

**Work Permit Requirements:**
- Job offer from Canadian employer
- LMIA or exemption code
- Proof of qualifications and work experience

**Processing Times:**
- Visitor visa: 2-4 weeks
- Study permit: 4-12 weeks  
- Work permit: 2-12 weeks

**Biometrics:**
Most applicants must provide biometrics (fingerprints and photo) as part of their application.""",
        "source_type": "guideline",
        "metadata": {
            "page_number": 2,
            "section": "overview",
            "document_type": "reference",
            "last_updated": "2024-02-01"
        }
    },
    
    {
        "id": "doc_family_sponsorship",
        "title": "Family Class Sponsorship Guide",
        "content": """Canadian citizens and permanent residents can sponsor certain family members for immigration:

**Eligible Relationships:**
- Spouse, common-law partner, or conjugal partner
- Dependent children under 22 years old
- Parents and grandparents
- Orphaned siblings, nephews, nieces, or grandchildren under 18

**Sponsor Requirements:**
- Must be 18 years or older
- Canadian citizen or permanent resident
- Meet income requirements (LICO)
- Sign undertaking to provide financial support
- Not receiving social assistance (except disability)

**Income Requirements:**
Sponsors must meet the Low Income Cut-Off (LICO) for:
- 3 consecutive years for parents/grandparents
- Current year for spouse/children

**Application Process:**
1. Submit sponsorship application (IMM 1344)
2. Principal applicant submits permanent residence application
3. Pay required fees
4. Attend interviews if requested
5. Medical exams and background checks

**Processing Times:**
- Spouse/children: 12-24 months
- Parents/grandparents: 24-36 months""",
        "source_type": "policy",
        "metadata": {
            "page_number": 3,
            "section": "family_class",
            "document_type": "guide",
            "last_updated": "2024-01-20"
        }
    },
    
    {
        "id": "doc_express_entry",
        "title": "Express Entry System Overview",
        "content": """The Express Entry system manages applications for three federal economic immigration programs:

**Eligible Programs:**
- Federal Skilled Worker Program
- Canadian Experience Class  
- Federal Skilled Trades Program

**Eligibility Requirements:**
- Language proficiency (English and/or French)
- Education credentials assessment
- Work experience in eligible occupations
- Proof of funds
- Medical exams and security checks

**Comprehensive Ranking System (CRS):**
Points awarded for:
- Age (maximum 110 points)
- Education (maximum 150 points)
- Language ability (maximum 160 points)
- Work experience (maximum 80 points)
- Arranged employment (50-200 points)
- Provincial nomination (600 points)

**Application Process:**
1. Create Express Entry profile
2. Receive Invitation to Apply (ITA)
3. Submit complete application within 60 days
4. Processing time: 6 months or less

**Required Documents:**
- Language test results
- Educational credential assessment
- Proof of work experience
- Police certificates
- Medical exams
- Proof of funds""",
        "source_type": "program",
        "metadata": {
            "page_number": 4,
            "section": "express_entry",
            "document_type": "overview",
            "last_updated": "2024-01-10"
        }
    },
    
    {
        "id": "doc_medical_exams",
        "title": "Medical Examination Requirements",
        "content": """Medical examinations are required for most immigration applications to Canada:

**When Required:**
- Permanent residence applications
- Work permits over 6 months
- Study permits over 6 months
- Visitor visas (certain countries)

**Examination Components:**
- Physical examination
- Chest X-ray
- Blood tests (if indicated)
- Urine tests (if indicated)
- Mental health assessment (if required)

**Designated Medical Practitioners:**
- Must use IRCC-approved panel physicians
- Cannot use family doctor or walk-in clinic
- Appointments must be booked in advance

**Validity Period:**
- Medical exams valid for 12 months
- Must land in Canada before expiry
- Re-examination required if expired

**Special Considerations:**
- Pregnant women may defer X-rays
- Children under 5 may have modified requirements
- Additional tests for certain occupations

**Costs:**
- Fees paid directly to panel physician
- Not covered by provincial health insurance
- Costs vary by country and examination type""",
        "source_type": "requirement",
        "metadata": {
            "page_number": 5,
            "section": "medical",
            "document_type": "requirement",
            "last_updated": "2024-01-25"
        }
    }
]

# Sample conversation scenarios for testing
SAMPLE_CONVERSATIONS = {
    "work_authorization_flow": {
        "conversation_id": "conv_123",
        "user_id": "user_456",
        "turns": [
            {
                "user_query": "What are the visa requirements for Canada?",
                "expected_memory_turns": 0,
                "expected_documents": 2
            },
            {
                "user_query": "How long does processing take?",
                "expected_memory_turns": 1,
                "expected_documents": 2
            },
            {
                "user_query": "Can I work while my application is being processed?",
                "expected_memory_turns": 2,
                "expected_documents": 1
            },
            {
                "user_query": "What documents do I need for work authorization?",
                "expected_memory_turns": 3,
                "expected_documents": 2
            }
        ]
    },
    
    "family_sponsorship_flow": {
        "conversation_id": "conv_456", 
        "user_id": "user_789",
        "turns": [
            {
                "user_query": "How do I sponsor my spouse for immigration?",
                "expected_memory_turns": 0,
                "expected_documents": 1
            },
            {
                "user_query": "What are the income requirements?",
                "expected_memory_turns": 1,
                "expected_documents": 1
            }
        ]
    },
    
    "new_user_no_memory": {
        "conversation_id": None,
        "user_id": "user_new",
        "turns": [
            {
                "user_query": "What is Express Entry?",
                "expected_memory_turns": 0,
                "expected_documents": 1
            }
        ]
    }
}

# Sample queries for different test scenarios
SAMPLE_QUERIES = {
    "work_related": [
        "What documents do I need for work authorization?",
        "How do I apply for a work permit?",
        "Can I work while my application is being processed?",
        "What is an LMIA and do I need one?",
        "How long does work permit processing take?"
    ],
    
    "visa_related": [
        "What are the visa requirements for Canada?",
        "How do I apply for a visitor visa?",
        "Do I need a visa to visit Canada?",
        "What documents are required for visa application?",
        "How long does visa processing take?"
    ],
    
    "family_related": [
        "How do I sponsor my spouse?",
        "What are the income requirements for sponsorship?",
        "Can I sponsor my parents?",
        "How long does family sponsorship take?",
        "What is the undertaking requirement?"
    ],
    
    "study_related": [
        "What do I need for a study permit?",
        "How do I apply to study in Canada?",
        "Can I work while studying?",
        "What are designated learning institutions?",
        "How much money do I need for studies?"
    ],
    
    "general": [
        "What is Express Entry?",
        "How do I immigrate to Canada?",
        "What are the different immigration programs?",
        "Do I need medical exams?",
        "How do I check my application status?"
    ],
    
    "edge_cases": [
        "",  # Empty query
        "a",  # Single character
        "What is the meaning of life?",  # Unrelated query
        "Tell me about artificial intelligence",  # Off-topic
        "How do I cook pasta?"  # Completely unrelated
    ]
}

# Test state templates for different scenarios
TEST_STATE_TEMPLATES = {
    "with_memory_and_documents": {
        "user_query": "What documents do I need for work authorization?",
        "conversation_id": "conv_123",
        "user_id": "user_456",
        "memory_context": """## Conversation History:
🔥 [Turn 3] User: Can I work while my application is being processed?
Assistant: Work authorization depends on your current status and visa type...

📝 [Turn 1] User: What are the visa requirements for Canada?
Assistant: For Canadian visas, you need to meet eligibility criteria...""",
        "raw_documents": [
            {
                "id": "doc_work_permit_guide",
                "title": "Work Permit Application Guide",
                "content": "To apply for a work permit, you must provide...",
                "source_type": "policy"
            }
        ]
    },
    
    "memory_only": {
        "user_query": "Tell me more about work permits",
        "conversation_id": "conv_123", 
        "user_id": "user_456",
        "memory_context": """## Conversation History:
📝 [Turn 4] User: What documents do I need for work permit?
Assistant: For work permits, you need a job offer, LMIA...""",
        "raw_documents": []
    },
    
    "documents_only": {
        "user_query": "What is Express Entry?",
        "conversation_id": None,
        "user_id": "user_new",
        "memory_context": "",
        "raw_documents": [
            {
                "id": "doc_express_entry",
                "title": "Express Entry System Overview",
                "content": "The Express Entry system manages applications...",
                "source_type": "program"
            }
        ]
    },
    
    "no_context": {
        "user_query": "Tell me about artificial intelligence",
        "conversation_id": None,
        "user_id": "user_new",
        "memory_context": "",
        "raw_documents": []
    }
}

# Memory configuration test scenarios
MEMORY_CONFIG_SCENARIOS = {
    "default": {
        "max_turns": 5,
        "score_threshold": 0.5,
        "include_recent_turns": 2,
        "max_context_length": 2000
    },
    
    "restrictive": {
        "max_turns": 2,
        "score_threshold": 0.8,
        "include_recent_turns": 1,
        "max_context_length": 500
    },
    
    "permissive": {
        "max_turns": 10,
        "score_threshold": 0.3,
        "include_recent_turns": 5,
        "max_context_length": 5000
    },
    
    "edge_cases": {
        "max_turns": 0,
        "score_threshold": 1.0,
        "include_recent_turns": 0,
        "max_context_length": 100
    }
}

# Expected response patterns for validation
EXPECTED_RESPONSE_PATTERNS = {
    "work_authorization": {
        "required_keywords": ["work permit", "documents", "LMIA", "passport"],
        "citation_pattern": r"\[Document \d+\]",
        "min_length": 100,
        "max_length": 2000
    },
    
    "visa_requirements": {
        "required_keywords": ["visa", "requirements", "passport", "application"],
        "citation_pattern": r"\[Document \d+\]",
        "min_length": 100,
        "max_length": 2000
    },
    
    "fallback": {
        "required_keywords": ["apologize", "information", "context"],
        "citation_pattern": None,
        "min_length": 50,
        "max_length": 500
    }
}
