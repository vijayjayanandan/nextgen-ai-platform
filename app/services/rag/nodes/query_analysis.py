# app/services/rag/nodes/query_analysis.py
import json
from typing import Dict, Any
from app.services.rag.nodes.base import RAGNode
from app.services.rag.workflow_state import RAGState, QueryType
from app.utils.llm_response_parser import parse_llm_json, validate_query_analysis
from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class QueryAnalysisNode(RAGNode):
    """Analyzes user query to determine type, intent, and entities"""
    
    def __init__(self, ollama_service):
        super().__init__("query_analysis")
        self.ollama = ollama_service
    
    async def execute(self, state: RAGState) -> RAGState:
        """Analyze the user query and extract metadata"""
        
        # Add null guards and input validation
        if not state.user_query or not isinstance(state.user_query, str):
            logger.warning("Invalid or empty query provided, using fallback classification")
            state.query_type = QueryType.SIMPLE
            state.intent = "Invalid or empty query"
            state.entities = []
            state.error_message = "Invalid or empty query provided"
            return state
        
        # Sanitize query for prompt
        sanitized_query = state.user_query.strip()
        if not sanitized_query:
            logger.warning("Empty query after sanitization, using fallback classification")
            state.query_type = QueryType.SIMPLE
            state.intent = "Empty query"
            state.entities = []
            state.error_message = "Empty query after sanitization"
            return state
        
        analysis_prompt = f"""Analyze the user query and return ONLY valid JSON with no explanations or markdown.

Query: "{sanitized_query}"

Return exactly this JSON structure:
{{
    "query_type": "simple",
    "intent": "brief description",
    "entities": ["entity1", "entity2"]
}}

Valid query_type values: simple, conversational, complex, code_related
IMPORTANT: Return ONLY the JSON object, no other text."""
        
        try:
            # Use configurable model for query analysis
            model_name = settings.RAG_QUERY_ANALYSIS_MODEL
            response = await self.ollama.generate(
                prompt=analysis_prompt,
                model=model_name,
                temperature=0.1,
                max_tokens=200
            )
            
            # Use robust JSON parsing
            fallback_data = {
                "query_type": "simple",
                "intent": "General information request",
                "entities": []
            }
            
            parse_result = parse_llm_json(
                response=response,
                expected_type=dict,
                validator=validate_query_analysis,
                fallback_value=fallback_data
            )
            
            if parse_result.fallback_used:
                logger.warning(f"Query analysis JSON parsing used fallback: {parse_result.error_message}")
            
            analysis = parse_result.data
            
            state.query_type = QueryType(analysis["query_type"])
            state.intent = analysis["intent"]
            state.entities = analysis["entities"]
            
            logger.info(f"Query analyzed: type={state.query_type}, entities={len(state.entities)}")
            
        except (KeyError, ValueError) as e:
            # Fallback to simple classification
            logger.warning(f"Query analysis failed: {e}, using fallback classification")
            
            state.query_type = self._fallback_classification(state.user_query)
            state.intent = "General information request"
            state.entities = self._extract_simple_entities(state.user_query)
            state.error_message = f"Query analysis failed: {e}"
        
        return state
    
    def _fallback_classification(self, query: str) -> QueryType:
        """Simple fallback classification based on keywords"""
        
        # Add null guard
        if not query or not isinstance(query, str):
            return QueryType.SIMPLE
        
        query_lower = query.lower()
        
        # Check for code-related keywords
        code_keywords = ["python", "code", "script", "function", "api", "json", "sql"]
        if any(keyword in query_lower for keyword in code_keywords):
            return QueryType.CODE_RELATED
        
        # Check for conversational indicators
        conversation_keywords = ["remember", "earlier", "before", "previous", "we discussed"]
        if any(keyword in query_lower for keyword in conversation_keywords):
            return QueryType.CONVERSATIONAL
        
        # Check for complex query indicators
        complex_keywords = ["compare", "analyze", "explain the difference", "pros and cons"]
        if any(keyword in query_lower for keyword in complex_keywords):
            return QueryType.COMPLEX
        
        return QueryType.SIMPLE
    
    def _extract_simple_entities(self, query: str) -> list:
        """Simple entity extraction using basic NLP"""
        
        # Add null guard
        if not query or not isinstance(query, str):
            return []
        
        # Basic entity extraction - in production, use spaCy or similar
        import re
        
        # Extract capitalized words (potential proper nouns)
        entities = re.findall(r'\b[A-Z][a-z]+\b', query)
        
        # Extract common immigration terms
        immigration_terms = [
            "visa", "permit", "citizenship", "immigration", "refugee", 
            "passport", "application", "processing", "eligibility"
        ]
        
        query_lower = query.lower()
        for term in immigration_terms:
            if term in query_lower:
                entities.append(term)
        
        return list(set(entities))  # Remove duplicates
