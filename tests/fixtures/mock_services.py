# tests/fixtures/mock_services.py
"""Production-grade mock services for RAG workflow testing"""

import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from app.services.retrieval.qdrant_service import SearchResult
from app.services.rag.nodes.memory_retrieval import MemoryTurn


@dataclass
class MockScenarioConfig:
    """Configuration for mock service behavior"""
    memory_enabled: bool = True
    documents_available: bool = True
    response_latency: float = 0.1
    error_rate: float = 0.0
    max_memory_turns: int = 5
    max_documents: int = 10
    memory_score_range: tuple = (0.5, 0.95)
    document_score_range: tuple = (0.6, 0.9)


class ProductionMockQdrantService:
    """Production-grade mock Qdrant service with configurable scenarios"""
    
    def __init__(self, scenario_config: Optional[MockScenarioConfig] = None):
        self.config = scenario_config or MockScenarioConfig()
        self.call_history = []
        self.memory_store = {}
        self.document_store = {}
        self._setup_default_data()
    
    def _setup_default_data(self):
        """Setup default test data"""
        from .sample_data import SAMPLE_MEMORY_TURNS, SAMPLE_DOCUMENTS
        
        # Setup memory data
        for turn_data in SAMPLE_MEMORY_TURNS:
            conv_id = turn_data["conversation_id"]
            if conv_id not in self.memory_store:
                self.memory_store[conv_id] = []
            
            self.memory_store[conv_id].append(SearchResult(
                id=turn_data["id"],
                score=turn_data["relevance_score"],
                content=f"User: {turn_data['user_message']}\nAssistant: {turn_data['assistant_message']}",
                metadata=turn_data
            ))
        
        # Setup document data
        for doc_data in SAMPLE_DOCUMENTS:
            self.document_store[doc_data["id"]] = SearchResult(
                id=doc_data["id"],
                score=0.8,  # Default score
                content=doc_data["content"],
                metadata={
                    "title": doc_data["title"],
                    "source_type": doc_data["source_type"],
                    **doc_data.get("metadata", {})
                }
            )
    
    async def search_memory(
        self,
        conversation_id: str,
        query: str,
        limit: int = 5,
        score_threshold: float = 0.5
    ) -> List[SearchResult]:
        """Mock memory search with realistic behavior"""
        
        self.call_history.append(("search_memory", conversation_id, query, limit, score_threshold))
        
        # Simulate latency
        await asyncio.sleep(self.config.response_latency)
        
        # Simulate errors
        if self.config.error_rate > 0 and self._should_error():
            raise Exception("Mock Qdrant connection error")
        
        # Return empty if memory disabled
        if not self.config.memory_enabled:
            return []
        
        # Get memory for conversation
        memory_turns = self.memory_store.get(conversation_id, [])
        
        # Filter by score threshold and simulate semantic relevance
        relevant_turns = []
        for turn in memory_turns:
            # Simulate semantic similarity scoring
            relevance_score = self._calculate_semantic_relevance(query, turn.content)
            if relevance_score >= score_threshold:
                # Update score to simulated relevance
                turn.score = relevance_score
                relevant_turns.append(turn)
        
        # Sort by relevance and limit
        relevant_turns.sort(key=lambda x: x.score, reverse=True)
        return relevant_turns[:min(limit, self.config.max_memory_turns)]
    
    async def search_documents(
        self,
        query: str,
        search_type: str = "hybrid",
        limit: int = 10,
        filters: Optional[Dict] = None,
        **kwargs
    ) -> List[SearchResult]:
        """Mock document search with different search types"""
        
        self.call_history.append(("search_documents", query, search_type, limit))
        
        # Simulate latency
        await asyncio.sleep(self.config.response_latency)
        
        # Simulate errors
        if self.config.error_rate > 0 and self._should_error():
            raise Exception("Mock document search error")
        
        # Return empty if documents disabled
        if not self.config.documents_available:
            return []
        
        # Get all documents
        all_docs = list(self.document_store.values())
        
        # Apply filters if provided
        if filters:
            all_docs = self._apply_filters(all_docs, filters)
        
        # Simulate different search types
        if search_type == "semantic":
            results = self._semantic_search(query, all_docs)
        elif search_type == "keyword":
            results = self._keyword_search(query, all_docs)
        elif search_type == "hybrid":
            semantic_results = self._semantic_search(query, all_docs)
            keyword_results = self._keyword_search(query, all_docs)
            results = self._fuse_results(semantic_results, keyword_results)
        else:
            results = all_docs
        
        # Sort by score and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:min(limit, self.config.max_documents)]
    
    async def upsert_memory_turn(
        self,
        conversation_id: str,
        turn: Dict[str, Any]
    ) -> bool:
        """Mock memory storage"""
        
        self.call_history.append(("upsert_memory_turn", conversation_id, turn))
        
        # Simulate latency
        await asyncio.sleep(self.config.response_latency * 0.5)
        
        # Add to memory store
        if conversation_id not in self.memory_store:
            self.memory_store[conversation_id] = []
        
        search_result = SearchResult(
            id=turn.get("id", f"turn_{len(self.memory_store[conversation_id])}"),
            score=0.9,  # New turns have high relevance
            content=f"User: {turn['user_message']}\nAssistant: {turn['assistant_message']}",
            metadata=turn
        )
        
        self.memory_store[conversation_id].append(search_result)
        return True
    
    def _calculate_semantic_relevance(self, query: str, content: str) -> float:
        """Simulate semantic relevance calculation"""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        # Simple word overlap simulation
        overlap = len(query_words.intersection(content_words))
        total_words = len(query_words.union(content_words))
        
        if total_words == 0:
            return 0.0
        
        # Base similarity with some randomness
        base_score = overlap / total_words
        
        # Add some realistic variance
        import random
        variance = random.uniform(-0.1, 0.1)
        score = max(0.0, min(1.0, base_score + variance))
        
        # Ensure score is within configured range
        min_score, max_score = self.config.memory_score_range
        return min_score + (score * (max_score - min_score))
    
    def _semantic_search(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        """Simulate semantic search"""
        results = []
        for doc in documents:
            score = self._calculate_semantic_relevance(query, doc.content)
            doc_copy = SearchResult(
                id=doc.id,
                score=score,
                content=doc.content,
                metadata=doc.metadata
            )
            results.append(doc_copy)
        return results
    
    def _keyword_search(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        """Simulate keyword/BM25 search"""
        results = []
        query_terms = query.lower().split()
        
        for doc in documents:
            # Simple BM25-like scoring
            content_lower = doc.content.lower()
            score = 0.0
            
            for term in query_terms:
                if term in content_lower:
                    # Term frequency simulation
                    tf = content_lower.count(term)
                    score += tf * 0.1
            
            # Normalize and add variance
            score = min(1.0, score)
            min_score, max_score = self.config.document_score_range
            final_score = min_score + (score * (max_score - min_score))
            
            doc_copy = SearchResult(
                id=doc.id,
                score=final_score,
                content=doc.content,
                metadata=doc.metadata
            )
            results.append(doc_copy)
        
        return results
    
    def _fuse_results(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult]
    ) -> List[SearchResult]:
        """Simulate Reciprocal Rank Fusion (RRF)"""
        
        # Create rank maps
        semantic_ranks = {result.id: i + 1 for i, result in enumerate(semantic_results)}
        keyword_ranks = {result.id: i + 1 for i, result in enumerate(keyword_results)}
        
        # Get all unique document IDs
        all_doc_ids = set(semantic_ranks.keys()) | set(keyword_ranks.keys())
        
        # Calculate RRF scores
        k = 60  # RRF parameter
        fused_results = []
        
        for doc_id in all_doc_ids:
            semantic_rank = semantic_ranks.get(doc_id, len(semantic_results) + 1)
            keyword_rank = keyword_ranks.get(doc_id, len(keyword_results) + 1)
            
            rrf_score = (1 / (k + semantic_rank)) + (1 / (k + keyword_rank))
            
            # Find the document
            doc = None
            for result in semantic_results + keyword_results:
                if result.id == doc_id:
                    doc = result
                    break
            
            if doc:
                fused_doc = SearchResult(
                    id=doc.id,
                    score=rrf_score,
                    content=doc.content,
                    metadata=doc.metadata
                )
                fused_results.append(fused_doc)
        
        return fused_results
    
    def _apply_filters(self, documents: List[SearchResult], filters: Dict) -> List[SearchResult]:
        """Apply metadata filters to documents"""
        filtered = []
        
        for doc in documents:
            include = True
            
            for key, value in filters.items():
                if key in doc.metadata:
                    if isinstance(value, list):
                        if doc.metadata[key] not in value:
                            include = False
                            break
                    else:
                        if doc.metadata[key] != value:
                            include = False
                            break
                else:
                    include = False
                    break
            
            if include:
                filtered.append(doc)
        
        return filtered
    
    def _should_error(self) -> bool:
        """Determine if an error should occur based on error rate"""
        import random
        return random.random() < self.config.error_rate
    
    def reset_call_history(self):
        """Reset call history for testing"""
        self.call_history = []
    
    def get_call_count(self, method_name: str) -> int:
        """Get number of calls to a specific method"""
        return len([call for call in self.call_history if call[0] == method_name])


class MockOllamaAdapter:
    """Deterministic LLM adapter for testing"""
    
    def __init__(self, response_templates: Optional[Dict[str, str]] = None):
        self.response_templates = response_templates or self._default_templates()
        self.call_history = []
        self.generation_count = 0
    
    def _default_templates(self) -> Dict[str, str]:
        """Default response templates based on prompt content"""
        return {
            "work_authorization": """Based on your previous questions about work authorization and the available documentation, here are the key documents you'll need:

**Primary Documents [Document 1]:**
- Valid passport
- Job offer letter from Canadian employer
- Labour Market Impact Assessment (LMIA) if required
- Educational credential assessment

**Supporting Documents [Document 2]:**
- Medical examination results
- Police certificates from countries where you've lived
- Proof of funds to support yourself
- Passport-style photographs

Since you previously asked about working while your application is being processed, note that work authorization is separate from your main immigration application and requires its own documentation process.""",
            
            "visa_requirements": """For Canadian visas, you need to meet several eligibility criteria [Document 1]:

**Basic Requirements:**
- Valid passport
- Completed application forms
- Application fees
- Biometric information

**Supporting Documentation [Document 2]:**
- Proof of financial support
- Medical examination (if required)
- Police certificates
- Purpose of visit documentation

The specific requirements may vary depending on your country of residence and the type of visa you're applying for.""",
            
            "general_immigration": """I can help you with Canadian immigration questions. Based on the available documentation [Document 1], immigration to Canada involves several pathways including:

- Express Entry system for skilled workers
- Provincial Nominee Programs (PNP)
- Family sponsorship
- Business and investor programs

Each pathway has specific requirements and documentation needs. Please let me know what specific aspect of immigration you'd like to know more about.""",
            
            "fallback": """I apologize, but I don't have sufficient information in the provided context to answer your specific question. Please provide more details about what you'd like to know about Canadian immigration, or rephrase your question to be more specific."""
        }
    
    async def generate(
        self,
        prompt: str,
        model: str = "deepseek-coder:6.7b",
        temperature: float = 0.3,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate deterministic responses based on prompt content"""
        
        self.call_history.append(("generate", prompt[:100], model, temperature, max_tokens))
        self.generation_count += 1
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        # Select response based on prompt content
        prompt_lower = prompt.lower()
        
        if "work authorization" in prompt_lower or "work permit" in prompt_lower:
            return self.response_templates["work_authorization"]
        elif "visa requirements" in prompt_lower or "visa" in prompt_lower:
            return self.response_templates["visa_requirements"]
        elif "immigration" in prompt_lower:
            return self.response_templates["general_immigration"]
        else:
            return self.response_templates["fallback"]
    
    def reset_call_history(self):
        """Reset call history for testing"""
        self.call_history = []
        self.generation_count = 0
    
    def get_generation_count(self) -> int:
        """Get total number of generations"""
        return self.generation_count


class MockEmbeddingService:
    """Mock embedding service for testing"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.call_history = []
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate mock embeddings"""
        self.call_history.append(("embed_text", text[:50]))
        
        # Generate deterministic embeddings based on text hash
        import hashlib
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to embedding vector
        embedding = []
        for i in range(0, len(text_hash), 2):
            hex_pair = text_hash[i:i+2]
            value = int(hex_pair, 16) / 255.0  # Normalize to 0-1
            embedding.append(value)
        
        # Pad or truncate to desired dimension
        while len(embedding) < self.embedding_dim:
            embedding.extend(embedding[:self.embedding_dim - len(embedding)])
        
        return embedding[:self.embedding_dim]
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate batch embeddings"""
        embeddings = []
        for text in texts:
            embedding = await self.embed_text(text)
            embeddings.append(embedding)
        return embeddings
    
    def reset_call_history(self):
        """Reset call history for testing"""
        self.call_history = []
