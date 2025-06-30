# app/services/rag/nodes/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any
from app.services.rag.workflow_state import RAGState


class RAGNode(ABC):
    """Base class for all RAG workflow nodes"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    async def execute(self, state: RAGState) -> RAGState:
        """Execute the node logic and return updated state"""
        pass
    
    async def __call__(self, state: RAGState) -> RAGState:
        """Make nodes callable for LangGraph integration"""
        return await self.execute(state)
