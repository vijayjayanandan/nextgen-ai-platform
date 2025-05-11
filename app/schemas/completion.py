from typing import List, Dict, Optional, Any, Union
from uuid import UUID
from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    """Schema for completion requests."""
    prompt: str
    model: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None
    # RAG specific parameters
    retrieve: Optional[bool] = False
    retrieval_options: Optional[Dict[str, Any]] = None


class CompletionResponseChoice(BaseModel):
    """Schema for completion response choices."""
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: str


class CompletionResponseUsage(BaseModel):
    """Schema for completion response usage."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    """Schema for completion responses."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionResponseChoice]
    usage: CompletionResponseUsage
    
    # For explainability
    source_documents: Optional[List[Dict[str, Any]]] = None


# Streaming completion schemas
class CompletionStreamResponseChoice(BaseModel):
    """Schema for streaming completion response choices."""
    text: str
    index: int
    logprobs: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


class CompletionStreamResponse(BaseModel):
    """Schema for streaming completion responses."""
    id: str
    object: str = "text_completion.chunk"
    created: int
    model: str
    choices: List[CompletionStreamResponseChoice]