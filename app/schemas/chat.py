from typing import List, Dict, Optional, Any, Union
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

from app.models.chat import MessageRole


class MessageBase(BaseModel):
    """Base schema for messages."""
    role: MessageRole
    content: str
    function_name: Optional[str] = None
    function_arguments: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MessageCreate(MessageBase):
    """Schema for creating messages."""
    conversation_id: UUID
    sequence: int


class MessageUpdate(BaseModel):
    """Schema for updating messages."""
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    source_documents: Optional[List[UUID]] = None


class MessageInDB(MessageBase):
    """Schema for messages in the database."""
    id: UUID
    conversation_id: UUID
    sequence: int
    created_at: datetime
    updated_at: datetime
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    source_documents: List[UUID] = Field(default_factory=list)

    class Config:
        from_attributes = True


class ConversationBase(BaseModel):
    """Base schema for conversations."""
    title: Optional[str] = None
    description: Optional[str] = None
    model_name: str
    model_params: Dict[str, Any] = Field(default_factory=dict)
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ConversationCreate(ConversationBase):
    """Schema for creating conversations."""
    user_id: UUID
    is_active: bool = True


class ConversationUpdate(BaseModel):
    """Schema for updating conversations."""
    title: Optional[str] = None
    description: Optional[str] = None
    model_name: Optional[str] = None
    model_params: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    system_prompt: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    retrieved_document_ids: Optional[List[UUID]] = None


class ConversationInDB(ConversationBase):
    """Schema for conversations in the database."""
    id: UUID
    user_id: UUID
    created_at: datetime
    updated_at: datetime
    is_active: bool
    retrieved_document_ids: List[UUID] = Field(default_factory=list)

    class Config:
        from_attributes = True


class ConversationWithMessages(ConversationInDB):
    """Schema for conversations with their messages."""
    messages: List[MessageInDB] = Field(default_factory=list)


# Chat API schemas
class ChatMessage(BaseModel):
    """Schema for chat messages in the API."""
    role: MessageRole
    content: str
    function_name: Optional[str] = None
    function_arguments: Optional[Dict[str, Any]] = None
    name: Optional[str] = None  # For named users/assistants


class ChatCompletionRequest(BaseModel):
    """Schema for chat completion requests."""
    messages: List[ChatMessage]
    model: str
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None
    functions: Optional[List[Dict[str, Any]]] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    stop: Optional[Union[str, List[str]]] = None
    # RAG specific parameters
    retrieve: Optional[bool] = False
    retrieval_options: Optional[Dict[str, Any]] = None


class FunctionCall(BaseModel):
    """Schema for function calls."""
    name: str
    arguments: str


class ChatCompletionResponseMessage(BaseModel):
    """Schema for chat completion response messages."""
    role: MessageRole
    content: Optional[str] = None
    function_call: Optional[FunctionCall] = None


class ChatCompletionResponseChoice(BaseModel):
    """Schema for chat completion response choices."""
    index: int
    message: ChatCompletionResponseMessage
    finish_reason: str


class ChatCompletionResponseUsage(BaseModel):
    """Schema for chat completion response usage."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Schema for chat completion responses."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage
    
    # For explainability
    source_documents: Optional[List[Dict[str, Any]]] = None


# Chat history management schemas
class ConversationSummary(BaseModel):
    """Schema for conversation summaries."""
    id: UUID
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int
    last_message_at: Optional[datetime]