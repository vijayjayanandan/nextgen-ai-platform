# app/services/tokenizer_service.py
from typing import Optional, Dict, Any
import tiktoken
from app.core.logging import get_logger

logger = get_logger(__name__)

class TokenizerService:
    """
    A service for counting tokens across different model types.
    Follows the Single Responsibility Principle by encapsulating all token 
    counting logic in one place.
    """
    
    def __init__(self):
        self._tokenizers = {}  # Cache for tokenizers
    
    async def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens for the given text based on model type.
        
        Args:
            text: The text to count tokens for
            model: The model identifier (e.g., "claude-3-7-sonnet", "gpt-4")
            
        Returns:
            Token count
        """
        if not text:
            return 0
            
        # Determine which tokenizer to use based on the model
        if model and any(prefix in model.lower() for prefix in ["claude", "anthropic"]):
            return await self._count_anthropic_tokens(text)
        elif model and any(prefix in model.lower() for prefix in ["gpt", "text-davinci", "openai"]):
            return await self._count_openai_tokens(text, model)
        else:
            # Use cl100k_base as a reasonable default for most modern models
            return await self._count_with_tiktoken(text, "cl100k_base")
    
    async def _count_anthropic_tokens(self, text: str) -> int:
        """
        Count tokens using Anthropic's tokenization approximation.
        For more accuracy, Anthropic recommends using cl100k_base.
        """
        try:
            # Use the same encoding OpenAI uses for GPT-4, which is similar to Claude's tokenizer
            return await self._count_with_tiktoken(text, "cl100k_base")
        except Exception as e:
            logger.warning(f"Error counting Anthropic tokens: {str(e)}")
            # Fallback to approximation
            return len(text) // 4 + 1
    
    async def _count_openai_tokens(self, text: str, model: str) -> int:
        """
        Count tokens using OpenAI's tiktoken library.
        """
        # Select the right encoding based on model
        if "gpt-4" in model or "gpt-3.5-turbo" in model:
            encoding_name = "cl100k_base"
        elif "text-davinci-003" in model:
            encoding_name = "p50k_base"
        elif any(model_type in model for model_type in ["text-davinci-001", "text-davinci-002", "text-curie"]):
            encoding_name = "r50k_base"
        else:
            encoding_name = "cl100k_base"  # Default to most modern tokenizer
            
        return await self._count_with_tiktoken(text, encoding_name)
    
    async def _count_with_tiktoken(self, text: str, encoding_name: str) -> int:
        """
        Count tokens using tiktoken with the specified encoding.
        """
        try:
            # Cache tokenizers to avoid recreating them
            if encoding_name not in self._tokenizers:
                self._tokenizers[encoding_name] = tiktoken.get_encoding(encoding_name)
                
            encoding = self._tokenizers[encoding_name]
            token_count = len(encoding.encode(text))
            
            logger.debug(f"Counted {token_count} tokens using {encoding_name}")
            return token_count
            
        except Exception as e:
            logger.warning(f"Error using tiktoken with {encoding_name}: {str(e)}")
            # Fallback to word-based approximation
            word_count = len(text.split())
            logger.warning(f"Falling back to word count: {word_count}")
            return word_count


# Create a singleton instance
_tokenizer_service = TokenizerService()

def get_tokenizer_service() -> TokenizerService:
    """Factory function to get the tokenizer service instance."""
    return _tokenizer_service