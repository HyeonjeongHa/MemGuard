"""
LLM Client for MemGuard.

Provides an OpenAI-compatible interface for chat completions and embeddings.
Set OPENAI_API_KEY (and optionally OPENAI_BASE_URL) as environment variables.
"""

import os
from typing import Dict, List, Optional

from openai import OpenAI


class LLMClient:
    """OpenAI-compatible LLM client for chat completions and embeddings."""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4o",
        embedding_model: str = "text-embedding-3-small",
        **kwargs
    ):
        """
        Initialize LLM client.
        
        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL (defaults to OPENAI_BASE_URL env var or OpenAI)
            model: Default model name
            embedding_model: Default embedding model name
        """
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL")

        self.model = model
        self.embedding_model = embedding_model
        
        if not self.api_key:
            raise ValueError(
                "API key required. Set via api_key parameter or OPENAI_API_KEY env var."
            )
        
        # Initialize OpenAI client
        client_kwargs = {"api_key": self.api_key}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url

        self.client = OpenAI(**client_kwargs)
        self.token_usage_input = 0
        self.token_usage_output = 0
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None,
        model: str = None,
        **kwargs
    ) -> str:
        """
        Send chat completion request.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            response_format: Optional response format (e.g., {"type": "json_object"})
            model: Model override (uses default if not specified)
            **kwargs: Additional OpenAI API parameters
        
        Returns:
            Response content as string
        """
        model = model or self.model
        
        request_kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            **kwargs,
        }
        
        if response_format:
            request_kwargs["response_format"] = response_format
        
        response = self.client.chat.completions.create(**request_kwargs)
        self.token_usage_input += response.usage.prompt_tokens
        self.token_usage_output += response.usage.completion_tokens
        return response.choices[0].message.content.strip()
    
    def get_embeddings(
        self,
        texts: List[str],
        model: str = None
    ) -> List[List[float]]:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            model: Embedding model override
        
        Returns:
            List of embedding vectors
        """
        model = model or self.embedding_model
        # text-embedding-3-small limit is 8192 tokens. Dense content (JSON, code,
        # non-English) can hit ~3 chars/token, so use 20k chars (~6600 tokens worst-case).
        MAX_CHARS = 20_000
        texts = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in texts]
        response = self.client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]


# Alias for backward compatibility
OpenAIClient = LLMClient


def create_llm_client(
    api_key: str = None,
    base_url: str = None,
    model: str = "gpt-4o",
    **kwargs
) -> LLMClient:
    """Factory function to create an LLM client (OpenAI or compatible endpoint)."""
    return LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        **kwargs
    )
