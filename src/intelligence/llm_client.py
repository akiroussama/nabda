"""
LLM client for Gemini 2.0 Flash.

Provides a unified interface for LLM interactions with retry, caching, and fallback support.
"""

import hashlib
import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Any, Callable, Literal

from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Suppress deprecation warning for google.generativeai
# TODO: Migrate to google.genai package when stable
warnings.filterwarnings("ignore", message=".*google.generativeai.*")

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig, HarmBlockThreshold, HarmCategory
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class RateLimitError(LLMClientError):
    """Rate limit exceeded."""
    pass


class ContentBlockedError(LLMClientError):
    """Content blocked by safety filters."""
    pass


class LLMResponseCache:
    """
    Simple in-memory cache for LLM responses.

    Stores responses with TTL and optional persistence.
    """

    def __init__(
        self,
        default_ttl: int = 3600,
        max_entries: int = 1000,
    ):
        """
        Initialize the cache.

        Args:
            default_ttl: Default time-to-live in seconds
            max_entries: Maximum cache entries
        """
        self._cache: dict[str, dict[str, Any]] = {}
        self._default_ttl = default_ttl
        self._max_entries = max_entries

    def get(self, key: str) -> str | None:
        """Get cached response if valid."""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if datetime.now() > entry["expires_at"]:
            del self._cache[key]
            return None

        logger.debug(f"Cache hit for key {key[:16]}...")
        return entry["response"]

    def set(
        self,
        key: str,
        response: str,
        ttl: int | None = None,
    ) -> None:
        """Cache a response."""
        # Evict oldest if at capacity
        if len(self._cache) >= self._max_entries:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k]["created_at"]
            )
            del self._cache[oldest_key]

        self._cache[key] = {
            "response": response,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(seconds=ttl or self._default_ttl),
        }
        logger.debug(f"Cached response for key {key[:16]}...")

    def invalidate(self, key: str) -> None:
        """Invalidate a cache entry."""
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        now = datetime.now()
        valid_entries = sum(
            1 for entry in self._cache.values()
            if entry["expires_at"] > now
        )
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid_entries,
            "max_entries": self._max_entries,
        }


class GeminiClient:
    """
    Client for Google Gemini 2.0 Flash API.

    Provides prompt execution with retry, caching, and safety handling.

    Example:
        >>> client = GeminiClient(api_key="...")
        >>> response = client.generate("Summarize this ticket...")
    """

    DEFAULT_MODEL = "gemini-2.0-flash-exp"

    DEFAULT_GENERATION_CONFIG = {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 2048,
    }

    # Safety settings - allow most content for analysis tasks
    DEFAULT_SAFETY_SETTINGS = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_ONLY_HIGH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_ONLY_HIGH",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_ONLY_HIGH",
    }

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        *,
        generation_config: dict[str, Any] | None = None,
        enable_cache: bool = True,
        cache_ttl: int = 3600,
    ):
        """
        Initialize the Gemini client.

        Args:
            api_key: Google API key (or from GOOGLE_API_KEY env)
            model: Model to use
            generation_config: Custom generation config
            enable_cache: Whether to cache responses
            cache_ttl: Cache TTL in seconds
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )

        self._api_key = api_key
        self._model_name = model or self.DEFAULT_MODEL
        self._generation_config = {
            **self.DEFAULT_GENERATION_CONFIG,
            **(generation_config or {}),
        }

        # Initialize the API
        if api_key:
            genai.configure(api_key=api_key)

        # Create model instance
        self._model = genai.GenerativeModel(
            model_name=self._model_name,
            generation_config=GenerationConfig(**self._generation_config),
        )

        # Response cache
        self._cache = LLMResponseCache(default_ttl=cache_ttl) if enable_cache else None
        self._enable_cache = enable_cache

        # Metrics
        self._total_requests = 0
        self._cache_hits = 0
        self._total_tokens = 0

        logger.info(f"GeminiClient initialized with model {self._model_name}")

    def _get_cache_key(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Generate cache key for prompt."""
        content = f"{system_prompt or ''}|{prompt}"
        return hashlib.sha256(content.encode()).hexdigest()

    @retry(
        retry=retry_if_exception_type((RateLimitError, ConnectionError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        use_cache: bool = True,
        cache_ttl: int | None = None,
        response_format: Literal["text", "json"] = "text",
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            use_cache: Whether to use response cache
            cache_ttl: Custom cache TTL
            response_format: Expected response format

        Returns:
            Generated response text
        """
        self._total_requests += 1

        # Check cache first
        if self._enable_cache and use_cache:
            cache_key = self._get_cache_key(prompt, system_prompt)
            cached = self._cache.get(cache_key)
            if cached:
                self._cache_hits += 1
                return cached

        # Build messages
        messages = []
        if system_prompt:
            # Gemini uses a different approach - prepend to prompt
            full_prompt = f"{system_prompt}\n\n---\n\n{prompt}"
        else:
            full_prompt = prompt

        # Add JSON format instruction if needed
        if response_format == "json":
            full_prompt += "\n\nRespond with valid JSON only, no markdown formatting."

        try:
            # Generate response
            response = self._model.generate_content(full_prompt)

            # Check for blocked content
            if response.prompt_feedback.block_reason:
                raise ContentBlockedError(
                    f"Content blocked: {response.prompt_feedback.block_reason}"
                )

            # Extract text
            if not response.candidates:
                raise LLMClientError("No response candidates returned")

            result = response.text

            # Track tokens if available
            if hasattr(response, 'usage_metadata'):
                self._total_tokens += getattr(
                    response.usage_metadata, 'total_token_count', 0
                )

            # Cache result
            if self._enable_cache and use_cache:
                self._cache.set(cache_key, result, ttl=cache_ttl)

            logger.debug(f"Generated response ({len(result)} chars)")
            return result

        except Exception as e:
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str:
                logger.warning("Rate limit hit, will retry...")
                raise RateLimitError(str(e))
            elif "blocked" in error_str or "safety" in error_str:
                raise ContentBlockedError(str(e))
            else:
                raise LLMClientError(f"Generation failed: {e}")

    def generate_json(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        use_cache: bool = True,
        default: dict | None = None,
    ) -> dict[str, Any]:
        """
        Generate and parse a JSON response.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            use_cache: Whether to use cache
            default: Default value if parsing fails

        Returns:
            Parsed JSON response
        """
        response = self.generate(
            prompt,
            system_prompt=system_prompt,
            use_cache=use_cache,
            response_format="json",
        )

        # Clean response (remove markdown code blocks if present)
        cleaned = response.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            if default is not None:
                return default
            raise LLMClientError(f"Invalid JSON response: {e}")

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Token count
        """
        try:
            result = self._model.count_tokens(text)
            return result.total_tokens
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            # Rough estimate
            return len(text) // 4

    def get_stats(self) -> dict[str, Any]:
        """Get client statistics."""
        stats = {
            "model": self._model_name,
            "total_requests": self._total_requests,
            "cache_hits": self._cache_hits,
            "cache_hit_rate": (
                self._cache_hits / self._total_requests * 100
                if self._total_requests > 0 else 0
            ),
            "total_tokens": self._total_tokens,
        }

        if self._cache:
            stats["cache"] = self._cache.stats()

        return stats

    def clear_cache(self) -> None:
        """Clear the response cache."""
        if self._cache:
            self._cache.clear()
            logger.info("LLM response cache cleared")


class MockGeminiClient:
    """
    Mock client for testing without API calls.

    Returns predefined or generated mock responses.
    """

    def __init__(self, responses: dict[str, str] | None = None):
        """
        Initialize mock client.

        Args:
            responses: Map of prompt patterns to responses
        """
        self._responses = responses or {}
        self._total_requests = 0

    def generate(
        self,
        prompt: str,
        **kwargs,
    ) -> str:
        """Generate mock response."""
        self._total_requests += 1

        # Check for matching pattern
        for pattern, response in self._responses.items():
            if pattern.lower() in prompt.lower():
                return response

        # Default mock response
        return json.dumps({
            "summary": "Mock summary for testing",
            "status": "ok",
            "recommendations": ["Mock recommendation 1", "Mock recommendation 2"],
        })

    def generate_json(self, prompt: str, **kwargs) -> dict[str, Any]:
        """Generate mock JSON response."""
        response = self.generate(prompt, **kwargs)
        return json.loads(response)

    def count_tokens(self, text: str) -> int:
        """Mock token count."""
        return len(text) // 4

    def get_stats(self) -> dict[str, Any]:
        """Get mock stats."""
        return {
            "model": "mock",
            "total_requests": self._total_requests,
        }

    def clear_cache(self) -> None:
        """No-op for mock."""
        pass


def create_llm_client(
    api_key: str | None = None,
    use_mock: bool = False,
    **kwargs,
) -> GeminiClient | MockGeminiClient:
    """
    Factory function to create LLM client.

    Args:
        api_key: Google API key
        use_mock: Whether to use mock client
        **kwargs: Additional arguments for client

    Returns:
        LLM client instance
    """
    if use_mock or not GENAI_AVAILABLE:
        logger.info("Using mock LLM client")
        return MockGeminiClient()

    return GeminiClient(api_key=api_key, **kwargs)
