"""Tests for LLM client."""

import json
import pytest
from datetime import datetime, timedelta

from src.intelligence.llm_client import (
    LLMResponseCache,
    MockGeminiClient,
    LLMClientError,
    create_llm_client,
)


class TestLLMResponseCache:
    """Tests for LLM response cache."""

    def test_set_and_get(self):
        """Test basic set and get operations."""
        cache = LLMResponseCache()

        cache.set("key1", "response1")
        result = cache.get("key1")

        assert result == "response1"

    def test_get_missing_key(self):
        """Test getting non-existent key."""
        cache = LLMResponseCache()

        result = cache.get("nonexistent")

        assert result is None

    def test_expiration(self):
        """Test that entries expire after TTL."""
        cache = LLMResponseCache(default_ttl=1)  # 1 second TTL

        cache.set("key1", "response1")

        # Manually expire the entry
        cache._cache["key1"]["expires_at"] = datetime.now() - timedelta(seconds=1)

        result = cache.get("key1")

        assert result is None

    def test_custom_ttl(self):
        """Test setting custom TTL per entry."""
        cache = LLMResponseCache(default_ttl=3600)

        cache.set("key1", "response1", ttl=1)

        # Should still be valid
        assert cache.get("key1") == "response1"

        # Manually expire
        cache._cache["key1"]["expires_at"] = datetime.now() - timedelta(seconds=1)

        assert cache.get("key1") is None

    def test_max_entries_eviction(self):
        """Test that oldest entries are evicted when at capacity."""
        cache = LLMResponseCache(max_entries=2)

        cache.set("key1", "response1")
        cache.set("key2", "response2")
        cache.set("key3", "response3")  # Should evict key1

        assert cache.get("key1") is None
        assert cache.get("key2") == "response2"
        assert cache.get("key3") == "response3"

    def test_invalidate(self):
        """Test invalidating a cache entry."""
        cache = LLMResponseCache()

        cache.set("key1", "response1")
        cache.invalidate("key1")

        assert cache.get("key1") is None

    def test_clear(self):
        """Test clearing all cache entries."""
        cache = LLMResponseCache()

        cache.set("key1", "response1")
        cache.set("key2", "response2")
        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_stats(self):
        """Test cache statistics."""
        cache = LLMResponseCache(max_entries=100)

        cache.set("key1", "response1")
        cache.set("key2", "response2")

        stats = cache.stats()

        assert stats["total_entries"] == 2
        assert stats["valid_entries"] == 2
        assert stats["max_entries"] == 100


class TestMockGeminiClient:
    """Tests for mock Gemini client."""

    def test_generate_default_response(self):
        """Test generating default mock response."""
        client = MockGeminiClient()

        response = client.generate("Test prompt")

        assert response is not None
        assert len(response) > 0

    def test_generate_with_pattern_match(self):
        """Test generating response with pattern matching."""
        responses = {
            "summarize": '{"summary": "Test summary"}',
            "risk": '{"risk_level": "high"}',
        }
        client = MockGeminiClient(responses=responses)

        response = client.generate("Please summarize this ticket")

        assert "Test summary" in response

    def test_generate_json(self):
        """Test generating JSON response."""
        client = MockGeminiClient()

        result = client.generate_json("Test prompt")

        assert isinstance(result, dict)

    def test_count_tokens(self):
        """Test token counting."""
        client = MockGeminiClient()

        count = client.count_tokens("Hello world!")

        assert isinstance(count, int)
        assert count > 0

    def test_get_stats(self):
        """Test getting statistics."""
        client = MockGeminiClient()

        client.generate("Test 1")
        client.generate("Test 2")

        stats = client.get_stats()

        assert stats["model"] == "mock"
        assert stats["total_requests"] == 2


class TestCreateLLMClient:
    """Tests for LLM client factory."""

    def test_create_mock_client(self):
        """Test creating mock client."""
        client = create_llm_client(use_mock=True)

        assert isinstance(client, MockGeminiClient)

    def test_create_without_api_key_returns_mock(self):
        """Test that missing API key returns mock client."""
        # When genai is not available or no API key, should return mock
        client = create_llm_client(api_key=None, use_mock=True)

        assert isinstance(client, MockGeminiClient)
