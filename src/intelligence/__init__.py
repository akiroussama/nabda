"""
Intelligence module for LLM-powered Jira insights.

Provides intelligent summarization, risk explanation,
and priority suggestions using Gemini 2.0 Flash.
"""

from src.intelligence.llm_client import (
    GeminiClient,
    LLMClientError,
    LLMResponseCache,
    MockGeminiClient,
    RateLimitError,
    ContentBlockedError,
    create_llm_client,
)
from src.intelligence.orchestrator import (
    JiraIntelligence,
    TicketSummary,
    RiskExplanation,
    PrioritySuggestion,
    WorkloadAssessment,
    create_intelligence_from_settings,
)
from src.intelligence.prompts import (
    PromptTemplateManager,
    create_default_templates,
)

__all__ = [
    # LLM Client
    "GeminiClient",
    "MockGeminiClient",
    "LLMResponseCache",
    "LLMClientError",
    "RateLimitError",
    "ContentBlockedError",
    "create_llm_client",
    # Orchestrator
    "JiraIntelligence",
    "TicketSummary",
    "RiskExplanation",
    "PrioritySuggestion",
    "WorkloadAssessment",
    "create_intelligence_from_settings",
    # Prompts
    "PromptTemplateManager",
    "create_default_templates",
]
