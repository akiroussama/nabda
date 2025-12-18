"""
Intelligence orchestration module.

Coordinates LLM calls with data sources to provide intelligent Jira insights.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from src.intelligence.llm_client import GeminiClient, MockGeminiClient, create_llm_client
from src.intelligence.prompts import PromptTemplateManager, create_default_templates


@dataclass
class TicketSummary:
    """Structured ticket summary."""
    issue_key: str
    summary: str
    status_assessment: str  # on_track, at_risk, blocked
    next_action: str
    key_blocker: str | None
    generated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue_key": self.issue_key,
            "summary": self.summary,
            "status_assessment": self.status_assessment,
            "next_action": self.next_action,
            "key_blocker": self.key_blocker,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class RiskExplanation:
    """Structured sprint risk explanation."""
    sprint_id: int
    sprint_name: str
    risk_score: float
    risk_level: str
    risk_summary: str
    main_concerns: list[str]
    root_causes: list[str]
    recommended_actions: list[dict[str, str]]
    outlook: str  # improving, stable, declining
    generated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "sprint_id": self.sprint_id,
            "sprint_name": self.sprint_name,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "risk_summary": self.risk_summary,
            "main_concerns": self.main_concerns,
            "root_causes": self.root_causes,
            "recommended_actions": self.recommended_actions,
            "outlook": self.outlook,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class PrioritySuggestion:
    """Structured priority suggestions."""
    sprint_id: int
    sprint_name: str
    must_complete: list[str]
    consider_deferring: list[str]
    reorder_suggestions: list[dict[str, str]]
    focus_recommendation: str
    risk_if_unchanged: str
    generated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "sprint_id": self.sprint_id,
            "sprint_name": self.sprint_name,
            "must_complete": self.must_complete,
            "consider_deferring": self.consider_deferring,
            "reorder_suggestions": self.reorder_suggestions,
            "focus_recommendation": self.focus_recommendation,
            "risk_if_unchanged": self.risk_if_unchanged,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class WorkloadAssessment:
    """Structured developer workload assessment."""
    developer_id: str
    developer_name: str
    assessment: str  # sustainable, concerning, critical
    summary: str
    immediate_concerns: list[str]
    recommendations: list[str]
    suggested_actions: list[dict[str, str]]
    generated_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "developer_id": self.developer_id,
            "developer_name": self.developer_name,
            "assessment": self.assessment,
            "summary": self.summary,
            "immediate_concerns": self.immediate_concerns,
            "recommendations": self.recommendations,
            "suggested_actions": self.suggested_actions,
            "generated_at": self.generated_at.isoformat(),
        }


class JiraIntelligence:
    """
    Orchestrates LLM-powered insights for Jira data.

    Provides intelligent summarization, risk explanation,
    and priority suggestions using Gemini 2.0 Flash.

    Example:
        >>> intel = JiraIntelligence(api_key="...")
        >>> summary = intel.summarize_ticket(ticket_data)
        >>> risk = intel.explain_sprint_risk(sprint_data, risk_score)
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        templates_dir: str | None = None,
        use_mock: bool = False,
        enable_cache: bool = True,
        cache_ttl: int = 1800,  # 30 minutes default
    ):
        """
        Initialize the intelligence orchestrator.

        Args:
            api_key: Google API key for Gemini
            templates_dir: Custom templates directory
            use_mock: Use mock LLM for testing
            enable_cache: Enable response caching
            cache_ttl: Cache TTL in seconds
        """
        # Initialize LLM client
        self._llm = create_llm_client(
            api_key=api_key,
            use_mock=use_mock,
            enable_cache=enable_cache,
            cache_ttl=cache_ttl,
        )

        # Initialize prompt manager
        self._prompts = PromptTemplateManager(templates_dir=templates_dir)

        # Ensure default templates exist
        if templates_dir:
            from pathlib import Path
            create_default_templates(Path(templates_dir))

        logger.info("JiraIntelligence initialized")

    def summarize_ticket(
        self,
        ticket: dict[str, Any],
        *,
        include_comments: bool = True,
        include_changelog: bool = True,
    ) -> TicketSummary:
        """
        Generate an intelligent summary of a ticket.

        Args:
            ticket: Ticket data dictionary
            include_comments: Include recent comments in context
            include_changelog: Include changelog in context

        Returns:
            TicketSummary object
        """
        logger.info(f"Summarizing ticket {ticket.get('key', 'unknown')}")

        # Prepare context
        context = {
            "ticket": {
                **ticket,
                "comments": ticket.get("comments", []) if include_comments else [],
                "changelog": ticket.get("changelog", []) if include_changelog else [],
            }
        }

        # Get system prompt
        system_prompt = self._get_system_prompt("base")

        # Render prompt
        if self._prompts.template_exists("ticket/summarize"):
            prompt = self._prompts.render("ticket/summarize", **context)
        else:
            # Use builtin template
            from src.intelligence.prompts import BUILTIN_TEMPLATES
            prompt = self._prompts.render_inline(
                BUILTIN_TEMPLATES["ticket/summarize"],
                **context
            )

        # Generate response
        response = self._llm.generate_json(
            prompt,
            system_prompt=system_prompt,
            default={
                "summary": "Unable to generate summary",
                "status_assessment": "unknown",
                "next_action": "Review ticket manually",
                "key_blocker": None,
            }
        )

        return TicketSummary(
            issue_key=ticket.get("key", "unknown"),
            summary=response.get("summary", ""),
            status_assessment=response.get("status_assessment", "unknown"),
            next_action=response.get("next_action", ""),
            key_blocker=response.get("key_blocker"),
            generated_at=datetime.now(),
        )

    def explain_sprint_risk(
        self,
        sprint: dict[str, Any],
        risk_score: dict[str, Any],
    ) -> RiskExplanation:
        """
        Generate a human-readable explanation of sprint risk.

        Args:
            sprint: Sprint feature data
            risk_score: Risk score from SprintRiskScorer

        Returns:
            RiskExplanation object
        """
        logger.info(f"Explaining risk for sprint {sprint.get('sprint_name', 'unknown')}")

        # Prepare context
        context = {
            "sprint": sprint,
            "risk_score": risk_score,
        }

        # Get system prompt
        system_prompt = self._get_system_prompt("analyst")

        # Render prompt
        if self._prompts.template_exists("sprint/explain_risk"):
            prompt = self._prompts.render("sprint/explain_risk", **context)
        else:
            from src.intelligence.prompts import BUILTIN_TEMPLATES
            prompt = self._prompts.render_inline(
                BUILTIN_TEMPLATES["sprint/explain_risk"],
                **context
            )

        # Generate response
        response = self._llm.generate_json(
            prompt,
            system_prompt=system_prompt,
            default={
                "risk_summary": "Unable to analyze risk",
                "main_concerns": [],
                "root_causes": [],
                "recommended_actions": [],
                "outlook": "unknown",
            }
        )

        return RiskExplanation(
            sprint_id=sprint.get("sprint_id", 0),
            sprint_name=sprint.get("sprint_name", "unknown"),
            risk_score=risk_score.get("score", 0),
            risk_level=risk_score.get("level", "unknown"),
            risk_summary=response.get("risk_summary", ""),
            main_concerns=response.get("main_concerns", []),
            root_causes=response.get("root_causes", []),
            recommended_actions=response.get("recommended_actions", []),
            outlook=response.get("outlook", "unknown"),
            generated_at=datetime.now(),
        )

    def suggest_priorities(
        self,
        sprint: dict[str, Any],
        tickets: list[dict[str, Any]],
    ) -> PrioritySuggestion:
        """
        Generate priority suggestions for a sprint.

        Args:
            sprint: Sprint data
            tickets: List of tickets in the sprint

        Returns:
            PrioritySuggestion object
        """
        logger.info(f"Suggesting priorities for sprint {sprint.get('sprint_name', 'unknown')}")

        # Prepare context
        context = {
            "sprint": sprint,
            "tickets": tickets,
        }

        # Get system prompt
        system_prompt = self._get_system_prompt("analyst")

        # Render prompt
        if self._prompts.template_exists("sprint/suggest_priorities"):
            prompt = self._prompts.render("sprint/suggest_priorities", **context)
        else:
            from src.intelligence.prompts import BUILTIN_TEMPLATES
            prompt = self._prompts.render_inline(
                BUILTIN_TEMPLATES["sprint/suggest_priorities"],
                **context
            )

        # Generate response
        response = self._llm.generate_json(
            prompt,
            system_prompt=system_prompt,
            default={
                "must_complete": [],
                "consider_deferring": [],
                "reorder_suggestion": [],
                "focus_recommendation": "Review priorities manually",
                "risk_if_unchanged": "Unknown",
            }
        )

        return PrioritySuggestion(
            sprint_id=sprint.get("sprint_id", 0),
            sprint_name=sprint.get("sprint_name", "unknown"),
            must_complete=response.get("must_complete", []),
            consider_deferring=response.get("consider_deferring", []),
            reorder_suggestions=response.get("reorder_suggestion", []),
            focus_recommendation=response.get("focus_recommendation", ""),
            risk_if_unchanged=response.get("risk_if_unchanged", ""),
            generated_at=datetime.now(),
        )

    def assess_developer_workload(
        self,
        developer: dict[str, Any],
    ) -> WorkloadAssessment:
        """
        Generate workload assessment for a developer.

        Args:
            developer: Developer metrics data

        Returns:
            WorkloadAssessment object
        """
        logger.info(f"Assessing workload for {developer.get('pseudonym', 'unknown')}")

        # Prepare context
        context = {
            "developer": developer,
        }

        # Get system prompt
        system_prompt = self._get_system_prompt("base")

        # Render prompt
        if self._prompts.template_exists("developer/workload_summary"):
            prompt = self._prompts.render("developer/workload_summary", **context)
        else:
            from src.intelligence.prompts import BUILTIN_TEMPLATES
            prompt = self._prompts.render_inline(
                BUILTIN_TEMPLATES["developer/workload_summary"],
                **context
            )

        # Generate response
        response = self._llm.generate_json(
            prompt,
            system_prompt=system_prompt,
            default={
                "assessment": "unknown",
                "summary": "Unable to assess workload",
                "immediate_concerns": [],
                "recommendations": [],
                "suggested_actions": [],
            }
        )

        return WorkloadAssessment(
            developer_id=developer.get("assignee_id", "unknown"),
            developer_name=developer.get("pseudonym", "unknown"),
            assessment=response.get("assessment", "unknown"),
            summary=response.get("summary", ""),
            immediate_concerns=response.get("immediate_concerns", []),
            recommendations=response.get("recommendations", []),
            suggested_actions=response.get("suggested_actions", []),
            generated_at=datetime.now(),
        )

    def generate_standup_summary(
        self,
        sprint: dict[str, Any],
        tickets: list[dict[str, Any]],
        team_updates: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Generate a daily standup summary.

        Args:
            sprint: Current sprint data
            tickets: Sprint tickets
            team_updates: Optional team member updates

        Returns:
            Formatted standup summary
        """
        logger.info(f"Generating standup summary for {sprint.get('sprint_name', 'unknown')}")

        prompt = f"""Generate a concise daily standup summary for the team.

Sprint: {sprint.get('sprint_name', 'N/A')}
Day: {sprint.get('days_elapsed', 0)} of {sprint.get('total_days', 14)}
Progress: {sprint.get('completion_rate', 0):.1f}% complete

Ticket Status:
- Done: {sprint.get('completed_issues', 0)}
- In Progress: {sprint.get('in_progress_issues', 0)}
- Blocked: {sprint.get('blocked_issues', 0)}
- To Do: {sprint.get('todo_issues', 0)}

Key tickets in progress:
"""
        in_progress = [t for t in tickets if t.get('status') == 'In Progress'][:5]
        for t in in_progress:
            prompt += f"- {t.get('key')}: {t.get('summary', '')[:50]}\n"

        blocked = [t for t in tickets if t.get('is_blocked', False)]
        if blocked:
            prompt += "\nBlocked items:\n"
            for t in blocked[:3]:
                prompt += f"- {t.get('key')}: {t.get('summary', '')[:50]}\n"

        prompt += """
Create a brief standup summary with:
1. Sprint health status (1 sentence)
2. Key accomplishments since yesterday
3. Today's focus areas
4. Blockers needing attention

Keep it under 200 words and suitable for reading aloud."""

        system_prompt = self._get_system_prompt("base")

        return self._llm.generate(
            prompt,
            system_prompt=system_prompt,
        )

    def _get_system_prompt(self, variant: str = "base") -> str:
        """Get system prompt, with fallback to builtin."""
        if self._prompts.template_exists(f"system/{variant}"):
            return self._prompts.render(f"system/{variant}")
        else:
            from src.intelligence.prompts import BUILTIN_TEMPLATES
            return BUILTIN_TEMPLATES.get(f"system/{variant}", BUILTIN_TEMPLATES["system/base"])

    def get_stats(self) -> dict[str, Any]:
        """Get intelligence module statistics."""
        return {
            "llm": self._llm.get_stats(),
            "templates": {
                "available": self._prompts.list_templates(),
            }
        }

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._llm.clear_cache()


def create_intelligence_from_settings() -> JiraIntelligence:
    """
    Create JiraIntelligence from application settings.

    Returns:
        Configured JiraIntelligence instance
    """
    from config.settings import get_settings

    settings = get_settings()

    return JiraIntelligence(
        api_key=settings.llm.api_key,
        enable_cache=True,
        cache_ttl=settings.llm.cache_ttl,
    )
