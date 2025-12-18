"""
Prompt template management system.

Uses Jinja2 for flexible, maintainable prompt templates.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Template, select_autoescape
from loguru import logger


class PromptTemplateManager:
    """
    Manages Jinja2 prompt templates for LLM interactions.

    Provides template loading, rendering, and caching with
    support for context injection and output formatting.

    Example:
        >>> manager = PromptTemplateManager()
        >>> prompt = manager.render("ticket/summarize", ticket_data=data)
    """

    DEFAULT_TEMPLATES_DIR = Path("prompts")

    def __init__(
        self,
        templates_dir: str | Path | None = None,
        *,
        auto_reload: bool = True,
    ):
        """
        Initialize the prompt template manager.

        Args:
            templates_dir: Directory containing prompt templates
            auto_reload: Whether to auto-reload templates on change
        """
        self._templates_dir = Path(templates_dir) if templates_dir else self.DEFAULT_TEMPLATES_DIR
        self._ensure_templates_dir()

        self._env = Environment(
            loader=FileSystemLoader(str(self._templates_dir)),
            autoescape=select_autoescape(default=False),
            auto_reload=auto_reload,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Register custom filters
        self._register_filters()

        # Template cache for inline templates
        self._inline_cache: dict[str, Template] = {}

        logger.debug(f"PromptTemplateManager initialized with templates from {self._templates_dir}")

    def _ensure_templates_dir(self) -> None:
        """Ensure templates directory exists with base templates."""
        self._templates_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in ["system", "ticket", "sprint", "developer"]:
            (self._templates_dir / subdir).mkdir(exist_ok=True)

    def _register_filters(self) -> None:
        """Register custom Jinja2 filters."""
        self._env.filters["json_pretty"] = lambda x: json.dumps(x, indent=2, default=str)
        self._env.filters["truncate_smart"] = self._truncate_smart
        self._env.filters["format_duration"] = self._format_duration
        self._env.filters["risk_emoji"] = self._risk_emoji
        self._env.filters["priority_emoji"] = self._priority_emoji

    @staticmethod
    def _truncate_smart(text: str, max_length: int = 500) -> str:
        """Truncate text at sentence boundary if possible."""
        if not text or len(text) <= max_length:
            return text or ""

        # Try to truncate at sentence boundary
        truncated = text[:max_length]
        last_period = truncated.rfind(".")
        last_newline = truncated.rfind("\n")

        cut_point = max(last_period, last_newline)
        if cut_point > max_length * 0.5:
            return truncated[:cut_point + 1].strip()

        return truncated.strip() + "..."

    @staticmethod
    def _format_duration(hours: float) -> str:
        """Format hours as human-readable duration."""
        if hours < 1:
            return f"{int(hours * 60)} minutes"
        elif hours < 24:
            return f"{hours:.1f} hours"
        else:
            days = hours / 24
            return f"{days:.1f} days"

    @staticmethod
    def _risk_emoji(level: str) -> str:
        """Get emoji for risk level."""
        return {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(level.lower(), "⚪")

    @staticmethod
    def _priority_emoji(priority: str) -> str:
        """Get emoji for priority level."""
        mapping = {
            "highest": "🔴",
            "high": "🟠",
            "medium": "🟡",
            "low": "🟢",
            "lowest": "⚪",
        }
        return mapping.get(priority.lower(), "⚪")

    def render(
        self,
        template_name: str,
        **context: Any,
    ) -> str:
        """
        Render a template with context.

        Args:
            template_name: Template name (e.g., "ticket/summarize")
            **context: Context variables for the template

        Returns:
            Rendered prompt string
        """
        # Add .j2 extension if not present
        if not template_name.endswith(".j2"):
            template_name = f"{template_name}.j2"

        try:
            template = self._env.get_template(template_name)
            rendered = template.render(**context)
            logger.debug(f"Rendered template {template_name} ({len(rendered)} chars)")
            return rendered.strip()
        except Exception as e:
            logger.error(f"Failed to render template {template_name}: {e}")
            raise

    def render_inline(
        self,
        template_string: str,
        **context: Any,
    ) -> str:
        """
        Render an inline template string.

        Args:
            template_string: Jinja2 template string
            **context: Context variables

        Returns:
            Rendered prompt string
        """
        # Cache compiled templates by hash
        template_hash = hashlib.md5(template_string.encode()).hexdigest()

        if template_hash not in self._inline_cache:
            self._inline_cache[template_hash] = self._env.from_string(template_string)

        template = self._inline_cache[template_hash]
        return template.render(**context).strip()

    def get_system_prompt(self, variant: str = "base") -> str:
        """
        Get system prompt for LLM.

        Args:
            variant: System prompt variant (base, analyst, advisor)

        Returns:
            System prompt string
        """
        return self.render(f"system/{variant}")

    def list_templates(self, category: str | None = None) -> list[str]:
        """
        List available templates.

        Args:
            category: Filter by category (ticket, sprint, developer)

        Returns:
            List of template names
        """
        templates = []

        if category:
            search_dir = self._templates_dir / category
            if search_dir.exists():
                templates = [
                    f"{category}/{f.stem}"
                    for f in search_dir.glob("*.j2")
                ]
        else:
            for subdir in self._templates_dir.iterdir():
                if subdir.is_dir():
                    templates.extend([
                        f"{subdir.name}/{f.stem}"
                        for f in subdir.glob("*.j2")
                    ])

        return sorted(templates)

    def template_exists(self, template_name: str) -> bool:
        """Check if a template exists."""
        if not template_name.endswith(".j2"):
            template_name = f"{template_name}.j2"

        template_path = self._templates_dir / template_name
        return template_path.exists()

    def get_template_hash(self, template_name: str, **context: Any) -> str:
        """
        Get hash for template + context combination (for caching).

        Args:
            template_name: Template name
            **context: Context variables

        Returns:
            Hash string
        """
        rendered = self.render(template_name, **context)
        return hashlib.sha256(rendered.encode()).hexdigest()[:16]


# Built-in prompt templates as fallbacks
BUILTIN_TEMPLATES = {
    "system/base": """You are an expert Jira project analyst and agile coach assistant.

Your role is to:
- Analyze project data and provide actionable insights
- Explain risks and blockers in clear, non-technical language
- Suggest priorities based on data-driven analysis
- Help teams improve their sprint planning and execution

Guidelines:
- Be concise and actionable
- Use data to support your recommendations
- Prioritize clarity over comprehensiveness
- When uncertain, acknowledge limitations
- Format responses for easy scanning (bullets, headers)

Current context: Analyzing Jira data for project management insights.""",

    "system/analyst": """You are a senior data analyst specializing in software development metrics.

Your expertise includes:
- Cycle time and lead time analysis
- Velocity trends and forecasting
- Bottleneck identification
- Team capacity planning

Approach:
- Lead with key insights
- Support claims with specific data points
- Provide actionable recommendations
- Consider team context and constraints""",

    "ticket/summarize": """Summarize this Jira ticket for a quick status update.

Ticket: {{ ticket.key }}
Type: {{ ticket.issue_type }}
Status: {{ ticket.status }}
Priority: {{ ticket.priority }}
Assignee: {{ ticket.assignee | default('Unassigned') }}

Summary: {{ ticket.summary }}

Description:
{{ ticket.description | truncate_smart(800) }}

{% if ticket.comments %}
Recent Comments ({{ ticket.comments | length }}):
{% for comment in ticket.comments[:3] %}
- {{ comment.author }}: {{ comment.body | truncate_smart(200) }}
{% endfor %}
{% endif %}

{% if ticket.changelog %}
Recent Changes:
{% for change in ticket.changelog[:5] %}
- {{ change.field }}: {{ change.from_value }} → {{ change.to_value }}
{% endfor %}
{% endif %}

Provide a 2-3 sentence summary covering:
1. What this ticket is about
2. Current progress/blockers
3. What needs to happen next

Format as JSON:
{
  "summary": "...",
  "status_assessment": "on_track|at_risk|blocked",
  "next_action": "...",
  "key_blocker": "..." or null
}""",

    "sprint/explain_risk": """Analyze the risk factors for this sprint and explain them clearly.

Sprint: {{ sprint.name }}
Progress: {{ sprint.days_elapsed }}/{{ sprint.total_days }} days ({{ sprint.progress_percent | round(1) }}%)

Completion Metrics:
- Completed: {{ sprint.completed_points }}/{{ sprint.total_points }} points ({{ sprint.completion_rate | round(1) }}%)
- Expected by now: {{ sprint.expected_completion_rate | round(1) }}%
- Gap: {{ sprint.completion_gap | round(1) }}%

Risk Indicators:
- Velocity ratio: {{ sprint.velocity_ratio | round(2) }}x historical average
- Blocked tickets: {{ sprint.blocked_issues }} ({{ (sprint.blocked_ratio * 100) | round(1) }}%)
- Remaining: {{ sprint.remaining_points }} points in {{ sprint.days_remaining }} days

Risk Score: {{ risk_score.score }}/100 ({{ risk_score.level | upper }})

Top Risk Factors:
{% for factor_name, factor_data in risk_score.factors.items() %}
{% if factor_data.contribution > 0.05 %}
- {{ factor_name }}: {{ (factor_data.contribution * 100) | round(1) }}% contribution
{% endif %}
{% endfor %}

Explain this risk assessment in plain language for a project manager:
1. What are the main concerns?
2. Why is the risk score at this level?
3. What specific actions could reduce risk?

Format as JSON:
{
  "risk_summary": "1-2 sentence overview",
  "main_concerns": ["concern1", "concern2", ...],
  "root_causes": ["cause1", "cause2", ...],
  "recommended_actions": [
    {"action": "...", "priority": "high|medium|low", "impact": "..."}
  ],
  "outlook": "improving|stable|declining"
}""",

    "sprint/suggest_priorities": """Based on the current sprint state, suggest priority adjustments.

Sprint: {{ sprint.name }}
Days Remaining: {{ sprint.days_remaining }}

Current Tickets:
{% for ticket in tickets %}
- {{ ticket.key }}: {{ ticket.summary | truncate(50) }}
  Type: {{ ticket.issue_type }} | Status: {{ ticket.status }} | Points: {{ ticket.story_points | default('?') }}
  {% if ticket.is_blocked %}⚠️ BLOCKED{% endif %}
{% endfor %}

Sprint Goals: {{ sprint.goal | default('Not specified') }}

Constraints:
- Available capacity: ~{{ sprint.days_remaining * 6 }} hours
- Team velocity: {{ sprint.avg_historical_velocity | round(1) }} points/sprint

Analyze the tickets and suggest:
1. Which tickets MUST be completed this sprint?
2. Which could be moved to next sprint?
3. Any re-ordering for optimal flow?

Format as JSON:
{
  "must_complete": ["KEY-1", "KEY-2"],
  "consider_deferring": ["KEY-3"],
  "reorder_suggestion": [
    {"ticket": "KEY-1", "reason": "..."}
  ],
  "focus_recommendation": "...",
  "risk_if_unchanged": "..."
}""",

    "developer/workload_summary": """Analyze this developer's workload and provide recommendations.

Developer: {{ developer.pseudonym }}
Current Status: {{ developer.status | upper }}
Workload Score: {{ developer.score }}/200 ({{ developer.relative_to_team | round(0) }}% of team average)

Current Work:
- WIP tickets: {{ developer.wip_count }} ({{ developer.wip_points }} points)
- Blocked: {{ developer.blocked_count }}
- Overdue: {{ developer.overdue_count }}

Recent Performance (30 days):
- Completed: {{ developer.completed_count_30d }} tickets ({{ developer.completed_points_30d }} points)
- Hours logged: {{ developer.hours_logged_30d | round(1) }}

Team Context:
- Team average WIP: {{ developer.team_avg_wip_points | round(1) }} points
- Team average completed: {{ developer.team_avg_completed_points | round(1) }} points

Provide a workload assessment:
1. Is this sustainable?
2. Any immediate concerns?
3. Recommendations?

Format as JSON:
{
  "assessment": "sustainable|concerning|critical",
  "summary": "...",
  "immediate_concerns": ["..."],
  "recommendations": ["..."],
  "suggested_actions": [
    {"action": "...", "priority": "high|medium|low"}
  ]
}"""
}


def create_default_templates(templates_dir: Path) -> None:
    """Create default template files if they don't exist."""
    templates_dir.mkdir(parents=True, exist_ok=True)

    for template_path, content in BUILTIN_TEMPLATES.items():
        file_path = templates_dir / f"{template_path}.j2"
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not file_path.exists():
            file_path.write_text(content.strip())
            logger.info(f"Created default template: {file_path}")
