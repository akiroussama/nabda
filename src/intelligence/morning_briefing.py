"""
Morning Briefing Intelligence Module.

Generates personalized morning briefings using LLM with evidence-based content.
"""

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any

from loguru import logger

from src.features.delta_engine import (
    DeltaEngine,
    DailyDelta,
    AttentionItem,
    ComparisonMetrics,
    TimeframeContext,
)
from src.intelligence.llm_client import GeminiClient, create_llm_client


@dataclass
class BriefingContext:
    """All context needed to generate a briefing."""
    timeframe: str
    project_key: str
    project_name: str
    current_time: datetime
    day_of_week: str

    # Sprint context
    sprint_info: dict[str, Any] | None = None
    is_sprint_start: bool = False
    is_sprint_end: bool = False
    sprint_day: int | None = None
    sprint_total_days: int | None = None

    # Yesterday's context (for continuity)
    yesterday_briefing: dict[str, Any] | None = None
    pending_recommendations: list[dict[str, Any]] = field(default_factory=list)

    # Current data
    delta: DailyDelta | None = None
    attention_items: list[AttentionItem] = field(default_factory=list)
    comparison_metrics: ComparisonMetrics | None = None

    # Team context
    team_members: list[dict[str, Any]] = field(default_factory=list)


class MorningBriefingGenerator:
    """
    Generates personalized morning briefings using LLM.

    Uses the Delta Engine for evidence and the LLM for natural language synthesis.
    """

    SYSTEM_PROMPT = """You are the Jira Copilot "Chief of Staff" - an AI assistant for Project Managers.

CORE RULES (NEVER VIOLATE):
1. NEVER hallucinate or invent information
2. EVERY claim must cite evidence: [EVIDENCE: PROJ-123] or [DATA: metric_name]
3. Be supportive but direct - PMs are busy
4. Celebrate wins before discussing problems
5. Always provide actionable recommendations
6. Use confident language when data supports it, hedge when uncertain
7. Respect the PM's time - be concise

PERSONALITY:
- Professional but warm
- Data-driven but human
- Proactive but not alarmist
- Supportive but honest

FORMAT RULES:
- Use markdown formatting
- Keep sentences short and scannable
- Use bullet points for lists
- Always cite evidence with [EVIDENCE: KEY-123] format"""

    def __init__(
        self,
        llm_client: GeminiClient | None = None,
        delta_engine: DeltaEngine | None = None,
    ):
        """
        Initialize the briefing generator.

        Args:
            llm_client: LLM client for generation
            delta_engine: Delta engine for computing changes
        """
        self._llm = llm_client
        self._delta_engine = delta_engine

    def generate_morning_brief(
        self,
        ctx: BriefingContext,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        """
        Generate a complete morning briefing.

        Args:
            ctx: Briefing context with all data
            use_cache: Whether to use cached responses

        Returns:
            Dictionary with briefing content and metadata
        """
        start_time = time.time()

        prompt = self._build_morning_brief_prompt(ctx)

        # Generate with LLM
        if self._llm:
            try:
                response = self._llm.generate(
                    prompt,
                    system_prompt=self.SYSTEM_PROMPT,
                    use_cache=use_cache,
                )
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                response = self._generate_fallback_brief(ctx)
        else:
            # No LLM available, use fallback
            response = self._generate_fallback_brief(ctx)

        generation_time = int((time.time() - start_time) * 1000)

        # Parse the response
        parsed = self._parse_briefing_response(response)

        return {
            "narrative": response,
            "parsed": parsed,
            "evidence_citations": self._extract_evidence_citations(response),
            "generation_time_ms": generation_time,
            "context": {
                "timeframe": ctx.timeframe,
                "project_key": ctx.project_key,
                "generated_at": datetime.now().isoformat(),
            },
        }

    def generate_decision_queue(
        self,
        ctx: BriefingContext,
        max_items: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Generate a prioritized decision queue.

        Args:
            ctx: Briefing context
            max_items: Maximum items to return

        Returns:
            List of prioritized decision items
        """
        if not ctx.attention_items:
            return []

        # Take top N items by attention score
        top_items = ctx.attention_items[:max_items]

        # Enrich with LLM-generated action drafts if available
        enriched_items = []
        for item in top_items:
            enriched = item.to_dict()

            # Add priority number
            enriched["priority"] = len(enriched_items) + 1

            # Add severity emoji
            severity_emoji = {
                "critical": "critical",
                "high": "high",
                "medium": "medium",
                "low": "low",
            }.get(item.severity.value, "")
            enriched["severity_display"] = severity_emoji

            enriched_items.append(enriched)

        return enriched_items

    def _build_morning_brief_prompt(self, ctx: BriefingContext) -> str:
        """Build the prompt for morning briefing generation."""

        # Build time context
        time_context = f"""
## TIME CONTEXT
- Today is {ctx.day_of_week}, {ctx.current_time.strftime('%B %d, %Y')}
- Time: {ctx.current_time.strftime('%I:%M %p')}"""

        if ctx.sprint_info:
            time_context += f"""
- Sprint: {ctx.sprint_info.get('sprint_name', 'Unknown')}
- Sprint day: {ctx.sprint_day} of {ctx.sprint_total_days}"""

            if ctx.is_sprint_end:
                time_context += "\n- Sprint ends in 2 days or less - focus on completion"
            elif ctx.is_sprint_start:
                time_context += "\n- Sprint just started - focus on planning and clarity"

        # Build continuity section
        continuity_section = ""
        if ctx.yesterday_briefing:
            continuity_section = f"""
## YESTERDAY'S CONTEXT (for narrative continuity)
Yesterday's summary: {ctx.yesterday_briefing.get('summary', 'N/A')}

IMPORTANT: Reference yesterday's items if relevant. For example:
- "Yesterday I flagged [ticket] as at-risk. Good news: it's now resolved."
- "The blocker I mentioned yesterday is still unresolved. This is now Day X."
"""

        if ctx.pending_recommendations:
            continuity_section += f"""
## PENDING RECOMMENDATIONS FROM PREVIOUS DAYS
{json.dumps(ctx.pending_recommendations[:5], indent=2)}
"""

        # Build delta section
        delta_section = ""
        if ctx.delta:
            delta_section = f"""
## WHAT CHANGED ({ctx.timeframe.upper()})
- Tickets completed: {ctx.delta.tickets_completed} ({ctx.delta.points_completed} points)
- Tickets created: {ctx.delta.tickets_created} ({ctx.delta.points_added} points)
- Active blockers: {ctx.delta.active_blockers}
- Status transitions: {ctx.delta.status_transitions}
- Regressions (Done -> reopened): {ctx.delta.regressions}
- After-hours activity: {ctx.delta.after_hours_events} events
- Weekend activity: {ctx.delta.weekend_events} events

### Completed Tickets
{json.dumps(ctx.delta.completed_tickets[:10], indent=2) if ctx.delta.completed_tickets else "None"}

### Blockers
{json.dumps(ctx.delta.blocker_tickets[:10], indent=2) if ctx.delta.blocker_tickets else "None"}
"""

        # Build comparison section
        comparison_section = ""
        if ctx.comparison_metrics:
            trend_emoji = {"up": "trending up", "down": "trending down", "stable": "stable"}[ctx.comparison_metrics.trend]
            comparison_section = f"""
## COMPARISON TO PREVIOUS PERIOD
- Current period: {ctx.comparison_metrics.current_tickets_completed} tickets, {ctx.comparison_metrics.current_points_completed} points
- Previous period: {ctx.comparison_metrics.previous_tickets_completed} tickets, {ctx.comparison_metrics.previous_points_completed} points
- Velocity change: {ctx.comparison_metrics.velocity_change_percent}% ({trend_emoji})
"""

        # Build attention items section
        attention_section = ""
        if ctx.attention_items:
            attention_section = f"""
## ITEMS NEEDING ATTENTION (ranked by severity)
{json.dumps([item.to_dict() for item in ctx.attention_items[:10]], indent=2)}
"""

        # The main prompt
        return f"""# Generate Morning Briefing for {ctx.project_name}

{time_context}

{continuity_section}

{delta_section}

{comparison_section}

{attention_section}

---

# OUTPUT FORMAT

Generate a morning briefing with this EXACT structure:

## Good Morning!

[One warm, personalized opening sentence that acknowledges the day/time context. If Monday, mention the week ahead. If Friday, acknowledge weekend prep.]

---

### The Good News
[2-3 bullet points of wins/progress. MUST cite evidence with [EVIDENCE: KEY-123].]
[If yesterday you flagged something that's now resolved, celebrate it here.]

### The Concern
[1-2 items that need attention but aren't critical. Cite evidence.]

### The Risk
[0-2 items that are critical/blocking. Cite evidence. If none, say "No critical risks today"]

### My Top Recommendation
[THE single most important action for this PM today. Be specific and actionable.]
[Include: What to do, Why it matters, Suggested approach]

---

### Quick Stats ({ctx.timeframe})
| Metric | Value | vs. Previous |
|--------|-------|--------------|
[Fill in key metrics with comparison]

---

### Heads Up
[1-2 things to watch for in the next period. Be predictive but evidence-based.]

REMEMBER:
- Every claim needs [EVIDENCE: TICKET-KEY] or [DATA: metric]
- Be concise - this should take 2 minutes to read
- Be warm but professional
- Prioritize ruthlessly - don't overwhelm"""

    def _generate_fallback_brief(self, ctx: BriefingContext) -> str:
        """Generate a basic briefing without LLM."""
        greeting = self._get_time_greeting()

        # Build basic stats
        completed = ctx.delta.tickets_completed if ctx.delta else 0
        points = ctx.delta.points_completed if ctx.delta else 0
        blockers = ctx.delta.active_blockers if ctx.delta else 0

        # Get top attention items
        critical_items = [
            item for item in ctx.attention_items
            if item.severity.value in ("critical", "high")
        ][:3]

        attention_text = ""
        if critical_items:
            attention_text = "\n".join([
                f"- {item.ticket_key}: {item.evidence.description} [EVIDENCE: {item.ticket_key}]"
                for item in critical_items
            ])
        else:
            attention_text = "No critical items today"

        return f"""## {greeting}!

Here's your {ctx.timeframe} update for {ctx.project_name}.

---

### The Good News
- **{completed} tickets completed** this period ({points} points) [DATA: tickets_completed]
- Team is actively working through the backlog

### The Concern
- {blockers} active blocker(s) in the project

### The Risk
{attention_text}

### My Top Recommendation
Review the items above and prioritize unblocking any critical issues.

---

### Quick Stats ({ctx.timeframe})
| Metric | Value |
|--------|-------|
| Completed | {completed} |
| Points | {points} |
| Blockers | {blockers} |

---

### Heads Up
- Monitor blocked items for resolution
- Check in with team members on stalled work
"""

    def _get_time_greeting(self) -> str:
        """Get time-appropriate greeting."""
        hour = datetime.now().hour
        if hour < 12:
            return "Good Morning"
        elif hour < 17:
            return "Good Afternoon"
        else:
            return "Good Evening"

    def _parse_briefing_response(self, response: str) -> dict[str, Any]:
        """Parse the briefing into structured sections."""
        sections = {
            "greeting": "",
            "good_news": [],
            "concerns": [],
            "risks": [],
            "top_recommendation": "",
            "quick_stats": [],
            "heads_up": [],
        }

        current_section = None
        current_content = []

        for line in response.split("\n"):
            line_lower = line.lower()

            if "good morning" in line_lower or "good afternoon" in line_lower or "good evening" in line_lower:
                sections["greeting"] = line.strip("#").strip()
                current_section = "greeting"
            elif "good news" in line_lower:
                current_section = "good_news"
            elif "concern" in line_lower:
                current_section = "concerns"
            elif "risk" in line_lower:
                current_section = "risks"
            elif "recommendation" in line_lower:
                current_section = "top_recommendation"
            elif "quick stats" in line_lower:
                current_section = "quick_stats"
            elif "heads up" in line_lower:
                current_section = "heads_up"
            elif current_section and line.strip():
                if current_section in ("good_news", "concerns", "risks", "heads_up"):
                    if line.strip().startswith(("-", "*", "+")):
                        sections[current_section].append(line.strip()[1:].strip())
                elif current_section == "top_recommendation":
                    sections[current_section] += line + "\n"
                elif current_section == "quick_stats":
                    if "|" in line and "---" not in line:
                        sections[current_section].append(line.strip())

        return sections

    def _extract_evidence_citations(self, text: str) -> list[str]:
        """Extract all [EVIDENCE: XXX-123] citations from text."""
        pattern = r"\[EVIDENCE:\s*([A-Z]+-\d+(?:,\s*[A-Z]+-\d+)*)\]"
        matches = re.findall(pattern, text)

        citations = []
        for match in matches:
            citations.extend([t.strip() for t in match.split(",")])

        return list(set(citations))


class MorningBriefingOrchestrator:
    """
    Orchestrates the complete morning briefing workflow.

    Combines Delta Engine, LLM, and database operations.
    """

    def __init__(
        self,
        db_path: str,
        llm_api_key: str | None = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            db_path: Path to DuckDB database
            llm_api_key: API key for LLM (optional)
        """
        import duckdb
        self._conn = duckdb.connect(db_path)
        self._delta_engine = DeltaEngine(self._conn)

        # Initialize LLM client
        if llm_api_key:
            self._llm = create_llm_client(api_key=llm_api_key)
        else:
            self._llm = create_llm_client(use_mock=True)

        self._generator = MorningBriefingGenerator(
            llm_client=self._llm,
            delta_engine=self._delta_engine,
        )

    def generate_briefing(
        self,
        project_key: str,
        timeframe: str = "daily",
        user_id: str = "default",
    ) -> dict[str, Any]:
        """
        Generate a complete morning briefing.

        Args:
            project_key: Jira project key
            timeframe: 'daily', 'weekly', or 'monthly'
            user_id: User identifier for personalization

        Returns:
            Complete briefing data
        """
        logger.info(f"Generating {timeframe} briefing for {project_key}")

        # Get timeframe context
        context = self._delta_engine.get_timeframe_context(timeframe)

        # Compute delta
        delta = self._delta_engine.compute_delta(project_key, context)

        # Get attention items
        attention_items = self._delta_engine.detect_attention_items(project_key, context)

        # Get comparison metrics
        comparison = self._delta_engine.get_comparison_metrics(project_key, context)

        # Get yesterday's briefing for continuity
        yesterday = self._delta_engine.get_yesterday_briefing(user_id, project_key)

        # Get pending recommendations
        pending = self._delta_engine.get_pending_recommendations(user_id, project_key)

        # Get sprint context
        sprint_info = self._delta_engine.get_sprint_context(project_key)

        # Build briefing context
        briefing_ctx = BriefingContext(
            timeframe=timeframe,
            project_key=project_key,
            project_name=project_key,  # Could lookup actual name
            current_time=datetime.now(),
            day_of_week=datetime.now().strftime("%A"),
            sprint_info=sprint_info,
            is_sprint_start=sprint_info.get("is_sprint_start", False) if sprint_info else False,
            is_sprint_end=sprint_info.get("is_sprint_end", False) if sprint_info else False,
            sprint_day=sprint_info.get("days_elapsed") if sprint_info else None,
            sprint_total_days=sprint_info.get("total_days") if sprint_info else None,
            yesterday_briefing=yesterday,
            pending_recommendations=pending,
            delta=delta,
            attention_items=attention_items,
            comparison_metrics=comparison,
        )

        # Generate the briefing
        briefing = self._generator.generate_morning_brief(briefing_ctx)

        # Generate decision queue
        decision_queue = self._generator.generate_decision_queue(briefing_ctx)

        # Save the briefing for future reference
        self._save_briefing(
            user_id=user_id,
            project_key=project_key,
            timeframe=timeframe,
            briefing=briefing,
        )

        # Save delta for analytics
        self._delta_engine.save_daily_delta(project_key, delta)

        return {
            "briefing": briefing,
            "decision_queue": decision_queue,
            "delta": delta.to_dict(),
            "attention_items": [item.to_dict() for item in attention_items],
            "comparison": comparison.to_dict(),
            "sprint_info": sprint_info,
            "context": {
                "timeframe": timeframe,
                "project_key": project_key,
                "generated_at": datetime.now().isoformat(),
                "user_id": user_id,
            },
        }

    def _save_briefing(
        self,
        user_id: str,
        project_key: str,
        timeframe: str,
        briefing: dict[str, Any],
    ) -> None:
        """Save briefing to database for memory/continuity."""
        try:
            self._conn.execute("""
                INSERT OR REPLACE INTO daily_briefings (
                    user_id, project_key, briefing_date, timeframe,
                    narrative_summary, key_highlights, recommendations,
                    metrics_snapshot, model_used, generation_time_ms
                ) VALUES (?, ?, CURRENT_DATE, ?, ?, ?, ?, ?, ?, ?)
            """, [
                user_id,
                project_key,
                timeframe,
                briefing.get("narrative", "")[:500],  # Summary
                json.dumps(briefing.get("parsed", {}).get("good_news", [])),
                json.dumps(briefing.get("parsed", {}).get("top_recommendation", "")),
                json.dumps(briefing.get("context", {})),
                "gemini",
                briefing.get("generation_time_ms", 0),
            ])
            logger.debug(f"Saved briefing for {user_id}/{project_key}")
        except Exception as e:
            logger.warning(f"Failed to save briefing: {e}")
