"""
Delta Engine - The Deterministic Brain for the Good Morning Dashboard.

Computes exactly what changed and identifies items that need PM attention.
All AI claims must trace back to evidence computed here.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Any

import duckdb
from loguru import logger


class AttentionSeverity(Enum):
    """Severity levels for attention items."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AttentionReason(Enum):
    """Reasons an item needs attention."""
    SILENT_BLOCKER = "silent_blocker"
    STATUS_CHURN = "status_churn"
    BLOCKED_TOO_LONG = "blocked_too_long"
    APPROACHING_DEADLINE = "approaching_deadline"
    SCOPE_CREEP = "scope_creep"
    NO_ASSIGNEE = "no_assignee"
    REGRESSION = "regression"
    OVERDUE = "overdue"
    AFTER_HOURS_SPIKE = "after_hours_spike"
    HIGH_WIP = "high_wip"


@dataclass
class Evidence:
    """Evidence supporting an alert - every claim must cite specific evidence."""
    ticket_keys: list[str]
    description: str
    metric_value: float | None = None
    comparison_value: float | None = None
    raw_data: dict[str, Any] | None = None


@dataclass
class AttentionItem:
    """An item requiring PM attention."""
    ticket_key: str
    ticket_summary: str
    reason: AttentionReason
    severity: AttentionSeverity
    evidence: Evidence
    suggested_action: str
    draft_message: str | None = None
    attention_score: float = 0.0
    assignee_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "ticket_key": self.ticket_key,
            "ticket_summary": self.ticket_summary,
            "reason": self.reason.value,
            "severity": self.severity.value,
            "evidence": {
                "ticket_keys": self.evidence.ticket_keys,
                "description": self.evidence.description,
                "metric_value": self.evidence.metric_value,
                "comparison_value": self.evidence.comparison_value,
            },
            "suggested_action": self.suggested_action,
            "draft_message": self.draft_message,
            "attention_score": self.attention_score,
            "assignee_name": self.assignee_name,
        }


@dataclass
class DailyDelta:
    """What changed in a time period."""
    period_start: date
    period_end: date

    # Counts
    tickets_created: int = 0
    tickets_completed: int = 0
    tickets_reopened: int = 0
    points_completed: float = 0.0
    points_added: float = 0.0
    points_removed: float = 0.0

    # Blockers
    new_blockers: int = 0
    resolved_blockers: int = 0
    active_blockers: int = 0

    # People signals
    after_hours_events: int = 0
    weekend_events: int = 0

    # Status transitions
    status_transitions: int = 0
    regressions: int = 0

    # Lists for drill-down
    completed_tickets: list[dict] = field(default_factory=list)
    created_tickets: list[dict] = field(default_factory=list)
    blocker_tickets: list[dict] = field(default_factory=list)
    status_changes: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class TimeframeContext:
    """Context that changes based on daily/weekly/monthly view."""
    timeframe: str  # 'daily', 'weekly', 'monthly'
    period_start: date
    period_end: date
    comparison_period_start: date
    comparison_period_end: date

    # Questions to answer
    primary_questions: list[str] = field(default_factory=list)

    # What to emphasize
    emphasis: list[str] = field(default_factory=list)


@dataclass
class ComparisonMetrics:
    """Comparison between current and previous periods."""
    current_tickets_completed: int = 0
    current_points_completed: float = 0.0
    previous_tickets_completed: int = 0
    previous_points_completed: float = 0.0
    velocity_change_percent: float = 0.0
    trend: str = "stable"  # 'up', 'down', 'stable'

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DeltaEngine:
    """
    The deterministic brain that computes exactly what changed.
    All AI claims must trace back to evidence computed here.
    """

    # Thresholds for detection
    SILENT_DAYS_THRESHOLD = 3
    BLOCKED_DAYS_THRESHOLD = 5
    STATUS_CHURN_THRESHOLD = 3
    HIGH_WIP_THRESHOLD = 5
    AFTER_HOURS_THRESHOLD = 5
    WEEKEND_EVENTS_THRESHOLD = 3

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        """
        Initialize the Delta Engine.

        Args:
            conn: DuckDB connection
        """
        self._conn = conn

    @classmethod
    def from_path(cls, db_path: str) -> "DeltaEngine":
        """Create engine from database path."""
        conn = duckdb.connect(db_path, read_only=True)
        return cls(conn)

    def get_timeframe_context(
        self,
        timeframe: str,
        as_of_date: date | None = None
    ) -> TimeframeContext:
        """Build the context for a given timeframe."""
        if as_of_date is None:
            as_of_date = date.today()

        if timeframe == "daily":
            return TimeframeContext(
                timeframe="daily",
                period_start=as_of_date - timedelta(days=1),
                period_end=as_of_date,
                comparison_period_start=as_of_date - timedelta(days=2),
                comparison_period_end=as_of_date - timedelta(days=1),
                primary_questions=[
                    "What moved yesterday?",
                    "Who is stuck right now?",
                    "What needs my attention today?",
                    "Are there any fires to fight?",
                    "What's the top blocker?",
                ],
                emphasis=["tactical", "blockers", "immediate_action"],
            )
        elif timeframe == "weekly":
            week_start = as_of_date - timedelta(days=as_of_date.weekday())
            return TimeframeContext(
                timeframe="weekly",
                period_start=week_start - timedelta(days=7),
                period_end=week_start,
                comparison_period_start=week_start - timedelta(days=14),
                comparison_period_end=week_start - timedelta(days=7),
                primary_questions=[
                    "Will we hit the sprint goal?",
                    "What consumed unplanned time?",
                    "Which tickets churned the most?",
                    "Who's overloaded?",
                    "What should I escalate?",
                ],
                emphasis=["sprint_health", "predictability", "team_load"],
            )
        else:  # monthly
            month_start = as_of_date.replace(day=1)
            last_month_start = (month_start - timedelta(days=1)).replace(day=1)
            return TimeframeContext(
                timeframe="monthly",
                period_start=last_month_start,
                period_end=month_start,
                comparison_period_start=(last_month_start - timedelta(days=1)).replace(day=1),
                comparison_period_end=last_month_start,
                primary_questions=[
                    "What are our systemic bottlenecks?",
                    "What's our strategy alignment score?",
                    "Where are we investing engineering time?",
                    "What are the burnout trends?",
                    "What should we change for next month?",
                ],
                emphasis=["strategic", "trends", "systemic_issues", "team_health"],
            )

    def compute_delta(
        self,
        project_key: str,
        context: TimeframeContext
    ) -> DailyDelta:
        """Compute what changed in the given timeframe."""
        delta = DailyDelta(
            period_start=context.period_start,
            period_end=context.period_end,
        )

        # 1. Tickets completed in the period
        try:
            completed = self._conn.execute("""
                SELECT
                    i.key,
                    i.summary,
                    i.story_points,
                    i.assignee_id,
                    u.pseudonym as assignee_name,
                    i.resolved as completed_at
                FROM issues i
                LEFT JOIN users u ON i.assignee_id = u.account_id
                WHERE i.resolved IS NOT NULL
                  AND i.project_key = ?
                  AND CAST(i.resolved AS DATE) BETWEEN ? AND ?
                ORDER BY i.resolved DESC
            """, [project_key, context.period_start, context.period_end]).fetchall()

            delta.tickets_completed = len(completed)
            delta.points_completed = sum(r[2] or 0 for r in completed)
            delta.completed_tickets = [
                {
                    "ticket_key": r[0],
                    "summary": r[1],
                    "points": r[2],
                    "assignee": r[4],
                    "completed_at": str(r[5]) if r[5] else None,
                }
                for r in completed
            ]
        except Exception as e:
            logger.warning(f"Failed to get completed tickets: {e}")

        # 2. Tickets created in the period
        try:
            created = self._conn.execute("""
                SELECT
                    key,
                    summary,
                    story_points,
                    issue_type,
                    created
                FROM issues
                WHERE project_key = ?
                  AND CAST(created AS DATE) BETWEEN ? AND ?
                ORDER BY created DESC
            """, [project_key, context.period_start, context.period_end]).fetchall()

            delta.tickets_created = len(created)
            delta.points_added = sum(r[2] or 0 for r in created)
            delta.created_tickets = [
                {
                    "ticket_key": r[0],
                    "summary": r[1],
                    "points": r[2],
                    "type": r[3],
                    "created_at": str(r[4]) if r[4] else None,
                }
                for r in created
            ]
        except Exception as e:
            logger.warning(f"Failed to get created tickets: {e}")

        # 3. Status changes and regressions
        try:
            status_changes = self._conn.execute("""
                SELECT
                    issue_key,
                    from_value,
                    to_value,
                    changed_at
                FROM issue_changelog
                WHERE field = 'status'
                  AND CAST(changed_at AS DATE) BETWEEN ? AND ?
                  AND issue_key IN (SELECT key FROM issues WHERE project_key = ?)
                ORDER BY changed_at DESC
            """, [context.period_start, context.period_end, project_key]).fetchall()

            delta.status_transitions = len(status_changes)

            # Detect regressions (Done/Resolved/Closed -> non-done status)
            done_statuses = {"Done", "Resolved", "Closed", "Complete"}
            for change in status_changes:
                from_status = change[1] or ""
                to_status = change[2] or ""
                if from_status in done_statuses and to_status not in done_statuses:
                    delta.regressions += 1

            delta.status_changes = [
                {
                    "ticket_key": r[0],
                    "from_status": r[1],
                    "to_status": r[2],
                    "when": str(r[3]) if r[3] else None,
                }
                for r in status_changes[:50]  # Limit to 50
            ]
        except Exception as e:
            logger.warning(f"Failed to get status changes: {e}")

        # 4. Current blockers
        try:
            blockers = self._conn.execute("""
                SELECT
                    key,
                    summary,
                    u.pseudonym as assignee_name,
                    updated
                FROM issues i
                LEFT JOIN users u ON i.assignee_id = u.account_id
                WHERE i.project_key = ?
                  AND i.status IN ('Blocked', 'On Hold', 'Waiting')
                  AND i.resolved IS NULL
            """, [project_key]).fetchall()

            delta.active_blockers = len(blockers)
            delta.blocker_tickets = [
                {
                    "ticket_key": r[0],
                    "summary": r[1],
                    "assignee": r[2],
                    "days_blocked": (date.today() - r[3].date()).days if r[3] else 0,
                }
                for r in blockers
            ]
        except Exception as e:
            logger.warning(f"Failed to get blockers: {e}")

        # 5. After-hours activity (events outside 9-18)
        try:
            after_hours = self._conn.execute("""
                SELECT COUNT(*)
                FROM issue_changelog
                WHERE CAST(changed_at AS DATE) BETWEEN ? AND ?
                  AND (EXTRACT(HOUR FROM changed_at) < 9 OR EXTRACT(HOUR FROM changed_at) >= 18)
                  AND issue_key IN (SELECT key FROM issues WHERE project_key = ?)
            """, [context.period_start, context.period_end, project_key]).fetchone()
            delta.after_hours_events = after_hours[0] if after_hours else 0
        except Exception as e:
            logger.warning(f"Failed to get after-hours events: {e}")

        # 6. Weekend activity
        try:
            weekend = self._conn.execute("""
                SELECT COUNT(*)
                FROM issue_changelog
                WHERE CAST(changed_at AS DATE) BETWEEN ? AND ?
                  AND EXTRACT(DOW FROM changed_at) IN (0, 6)
                  AND issue_key IN (SELECT key FROM issues WHERE project_key = ?)
            """, [context.period_start, context.period_end, project_key]).fetchone()
            delta.weekend_events = weekend[0] if weekend else 0
        except Exception as e:
            logger.warning(f"Failed to get weekend events: {e}")

        return delta

    def detect_attention_items(
        self,
        project_key: str,
        context: TimeframeContext
    ) -> list[AttentionItem]:
        """Identify all items that need PM attention."""
        attention_items: list[AttentionItem] = []

        # 1. Silent tickets (no update in N+ days while in progress)
        try:
            silent = self._conn.execute("""
                SELECT
                    i.key,
                    i.summary,
                    i.status,
                    u.pseudonym as assignee_name,
                    DATE_DIFF('day', i.updated, CURRENT_TIMESTAMP) as days_silent
                FROM issues i
                LEFT JOIN users u ON i.assignee_id = u.account_id
                WHERE i.project_key = ?
                  AND i.status NOT IN ('Done', 'Resolved', 'Closed', 'Backlog', 'To Do')
                  AND i.resolved IS NULL
                  AND i.updated < CURRENT_DATE - INTERVAL ? DAY
                ORDER BY days_silent DESC
            """, [project_key, self.SILENT_DAYS_THRESHOLD]).fetchall()

            for row in silent:
                days_silent = row[4]
                severity = (
                    AttentionSeverity.CRITICAL if days_silent >= 7
                    else AttentionSeverity.HIGH if days_silent >= 5
                    else AttentionSeverity.MEDIUM
                )

                attention_items.append(AttentionItem(
                    ticket_key=row[0],
                    ticket_summary=row[1],
                    reason=AttentionReason.SILENT_BLOCKER,
                    severity=severity,
                    evidence=Evidence(
                        ticket_keys=[row[0]],
                        description=f"No updates for {days_silent} days while in '{row[2]}'",
                        metric_value=days_silent,
                    ),
                    suggested_action=f"Ping {row[3] or 'assignee'} for status update",
                    draft_message=self._draft_nudge_message(row[0], row[3], days_silent),
                    attention_score=days_silent * 10 + (20 if severity == AttentionSeverity.CRITICAL else 0),
                    assignee_name=row[3],
                ))
        except Exception as e:
            logger.warning(f"Failed to detect silent tickets: {e}")

        # 2. Status churn (bouncing between statuses)
        try:
            churn = self._conn.execute("""
                WITH status_counts AS (
                    SELECT
                        issue_key,
                        COUNT(*) as transitions_last_week
                    FROM issue_changelog
                    WHERE field = 'status'
                      AND changed_at >= CURRENT_DATE - INTERVAL '7 days'
                      AND issue_key IN (SELECT key FROM issues WHERE project_key = ?)
                    GROUP BY issue_key
                    HAVING COUNT(*) >= ?
                )
                SELECT
                    sc.issue_key,
                    i.summary,
                    sc.transitions_last_week
                FROM status_counts sc
                JOIN issues i ON sc.issue_key = i.key
            """, [project_key, self.STATUS_CHURN_THRESHOLD]).fetchall()

            for row in churn:
                severity = AttentionSeverity.HIGH if row[2] >= 5 else AttentionSeverity.MEDIUM

                attention_items.append(AttentionItem(
                    ticket_key=row[0],
                    ticket_summary=row[1],
                    reason=AttentionReason.STATUS_CHURN,
                    severity=severity,
                    evidence=Evidence(
                        ticket_keys=[row[0]],
                        description=f"{row[2]} status changes in 7 days",
                        metric_value=row[2],
                    ),
                    suggested_action="Investigate process issue or unclear requirements",
                    attention_score=row[2] * 8,
                ))
        except Exception as e:
            logger.warning(f"Failed to detect status churn: {e}")

        # 3. Blocked too long
        try:
            long_blocked = self._conn.execute("""
                SELECT
                    i.key,
                    i.summary,
                    u.pseudonym as assignee_name,
                    DATE_DIFF('day', i.updated, CURRENT_TIMESTAMP) as days_blocked
                FROM issues i
                LEFT JOIN users u ON i.assignee_id = u.account_id
                WHERE i.project_key = ?
                  AND i.status IN ('Blocked', 'On Hold', 'Waiting')
                  AND i.resolved IS NULL
                  AND DATE_DIFF('day', i.updated, CURRENT_TIMESTAMP) >= ?
                ORDER BY days_blocked DESC
            """, [project_key, self.BLOCKED_DAYS_THRESHOLD]).fetchall()

            for row in long_blocked:
                days_blocked = row[3]
                severity = AttentionSeverity.CRITICAL if days_blocked >= 10 else AttentionSeverity.HIGH

                attention_items.append(AttentionItem(
                    ticket_key=row[0],
                    ticket_summary=row[1],
                    reason=AttentionReason.BLOCKED_TOO_LONG,
                    severity=severity,
                    evidence=Evidence(
                        ticket_keys=[row[0]],
                        description=f"Blocked for {days_blocked} days",
                        metric_value=days_blocked,
                    ),
                    suggested_action="Escalate blocker or find workaround",
                    draft_message=self._draft_escalation_message(row[0], row[1], days_blocked),
                    attention_score=days_blocked * 12,
                    assignee_name=row[2],
                ))
        except Exception as e:
            logger.warning(f"Failed to detect long-blocked tickets: {e}")

        # 4. High WIP per developer
        try:
            high_wip = self._conn.execute("""
                SELECT
                    u.pseudonym as assignee_name,
                    COUNT(*) as wip_count,
                    ARRAY_AGG(i.key) as ticket_keys
                FROM issues i
                JOIN users u ON i.assignee_id = u.account_id
                WHERE i.project_key = ?
                  AND i.status IN ('In Progress', 'In Review', 'In Development')
                  AND i.resolved IS NULL
                GROUP BY u.pseudonym
                HAVING COUNT(*) >= ?
            """, [project_key, self.HIGH_WIP_THRESHOLD]).fetchall()

            for row in high_wip:
                attention_items.append(AttentionItem(
                    ticket_key="TEAM",
                    ticket_summary=f"High WIP: {row[0]}",
                    reason=AttentionReason.HIGH_WIP,
                    severity=AttentionSeverity.MEDIUM,
                    evidence=Evidence(
                        ticket_keys=row[2][:5] if row[2] else [],
                        description=f"{row[1]} items in progress for {row[0]}",
                        metric_value=row[1],
                    ),
                    suggested_action=f"Check in with {row[0]} about workload",
                    attention_score=row[1] * 5,
                    assignee_name=row[0],
                ))
        except Exception as e:
            logger.warning(f"Failed to detect high WIP: {e}")

        # 5. No assignee on in-progress tickets
        try:
            no_assignee = self._conn.execute("""
                SELECT
                    key,
                    summary,
                    status
                FROM issues
                WHERE project_key = ?
                  AND assignee_id IS NULL
                  AND status NOT IN ('Done', 'Resolved', 'Closed', 'Backlog')
                  AND resolved IS NULL
            """, [project_key]).fetchall()

            for row in no_assignee:
                attention_items.append(AttentionItem(
                    ticket_key=row[0],
                    ticket_summary=row[1],
                    reason=AttentionReason.NO_ASSIGNEE,
                    severity=AttentionSeverity.MEDIUM,
                    evidence=Evidence(
                        ticket_keys=[row[0]],
                        description=f"No assignee while in '{row[2]}'",
                    ),
                    suggested_action="Assign ticket to a team member",
                    attention_score=30,
                ))
        except Exception as e:
            logger.warning(f"Failed to detect unassigned tickets: {e}")

        # Sort by attention score
        attention_items.sort(key=lambda x: x.attention_score, reverse=True)

        return attention_items

    def get_comparison_metrics(
        self,
        project_key: str,
        context: TimeframeContext
    ) -> ComparisonMetrics:
        """Get current vs previous period metrics for comparison."""
        metrics = ComparisonMetrics()

        try:
            # Current period
            current = self._conn.execute("""
                SELECT
                    COUNT(*) as tickets_completed,
                    COALESCE(SUM(story_points), 0) as points_completed
                FROM issues
                WHERE project_key = ?
                  AND resolved IS NOT NULL
                  AND CAST(resolved AS DATE) BETWEEN ? AND ?
            """, [project_key, context.period_start, context.period_end]).fetchone()

            metrics.current_tickets_completed = current[0] if current else 0
            metrics.current_points_completed = current[1] if current else 0

            # Previous period
            previous = self._conn.execute("""
                SELECT
                    COUNT(*) as tickets_completed,
                    COALESCE(SUM(story_points), 0) as points_completed
                FROM issues
                WHERE project_key = ?
                  AND resolved IS NOT NULL
                  AND CAST(resolved AS DATE) BETWEEN ? AND ?
            """, [
                project_key,
                context.comparison_period_start,
                context.comparison_period_end,
            ]).fetchone()

            metrics.previous_tickets_completed = previous[0] if previous else 0
            metrics.previous_points_completed = previous[1] if previous else 0

            # Calculate velocity trend
            if metrics.previous_points_completed > 0:
                velocity_change = (
                    (metrics.current_points_completed - metrics.previous_points_completed)
                    / metrics.previous_points_completed
                ) * 100
                metrics.velocity_change_percent = round(velocity_change, 1)
                metrics.trend = (
                    "up" if velocity_change > 5
                    else "down" if velocity_change < -5
                    else "stable"
                )
        except Exception as e:
            logger.warning(f"Failed to compute comparison metrics: {e}")

        return metrics

    def get_yesterday_briefing(
        self,
        user_id: str,
        project_key: str
    ) -> dict[str, Any] | None:
        """Retrieve yesterday's briefing for narrative continuity."""
        try:
            result = self._conn.execute("""
                SELECT
                    briefing_date,
                    narrative_summary,
                    key_highlights,
                    recommendations
                FROM daily_briefings
                WHERE user_id = ?
                  AND project_key = ?
                  AND briefing_date = CURRENT_DATE - INTERVAL '1 day'
                  AND timeframe = 'daily'
            """, [user_id, project_key]).fetchone()

            if result:
                return {
                    "date": str(result[0]),
                    "summary": result[1],
                    "highlights": json.loads(result[2]) if result[2] else [],
                    "recommendations": json.loads(result[3]) if result[3] else [],
                }
        except Exception as e:
            logger.warning(f"Failed to get yesterday's briefing: {e}")

        return None

    def get_pending_recommendations(
        self,
        user_id: str,
        project_key: str
    ) -> list[dict[str, Any]]:
        """Get recommendations that weren't acted upon."""
        try:
            results = self._conn.execute("""
                SELECT
                    ra.recommendation_text,
                    ra.ticket_keys,
                    ra.created_at
                FROM recommendation_actions ra
                JOIN daily_briefings db ON ra.briefing_id = db.id
                WHERE db.user_id = ?
                  AND db.project_key = ?
                  AND ra.action_taken = 'pending'
                  AND db.briefing_date >= CURRENT_DATE - INTERVAL '7 days'
                ORDER BY ra.created_at DESC
            """, [user_id, project_key]).fetchall()

            return [
                {
                    "recommendation": r[0],
                    "ticket_keys": r[1],
                    "created_at": str(r[2]),
                }
                for r in results
            ]
        except Exception as e:
            logger.warning(f"Failed to get pending recommendations: {e}")
            return []

    def get_sprint_context(
        self,
        project_key: str
    ) -> dict[str, Any] | None:
        """Get active sprint information."""
        try:
            result = self._conn.execute("""
                SELECT
                    s.id,
                    s.name,
                    s.start_date,
                    s.end_date,
                    s.goal
                FROM sprints s
                WHERE s.state = 'active'
                ORDER BY s.start_date DESC
                LIMIT 1
            """).fetchone()

            if result:
                start_date = result[2]
                end_date = result[3]

                # Make dates naive for comparison
                if hasattr(start_date, 'date'):
                    start_date = start_date.date() if hasattr(start_date, 'date') else start_date
                if hasattr(end_date, 'date'):
                    end_date = end_date.date() if hasattr(end_date, 'date') else end_date

                today = date.today()
                total_days = (end_date - start_date).days if start_date and end_date else 14
                days_elapsed = (today - start_date).days if start_date else 0
                days_remaining = (end_date - today).days if end_date else 0

                return {
                    "sprint_id": result[0],
                    "sprint_name": result[1],
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                    "goal": result[4],
                    "total_days": total_days,
                    "days_elapsed": max(0, days_elapsed),
                    "days_remaining": max(0, days_remaining),
                    "is_sprint_start": days_elapsed <= 2,
                    "is_sprint_end": days_remaining <= 2,
                }
        except Exception as e:
            logger.warning(f"Failed to get sprint context: {e}")

        return None

    def save_daily_delta(
        self,
        project_key: str,
        delta: DailyDelta
    ) -> None:
        """Persist the computed delta to the database."""
        try:
            self._conn.execute("""
                INSERT OR REPLACE INTO daily_deltas (
                    project_key, delta_date,
                    tickets_created, tickets_completed, tickets_reopened,
                    points_completed, points_added, points_removed,
                    new_blockers, resolved_blockers, active_blockers,
                    after_hours_events, weekend_events,
                    status_transitions, regressions,
                    completed_ticket_keys, created_ticket_keys, blocker_ticket_keys,
                    computed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, [
                project_key,
                delta.period_end,
                delta.tickets_created,
                delta.tickets_completed,
                delta.tickets_reopened,
                delta.points_completed,
                delta.points_added,
                delta.points_removed,
                delta.new_blockers,
                delta.resolved_blockers,
                delta.active_blockers,
                delta.after_hours_events,
                delta.weekend_events,
                delta.status_transitions,
                delta.regressions,
                json.dumps([t["ticket_key"] for t in delta.completed_tickets]),
                json.dumps([t["ticket_key"] for t in delta.created_tickets]),
                json.dumps([t["ticket_key"] for t in delta.blocker_tickets]),
            ])
            logger.debug(f"Saved daily delta for {project_key} on {delta.period_end}")
        except Exception as e:
            logger.warning(f"Failed to save daily delta: {e}")

    def _draft_nudge_message(
        self,
        ticket_key: str,
        assignee: str | None,
        days_silent: int
    ) -> str:
        """Draft a friendly nudge message."""
        return f"""Hi {assignee or 'there'}!

Quick check-in on {ticket_key} - I noticed it hasn't had an update in {days_silent} days.

No pressure, just wanted to see:
- Is everything going okay?
- Are you blocked on anything I can help with?
- Do we need to adjust the timeline?

Let me know!"""

    def _draft_escalation_message(
        self,
        ticket_key: str,
        summary: str,
        days_blocked: int
    ) -> str:
        """Draft an escalation message for leadership."""
        return f"""Escalation Required: {ticket_key}

Ticket: {ticket_key} - {summary}
Blocked for: {days_blocked} days

Impact: [Describe downstream impact]

Requested Action: [What you need from leadership]

Recommended Resolution Path:
1. [Option A]
2. [Option B]

Please advise on how to proceed."""
