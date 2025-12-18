"""
Ticket feature extraction module.

Extracts features from Jira issues for ML model training and prediction.
"""

from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger


class TicketFeatureExtractor:
    """
    Extracts features from Jira tickets for ML models.

    Features include numerical, categorical, temporal, and developer-related
    attributes useful for predicting ticket duration.

    Example:
        >>> extractor = TicketFeatureExtractor(conn)
        >>> features = extractor.extract_features("PROJ-123")
        >>> df = extractor.extract_batch(issue_keys)
    """

    # Status categories for cycle time calculation
    IN_PROGRESS_STATUSES = {
        "in progress", "in development", "in review", "code review",
        "testing", "qa", "in testing", "review"
    }

    DONE_STATUSES = {"done", "closed", "resolved", "released", "completed"}

    BLOCKED_STATUSES = {"blocked", "on hold", "waiting", "impediment"}

    def __init__(
        self,
        conn,
        *,
        developer_stats: pd.DataFrame | None = None,
    ):
        """
        Initialize the feature extractor.

        Args:
            conn: DuckDB connection
            developer_stats: Pre-computed developer statistics DataFrame
        """
        self._conn = conn
        self._developer_stats = developer_stats

    def extract_features(self, issue_key: str) -> dict[str, Any] | None:
        """
        Extract features for a single issue.

        Args:
            issue_key: Jira issue key

        Returns:
            Dictionary of features or None if issue not found
        """
        query = """
        SELECT
            key,
            summary,
            description,
            issue_type,
            status,
            priority,
            assignee_id,
            created,
            updated,
            resolved,
            story_points,
            sprint_id,
            sprint_name,
            epic_key,
            components,
            labels,
            subtask_count,
            link_count,
            attachment_count,
            project_key,
            lead_time_hours,
            cycle_time_hours
        FROM issues
        WHERE key = ?
        """

        result = self._conn.execute(query, [issue_key]).fetchone()
        if not result:
            return None

        columns = [
            "key", "summary", "description", "issue_type", "status", "priority",
            "assignee_id", "created", "updated", "resolved", "story_points",
            "sprint_id", "sprint_name", "epic_key", "components", "labels",
            "subtask_count", "link_count", "attachment_count", "project_key",
            "lead_time_hours", "cycle_time_hours"
        ]
        issue = dict(zip(columns, result))

        return self._compute_features(issue)

    def extract_batch(
        self,
        project_key: str | None = None,
        *,
        resolved_only: bool = True,
        min_cycle_time_hours: float = 1.0,
        max_cycle_time_hours: float = 500.0,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Extract features for multiple issues.

        Args:
            project_key: Filter by project
            resolved_only: Only include resolved issues (for training)
            min_cycle_time_hours: Minimum cycle time filter
            max_cycle_time_hours: Maximum cycle time filter
            limit: Maximum number of issues

        Returns:
            DataFrame with features for each issue
        """
        conditions = ["1=1"]
        params: list[Any] = []

        if project_key:
            conditions.append("project_key = ?")
            params.append(project_key)

        if resolved_only:
            conditions.append("resolved IS NOT NULL")
            conditions.append("cycle_time_hours IS NOT NULL")
            conditions.append("cycle_time_hours >= ?")
            params.append(min_cycle_time_hours)
            conditions.append("cycle_time_hours <= ?")
            params.append(max_cycle_time_hours)

        where_clause = " AND ".join(conditions)
        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
        SELECT
            key,
            summary,
            description,
            issue_type,
            status,
            priority,
            assignee_id,
            created,
            updated,
            resolved,
            story_points,
            sprint_id,
            sprint_name,
            epic_key,
            components,
            labels,
            subtask_count,
            link_count,
            attachment_count,
            project_key,
            lead_time_hours,
            cycle_time_hours
        FROM issues
        WHERE {where_clause}
        ORDER BY created ASC
        {limit_clause}
        """

        df = self._conn.execute(query, params).df()

        if df.empty:
            logger.warning("No issues found matching criteria")
            return pd.DataFrame()

        logger.info(f"Extracting features for {len(df)} issues")

        # Pre-load developer stats if not provided
        if self._developer_stats is None:
            self._developer_stats = self._compute_developer_stats()

        # Extract features for each row
        features_list = []
        for _, row in df.iterrows():
            issue = row.to_dict()
            features = self._compute_features(issue)
            features_list.append(features)

        return pd.DataFrame(features_list)

    def _compute_features(self, issue: dict[str, Any]) -> dict[str, Any]:
        """
        Compute all features for a single issue.

        Args:
            issue: Issue dictionary

        Returns:
            Dictionary of computed features
        """
        features = {
            "issue_key": issue.get("key"),
        }

        # === Numerical Features ===
        features["story_points"] = self._safe_float(issue.get("story_points"), default=0.0)

        # Description length
        description = issue.get("description") or ""
        features["description_length"] = len(description)
        features["description_word_count"] = len(description.split()) if description else 0

        # Summary length
        summary = issue.get("summary") or ""
        features["summary_length"] = len(summary)
        features["summary_word_count"] = len(summary.split()) if summary else 0

        # Counts
        components = issue.get("components") or []
        labels = issue.get("labels") or []

        features["num_components"] = len(components) if isinstance(components, list) else 0
        features["num_labels"] = len(labels) if isinstance(labels, list) else 0
        features["num_subtasks"] = self._safe_int(issue.get("subtask_count"), default=0)
        features["num_links"] = self._safe_int(issue.get("link_count"), default=0)
        features["has_attachments"] = 1 if self._safe_int(issue.get("attachment_count"), 0) > 0 else 0
        features["has_epic"] = 1 if issue.get("epic_key") else 0
        features["has_sprint"] = 1 if issue.get("sprint_id") else 0

        # === Categorical Features ===
        features["issue_type"] = issue.get("issue_type") or "Unknown"
        features["priority"] = issue.get("priority") or "Medium"
        features["status"] = issue.get("status") or "Unknown"

        # Primary component
        if components and isinstance(components, list) and len(components) > 0:
            features["component_primary"] = components[0]
        else:
            features["component_primary"] = "None"

        # === Temporal Features ===
        created = issue.get("created")
        if created:
            if isinstance(created, str):
                created = datetime.fromisoformat(created.replace("Z", "+00:00"))
            if hasattr(created, "weekday"):
                features["created_day_of_week"] = created.weekday()  # 0=Monday
                features["created_hour"] = created.hour
                features["created_month"] = created.month
                features["created_quarter"] = (created.month - 1) // 3 + 1
                features["is_created_weekend"] = 1 if created.weekday() >= 5 else 0
            else:
                features["created_day_of_week"] = 0
                features["created_hour"] = 12
                features["created_month"] = 1
                features["created_quarter"] = 1
                features["is_created_weekend"] = 0
        else:
            features["created_day_of_week"] = 0
            features["created_hour"] = 12
            features["created_month"] = 1
            features["created_quarter"] = 1
            features["is_created_weekend"] = 0

        # Sprint day created (1-14 for 2-week sprint)
        features["sprint_day_created"] = self._calculate_sprint_day(issue)

        # === Developer Features ===
        assignee_id = issue.get("assignee_id")
        if assignee_id and self._developer_stats is not None:
            dev_stats = self._developer_stats[
                self._developer_stats["assignee_id"] == assignee_id
            ]
            if not dev_stats.empty:
                row = dev_stats.iloc[0]
                features["assignee_avg_cycle_time_30d"] = self._safe_float(
                    row.get("avg_cycle_time_hours"), default=48.0
                )
                features["assignee_completion_rate_30d"] = self._safe_float(
                    row.get("completion_rate"), default=0.5
                )
                features["assignee_current_wip"] = self._safe_int(
                    row.get("wip_count"), default=3
                )
                features["assignee_total_completed_30d"] = self._safe_int(
                    row.get("completed_count"), default=5
                )
            else:
                features["assignee_avg_cycle_time_30d"] = 48.0
                features["assignee_completion_rate_30d"] = 0.5
                features["assignee_current_wip"] = 3
                features["assignee_total_completed_30d"] = 5
        else:
            # Default values for unassigned or unknown
            features["assignee_avg_cycle_time_30d"] = 48.0
            features["assignee_completion_rate_30d"] = 0.5
            features["assignee_current_wip"] = 3
            features["assignee_total_completed_30d"] = 5

        features["is_assigned"] = 1 if assignee_id else 0

        # === Target Variables ===
        features["cycle_time_hours"] = self._safe_float(issue.get("cycle_time_hours"))
        features["lead_time_hours"] = self._safe_float(issue.get("lead_time_hours"))

        return features

    def _compute_developer_stats(self) -> pd.DataFrame:
        """Compute developer statistics for feature enrichment."""
        query = """
        WITH completed AS (
            SELECT
                assignee_id,
                COUNT(*) as completed_count,
                AVG(cycle_time_hours) as avg_cycle_time_hours
            FROM issues
            WHERE resolved IS NOT NULL
              AND resolved >= CURRENT_DATE - INTERVAL '30' DAY
              AND assignee_id IS NOT NULL
            GROUP BY assignee_id
        ),
        wip AS (
            SELECT
                assignee_id,
                COUNT(*) as wip_count
            FROM issues
            WHERE resolved IS NULL
              AND status NOT IN ('To Do', 'Backlog', 'Open')
              AND assignee_id IS NOT NULL
            GROUP BY assignee_id
        ),
        total AS (
            SELECT
                assignee_id,
                COUNT(*) as total_assigned
            FROM issues
            WHERE created >= CURRENT_DATE - INTERVAL '30' DAY
              AND assignee_id IS NOT NULL
            GROUP BY assignee_id
        )
        SELECT
            COALESCE(c.assignee_id, w.assignee_id, t.assignee_id) as assignee_id,
            COALESCE(c.completed_count, 0) as completed_count,
            COALESCE(c.avg_cycle_time_hours, 48.0) as avg_cycle_time_hours,
            COALESCE(w.wip_count, 0) as wip_count,
            COALESCE(t.total_assigned, 0) as total_assigned,
            CASE
                WHEN COALESCE(t.total_assigned, 0) > 0
                THEN CAST(COALESCE(c.completed_count, 0) AS FLOAT) / t.total_assigned
                ELSE 0.5
            END as completion_rate
        FROM completed c
        FULL OUTER JOIN wip w ON c.assignee_id = w.assignee_id
        FULL OUTER JOIN total t ON COALESCE(c.assignee_id, w.assignee_id) = t.assignee_id
        """

        return self._conn.execute(query).df()

    def _calculate_sprint_day(self, issue: dict[str, Any]) -> int:
        """Calculate which day of the sprint the issue was created."""
        sprint_id = issue.get("sprint_id")
        created = issue.get("created")

        if not sprint_id or not created:
            return 7  # Default to middle of sprint

        # Get sprint start date
        result = self._conn.execute(
            "SELECT start_date FROM sprints WHERE id = ?",
            [sprint_id]
        ).fetchone()

        if not result or not result[0]:
            return 7

        sprint_start = result[0]
        if isinstance(sprint_start, str):
            sprint_start = datetime.fromisoformat(sprint_start.replace("Z", "+00:00"))
        if isinstance(created, str):
            created = datetime.fromisoformat(created.replace("Z", "+00:00"))

        # Make naive for comparison
        if hasattr(sprint_start, 'tzinfo') and sprint_start.tzinfo:
            sprint_start = sprint_start.replace(tzinfo=None)
        if hasattr(created, 'tzinfo') and created.tzinfo:
            created = created.replace(tzinfo=None)

        try:
            days = (created - sprint_start).days + 1
            return max(1, min(14, days))  # Clamp to 1-14
        except Exception:
            return 7

    @staticmethod
    def _safe_float(value: Any, default: float | None = None) -> float | None:
        """Safely convert to float."""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """Safely convert to int."""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    @classmethod
    def get_feature_columns(cls) -> dict[str, list[str]]:
        """
        Get lists of feature columns by type.

        Returns:
            Dictionary with 'numerical', 'categorical', and 'temporal' keys
        """
        return {
            "numerical": [
                "story_points",
                "description_length",
                "description_word_count",
                "summary_length",
                "summary_word_count",
                "num_components",
                "num_labels",
                "num_subtasks",
                "num_links",
                "has_attachments",
                "has_epic",
                "has_sprint",
                "is_assigned",
                "assignee_avg_cycle_time_30d",
                "assignee_completion_rate_30d",
                "assignee_current_wip",
                "assignee_total_completed_30d",
            ],
            "categorical": [
                "issue_type",
                "priority",
                "component_primary",
            ],
            "temporal": [
                "created_day_of_week",
                "created_hour",
                "created_month",
                "created_quarter",
                "is_created_weekend",
                "sprint_day_created",
            ],
            "target": [
                "cycle_time_hours",
                "lead_time_hours",
            ],
        }
