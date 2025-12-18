"""
Sprint feature extraction module.

Extracts health and progress metrics for sprints.
"""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger


class SprintFeatureExtractor:
    """
    Extracts features for sprint health assessment and risk prediction.

    Computes metrics like completion rate, velocity ratio, blocked tickets,
    and scope creep for sprint risk scoring.

    Example:
        >>> extractor = SprintFeatureExtractor(conn)
        >>> features = extractor.extract_features(sprint_id=123)
        >>> burndown = extractor.get_burndown_data(sprint_id=123)
    """

    DONE_STATUSES = {"done", "closed", "resolved", "released", "completed"}
    BLOCKED_STATUSES = {"blocked", "on hold", "waiting", "impediment"}

    def __init__(self, conn):
        """
        Initialize the sprint feature extractor.

        Args:
            conn: DuckDB connection
        """
        self._conn = conn

    def extract_features(self, sprint_id: int) -> dict[str, Any] | None:
        """
        Extract features for a sprint.

        Args:
            sprint_id: Sprint ID

        Returns:
            Dictionary of sprint features or None if not found
        """
        # Get sprint info
        sprint_info = self._get_sprint_info(sprint_id)
        if not sprint_info:
            return None

        # Get issue metrics
        issue_metrics = self._get_issue_metrics(sprint_id)

        # Get historical velocity
        velocity_stats = self._get_velocity_stats(sprint_info.get("board_id"))

        # Calculate derived features
        features = self._compute_features(sprint_info, issue_metrics, velocity_stats)

        return features

    def _get_sprint_info(self, sprint_id: int) -> dict[str, Any] | None:
        """Get basic sprint information."""
        query = """
        SELECT
            id,
            name,
            state,
            start_date,
            end_date,
            complete_date,
            goal,
            board_id
        FROM sprints
        WHERE id = ?
        """

        result = self._conn.execute(query, [sprint_id]).fetchone()
        if not result:
            return None

        columns = ["id", "name", "state", "start_date", "end_date",
                   "complete_date", "goal", "board_id"]
        return dict(zip(columns, result))

    def _get_issue_metrics(self, sprint_id: int) -> dict[str, Any]:
        """Get aggregated issue metrics for a sprint."""
        query = """
        SELECT
            COUNT(*) as total_issues,
            COUNT(CASE WHEN status IN ('Done', 'Closed', 'Resolved', 'Released') THEN 1 END) as completed_issues,
            COUNT(CASE WHEN status NOT IN ('Done', 'Closed', 'Resolved', 'Released') THEN 1 END) as remaining_issues,
            COUNT(CASE WHEN status IN ('Blocked', 'On Hold', 'Waiting') THEN 1 END) as blocked_issues,
            COUNT(CASE WHEN status IN ('In Progress', 'In Development', 'In Review', 'Testing') THEN 1 END) as in_progress_issues,
            COUNT(CASE WHEN status IN ('To Do', 'Open', 'Backlog', 'New') THEN 1 END) as todo_issues,
            COALESCE(SUM(story_points), 0) as total_points,
            COALESCE(SUM(CASE WHEN status IN ('Done', 'Closed', 'Resolved', 'Released') THEN story_points ELSE 0 END), 0) as completed_points,
            COALESCE(SUM(CASE WHEN status NOT IN ('Done', 'Closed', 'Resolved', 'Released') THEN story_points ELSE 0 END), 0) as remaining_points,
            COALESCE(SUM(CASE WHEN status IN ('Blocked', 'On Hold', 'Waiting') THEN story_points ELSE 0 END), 0) as blocked_points,
            AVG(CASE WHEN cycle_time_hours IS NOT NULL THEN cycle_time_hours END) as avg_cycle_time,
            COUNT(DISTINCT assignee_id) as unique_assignees
        FROM issues
        WHERE sprint_id = ?
        """

        result = self._conn.execute(query, [sprint_id]).fetchone()

        columns = [
            "total_issues", "completed_issues", "remaining_issues",
            "blocked_issues", "in_progress_issues", "todo_issues",
            "total_points", "completed_points", "remaining_points",
            "blocked_points", "avg_cycle_time", "unique_assignees"
        ]
        return dict(zip(columns, result))

    def _get_velocity_stats(self, board_id: int | None, n_sprints: int = 5) -> dict[str, Any]:
        """Get historical velocity statistics."""
        if not board_id:
            return {"avg_velocity": 0, "std_velocity": 0, "sprints_count": 0}

        query = """
        SELECT
            COALESCE(SUM(CASE WHEN i.status IN ('Done', 'Closed', 'Resolved') THEN i.story_points ELSE 0 END), 0) as velocity
        FROM sprints s
        LEFT JOIN issues i ON s.id = i.sprint_id
        WHERE s.board_id = ?
          AND s.state = 'closed'
        GROUP BY s.id
        ORDER BY s.end_date DESC
        LIMIT ?
        """

        df = self._conn.execute(query, [board_id, n_sprints]).df()

        if df.empty:
            return {"avg_velocity": 0, "std_velocity": 0, "sprints_count": 0}

        return {
            "avg_velocity": df["velocity"].mean(),
            "std_velocity": df["velocity"].std() if len(df) > 1 else 0,
            "sprints_count": len(df),
        }

    def _compute_features(
        self,
        sprint_info: dict[str, Any],
        issue_metrics: dict[str, Any],
        velocity_stats: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute all sprint features."""
        now = datetime.now()

        # Parse dates
        start_date = sprint_info.get("start_date")
        end_date = sprint_info.get("end_date")

        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

        # Make naive for comparison
        if start_date and hasattr(start_date, 'tzinfo') and start_date.tzinfo:
            start_date = start_date.replace(tzinfo=None)
        if end_date and hasattr(end_date, 'tzinfo') and end_date.tzinfo:
            end_date = end_date.replace(tzinfo=None)

        # Time calculations
        total_days = 0
        days_elapsed = 0
        days_remaining = 0

        if start_date and end_date:
            total_days = (end_date - start_date).days
            if total_days > 0:
                days_elapsed = min((now - start_date).days, total_days)
                days_remaining = max((end_date - now).days, 0)

        # Progress metrics
        total_points = issue_metrics["total_points"] or 0
        completed_points = issue_metrics["completed_points"] or 0
        remaining_points = issue_metrics["remaining_points"] or 0

        # Completion rates
        completion_rate = (completed_points / total_points * 100) if total_points > 0 else 0
        expected_completion_rate = (days_elapsed / total_days * 100) if total_days > 0 else 0

        # Completion gap (negative = behind schedule)
        completion_gap = completion_rate - expected_completion_rate

        # Velocity ratio (current vs historical)
        avg_velocity = velocity_stats["avg_velocity"] or 0
        velocity_ratio = (completed_points / avg_velocity) if avg_velocity > 0 else 1.0

        # Blocked ratio
        blocked_issues = issue_metrics["blocked_issues"] or 0
        total_issues = issue_metrics["total_issues"] or 0
        blocked_ratio = (blocked_issues / total_issues) if total_issues > 0 else 0

        # Urgency factor (increases as deadline approaches)
        urgency_factor = 1.0
        if days_remaining > 0 and remaining_points > 0:
            # Points per remaining day needed vs historical rate
            points_per_day_needed = remaining_points / days_remaining
            historical_rate = avg_velocity / 14  # Assuming 14-day sprints
            if historical_rate > 0:
                urgency_factor = points_per_day_needed / historical_rate

        features = {
            # Sprint identification
            "sprint_id": sprint_info["id"],
            "sprint_name": sprint_info["name"],
            "sprint_state": sprint_info["state"],
            "board_id": sprint_info["board_id"],

            # Time metrics
            "total_days": total_days,
            "days_elapsed": days_elapsed,
            "days_remaining": days_remaining,
            "progress_percent": (days_elapsed / total_days * 100) if total_days > 0 else 0,

            # Issue counts
            "total_issues": issue_metrics["total_issues"],
            "completed_issues": issue_metrics["completed_issues"],
            "remaining_issues": issue_metrics["remaining_issues"],
            "blocked_issues": issue_metrics["blocked_issues"],
            "in_progress_issues": issue_metrics["in_progress_issues"],
            "todo_issues": issue_metrics["todo_issues"],

            # Story points
            "total_points": total_points,
            "completed_points": completed_points,
            "remaining_points": remaining_points,
            "blocked_points": issue_metrics["blocked_points"] or 0,

            # Completion metrics
            "completion_rate": round(completion_rate, 1),
            "expected_completion_rate": round(expected_completion_rate, 1),
            "completion_gap": round(completion_gap, 1),

            # Velocity metrics
            "avg_historical_velocity": round(avg_velocity, 1),
            "velocity_ratio": round(velocity_ratio, 2),

            # Risk indicators
            "blocked_ratio": round(blocked_ratio, 3),
            "urgency_factor": round(urgency_factor, 2),

            # Team metrics
            "unique_assignees": issue_metrics["unique_assignees"],
            "avg_cycle_time_hours": round(issue_metrics["avg_cycle_time"] or 0, 1),

            # Scope creep (would need changelog analysis - placeholder)
            "scope_creep_ratio": 0.0,

            # Historical context
            "historical_sprints_count": velocity_stats["sprints_count"],
        }

        return features

    def extract_batch(
        self,
        board_id: int | None = None,
        *,
        state: str | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Extract features for multiple sprints.

        Args:
            board_id: Optional board filter
            state: Filter by state (active, closed, future)
            limit: Maximum sprints to process

        Returns:
            DataFrame with sprint features
        """
        conditions = ["1=1"]
        params: list[Any] = []

        if board_id:
            conditions.append("board_id = ?")
            params.append(board_id)

        if state:
            conditions.append("state = ?")
            params.append(state)

        where_clause = " AND ".join(conditions)
        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
        SELECT id
        FROM sprints
        WHERE {where_clause}
        ORDER BY start_date DESC
        {limit_clause}
        """

        sprint_ids = self._conn.execute(query, params).fetchall()

        features_list = []
        for (sprint_id,) in sprint_ids:
            features = self.extract_features(sprint_id)
            if features:
                features_list.append(features)

        return pd.DataFrame(features_list)

    def get_burndown_data(self, sprint_id: int) -> pd.DataFrame:
        """
        Get burndown chart data for a sprint.

        Args:
            sprint_id: Sprint ID

        Returns:
            DataFrame with daily remaining points
        """
        sprint_info = self._get_sprint_info(sprint_id)
        if not sprint_info:
            return pd.DataFrame()

        start_date = sprint_info.get("start_date")
        end_date = sprint_info.get("end_date")

        if not start_date or not end_date:
            return pd.DataFrame()

        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

        # Make naive
        if hasattr(start_date, 'tzinfo') and start_date.tzinfo:
            start_date = start_date.replace(tzinfo=None)
        if hasattr(end_date, 'tzinfo') and end_date.tzinfo:
            end_date = end_date.replace(tzinfo=None)

        # Get total committed points
        total_query = """
        SELECT COALESCE(SUM(story_points), 0) as total_points
        FROM issues
        WHERE sprint_id = ?
        """
        total_result = self._conn.execute(total_query, [sprint_id]).fetchone()
        total_points = total_result[0] if total_result else 0

        # Get completed points by day from changelog
        completion_query = """
        SELECT
            DATE_TRUNC('day', c.changed_at) as completion_date,
            COALESCE(SUM(i.story_points), 0) as points_completed
        FROM issue_changelog c
        JOIN issues i ON c.issue_key = i.key
        WHERE i.sprint_id = ?
          AND c.field = 'status'
          AND c.to_value IN ('Done', 'Closed', 'Resolved', 'Released')
        GROUP BY DATE_TRUNC('day', c.changed_at)
        ORDER BY completion_date
        """

        completions = self._conn.execute(completion_query, [sprint_id]).df()

        # Build daily burndown
        dates = []
        ideal_remaining = []
        actual_remaining = []

        current_date = start_date
        total_days = (end_date - start_date).days
        cumulative_completed = 0

        while current_date <= end_date:
            dates.append(current_date)

            # Ideal burndown (linear)
            days_passed = (current_date - start_date).days
            ideal = total_points * (1 - days_passed / total_days) if total_days > 0 else 0
            ideal_remaining.append(max(0, ideal))

            # Actual remaining
            if not completions.empty:
                day_completions = completions[
                    completions["completion_date"].dt.date <= current_date.date()
                ]["points_completed"].sum()
                cumulative_completed = day_completions

            actual_remaining.append(max(0, total_points - cumulative_completed))

            current_date += timedelta(days=1)

        return pd.DataFrame({
            "date": dates,
            "ideal_remaining": ideal_remaining,
            "actual_remaining": actual_remaining,
        })

    def get_active_sprint_features(self, board_id: int | None = None) -> dict[str, Any] | None:
        """
        Get features for the currently active sprint.

        Args:
            board_id: Optional board filter

        Returns:
            Sprint features or None if no active sprint
        """
        conditions = ["state = 'active'"]
        params: list[Any] = []

        if board_id:
            conditions.append("board_id = ?")
            params.append(board_id)

        query = f"""
        SELECT id
        FROM sprints
        WHERE {" AND ".join(conditions)}
        ORDER BY start_date DESC
        LIMIT 1
        """

        result = self._conn.execute(query, params).fetchone()
        if not result:
            return None

        return self.extract_features(result[0])

    @classmethod
    def get_feature_columns(cls) -> list[str]:
        """Get list of feature columns for ML training."""
        return [
            "days_elapsed",
            "days_remaining",
            "progress_percent",
            "total_issues",
            "completed_issues",
            "remaining_issues",
            "blocked_issues",
            "in_progress_issues",
            "total_points",
            "completed_points",
            "remaining_points",
            "blocked_points",
            "completion_rate",
            "expected_completion_rate",
            "completion_gap",
            "velocity_ratio",
            "blocked_ratio",
            "urgency_factor",
            "unique_assignees",
            "scope_creep_ratio",
        ]
