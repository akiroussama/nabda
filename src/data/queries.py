"""
Analytical queries module for Jira data.

Provides reusable SQL queries for common analytical operations.
"""

from datetime import datetime, timedelta
from typing import Any

import duckdb
import pandas as pd
from loguru import logger


class JiraQueries:
    """
    Provides analytical queries on Jira data stored in DuckDB.

    All methods return pandas DataFrames for easy analysis and visualization.

    Example:
        >>> queries = JiraQueries(conn)
        >>> df = queries.get_issues_with_metrics("PROJ")
        >>> sprint_summary = queries.get_sprint_summary(sprint_id=123)
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        """
        Initialize the queries helper.

        Args:
            conn: DuckDB connection
        """
        self._conn = conn

    def get_issues_with_metrics(
        self,
        project_key: str | None = None,
        *,
        status: str | None = None,
        assignee_id: str | None = None,
        sprint_id: int | None = None,
        include_resolved: bool = True,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Get issues with calculated metrics (lead time, cycle time).

        Args:
            project_key: Filter by project key
            status: Filter by status
            assignee_id: Filter by assignee
            sprint_id: Filter by sprint
            include_resolved: Whether to include resolved issues
            limit: Maximum number of issues to return

        Returns:
            DataFrame with issues and metrics
        """
        conditions = ["1=1"]
        params: list[Any] = []

        if project_key:
            conditions.append("project_key = ?")
            params.append(project_key)

        if status:
            conditions.append("status = ?")
            params.append(status)

        if assignee_id:
            conditions.append("assignee_id = ?")
            params.append(assignee_id)

        if sprint_id:
            conditions.append("sprint_id = ?")
            params.append(sprint_id)

        if not include_resolved:
            conditions.append("resolved IS NULL")

        where_clause = " AND ".join(conditions)
        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
        SELECT
            key,
            summary,
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
            lead_time_hours,
            cycle_time_hours,
            CASE
                WHEN lead_time_hours IS NOT NULL THEN lead_time_hours / 24.0
                ELSE NULL
            END as lead_time_days,
            CASE
                WHEN cycle_time_hours IS NOT NULL THEN cycle_time_hours / 24.0
                ELSE NULL
            END as cycle_time_days
        FROM issues
        WHERE {where_clause}
        ORDER BY created DESC
        {limit_clause}
        """

        return self._conn.execute(query, params).df()

    def get_sprint_summary(self, sprint_id: int) -> dict[str, Any]:
        """
        Get summary metrics for a sprint.

        Args:
            sprint_id: Sprint ID

        Returns:
            Dictionary with sprint summary metrics
        """
        # Get sprint info
        sprint_query = """
        SELECT
            id,
            name,
            state,
            start_date,
            end_date,
            complete_date,
            goal
        FROM sprints
        WHERE id = ?
        """
        sprint_result = self._conn.execute(sprint_query, [sprint_id]).fetchone()

        if not sprint_result:
            return {}

        sprint_info = {
            "id": sprint_result[0],
            "name": sprint_result[1],
            "state": sprint_result[2],
            "start_date": sprint_result[3],
            "end_date": sprint_result[4],
            "complete_date": sprint_result[5],
            "goal": sprint_result[6],
        }

        # Get issue metrics for the sprint
        metrics_query = """
        SELECT
            COUNT(*) as total_issues,
            COUNT(CASE WHEN status IN ('Done', 'Closed', 'Resolved') THEN 1 END) as completed_issues,
            COUNT(CASE WHEN status NOT IN ('Done', 'Closed', 'Resolved') THEN 1 END) as remaining_issues,
            COALESCE(SUM(story_points), 0) as total_points,
            COALESCE(SUM(CASE WHEN status IN ('Done', 'Closed', 'Resolved') THEN story_points ELSE 0 END), 0) as completed_points,
            COALESCE(SUM(CASE WHEN status NOT IN ('Done', 'Closed', 'Resolved') THEN story_points ELSE 0 END), 0) as remaining_points,
            AVG(cycle_time_hours) as avg_cycle_time_hours,
            COUNT(CASE WHEN status IN ('Blocked', 'On Hold') THEN 1 END) as blocked_count
        FROM issues
        WHERE sprint_id = ?
        """
        metrics_result = self._conn.execute(metrics_query, [sprint_id]).fetchone()

        # Calculate derived metrics
        total_points = metrics_result[3] or 0
        completed_points = metrics_result[4] or 0
        completion_rate = (completed_points / total_points * 100) if total_points > 0 else 0

        # Calculate days elapsed/remaining
        now = datetime.now()
        start_date = sprint_info["start_date"]
        end_date = sprint_info["end_date"]

        days_elapsed = 0
        days_remaining = 0
        total_days = 0

        if start_date and end_date:
            if isinstance(start_date, str):
                start_date = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            if isinstance(end_date, str):
                end_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))

            # Make naive for comparison
            if start_date.tzinfo:
                start_date = start_date.replace(tzinfo=None)
            if end_date.tzinfo:
                end_date = end_date.replace(tzinfo=None)

            total_days = (end_date - start_date).days
            days_elapsed = min((now - start_date).days, total_days)
            days_remaining = max((end_date - now).days, 0)

        return {
            **sprint_info,
            "total_issues": metrics_result[0],
            "completed_issues": metrics_result[1],
            "remaining_issues": metrics_result[2],
            "total_points": total_points,
            "completed_points": completed_points,
            "remaining_points": metrics_result[5] or 0,
            "completion_rate": round(completion_rate, 1),
            "avg_cycle_time_hours": round(metrics_result[6] or 0, 1),
            "blocked_count": metrics_result[7],
            "days_elapsed": days_elapsed,
            "days_remaining": days_remaining,
            "total_days": total_days,
        }

    def get_developer_workload(
        self,
        days: int = 30,
        *,
        project_key: str | None = None,
    ) -> pd.DataFrame:
        """
        Get workload metrics for each developer.

        Args:
            days: Number of days to look back for metrics
            project_key: Optional project filter

        Returns:
            DataFrame with developer workload metrics
        """
        since_date = datetime.now() - timedelta(days=days)
        project_filter = "AND i.project_key = ?" if project_key else ""
        params: list[Any] = [since_date]
        if project_key:
            params.append(project_key)

        query = f"""
        WITH developer_metrics AS (
            SELECT
                assignee_id,
                COUNT(*) as total_assigned,
                COUNT(CASE WHEN status NOT IN ('Done', 'Closed', 'Resolved') THEN 1 END) as in_progress,
                COUNT(CASE WHEN status IN ('Done', 'Closed', 'Resolved') AND resolved >= ? THEN 1 END) as completed_recently,
                COALESCE(SUM(CASE WHEN status NOT IN ('Done', 'Closed', 'Resolved') THEN story_points ELSE 0 END), 0) as wip_points,
                COALESCE(SUM(CASE WHEN status IN ('Done', 'Closed', 'Resolved') AND resolved >= ? THEN story_points ELSE 0 END), 0) as completed_points,
                AVG(CASE WHEN cycle_time_hours IS NOT NULL AND resolved >= ? THEN cycle_time_hours END) as avg_cycle_time
            FROM issues i
            WHERE assignee_id IS NOT NULL
            {project_filter}
            GROUP BY assignee_id
        ),
        worklog_metrics AS (
            SELECT
                author_id as assignee_id,
                SUM(time_spent_seconds) / 3600.0 as hours_logged
            FROM worklogs
            WHERE started >= ?
            GROUP BY author_id
        )
        SELECT
            dm.assignee_id,
            u.pseudonym,
            dm.total_assigned,
            dm.in_progress,
            dm.completed_recently as completed_{days}d,
            dm.wip_points,
            dm.completed_points as completed_points_{days}d,
            COALESCE(wm.hours_logged, 0) as hours_logged_{days}d,
            ROUND(dm.avg_cycle_time, 1) as avg_cycle_time_hours
        FROM developer_metrics dm
        LEFT JOIN worklog_metrics wm ON dm.assignee_id = wm.assignee_id
        LEFT JOIN users u ON dm.assignee_id = u.account_id
        ORDER BY dm.wip_points DESC, dm.in_progress DESC
        """

        # Add date params for each reference
        full_params = [since_date, since_date, since_date]
        if project_key:
            full_params.append(project_key)
        full_params.append(since_date)

        return self._conn.execute(query, full_params).df()

    def get_velocity_history(
        self,
        n_sprints: int = 10,
        *,
        board_id: int | None = None,
    ) -> pd.DataFrame:
        """
        Get velocity (completed story points) history by sprint.

        Args:
            n_sprints: Number of sprints to include
            board_id: Optional board ID filter

        Returns:
            DataFrame with velocity history
        """
        board_filter = "AND s.board_id = ?" if board_id else ""
        params: list[Any] = []
        if board_id:
            params.append(board_id)

        query = f"""
        SELECT
            s.id as sprint_id,
            s.name as sprint_name,
            s.state,
            s.start_date,
            s.end_date,
            COALESCE(SUM(i.story_points), 0) as committed_points,
            COALESCE(SUM(CASE WHEN i.status IN ('Done', 'Closed', 'Resolved') THEN i.story_points ELSE 0 END), 0) as completed_points,
            COUNT(i.key) as total_issues,
            COUNT(CASE WHEN i.status IN ('Done', 'Closed', 'Resolved') THEN 1 END) as completed_issues
        FROM sprints s
        LEFT JOIN issues i ON s.id = i.sprint_id
        WHERE s.state = 'closed'
        {board_filter}
        GROUP BY s.id, s.name, s.state, s.start_date, s.end_date
        ORDER BY s.end_date DESC
        LIMIT ?
        """
        params.append(n_sprints)

        df = self._conn.execute(query, params).df()

        # Calculate completion rate
        if not df.empty:
            df["completion_rate"] = (
                df["completed_points"] / df["committed_points"] * 100
            ).fillna(0).round(1)

        return df

    def get_active_sprint(self, board_id: int | None = None) -> dict[str, Any] | None:
        """
        Get the currently active sprint.

        Args:
            board_id: Optional board ID filter

        Returns:
            Sprint info dictionary or None
        """
        board_filter = "AND board_id = ?" if board_id else ""
        params: list[Any] = []
        if board_id:
            params.append(board_id)

        query = f"""
        SELECT id, name, state, start_date, end_date, goal, board_id
        FROM sprints
        WHERE state = 'active'
        {board_filter}
        ORDER BY start_date DESC
        LIMIT 1
        """

        result = self._conn.execute(query, params).fetchone()

        if result:
            return {
                "id": result[0],
                "name": result[1],
                "state": result[2],
                "start_date": result[3],
                "end_date": result[4],
                "goal": result[5],
                "board_id": result[6],
            }
        return None

    def get_blocked_tickets(
        self,
        sprint_id: int | None = None,
        *,
        project_key: str | None = None,
    ) -> pd.DataFrame:
        """
        Get currently blocked tickets.

        Args:
            sprint_id: Optional sprint filter
            project_key: Optional project filter

        Returns:
            DataFrame with blocked tickets
        """
        conditions = ["status IN ('Blocked', 'On Hold', 'Waiting')"]
        params: list[Any] = []

        if sprint_id:
            conditions.append("sprint_id = ?")
            params.append(sprint_id)

        if project_key:
            conditions.append("project_key = ?")
            params.append(project_key)

        where_clause = " AND ".join(conditions)

        query = f"""
        SELECT
            key,
            summary,
            status,
            priority,
            assignee_id,
            created,
            updated,
            story_points,
            sprint_name
        FROM issues
        WHERE {where_clause}
        ORDER BY priority DESC, updated DESC
        """

        return self._conn.execute(query, params).df()

    def get_cycle_time_distribution(
        self,
        project_key: str | None = None,
        *,
        issue_type: str | None = None,
        days: int = 90,
    ) -> pd.DataFrame:
        """
        Get cycle time distribution for completed issues.

        Args:
            project_key: Optional project filter
            issue_type: Optional issue type filter
            days: Number of days to look back

        Returns:
            DataFrame with cycle time data
        """
        conditions = [
            "cycle_time_hours IS NOT NULL",
            "resolved IS NOT NULL",
            f"resolved >= CURRENT_DATE - INTERVAL '{days}' DAY",
        ]
        params: list[Any] = []

        if project_key:
            conditions.append("project_key = ?")
            params.append(project_key)

        if issue_type:
            conditions.append("issue_type = ?")
            params.append(issue_type)

        where_clause = " AND ".join(conditions)

        query = f"""
        SELECT
            key,
            summary,
            issue_type,
            priority,
            story_points,
            cycle_time_hours,
            cycle_time_hours / 24.0 as cycle_time_days,
            resolved
        FROM issues
        WHERE {where_clause}
        ORDER BY resolved DESC
        """

        return self._conn.execute(query, params).df()

    def get_issue_type_metrics(
        self,
        project_key: str | None = None,
        *,
        days: int = 90,
    ) -> pd.DataFrame:
        """
        Get metrics grouped by issue type.

        Args:
            project_key: Optional project filter
            days: Number of days to look back for resolved issues

        Returns:
            DataFrame with metrics by issue type
        """
        project_filter = "AND project_key = ?" if project_key else ""
        params: list[Any] = []
        if project_key:
            params.append(project_key)

        query = f"""
        SELECT
            issue_type,
            COUNT(*) as total_count,
            COUNT(CASE WHEN resolved IS NOT NULL THEN 1 END) as resolved_count,
            AVG(story_points) as avg_story_points,
            AVG(cycle_time_hours) as avg_cycle_time_hours,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cycle_time_hours) as median_cycle_time_hours,
            PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY cycle_time_hours) as p90_cycle_time_hours
        FROM issues
        WHERE 1=1 {project_filter}
        GROUP BY issue_type
        ORDER BY total_count DESC
        """

        return self._conn.execute(query, params).df()


def create_queries_from_settings() -> JiraQueries:
    """
    Create a JiraQueries instance from application settings.

    Returns:
        Configured JiraQueries instance
    """
    from src.data.schema import get_connection

    conn = get_connection()
    return JiraQueries(conn)
