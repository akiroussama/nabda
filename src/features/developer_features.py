"""
Developer feature extraction module.

Extracts workload and performance metrics for developers.
"""

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from loguru import logger


class DeveloperFeatureExtractor:
    """
    Extracts workload and performance features for developers.

    Computes metrics like WIP count, cycle time averages, worklog hours,
    and identifies developers at risk of overload.

    Example:
        >>> extractor = DeveloperFeatureExtractor(conn)
        >>> df = extractor.extract_all_developers()
        >>> at_risk = extractor.get_at_risk_developers()
    """

    # Thresholds for risk detection
    DEFAULT_OVERLOAD_THRESHOLD = 1.3  # 130% of team average
    DEFAULT_UNDERLOAD_THRESHOLD = 0.7  # 70% of team average
    DEFAULT_HIGH_WIP_THRESHOLD = 5

    def __init__(
        self,
        conn,
        *,
        overload_threshold: float = DEFAULT_OVERLOAD_THRESHOLD,
        underload_threshold: float = DEFAULT_UNDERLOAD_THRESHOLD,
        high_wip_threshold: int = DEFAULT_HIGH_WIP_THRESHOLD,
    ):
        """
        Initialize the developer feature extractor.

        Args:
            conn: DuckDB connection
            overload_threshold: Ratio above team average for overload flag
            underload_threshold: Ratio below team average for underload flag
            high_wip_threshold: WIP count considered high
        """
        self._conn = conn
        self._overload_threshold = overload_threshold
        self._underload_threshold = underload_threshold
        self._high_wip_threshold = high_wip_threshold

    def extract_all_developers(
        self,
        project_key: str | None = None,
        *,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Extract features for all developers.

        Args:
            project_key: Optional project filter
            days: Number of days for rolling metrics

        Returns:
            DataFrame with developer features
        """
        since_date = datetime.now() - timedelta(days=days)
        short_window = datetime.now() - timedelta(days=7)

        project_filter = "AND i.project_key = ?" if project_key else ""
        params: list[Any] = [since_date]
        if project_key:
            params.append(project_key)

        query = f"""
        WITH developer_base AS (
            SELECT DISTINCT assignee_id
            FROM issues
            WHERE assignee_id IS NOT NULL
            {project_filter.replace('i.', '')}
        ),
        completed_metrics AS (
            SELECT
                assignee_id,
                COUNT(*) as completed_count_{days}d,
                COALESCE(SUM(story_points), 0) as completed_points_{days}d,
                AVG(cycle_time_hours) as avg_cycle_time_{days}d,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY cycle_time_hours) as median_cycle_time_{days}d
            FROM issues i
            WHERE resolved IS NOT NULL
              AND resolved >= ?
              AND assignee_id IS NOT NULL
              AND cycle_time_hours IS NOT NULL
              {project_filter}
            GROUP BY assignee_id
        ),
        wip_metrics AS (
            SELECT
                assignee_id,
                COUNT(*) as wip_count,
                COALESCE(SUM(story_points), 0) as wip_points
            FROM issues i
            WHERE resolved IS NULL
              AND status NOT IN ('To Do', 'Backlog', 'Open', 'New')
              AND assignee_id IS NOT NULL
              {project_filter}
            GROUP BY assignee_id
        ),
        assigned_metrics AS (
            SELECT
                assignee_id,
                COUNT(*) as total_assigned,
                COUNT(CASE WHEN resolved IS NULL THEN 1 END) as unresolved_count,
                COUNT(CASE WHEN status IN ('Blocked', 'On Hold', 'Waiting') THEN 1 END) as blocked_count
            FROM issues i
            WHERE assignee_id IS NOT NULL
              {project_filter}
            GROUP BY assignee_id
        ),
        worklog_metrics AS (
            SELECT
                author_id as assignee_id,
                SUM(time_spent_seconds) / 3600.0 as hours_logged_{days}d,
                SUM(CASE WHEN started >= ? THEN time_spent_seconds ELSE 0 END) / 3600.0 as hours_logged_7d,
                COUNT(DISTINCT DATE_TRUNC('day', started)) as active_days
            FROM worklogs
            WHERE started >= ?
            GROUP BY author_id
        ),
        overdue_metrics AS (
            SELECT
                assignee_id,
                COUNT(*) as overdue_count
            FROM issues i
            WHERE resolved IS NULL
              AND assignee_id IS NOT NULL
              AND updated < CURRENT_DATE - INTERVAL '7' DAY
              AND status NOT IN ('Done', 'Closed', 'Resolved', 'To Do', 'Backlog')
              {project_filter}
            GROUP BY assignee_id
        )
        SELECT
            db.assignee_id,
            u.pseudonym,
            COALESCE(cm.completed_count_{days}d, 0) as completed_count_{days}d,
            COALESCE(cm.completed_points_{days}d, 0) as completed_points_{days}d,
            COALESCE(cm.avg_cycle_time_{days}d, 0) as avg_cycle_time_{days}d,
            COALESCE(cm.median_cycle_time_{days}d, 0) as median_cycle_time_{days}d,
            COALESCE(wm.wip_count, 0) as wip_count,
            COALESCE(wm.wip_points, 0) as wip_points,
            COALESCE(am.total_assigned, 0) as total_assigned,
            COALESCE(am.unresolved_count, 0) as unresolved_count,
            COALESCE(am.blocked_count, 0) as blocked_count,
            COALESCE(wlm.hours_logged_{days}d, 0) as hours_logged_{days}d,
            COALESCE(wlm.hours_logged_7d, 0) as hours_logged_7d,
            COALESCE(wlm.active_days, 0) as active_days,
            COALESCE(om.overdue_count, 0) as overdue_count
        FROM developer_base db
        LEFT JOIN users u ON db.assignee_id = u.account_id
        LEFT JOIN completed_metrics cm ON db.assignee_id = cm.assignee_id
        LEFT JOIN wip_metrics wm ON db.assignee_id = wm.assignee_id
        LEFT JOIN assigned_metrics am ON db.assignee_id = am.assignee_id
        LEFT JOIN worklog_metrics wlm ON db.assignee_id = wlm.assignee_id
        LEFT JOIN overdue_metrics om ON db.assignee_id = om.assignee_id
        ORDER BY wip_points DESC
        """

        # Build params
        full_params = [since_date]
        if project_key:
            full_params.append(project_key)
        full_params.extend([short_window, since_date])
        if project_key:
            full_params.append(project_key)

        df = self._conn.execute(query, full_params).df()

        if df.empty:
            logger.warning("No developer data found")
            return df

        # Compute derived metrics
        df = self._compute_derived_metrics(df, days)

        logger.info(f"Extracted features for {len(df)} developers")
        return df

    def _compute_derived_metrics(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        """Compute derived metrics and risk flags."""
        # Completion rate
        df["completion_rate"] = df.apply(
            lambda r: r[f"completed_count_{days}d"] / r["total_assigned"]
            if r["total_assigned"] > 0 else 0,
            axis=1
        )

        # Compute team averages
        team_avg_wip = df["wip_points"].mean() if len(df) > 0 else 0
        team_avg_completed = df[f"completed_points_{days}d"].mean() if len(df) > 0 else 0
        team_avg_hours = df[f"hours_logged_{days}d"].mean() if len(df) > 0 else 0

        # Workload score (composite)
        df["workload_score"] = (
            df["wip_points"] * 0.35 +
            df[f"hours_logged_{days}d"] * 0.25 +
            df["unresolved_count"] * 0.25 +
            df["overdue_count"] * 0.15
        )

        # Relative workload (compared to team average)
        team_avg_workload = df["workload_score"].mean()
        if team_avg_workload > 0:
            df["workload_relative"] = df["workload_score"] / team_avg_workload
        else:
            df["workload_relative"] = 1.0

        # Risk flags
        df["is_overloaded"] = (
            (df["workload_relative"] > self._overload_threshold) |
            (df["wip_count"] > self._high_wip_threshold)
        ).astype(int)

        df["is_underloaded"] = (
            (df["workload_relative"] < self._underload_threshold) &
            (df["wip_count"] < 2)
        ).astype(int)

        df["at_risk"] = (
            (df["is_overloaded"] == 1) |
            (df["overdue_count"] > 2) |
            (df["blocked_count"] > 2)
        ).astype(int)

        # Team averages for reference
        df["team_avg_wip_points"] = team_avg_wip
        df["team_avg_completed_points"] = team_avg_completed
        df["team_avg_hours_logged"] = team_avg_hours

        return df

    def get_at_risk_developers(
        self,
        project_key: str | None = None,
        *,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get developers flagged as at-risk.

        Args:
            project_key: Optional project filter
            days: Days for metric calculation

        Returns:
            DataFrame with at-risk developers
        """
        df = self.extract_all_developers(project_key, days=days)
        return df[df["at_risk"] == 1].copy()

    def get_overloaded_developers(
        self,
        project_key: str | None = None,
        *,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get overloaded developers.

        Args:
            project_key: Optional project filter
            days: Days for metric calculation

        Returns:
            DataFrame with overloaded developers
        """
        df = self.extract_all_developers(project_key, days=days)
        return df[df["is_overloaded"] == 1].copy()

    def get_available_developers(
        self,
        project_key: str | None = None,
        *,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Get developers with capacity for more work.

        Args:
            project_key: Optional project filter
            days: Days for metric calculation

        Returns:
            DataFrame with available developers, sorted by availability
        """
        df = self.extract_all_developers(project_key, days=days)

        # Filter to those not overloaded
        available = df[df["is_overloaded"] == 0].copy()

        # Sort by workload (lowest first)
        available = available.sort_values("workload_relative", ascending=True)

        return available

    def get_developer_summary(self, assignee_id: str) -> dict[str, Any] | None:
        """
        Get detailed summary for a single developer.

        Args:
            assignee_id: Developer's pseudonymized ID

        Returns:
            Dictionary with developer summary or None
        """
        df = self.extract_all_developers()

        dev = df[df["assignee_id"] == assignee_id]
        if dev.empty:
            return None

        row = dev.iloc[0]
        return row.to_dict()

    def get_team_health_score(
        self,
        project_key: str | None = None,
        *,
        days: int = 30,
    ) -> dict[str, Any]:
        """
        Calculate overall team health metrics.

        Args:
            project_key: Optional project filter
            days: Days for metric calculation

        Returns:
            Dictionary with team health metrics
        """
        df = self.extract_all_developers(project_key, days=days)

        if df.empty:
            return {
                "total_developers": 0,
                "health_score": 0,
                "overloaded_count": 0,
                "underloaded_count": 0,
                "at_risk_count": 0,
            }

        total = len(df)
        overloaded = df["is_overloaded"].sum()
        underloaded = df["is_underloaded"].sum()
        at_risk = df["at_risk"].sum()

        # Health score: 100 - (overloaded% * 40) - (at_risk% * 30) - (underloaded% * 10)
        health_score = max(0, 100 - (
            (overloaded / total) * 40 +
            (at_risk / total) * 30 +
            (underloaded / total) * 10
        ) * 100)

        return {
            "total_developers": total,
            "health_score": round(health_score, 1),
            "overloaded_count": int(overloaded),
            "overloaded_percent": round(overloaded / total * 100, 1),
            "underloaded_count": int(underloaded),
            "underloaded_percent": round(underloaded / total * 100, 1),
            "at_risk_count": int(at_risk),
            "at_risk_percent": round(at_risk / total * 100, 1),
            "avg_wip_count": round(df["wip_count"].mean(), 1),
            "avg_workload_score": round(df["workload_score"].mean(), 1),
            "total_wip_points": round(df["wip_points"].sum(), 1),
        }
