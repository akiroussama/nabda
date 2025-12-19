"""Tests for CLI formatters."""

import pytest
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.cli.formatters import (
    get_risk_style,
    get_risk_emoji,
    get_priority_style,
    get_status_style,
    format_duration,
    format_percentage,
    create_issues_table,
    create_sprint_summary_panel,
    create_developer_table,
    create_prediction_panel,
    create_recommendations_panel,
    create_metrics_row,
)


class TestRiskFormatters:
    """Tests for risk formatting functions."""

    def test_get_risk_style_low(self):
        """Test low risk style."""
        assert get_risk_style("low") == "green"
        assert get_risk_style("LOW") == "green"

    def test_get_risk_style_medium(self):
        """Test medium risk style."""
        assert get_risk_style("medium") == "yellow"

    def test_get_risk_style_high(self):
        """Test high risk style."""
        assert get_risk_style("high") == "red bold"

    def test_get_risk_style_unknown(self):
        """Test unknown risk style."""
        assert get_risk_style("unknown") == "white"

    def test_get_risk_emoji_low(self):
        """Test low risk emoji."""
        assert get_risk_emoji("low") == "🟢"

    def test_get_risk_emoji_medium(self):
        """Test medium risk emoji."""
        assert get_risk_emoji("medium") == "🟡"

    def test_get_risk_emoji_high(self):
        """Test high risk emoji."""
        assert get_risk_emoji("high") == "🔴"


class TestPriorityFormatters:
    """Tests for priority formatting."""

    def test_get_priority_style_highest(self):
        """Test highest priority style."""
        assert get_priority_style("highest") == "red bold"

    def test_get_priority_style_high(self):
        """Test high priority style."""
        assert get_priority_style("high") == "red"

    def test_get_priority_style_medium(self):
        """Test medium priority style."""
        assert get_priority_style("medium") == "yellow"

    def test_get_priority_style_low(self):
        """Test low priority style."""
        assert get_priority_style("low") == "green"

    def test_get_priority_style_lowest(self):
        """Test lowest priority style."""
        assert get_priority_style("lowest") == "dim"


class TestStatusFormatters:
    """Tests for status formatting."""

    def test_get_status_style_done(self):
        """Test done status style."""
        assert get_status_style("Done") == "green"
        assert get_status_style("Closed") == "green"
        assert get_status_style("Resolved") == "green"

    def test_get_status_style_in_progress(self):
        """Test in progress status style."""
        assert get_status_style("In Progress") == "cyan"
        assert get_status_style("In Review") == "cyan"

    def test_get_status_style_blocked(self):
        """Test blocked status style."""
        assert get_status_style("Blocked") == "red"
        assert get_status_style("On Hold") == "red"

    def test_get_status_style_to_do(self):
        """Test to do status style."""
        assert get_status_style("To Do") == "dim"
        assert get_status_style("Open") == "dim"


class TestDurationFormatters:
    """Tests for duration formatting."""

    def test_format_duration_minutes(self):
        """Test formatting minutes."""
        assert format_duration(0.5) == "30m"
        assert format_duration(0.25) == "15m"

    def test_format_duration_hours(self):
        """Test formatting hours."""
        assert format_duration(2.5) == "2.5h"
        assert format_duration(8.0) == "8.0h"

    def test_format_duration_days(self):
        """Test formatting days."""
        assert format_duration(48.0) == "2.0d"
        assert format_duration(24.0) == "1.0d"

    def test_format_percentage(self):
        """Test percentage formatting."""
        assert format_percentage(50.0) == "50.0%"
        assert format_percentage(33.333, 2) == "33.33%"


class TestTableCreation:
    """Tests for table creation functions."""

    def test_create_issues_table(self):
        """Test creating issues table."""
        issues = [
            {
                "key": "PROJ-1",
                "issue_type": "Story",
                "status": "In Progress",
                "priority": "High",
                "summary": "Test issue",
                "story_points": 5,
                "assignee": "dev_001",
            },
            {
                "key": "PROJ-2",
                "issue_type": "Bug",
                "status": "To Do",
                "priority": "Medium",
                "summary": "Another issue",
                "story_points": 3,
                "assignee": None,
            },
        ]

        table = create_issues_table(issues)

        assert isinstance(table, Table)
        assert table.title == "Issues"
        assert table.row_count == 2

    def test_create_issues_table_custom_title(self):
        """Test creating issues table with custom title."""
        issues = [{"key": "PROJ-1", "status": "Done"}]
        table = create_issues_table(issues, title="Sprint Issues")

        assert table.title == "Sprint Issues"

    def test_create_developer_table(self):
        """Test creating developer table."""
        developers = [
            {
                "pseudonym": "dev_001",
                "wip_count": 3,
                "wip_points": 8,
                "blocked_count": 1,
                "workload_relative": 1.2,
                "status": "high",
            },
            {
                "assignee_id": "user-123",
                "wip_count": 2,
                "wip_points": 5,
                "blocked_count": 0,
                "workload_relative": 0.8,
                "status": "optimal",
            },
        ]

        table = create_developer_table(developers)

        assert isinstance(table, Table)
        assert table.row_count == 2


class TestPanelCreation:
    """Tests for panel creation functions."""

    def test_create_sprint_summary_panel(self):
        """Test creating sprint summary panel."""
        sprint = {
            "sprint_name": "Sprint 1",
            "days_elapsed": 5,
            "total_days": 10,
            "progress_percent": 50,
            "completed_points": 20,
            "total_points": 40,
            "completion_rate": 50,
            "remaining_points": 20,
            "days_remaining": 5,
        }

        risk_score = {
            "score": 45.0,
            "level": "medium",
        }

        panel = create_sprint_summary_panel(sprint, risk_score)

        assert isinstance(panel, Panel)
        assert "Sprint Health" in panel.title

    def test_create_sprint_summary_panel_no_risk(self):
        """Test creating sprint summary panel without risk."""
        sprint = {
            "sprint_name": "Sprint 1",
            "days_elapsed": 5,
            "total_days": 10,
            "progress_percent": 50,
            "completed_points": 20,
            "total_points": 40,
            "completion_rate": 50,
            "remaining_points": 20,
            "days_remaining": 5,
        }

        panel = create_sprint_summary_panel(sprint)

        assert isinstance(panel, Panel)

    def test_create_prediction_panel(self):
        """Test creating prediction panel."""
        prediction = {
            "issue_key": "PROJ-123",
            "predicted_hours": 16.5,
            "predicted_days": 2.0,
            "confidence_interval": {
                "lower_hours": 12.0,
                "upper_hours": 24.0,
            },
            "model_type": "lightgbm",
        }

        panel = create_prediction_panel(prediction)

        assert isinstance(panel, Panel)
        assert "Duration Prediction" in panel.title

    def test_create_recommendations_panel(self):
        """Test creating recommendations panel."""
        recommendations = [
            {
                "action": "Reduce WIP",
                "rationale": "Too many items in progress",
                "priority": "high",
            },
            {
                "action": "Clear blockers",
                "rationale": "Blocked items are slowing progress",
                "priority": "medium",
            },
        ]

        panel = create_recommendations_panel(recommendations)

        assert isinstance(panel, Panel)
        assert "Recommendations" in panel.title


class TestMetricsRow:
    """Tests for metrics row creation."""

    def test_create_metrics_row(self):
        """Test creating metrics row."""
        metrics = {
            "completion": 75.5,
            "velocity": 42,
            "risk": "medium",
        }

        table = create_metrics_row(metrics)

        assert isinstance(table, Table)
