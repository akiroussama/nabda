"""Tests for CLI application."""

import pytest
from typer.testing import CliRunner

from src.cli.app import app


@pytest.fixture
def runner():
    """Create CLI test runner."""
    return CliRunner()


class TestCLIApp:
    """Tests for main CLI application."""

    def test_app_help(self, runner):
        """Test app shows help."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Jira AI Co-pilot" in result.output

    def test_app_version(self, runner):
        """Test app shows version."""
        result = runner.invoke(app, ["--version"])

        assert result.exit_code == 0
        assert "v0.1.0" in result.output

    def test_app_no_args_shows_help(self, runner):
        """Test app with no args shows help."""
        result = runner.invoke(app, [])

        assert result.exit_code == 0
        assert "Usage:" in result.output

    def test_sync_command_group_exists(self, runner):
        """Test sync command group exists."""
        result = runner.invoke(app, ["sync", "--help"])

        assert result.exit_code == 0
        assert "Synchronize" in result.output

    def test_analyze_command_group_exists(self, runner):
        """Test analyze command group exists."""
        result = runner.invoke(app, ["analyze", "--help"])

        assert result.exit_code == 0
        assert "Analyze" in result.output

    def test_predict_command_group_exists(self, runner):
        """Test predict command group exists."""
        result = runner.invoke(app, ["predict", "--help"])

        assert result.exit_code == 0
        assert "predictions" in result.output.lower()

    def test_report_command_group_exists(self, runner):
        """Test report command group exists."""
        result = runner.invoke(app, ["report", "--help"])

        assert result.exit_code == 0
        assert "report" in result.output.lower()


class TestStatusCommand:
    """Tests for status command."""

    def test_status_help(self, runner):
        """Test status command help."""
        result = runner.invoke(app, ["status", "--help"])

        assert result.exit_code == 0
        assert "status" in result.output.lower()


class TestInitCommand:
    """Tests for init command."""

    def test_init_help(self, runner):
        """Test init command help."""
        result = runner.invoke(app, ["init", "--help"])

        assert result.exit_code == 0
        assert "Initialize" in result.output


class TestTrainCommand:
    """Tests for train command."""

    def test_train_help(self, runner):
        """Test train command help."""
        result = runner.invoke(app, ["train", "--help"])

        assert result.exit_code == 0
        assert "Train" in result.output


class TestDashboardCommand:
    """Tests for dashboard command."""

    def test_dashboard_help(self, runner):
        """Test dashboard command help."""
        result = runner.invoke(app, ["dashboard", "--help"])

        assert result.exit_code == 0
        assert "dashboard" in result.output.lower()


class TestSyncCommands:
    """Tests for sync subcommands."""

    def test_sync_full_help(self, runner):
        """Test sync full command help."""
        result = runner.invoke(app, ["sync", "full", "--help"])

        assert result.exit_code == 0
        assert "full" in result.output.lower()

    def test_sync_issues_help(self, runner):
        """Test sync issues command help."""
        result = runner.invoke(app, ["sync", "issues", "--help"])

        assert result.exit_code == 0
        assert "issues" in result.output.lower()

    def test_sync_sprints_help(self, runner):
        """Test sync sprints command help."""
        result = runner.invoke(app, ["sync", "sprints", "--help"])

        assert result.exit_code == 0
        assert "sprint" in result.output.lower()

    def test_sync_status_help(self, runner):
        """Test sync status command help."""
        result = runner.invoke(app, ["sync", "status", "--help"])

        assert result.exit_code == 0


class TestAnalyzeCommands:
    """Tests for analyze subcommands."""

    def test_analyze_sprint_help(self, runner):
        """Test analyze sprint command help."""
        result = runner.invoke(app, ["analyze", "sprint", "--help"])

        assert result.exit_code == 0
        assert "sprint" in result.output.lower()

    def test_analyze_workload_help(self, runner):
        """Test analyze workload command help."""
        result = runner.invoke(app, ["analyze", "workload", "--help"])

        assert result.exit_code == 0
        assert "workload" in result.output.lower()

    def test_analyze_blockers_help(self, runner):
        """Test analyze blockers command help."""
        result = runner.invoke(app, ["analyze", "blockers", "--help"])

        assert result.exit_code == 0
        assert "block" in result.output.lower()

    def test_analyze_velocity_help(self, runner):
        """Test analyze velocity command help."""
        result = runner.invoke(app, ["analyze", "velocity", "--help"])

        assert result.exit_code == 0
        assert "velocity" in result.output.lower()


class TestPredictCommands:
    """Tests for predict subcommands."""

    def test_predict_ticket_help(self, runner):
        """Test predict ticket command help."""
        result = runner.invoke(app, ["predict", "ticket", "--help"])

        assert result.exit_code == 0
        assert "ticket" in result.output.lower()

    def test_predict_risk_help(self, runner):
        """Test predict risk command help."""
        result = runner.invoke(app, ["predict", "risk", "--help"])

        assert result.exit_code == 0
        assert "risk" in result.output.lower()

    def test_predict_priorities_help(self, runner):
        """Test predict priorities command help."""
        result = runner.invoke(app, ["predict", "priorities", "--help"])

        assert result.exit_code == 0
        assert "priorit" in result.output.lower()

    def test_predict_batch_help(self, runner):
        """Test predict batch command help."""
        result = runner.invoke(app, ["predict", "batch", "--help"])

        assert result.exit_code == 0
        assert "batch" in result.output.lower()


class TestReportCommands:
    """Tests for report subcommands."""

    def test_report_standup_help(self, runner):
        """Test report standup command help."""
        result = runner.invoke(app, ["report", "standup", "--help"])

        assert result.exit_code == 0
        assert "standup" in result.output.lower()

    def test_report_sprint_health_help(self, runner):
        """Test report sprint-health command help."""
        result = runner.invoke(app, ["report", "sprint-health", "--help"])

        assert result.exit_code == 0
        assert "sprint" in result.output.lower()

    def test_report_workload_help(self, runner):
        """Test report workload command help."""
        result = runner.invoke(app, ["report", "workload", "--help"])

        assert result.exit_code == 0
        assert "workload" in result.output.lower()

    def test_report_export_help(self, runner):
        """Test report export command help."""
        result = runner.invoke(app, ["report", "export", "--help"])

        assert result.exit_code == 0
        assert "export" in result.output.lower()
