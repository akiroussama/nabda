"""Tests for intelligence orchestrator."""

import pytest
from datetime import datetime

from src.intelligence.orchestrator import (
    JiraIntelligence,
    TicketSummary,
    RiskExplanation,
    PrioritySuggestion,
    WorkloadAssessment,
)


class TestJiraIntelligence:
    """Tests for JiraIntelligence orchestrator."""

    @pytest.fixture
    def intel(self, tmp_path):
        """Create intelligence instance with mock client."""
        return JiraIntelligence(
            use_mock=True,
            templates_dir=str(tmp_path / "prompts"),
        )

    @pytest.fixture
    def sample_ticket(self):
        """Sample ticket data for testing."""
        return {
            "key": "PROJ-123",
            "summary": "Implement user authentication",
            "description": "Add login and registration functionality",
            "issue_type": "Story",
            "status": "In Progress",
            "priority": "High",
            "assignee": "dev_001",
            "comments": [
                {"author": "dev_001", "body": "Started working on this"},
            ],
            "changelog": [
                {"field": "status", "from_value": "To Do", "to_value": "In Progress"},
            ],
        }

    @pytest.fixture
    def sample_sprint(self, sample_sprint_features):
        """Use sprint features fixture from conftest."""
        return sample_sprint_features

    @pytest.fixture
    def sample_risk_score(self):
        """Sample risk score for testing."""
        return {
            "score": 65.0,
            "level": "medium",
            "factors": {
                "completion_gap": {"contribution": 0.15},
                "velocity_ratio": {"contribution": 0.10},
            },
        }

    def test_init_with_mock(self, intel):
        """Test initialization with mock client."""
        assert intel is not None
        stats = intel.get_stats()
        assert stats["llm"]["model"] == "mock"

    def test_summarize_ticket(self, intel, sample_ticket):
        """Test ticket summarization."""
        result = intel.summarize_ticket(sample_ticket)

        assert isinstance(result, TicketSummary)
        assert result.issue_key == "PROJ-123"
        assert result.summary is not None
        assert result.generated_at is not None

    def test_summarize_ticket_to_dict(self, intel, sample_ticket):
        """Test ticket summary serialization."""
        result = intel.summarize_ticket(sample_ticket)
        result_dict = result.to_dict()

        assert "issue_key" in result_dict
        assert "summary" in result_dict
        assert "status_assessment" in result_dict
        assert "generated_at" in result_dict

    def test_explain_sprint_risk(self, intel, sample_sprint, sample_risk_score):
        """Test sprint risk explanation."""
        result = intel.explain_sprint_risk(sample_sprint, sample_risk_score)

        assert isinstance(result, RiskExplanation)
        assert result.sprint_id == sample_sprint["sprint_id"]
        assert result.risk_score == 65.0
        assert result.generated_at is not None

    def test_explain_sprint_risk_to_dict(self, intel, sample_sprint, sample_risk_score):
        """Test risk explanation serialization."""
        result = intel.explain_sprint_risk(sample_sprint, sample_risk_score)
        result_dict = result.to_dict()

        assert "sprint_id" in result_dict
        assert "risk_summary" in result_dict
        assert "recommended_actions" in result_dict
        assert "outlook" in result_dict

    def test_suggest_priorities(self, intel, sample_sprint):
        """Test priority suggestions."""
        tickets = [
            {"key": "PROJ-1", "summary": "Task 1", "status": "In Progress", "issue_type": "Story", "priority": "High", "story_points": 5},
            {"key": "PROJ-2", "summary": "Task 2", "status": "To Do", "issue_type": "Bug", "priority": "Highest", "story_points": 2},
        ]

        result = intel.suggest_priorities(sample_sprint, tickets)

        assert isinstance(result, PrioritySuggestion)
        assert result.sprint_id == sample_sprint["sprint_id"]
        assert result.generated_at is not None

    def test_suggest_priorities_to_dict(self, intel, sample_sprint):
        """Test priority suggestions serialization."""
        tickets = [
            {"key": "PROJ-1", "summary": "Task 1", "status": "In Progress", "issue_type": "Story", "priority": "High", "story_points": 5},
        ]

        result = intel.suggest_priorities(sample_sprint, tickets)
        result_dict = result.to_dict()

        assert "must_complete" in result_dict
        assert "consider_deferring" in result_dict
        assert "focus_recommendation" in result_dict

    def test_assess_developer_workload(self, intel, sample_developer_features):
        """Test developer workload assessment."""
        # Add required fields for assessment
        developer = {
            **sample_developer_features,
            "status": "high",
            "score": 120,
            "relative_to_team": 1.2,
            "team_avg_wip_points": 12,
            "team_avg_completed_points": 20,
        }

        result = intel.assess_developer_workload(developer)

        assert isinstance(result, WorkloadAssessment)
        assert result.developer_id == developer["assignee_id"]
        assert result.generated_at is not None

    def test_assess_developer_workload_to_dict(self, intel, sample_developer_features):
        """Test workload assessment serialization."""
        developer = {
            **sample_developer_features,
            "status": "optimal",
            "score": 80,
            "relative_to_team": 0.9,
            "team_avg_wip_points": 15,
            "team_avg_completed_points": 25,
        }

        result = intel.assess_developer_workload(developer)
        result_dict = result.to_dict()

        assert "developer_id" in result_dict
        assert "assessment" in result_dict
        assert "recommendations" in result_dict

    def test_generate_standup_summary(self, intel, sample_sprint):
        """Test standup summary generation."""
        tickets = [
            {"key": "PROJ-1", "summary": "Task 1", "status": "In Progress", "is_blocked": False},
            {"key": "PROJ-2", "summary": "Blocked task", "status": "Blocked", "is_blocked": True},
        ]

        result = intel.generate_standup_summary(sample_sprint, tickets)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_stats(self, intel, sample_ticket):
        """Test getting intelligence statistics."""
        # Make some calls
        intel.summarize_ticket(sample_ticket)

        stats = intel.get_stats()

        assert "llm" in stats
        assert "templates" in stats
        assert stats["llm"]["total_requests"] > 0

    def test_clear_cache(self, intel):
        """Test clearing cache."""
        # Should not raise
        intel.clear_cache()


class TestDataClasses:
    """Tests for intelligence data classes."""

    def test_ticket_summary_creation(self):
        """Test TicketSummary creation."""
        summary = TicketSummary(
            issue_key="PROJ-1",
            summary="Test summary",
            status_assessment="on_track",
            next_action="Continue work",
            key_blocker=None,
            generated_at=datetime.now(),
        )

        assert summary.issue_key == "PROJ-1"
        assert summary.status_assessment == "on_track"
        assert summary.key_blocker is None

    def test_risk_explanation_creation(self):
        """Test RiskExplanation creation."""
        explanation = RiskExplanation(
            sprint_id=1,
            sprint_name="Sprint 1",
            risk_score=45.0,
            risk_level="medium",
            risk_summary="Moderate risk",
            main_concerns=["Behind schedule"],
            root_causes=["Scope creep"],
            recommended_actions=[{"action": "Reduce scope", "priority": "high"}],
            outlook="stable",
            generated_at=datetime.now(),
        )

        assert explanation.sprint_id == 1
        assert explanation.risk_score == 45.0
        assert len(explanation.main_concerns) == 1

    def test_priority_suggestion_creation(self):
        """Test PrioritySuggestion creation."""
        suggestion = PrioritySuggestion(
            sprint_id=1,
            sprint_name="Sprint 1",
            must_complete=["PROJ-1", "PROJ-2"],
            consider_deferring=["PROJ-3"],
            reorder_suggestions=[],
            focus_recommendation="Focus on must-complete items",
            risk_if_unchanged="Sprint may fail",
            generated_at=datetime.now(),
        )

        assert suggestion.sprint_id == 1
        assert len(suggestion.must_complete) == 2
        assert len(suggestion.consider_deferring) == 1

    def test_workload_assessment_creation(self):
        """Test WorkloadAssessment creation."""
        assessment = WorkloadAssessment(
            developer_id="user-001",
            developer_name="dev_001",
            assessment="sustainable",
            summary="Workload is manageable",
            immediate_concerns=[],
            recommendations=["Keep current pace"],
            suggested_actions=[],
            generated_at=datetime.now(),
        )

        assert assessment.developer_id == "user-001"
        assert assessment.assessment == "sustainable"
        assert len(assessment.immediate_concerns) == 0
