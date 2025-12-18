"""Tests for workload scoring model."""

import numpy as np
import pandas as pd
import pytest

from src.models.workload_scorer import WorkloadScorer


class TestWorkloadScorer:
    """Tests for WorkloadScorer model."""

    def test_init_default_weights(self):
        """Test default weights are set."""
        scorer = WorkloadScorer()

        assert "wip_points" in scorer.weights
        assert "hours_logged" in scorer.weights
        assert sum(scorer.weights.values()) == pytest.approx(1.0)

    def test_init_custom_weights(self):
        """Test custom weights."""
        custom_weights = {"wip_points": 0.5, "hours_logged": 0.5}
        scorer = WorkloadScorer(weights=custom_weights)

        assert scorer.weights["wip_points"] == 0.5
        assert scorer.weights["hours_logged"] == 0.5

    def test_set_team_baselines(self):
        """Test setting team baselines."""
        scorer = WorkloadScorer()
        df = pd.DataFrame({
            "wip_points": [10, 20, 30],
            "hours_logged_30d": [40, 50, 60],
            "unresolved_count": [3, 5, 7],
            "blocked_count": [1, 1, 2],
            "overdue_count": [0, 1, 1],
        })

        scorer.set_team_baselines(df)

        assert scorer._team_baselines is not None
        assert scorer._team_baselines["avg_wip_points"] == 20.0
        assert scorer._team_baselines["team_size"] == 3

    def test_score_returns_dict(self, sample_developer_features):
        """Test that score returns a dictionary."""
        scorer = WorkloadScorer()

        result = scorer.score(sample_developer_features)

        assert isinstance(result, dict)
        assert "score" in result
        assert "status" in result
        assert "factors" in result

    def test_score_range(self, sample_developer_features):
        """Test that score is in valid range."""
        scorer = WorkloadScorer()

        result = scorer.score(sample_developer_features)

        assert result["score"] >= 0

    def test_score_status_underloaded(self):
        """Test underloaded status detection."""
        scorer = WorkloadScorer()
        features = {
            "wip_points": 5,  # Low
            "hours_logged_30d": 10,  # Low
            "unresolved_count": 1,
            "blocked_count": 0,
            "overdue_count": 0,
        }

        result = scorer.score(features)

        assert result["status"] == "underloaded"

    def test_score_status_overloaded(self):
        """Test overloaded status detection."""
        scorer = WorkloadScorer()
        # Set high baselines so normal values seem low
        scorer._team_baselines = {
            "avg_wip_points": 10,
            "avg_hours_logged": 20,
            "avg_unresolved": 2,
            "avg_blocked": 0.5,
            "avg_overdue": 0.5,
        }
        features = {
            "wip_points": 40,  # 4x average
            "hours_logged_30d": 80,  # 4x average
            "unresolved_count": 8,
            "blocked_count": 2,
            "overdue_count": 2,
        }

        result = scorer.score(features)

        assert result["status"] == "overloaded"

    def test_score_contains_factors(self, sample_developer_features):
        """Test that score contains factor breakdown."""
        scorer = WorkloadScorer()

        result = scorer.score(sample_developer_features)

        assert "factors" in result
        assert "wip_points" in result["factors"]
        assert "hours_logged" in result["factors"]

    def test_factor_structure(self, sample_developer_features):
        """Test factor structure."""
        scorer = WorkloadScorer()

        result = scorer.score(sample_developer_features)

        factor = result["factors"]["wip_points"]
        assert "value" in factor
        assert "team_average" in factor
        assert "ratio" in factor
        assert "normalized" in factor

    def test_score_generates_recommendations(self):
        """Test that score generates recommendations for overloaded."""
        scorer = WorkloadScorer()
        scorer._team_baselines = {
            "avg_wip_points": 10,
            "avg_hours_logged": 20,
            "avg_unresolved": 2,
            "avg_blocked": 0.5,
            "avg_overdue": 0.5,
        }
        features = {
            "wip_points": 40,
            "hours_logged_30d": 80,
            "unresolved_count": 8,
            "blocked_count": 3,  # High blocked
            "overdue_count": 2,  # Overdue items
        }

        result = scorer.score(features)

        assert "recommendations" in result
        if result["status"] == "overloaded":
            assert len(result["recommendations"]) > 0

    def test_score_team_returns_dict(self):
        """Test that score_team returns dictionary."""
        scorer = WorkloadScorer()
        df = pd.DataFrame({
            "assignee_id": ["user-001", "user-002", "user-003"],
            "pseudonym": ["dev_001", "dev_002", "dev_003"],
            "wip_points": [10, 20, 30],
            "hours_logged_30d": [40, 50, 60],
            "unresolved_count": [3, 5, 7],
            "blocked_count": [1, 1, 2],
            "overdue_count": [0, 1, 1],
        })

        result = scorer.score_team(df)

        assert isinstance(result, dict)
        assert "team_size" in result
        assert "distribution" in result
        assert "balance_score" in result

    def test_score_team_calculates_balance(self):
        """Test that score_team calculates balance score."""
        scorer = WorkloadScorer()
        # Balanced team
        df = pd.DataFrame({
            "assignee_id": ["user-001", "user-002", "user-003"],
            "pseudonym": ["dev_001", "dev_002", "dev_003"],
            "wip_points": [15, 15, 15],
            "hours_logged_30d": [40, 40, 40],
            "unresolved_count": [5, 5, 5],
            "blocked_count": [1, 1, 1],
            "overdue_count": [0, 0, 0],
        })

        result = scorer.score_team(df)

        # Perfectly balanced team should have high balance score
        assert result["balance_score"] > 50

    def test_score_team_individual_scores(self):
        """Test that score_team returns individual scores."""
        scorer = WorkloadScorer()
        df = pd.DataFrame({
            "assignee_id": ["user-001", "user-002"],
            "pseudonym": ["dev_001", "dev_002"],
            "wip_points": [10, 20],
            "hours_logged_30d": [40, 50],
            "unresolved_count": [3, 5],
            "blocked_count": [1, 1],
            "overdue_count": [0, 1],
        })

        result = scorer.score_team(df)

        assert "individual_scores" in result
        assert len(result["individual_scores"]) == 2

    def test_identify_reassignment_candidates(self):
        """Test identifying reassignment candidates."""
        scorer = WorkloadScorer()
        # Create unbalanced team
        df = pd.DataFrame({
            "assignee_id": ["user-001", "user-002", "user-003"],
            "pseudonym": ["dev_001", "dev_002", "dev_003"],
            "wip_points": [5, 15, 50],  # Last one overloaded
            "hours_logged_30d": [20, 40, 100],
            "unresolved_count": [2, 5, 15],
            "blocked_count": [0, 1, 3],
            "overdue_count": [0, 0, 2],
        })

        result = scorer.identify_reassignment_candidates(df)

        assert "give" in result
        assert "receive" in result
        # user-003 should be in give (overloaded)
        # user-001 should be in receive (capacity available)

    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        scorer = WorkloadScorer()
        df = pd.DataFrame()

        result = scorer.score_team(df)

        assert result["team_size"] == 0
        assert result["balance_score"] == 0
