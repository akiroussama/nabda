"""Tests for sprint risk scoring model."""

import numpy as np
import pandas as pd
import pytest

from src.models.sprint_risk import SprintRiskScorer


class TestSprintRiskScorer:
    """Tests for SprintRiskScorer model."""

    def test_init_default_mode(self):
        """Test default mode is hybrid."""
        scorer = SprintRiskScorer()

        assert scorer.mode == "hybrid"

    def test_init_custom_mode(self):
        """Test custom mode."""
        scorer = SprintRiskScorer(mode="rule_based")

        assert scorer.mode == "rule_based"

    def test_init_custom_weights(self):
        """Test custom weights."""
        custom_weights = {"completion_gap": 0.5, "velocity_ratio": 0.5}
        scorer = SprintRiskScorer(weights=custom_weights)

        assert scorer.weights["completion_gap"] == 0.5
        assert scorer.weights["velocity_ratio"] == 0.5

    def test_score_returns_dict(self, sample_sprint_features):
        """Test that score returns a dictionary."""
        scorer = SprintRiskScorer(mode="rule_based")

        result = scorer.score(sample_sprint_features)

        assert isinstance(result, dict)
        assert "score" in result
        assert "level" in result
        assert "factors" in result

    def test_score_range(self, sample_sprint_features):
        """Test that score is in valid range."""
        scorer = SprintRiskScorer(mode="rule_based")

        result = scorer.score(sample_sprint_features)

        assert 0 <= result["score"] <= 100

    def test_score_level_low(self):
        """Test low risk level detection."""
        scorer = SprintRiskScorer(mode="rule_based")
        features = {
            "completion_gap": 10,  # Ahead of schedule
            "velocity_ratio": 1.2,  # Good velocity
            "blocked_ratio": 0,
            "scope_creep_ratio": 0,
            "days_remaining": 10,
            "remaining_points": 5,
            "total_points": 30,
        }

        result = scorer.score(features)

        assert result["level"] == "low"

    def test_score_level_high(self):
        """Test high risk level detection."""
        scorer = SprintRiskScorer(mode="rule_based")
        features = {
            "completion_gap": -30,  # Behind schedule
            "velocity_ratio": 0.4,  # Low velocity
            "blocked_ratio": 0.25,  # Many blocked
            "scope_creep_ratio": 0.3,  # Scope creep
            "days_remaining": 2,  # Little time left
            "remaining_points": 20,
            "total_points": 30,
        }

        result = scorer.score(features)

        assert result["level"] == "high"

    def test_score_contains_factors(self, sample_sprint_features):
        """Test that score contains factor breakdown."""
        scorer = SprintRiskScorer(mode="rule_based")

        result = scorer.score(sample_sprint_features)

        assert "factors" in result
        assert "completion_gap" in result["factors"]
        assert "velocity_ratio" in result["factors"]

    def test_factor_structure(self, sample_sprint_features):
        """Test factor structure."""
        scorer = SprintRiskScorer(mode="rule_based")

        result = scorer.score(sample_sprint_features)

        factor = result["factors"]["completion_gap"]
        assert "value" in factor
        assert "factor" in factor
        assert "weight" in factor
        assert "contribution" in factor

    def test_score_generates_recommendations(self, sample_sprint_features):
        """Test that score generates recommendations."""
        scorer = SprintRiskScorer(mode="rule_based")
        # Create high risk scenario
        features = sample_sprint_features.copy()
        features["completion_gap"] = -25
        features["blocked_ratio"] = 0.2
        features["blocked_issues"] = 3

        result = scorer.score(features)

        assert "recommendations" in result
        # High risk should have recommendations
        if result["level"] == "high":
            assert len(result["recommendations"]) > 0

    def test_recommendation_structure(self, sample_sprint_features):
        """Test recommendation structure."""
        scorer = SprintRiskScorer(mode="rule_based")
        features = sample_sprint_features.copy()
        features["completion_gap"] = -25

        result = scorer.score(features)

        if result["recommendations"]:
            rec = result["recommendations"][0]
            assert "action" in rec
            assert "priority" in rec
            assert "rationale" in rec

    def test_train_ml_model(self):
        """Test ML model training."""
        scorer = SprintRiskScorer(mode="hybrid")

        # Create sample training data
        np.random.seed(42)
        n_samples = 50

        X = pd.DataFrame({
            "completion_gap": np.random.uniform(-30, 30, n_samples),
            "velocity_ratio": np.random.uniform(0.5, 1.5, n_samples),
            "blocked_ratio": np.random.uniform(0, 0.3, n_samples),
            "days_remaining": np.random.randint(1, 14, n_samples),
            "remaining_points": np.random.uniform(0, 30, n_samples),
        })

        # Target: completed on time
        y = pd.Series((X["completion_gap"] > -10).astype(int))

        metrics = scorer.train_ml_model(X, y, calibrate=False)

        assert "accuracy" in metrics
        assert scorer.ml_model_ is not None

    def test_hybrid_score_uses_both(self):
        """Test that hybrid mode uses both rule-based and ML scores."""
        scorer = SprintRiskScorer(mode="hybrid")

        # Train ML component
        np.random.seed(42)
        X = pd.DataFrame({
            "completion_gap": np.random.uniform(-30, 30, 30),
            "velocity_ratio": np.random.uniform(0.5, 1.5, 30),
            "blocked_ratio": np.random.uniform(0, 0.3, 30),
        })
        y = pd.Series((X["completion_gap"] > -10).astype(int))
        scorer.train_ml_model(X, y, calibrate=False)

        # Score with features
        features = {
            "completion_gap": 0,
            "velocity_ratio": 1.0,
            "blocked_ratio": 0.1,
            "scope_creep_ratio": 0,
            "days_remaining": 7,
            "remaining_points": 10,
            "total_points": 20,
        }

        result = scorer.score(features)

        assert result["rule_based_score"] is not None
        assert result["ml_score"] is not None

    def test_save_and_load(self, temp_models_dir):
        """Test model save and load."""
        scorer = SprintRiskScorer(mode="rule_based")

        # Save
        save_path = temp_models_dir / "sprint_risk.pkl"
        scorer.save(save_path)

        # Load
        loaded_scorer = SprintRiskScorer.load(save_path)

        assert loaded_scorer.mode == scorer.mode
        assert loaded_scorer.weights == scorer.weights
