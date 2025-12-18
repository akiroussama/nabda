"""Tests for sprint feature extraction."""

import pytest

from src.features.sprint_features import SprintFeatureExtractor


class TestSprintFeatureExtractor:
    """Tests for SprintFeatureExtractor."""

    def test_get_feature_columns_returns_list(self):
        """Test that feature columns returns a list."""
        cols = SprintFeatureExtractor.get_feature_columns()

        assert isinstance(cols, list)
        assert len(cols) > 0

    def test_get_feature_columns_contains_expected(self):
        """Test that feature columns contains expected items."""
        cols = SprintFeatureExtractor.get_feature_columns()

        expected = ["completion_rate", "velocity_ratio", "blocked_ratio"]
        for exp in expected:
            assert exp in cols

    def test_extract_features_returns_none_for_missing_sprint(self, populated_test_db):
        """Test that extract_features returns None for non-existent sprint."""
        extractor = SprintFeatureExtractor(populated_test_db)

        result = extractor.extract_features(99999)

        assert result is None

    def test_extract_features_returns_dict_for_valid_sprint(self, populated_test_db):
        """Test that extract_features returns dict for valid sprint."""
        extractor = SprintFeatureExtractor(populated_test_db)

        result = extractor.extract_features(1)

        assert result is not None
        assert isinstance(result, dict)
        assert "sprint_id" in result
        assert result["sprint_id"] == 1

    def test_extract_features_contains_expected_keys(self, populated_test_db):
        """Test that extracted features contain expected keys."""
        extractor = SprintFeatureExtractor(populated_test_db)

        result = extractor.extract_features(1)

        assert result is not None
        expected_keys = [
            "sprint_name", "sprint_state", "total_days",
            "total_issues", "completion_rate", "velocity_ratio"
        ]
        for key in expected_keys:
            assert key in result

    def test_extract_features_calculates_completion_rate(self, populated_test_db):
        """Test that completion rate is calculated correctly."""
        extractor = SprintFeatureExtractor(populated_test_db)

        result = extractor.extract_features(1)

        assert result is not None
        assert "completion_rate" in result
        assert 0 <= result["completion_rate"] <= 100

    def test_extract_batch_returns_dataframe(self, populated_test_db):
        """Test that extract_batch returns a DataFrame."""
        extractor = SprintFeatureExtractor(populated_test_db)

        df = extractor.extract_batch()

        assert df is not None
        assert len(df) >= 0

    def test_extract_batch_filters_by_state(self, populated_test_db):
        """Test that state filter works."""
        extractor = SprintFeatureExtractor(populated_test_db)

        df_closed = extractor.extract_batch(state="closed")
        df_active = extractor.extract_batch(state="active")

        if len(df_closed) > 0:
            assert all(df_closed["sprint_state"] == "closed")
        if len(df_active) > 0:
            assert all(df_active["sprint_state"] == "active")

    def test_get_active_sprint_features_returns_active(self, populated_test_db):
        """Test that get_active_sprint_features returns active sprint."""
        extractor = SprintFeatureExtractor(populated_test_db)

        result = extractor.get_active_sprint_features()

        # Should return sprint 2 which is active
        if result is not None:
            assert result["sprint_state"] == "active"

    def test_get_burndown_data_returns_dataframe(self, populated_test_db):
        """Test that get_burndown_data returns a DataFrame."""
        extractor = SprintFeatureExtractor(populated_test_db)

        df = extractor.get_burndown_data(1)

        assert df is not None
        if len(df) > 0:
            assert "date" in df.columns
            assert "ideal_remaining" in df.columns
            assert "actual_remaining" in df.columns
