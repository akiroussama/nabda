"""Tests for ticket feature extraction."""

import pytest

from src.features.ticket_features import TicketFeatureExtractor


class TestTicketFeatureExtractor:
    """Tests for TicketFeatureExtractor."""

    def test_get_feature_columns_returns_all_categories(self):
        """Test that feature columns returns all expected categories."""
        cols = TicketFeatureExtractor.get_feature_columns()

        assert "numerical" in cols
        assert "categorical" in cols
        assert "temporal" in cols

        assert len(cols["numerical"]) > 0
        assert len(cols["categorical"]) > 0
        assert len(cols["temporal"]) > 0

    def test_get_feature_columns_no_duplicates(self):
        """Test that there are no duplicate feature columns."""
        cols = TicketFeatureExtractor.get_feature_columns()

        all_cols = cols["numerical"] + cols["categorical"] + cols["temporal"]
        assert len(all_cols) == len(set(all_cols))

    def test_extract_features_returns_none_for_missing_issue(self, populated_test_db):
        """Test that extract_features returns None for non-existent issue."""
        extractor = TicketFeatureExtractor(populated_test_db)

        result = extractor.extract_features("NONEXISTENT-999")

        assert result is None

    def test_extract_features_returns_dict_for_valid_issue(self, populated_test_db):
        """Test that extract_features returns dict for valid issue."""
        extractor = TicketFeatureExtractor(populated_test_db)

        result = extractor.extract_features("PROJ-1")

        assert result is not None
        assert isinstance(result, dict)
        assert "issue_key" in result
        assert result["issue_key"] == "PROJ-1"

    def test_extract_features_contains_expected_keys(self, populated_test_db):
        """Test that extracted features contain expected keys."""
        extractor = TicketFeatureExtractor(populated_test_db)
        cols = TicketFeatureExtractor.get_feature_columns()

        result = extractor.extract_features("PROJ-1")

        assert result is not None
        # Check that most feature columns are present
        all_cols = cols["numerical"] + cols["categorical"] + cols["temporal"]
        present_cols = [c for c in all_cols if c in result]
        assert len(present_cols) > len(all_cols) * 0.5  # At least 50% present

    def test_extract_batch_returns_dataframe(self, populated_test_db):
        """Test that extract_batch returns a DataFrame."""
        extractor = TicketFeatureExtractor(populated_test_db)

        df = extractor.extract_batch(project_key="PROJ", resolved_only=True)

        assert df is not None
        assert len(df) > 0
        assert "issue_key" in df.columns

    def test_extract_batch_resolved_only_filters(self, populated_test_db):
        """Test that resolved_only parameter filters correctly."""
        extractor = TicketFeatureExtractor(populated_test_db)

        df_resolved = extractor.extract_batch(project_key="PROJ", resolved_only=True)
        df_all = extractor.extract_batch(project_key="PROJ", resolved_only=False)

        assert len(df_resolved) <= len(df_all)

    def test_extract_batch_respects_cycle_time_limits(self, populated_test_db):
        """Test that cycle time limits are respected."""
        extractor = TicketFeatureExtractor(populated_test_db)

        df = extractor.extract_batch(
            project_key="PROJ",
            resolved_only=True,
            min_cycle_time_hours=50.0,
            max_cycle_time_hours=100.0,
        )

        # Should filter based on cycle time
        if len(df) > 0 and "cycle_time_hours" in df.columns:
            assert df["cycle_time_hours"].min() >= 50.0
            assert df["cycle_time_hours"].max() <= 100.0

    def test_extract_batch_limit_parameter(self, populated_test_db):
        """Test that limit parameter works correctly."""
        extractor = TicketFeatureExtractor(populated_test_db)

        df = extractor.extract_batch(project_key="PROJ", resolved_only=True, limit=2)

        assert len(df) <= 2
