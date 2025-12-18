"""
Feature engineering pipeline module.

Orchestrates feature extraction and prepares data for ML models.
"""

from datetime import datetime
from typing import Any, Literal

import pandas as pd
from loguru import logger
from scipy.stats import mstats

from src.features.developer_features import DeveloperFeatureExtractor
from src.features.sprint_features import SprintFeatureExtractor
from src.features.ticket_features import TicketFeatureExtractor


class FeaturePipeline:
    """
    Orchestrates feature extraction and data preparation for ML models.

    Handles feature extraction, missing value imputation, outlier handling,
    and categorical encoding.

    Example:
        >>> pipeline = FeaturePipeline(conn)
        >>> X_train, y_train = pipeline.build_ticket_training_set("PROJ")
        >>> sprint_features = pipeline.build_sprint_features(sprint_id=123)
    """

    # Default target variable
    DEFAULT_TARGET = "cycle_time_hours"

    # Outlier handling limits
    WINSORIZE_LIMITS = (0.01, 0.01)

    def __init__(self, conn):
        """
        Initialize the feature pipeline.

        Args:
            conn: DuckDB connection
        """
        self._conn = conn
        self._ticket_extractor = TicketFeatureExtractor(conn)
        self._developer_extractor = DeveloperFeatureExtractor(conn)
        self._sprint_extractor = SprintFeatureExtractor(conn)

    def build_ticket_training_set(
        self,
        project_key: str | None = None,
        *,
        target: str = DEFAULT_TARGET,
        min_cycle_time_hours: float = 1.0,
        max_cycle_time_hours: float = 500.0,
        handle_outliers: Literal["winsorize", "clip", "none"] = "winsorize",
        handle_missing: Literal["median", "mean", "drop"] = "median",
        limit: int | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build training dataset for ticket duration estimation.

        Args:
            project_key: Filter by project
            target: Target variable name
            min_cycle_time_hours: Minimum cycle time to include
            max_cycle_time_hours: Maximum cycle time to include
            handle_outliers: How to handle outliers in target
            handle_missing: How to handle missing values
            limit: Maximum samples

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Building ticket training set")

        # Extract raw features
        df = self._ticket_extractor.extract_batch(
            project_key,
            resolved_only=True,
            min_cycle_time_hours=min_cycle_time_hours,
            max_cycle_time_hours=max_cycle_time_hours,
            limit=limit,
        )

        if df.empty:
            logger.warning("No training data available")
            return pd.DataFrame(), pd.Series(dtype=float)

        logger.info(f"Extracted {len(df)} samples")

        # Get feature columns
        feature_cols = self._get_ticket_feature_columns()

        # Ensure all feature columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Handle missing values
        df = self._handle_missing_values(df, feature_cols, method=handle_missing)

        # Extract target
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found")

        y = df[target].copy()

        # Handle outliers in target
        if handle_outliers == "winsorize":
            y = pd.Series(
                mstats.winsorize(y.values, limits=self.WINSORIZE_LIMITS),
                index=y.index
            )
        elif handle_outliers == "clip":
            lower = y.quantile(0.01)
            upper = y.quantile(0.99)
            y = y.clip(lower, upper)

        # Extract features
        X = df[feature_cols].copy()

        # Encode categorical features
        X = self._encode_categoricals(X)

        logger.info(f"Training set: {len(X)} samples, {len(X.columns)} features")

        return X, y

    def build_prediction_features(
        self,
        issue_key: str,
    ) -> pd.DataFrame | None:
        """
        Build features for a single issue prediction.

        Args:
            issue_key: Jira issue key

        Returns:
            DataFrame with single row of features or None
        """
        features = self._ticket_extractor.extract_features(issue_key)
        if not features:
            return None

        df = pd.DataFrame([features])

        # Get feature columns
        feature_cols = self._get_ticket_feature_columns()

        # Ensure all columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Select and encode
        X = df[feature_cols].copy()
        X = self._encode_categoricals(X)

        return X

    def build_sprint_features(
        self,
        sprint_id: int,
    ) -> dict[str, Any] | None:
        """
        Build features for sprint risk assessment.

        Args:
            sprint_id: Sprint ID

        Returns:
            Dictionary of sprint features or None
        """
        return self._sprint_extractor.extract_features(sprint_id)

    def build_sprint_training_set(
        self,
        board_id: int | None = None,
        *,
        limit: int | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Build training dataset for sprint risk prediction.

        Args:
            board_id: Optional board filter
            limit: Maximum sprints

        Returns:
            Tuple of (features DataFrame, target Series)
        """
        logger.info("Building sprint training set")

        # Get closed sprints
        df = self._sprint_extractor.extract_batch(
            board_id=board_id,
            state="closed",
            limit=limit,
        )

        if df.empty:
            logger.warning("No sprint training data available")
            return pd.DataFrame(), pd.Series(dtype=int)

        # Calculate target: was sprint completed on time?
        # Definition: completion_rate >= 80% at end of sprint
        df["completed_on_time"] = (df["completion_rate"] >= 80).astype(int)

        # Get feature columns
        feature_cols = SprintFeatureExtractor.get_feature_columns()

        # Ensure columns exist
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        X = df[feature_cols].copy()
        y = df["completed_on_time"].copy()

        # Fill missing values
        X = X.fillna(0)

        logger.info(f"Sprint training set: {len(X)} samples")

        return X, y

    def build_developer_features(
        self,
        project_key: str | None = None,
        *,
        days: int = 30,
    ) -> pd.DataFrame:
        """
        Build developer workload features.

        Args:
            project_key: Optional project filter
            days: Days for rolling metrics

        Returns:
            DataFrame with developer features
        """
        return self._developer_extractor.extract_all_developers(
            project_key=project_key,
            days=days,
        )

    def save_features_to_db(
        self,
        features_df: pd.DataFrame,
        feature_set: str,
        *,
        target_values: pd.Series | None = None,
    ) -> int:
        """
        Save extracted features to the ml_features table.

        Args:
            features_df: DataFrame with features
            feature_set: Name of the feature set
            target_values: Optional target values

        Returns:
            Number of records saved
        """
        import json

        if features_df.empty:
            return 0

        records = []
        for i, (idx, row) in enumerate(features_df.iterrows()):
            issue_key = row.get("issue_key", f"unknown_{i}")
            features_json = json.dumps(row.to_dict(), default=str)

            target_value = None
            if target_values is not None and idx in target_values.index:
                target_value = float(target_values.loc[idx])

            records.append({
                "issue_key": issue_key,
                "feature_set": feature_set,
                "computed_at": datetime.now(),
                "features": features_json,
                "target_value": target_value,
            })

        # Delete existing records for this feature set
        self._conn.execute(
            "DELETE FROM ml_features WHERE feature_set = ?",
            [feature_set]
        )

        # Insert new records
        df_records = pd.DataFrame(records)

        # Get next ID
        result = self._conn.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM ml_features"
        ).fetchone()
        start_id = result[0] if result else 1

        df_records["id"] = range(start_id, start_id + len(df_records))

        self._conn.execute(
            "INSERT INTO ml_features SELECT id, issue_key, feature_set, computed_at, features, target_value FROM df_records"
        )

        logger.info(f"Saved {len(records)} feature records to database")
        return len(records)

    def _get_ticket_feature_columns(self) -> list[str]:
        """Get ordered list of ticket feature columns."""
        cols = TicketFeatureExtractor.get_feature_columns()

        return (
            cols["numerical"] +
            cols["categorical"] +
            cols["temporal"]
        )

    def _handle_missing_values(
        self,
        df: pd.DataFrame,
        columns: list[str],
        method: str = "median",
    ) -> pd.DataFrame:
        """Handle missing values in feature columns."""
        df = df.copy()

        for col in columns:
            if col not in df.columns:
                continue

            if df[col].isna().any():
                if method == "median":
                    fill_value = df[col].median()
                elif method == "mean":
                    fill_value = df[col].mean()
                else:
                    continue  # drop handled separately

                if pd.isna(fill_value):
                    fill_value = 0

                df[col] = df[col].fillna(fill_value)

        if method == "drop":
            df = df.dropna(subset=columns)

        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for LightGBM.

        LightGBM handles categoricals natively when dtype is 'category'.
        """
        df = df.copy()

        categorical_cols = TicketFeatureExtractor.get_feature_columns()["categorical"]

        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype("category")

        return df


def create_pipeline_from_settings() -> FeaturePipeline:
    """
    Create a FeaturePipeline from application settings.

    Returns:
        Configured FeaturePipeline instance
    """
    from src.data.schema import get_connection

    conn = get_connection()
    return FeaturePipeline(conn)
