"""
Ticket duration estimation model.

Predicts the cycle time (duration) of Jira tickets using ML models.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


class MeanByTypeBaseline(BaseEstimator, RegressorMixin):
    """
    Baseline model that predicts mean cycle time by issue type.

    Falls back to global mean for unseen types.
    """

    def __init__(self):
        self.type_means_: dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MeanByTypeBaseline":
        """Fit the model."""
        self.global_mean_ = float(y.mean())

        if "issue_type" in X.columns:
            df = pd.DataFrame({"issue_type": X["issue_type"], "target": y})
            self.type_means_ = df.groupby("issue_type")["target"].mean().to_dict()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict cycle times."""
        if "issue_type" not in X.columns:
            return np.full(len(X), self.global_mean_)

        predictions = X["issue_type"].map(self.type_means_).fillna(self.global_mean_)
        return predictions.values


class TicketEstimator:
    """
    Ticket duration estimation model.

    Supports multiple model types: baseline (mean by type),
    linear regression, and LightGBM.

    Example:
        >>> estimator = TicketEstimator()
        >>> estimator.train(X_train, y_train)
        >>> predictions = estimator.predict(X_test)
        >>> estimator.save("models/ticket_estimator.pkl")
    """

    MODEL_TYPES = ["baseline", "linear", "lightgbm"]

    DEFAULT_LGBM_PARAMS = {
        "objective": "regression",
        "metric": "mae",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 20,
        "verbose": -1,
        "n_estimators": 200,
        "random_state": 42,
    }

    def __init__(
        self,
        model_type: Literal["baseline", "linear", "lightgbm"] = "lightgbm",
        lgbm_params: dict[str, Any] | None = None,
    ):
        """
        Initialize the ticket estimator.

        Args:
            model_type: Type of model to use
            lgbm_params: LightGBM parameters (overrides defaults)
        """
        self.model_type = model_type
        self.lgbm_params = {**self.DEFAULT_LGBM_PARAMS, **(lgbm_params or {})}
        self.model_: BaseEstimator | None = None
        self.feature_names_: list[str] | None = None
        self.feature_importances_: dict[str, float] | None = None
        self.metrics_: dict[str, float] | None = None
        self.trained_at_: str | None = None

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        early_stopping_rounds: int = 50,
    ) -> dict[str, float]:
        """
        Train the model.

        Args:
            X: Feature DataFrame
            y: Target Series
            eval_set: Optional validation set for early stopping
            early_stopping_rounds: Rounds for early stopping (LightGBM)

        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training {self.model_type} model with {len(X)} samples")

        self.feature_names_ = list(X.columns)

        # Create model instance
        if self.model_type == "baseline":
            self.model_ = MeanByTypeBaseline()
            self.model_.fit(X, y)

        elif self.model_type == "linear":
            # Convert categoricals to dummies for linear model
            X_encoded = pd.get_dummies(X, drop_first=True)
            self.model_ = LinearRegression()
            self.model_.fit(X_encoded, y)

        elif self.model_type == "lightgbm":
            # Ensure categorical columns are properly typed
            categorical_features = [
                col for col in X.columns
                if X[col].dtype.name == "category" or col in ["issue_type", "priority", "component_primary"]
            ]

            self.model_ = LGBMRegressor(**self.lgbm_params)

            fit_params = {"categorical_feature": categorical_features}

            if eval_set:
                X_val, y_val = eval_set
                fit_params["eval_set"] = [(X_val, y_val)]
                fit_params["callbacks"] = [
                    # early_stopping callback
                ]

            self.model_.fit(X, y, **fit_params)

            # Extract feature importances
            self.feature_importances_ = dict(
                zip(self.feature_names_, self.model_.feature_importances_)
            )

        # Calculate training metrics
        y_pred = self.predict(X)
        self.metrics_ = self._calculate_metrics(y, y_pred)
        self.trained_at_ = pd.Timestamp.now().isoformat()

        logger.info(f"Training complete. MAE: {self.metrics_['mae']:.2f} hours")

        return self.metrics_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cycle times.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions in hours
        """
        if self.model_ is None:
            raise ValueError("Model not trained. Call train() first.")

        if self.model_type == "linear":
            X_encoded = pd.get_dummies(X, drop_first=True)
            # Align columns with training
            return self.model_.predict(X_encoded)

        return self.model_.predict(X)

    def predict_single(
        self,
        X: pd.DataFrame,
        *,
        return_confidence: bool = False,
    ) -> dict[str, Any]:
        """
        Predict for a single ticket with additional info.

        Args:
            X: Single-row feature DataFrame
            return_confidence: Whether to estimate confidence interval

        Returns:
            Dictionary with prediction and metadata
        """
        prediction = float(self.predict(X)[0])

        result = {
            "predicted_hours": round(prediction, 1),
            "predicted_days": round(prediction / 24, 1),
            "model_type": self.model_type,
        }

        # Rough confidence interval based on historical MAE
        if return_confidence and self.metrics_:
            mae = self.metrics_.get("mae", 24)
            result["confidence_interval"] = {
                "lower_hours": max(0, round(prediction - mae * 1.5, 1)),
                "upper_hours": round(prediction + mae * 1.5, 1),
            }

        return result

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            X: Feature DataFrame
            y: True values

        Returns:
            Dictionary of metrics
        """
        y_pred = self.predict(X)
        return self._calculate_metrics(y, y_pred)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> dict[str, Any]:
        """
        Perform time-series cross-validation.

        Args:
            X: Feature DataFrame (should be sorted by time)
            y: Target Series
            n_splits: Number of CV splits

        Returns:
            Dictionary with fold results and aggregated metrics
        """
        logger.info(f"Running {n_splits}-fold time series cross-validation")

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Train new model for this fold
            fold_model = TicketEstimator(
                model_type=self.model_type,
                lgbm_params=self.lgbm_params,
            )
            fold_model.train(X_train, y_train)

            # Evaluate
            metrics = fold_model.evaluate(X_test, y_test)
            metrics["fold"] = fold
            metrics["train_size"] = len(X_train)
            metrics["test_size"] = len(X_test)
            fold_results.append(metrics)

            logger.debug(f"Fold {fold}: MAE={metrics['mae']:.2f}")

        # Aggregate results
        results_df = pd.DataFrame(fold_results)

        return {
            "folds": fold_results,
            "mean_mae": results_df["mae"].mean(),
            "std_mae": results_df["mae"].std(),
            "mean_rmse": results_df["rmse"].mean(),
            "mean_r2": results_df["r2"].mean(),
        }

    def save(self, path: str | Path) -> None:
        """
        Save model to disk.

        Args:
            path: Path to save model
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self.model_,
            "model_type": self.model_type,
            "lgbm_params": self.lgbm_params,
            "feature_names": self.feature_names_,
            "feature_importances": self.feature_importances_,
            "metrics": self.metrics_,
            "trained_at": self.trained_at_,
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TicketEstimator":
        """
        Load model from disk.

        Args:
            path: Path to model file

        Returns:
            Loaded TicketEstimator instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        model_data = joblib.load(path)

        instance = cls(
            model_type=model_data["model_type"],
            lgbm_params=model_data.get("lgbm_params", {}),
        )
        instance.model_ = model_data["model"]
        instance.feature_names_ = model_data.get("feature_names")
        instance.feature_importances_ = model_data.get("feature_importances")
        instance.metrics_ = model_data.get("metrics")
        instance.trained_at_ = model_data.get("trained_at")

        logger.info(f"Model loaded from {path}")
        return instance

    def get_feature_importance(self, top_n: int = 10) -> list[tuple[str, float]]:
        """
        Get top feature importances.

        Args:
            top_n: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        if not self.feature_importances_:
            return []

        sorted_features = sorted(
            self.feature_importances_.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return sorted_features[:top_n]

    @staticmethod
    def _calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate regression metrics."""
        return {
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": float(r2_score(y_true, y_pred)),
            "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
        }


def predict_ticket(issue_key: str) -> dict[str, Any]:
    """
    Predict duration for a specific ticket.

    Args:
        issue_key: Jira issue key

    Returns:
        Prediction results
    """
    from src.features.pipeline import create_pipeline_from_settings

    # Load model
    model_path = Path("models/ticket_estimator.pkl")
    if not model_path.exists():
        return {"error": "Model not found. Run 'make train' first."}

    model = TicketEstimator.load(model_path)

    # Build features
    pipeline = create_pipeline_from_settings()
    X = pipeline.build_prediction_features(issue_key)

    if X is None:
        return {"error": f"Issue {issue_key} not found"}

    # Predict
    result = model.predict_single(X, return_confidence=True)
    result["issue_key"] = issue_key

    return result


def main():
    """CLI entry point for predictions."""
    parser = argparse.ArgumentParser(description="Ticket Duration Prediction")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict ticket duration")
    predict_parser.add_argument("issue_key", help="Jira issue key (e.g., PROJ-123)")

    args = parser.parse_args()

    if args.command == "predict":
        result = predict_ticket(args.issue_key)

        if "error" in result:
            print(f"❌ {result['error']}", file=sys.stderr)
            return 1

        print(f"\n📊 Prediction for {result['issue_key']}")
        print(f"   Estimated duration: {result['predicted_hours']:.1f} hours ({result['predicted_days']:.1f} days)")

        if "confidence_interval" in result:
            ci = result["confidence_interval"]
            print(f"   Confidence interval: {ci['lower_hours']:.1f} - {ci['upper_hours']:.1f} hours")

        print(f"   Model: {result['model_type']}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
