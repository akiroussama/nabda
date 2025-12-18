"""Tests for ticket duration estimation model."""

import numpy as np
import pandas as pd
import pytest

from src.models.ticket_estimator import MeanByTypeBaseline, TicketEstimator


class TestMeanByTypeBaseline:
    """Tests for MeanByTypeBaseline model."""

    def test_fit_calculates_global_mean(self):
        """Test that fit calculates global mean."""
        model = MeanByTypeBaseline()
        X = pd.DataFrame({"issue_type": ["Story", "Bug", "Task"]})
        y = pd.Series([10.0, 20.0, 30.0])

        model.fit(X, y)

        assert model.global_mean_ == 20.0

    def test_fit_calculates_type_means(self):
        """Test that fit calculates means by type."""
        model = MeanByTypeBaseline()
        X = pd.DataFrame({"issue_type": ["Story", "Story", "Bug"]})
        y = pd.Series([10.0, 20.0, 30.0])

        model.fit(X, y)

        assert model.type_means_["Story"] == 15.0
        assert model.type_means_["Bug"] == 30.0

    def test_predict_uses_type_means(self):
        """Test that predict uses type means when available."""
        model = MeanByTypeBaseline()
        X_train = pd.DataFrame({"issue_type": ["Story", "Story", "Bug"]})
        y_train = pd.Series([10.0, 20.0, 30.0])
        model.fit(X_train, y_train)

        X_test = pd.DataFrame({"issue_type": ["Story", "Bug"]})
        predictions = model.predict(X_test)

        assert predictions[0] == 15.0  # Story mean
        assert predictions[1] == 30.0  # Bug mean

    def test_predict_uses_global_mean_for_unknown_type(self):
        """Test that predict uses global mean for unknown types."""
        model = MeanByTypeBaseline()
        X_train = pd.DataFrame({"issue_type": ["Story", "Bug"]})
        y_train = pd.Series([10.0, 30.0])
        model.fit(X_train, y_train)

        X_test = pd.DataFrame({"issue_type": ["Task"]})
        predictions = model.predict(X_test)

        assert predictions[0] == 20.0  # Global mean


class TestTicketEstimator:
    """Tests for TicketEstimator model."""

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100

        X = pd.DataFrame({
            "issue_type": np.random.choice(["Story", "Bug", "Task"], n_samples),
            "priority": np.random.choice(["High", "Medium", "Low"], n_samples),
            "story_points": np.random.uniform(1, 13, n_samples),
            "description_length": np.random.randint(50, 500, n_samples),
            "components_count": np.random.randint(0, 3, n_samples),
        })

        # Target roughly correlated with story_points
        y = pd.Series(X["story_points"] * 8 + np.random.normal(0, 10, n_samples))
        y = y.clip(1, 200)

        return X, y

    def test_init_default_model_type(self):
        """Test default model type is lightgbm."""
        model = TicketEstimator()

        assert model.model_type == "lightgbm"

    def test_init_custom_model_type(self):
        """Test custom model type."""
        model = TicketEstimator(model_type="baseline")

        assert model.model_type == "baseline"

    def test_train_baseline_model(self, sample_training_data):
        """Test training baseline model."""
        X, y = sample_training_data
        model = TicketEstimator(model_type="baseline")

        metrics = model.train(X, y)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert model.model_ is not None

    def test_train_linear_model(self, sample_training_data):
        """Test training linear model."""
        X, y = sample_training_data
        model = TicketEstimator(model_type="linear")

        metrics = model.train(X, y)

        assert "mae" in metrics
        assert model.model_ is not None

    def test_train_lightgbm_model(self, sample_training_data):
        """Test training lightgbm model."""
        X, y = sample_training_data
        model = TicketEstimator(model_type="lightgbm")

        metrics = model.train(X, y)

        assert "mae" in metrics
        assert model.feature_importances_ is not None

    def test_predict_returns_array(self, sample_training_data):
        """Test that predict returns numpy array."""
        X, y = sample_training_data
        model = TicketEstimator(model_type="baseline")
        model.train(X, y)

        predictions = model.predict(X)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)

    def test_predict_raises_without_training(self, sample_training_data):
        """Test that predict raises error without training."""
        X, _ = sample_training_data
        model = TicketEstimator()

        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(X)

    def test_predict_single_returns_dict(self, sample_training_data):
        """Test that predict_single returns dictionary."""
        X, y = sample_training_data
        model = TicketEstimator(model_type="baseline")
        model.train(X, y)

        result = model.predict_single(X.head(1))

        assert isinstance(result, dict)
        assert "predicted_hours" in result
        assert "predicted_days" in result
        assert "model_type" in result

    def test_predict_single_with_confidence(self, sample_training_data):
        """Test predict_single with confidence interval."""
        X, y = sample_training_data
        model = TicketEstimator(model_type="baseline")
        model.train(X, y)

        result = model.predict_single(X.head(1), return_confidence=True)

        assert "confidence_interval" in result
        assert "lower_hours" in result["confidence_interval"]
        assert "upper_hours" in result["confidence_interval"]

    def test_evaluate_returns_metrics(self, sample_training_data):
        """Test that evaluate returns metrics."""
        X, y = sample_training_data
        model = TicketEstimator(model_type="baseline")
        model.train(X, y)

        metrics = model.evaluate(X, y)

        assert "mae" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics

    def test_cross_validate_returns_results(self, sample_training_data):
        """Test that cross_validate returns results."""
        X, y = sample_training_data
        model = TicketEstimator(model_type="baseline")

        results = model.cross_validate(X, y, n_splits=3)

        assert "folds" in results
        assert "mean_mae" in results
        assert len(results["folds"]) == 3

    def test_save_and_load(self, sample_training_data, temp_models_dir):
        """Test model save and load."""
        X, y = sample_training_data
        model = TicketEstimator(model_type="baseline")
        model.train(X, y)

        # Save
        save_path = temp_models_dir / "test_model.pkl"
        model.save(save_path)

        # Load
        loaded_model = TicketEstimator.load(save_path)

        # Compare predictions
        original_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)

        np.testing.assert_array_almost_equal(original_pred, loaded_pred)

    def test_get_feature_importance(self, sample_training_data):
        """Test feature importance extraction."""
        X, y = sample_training_data
        model = TicketEstimator(model_type="lightgbm")
        model.train(X, y)

        importance = model.get_feature_importance(top_n=3)

        assert isinstance(importance, list)
        assert len(importance) <= 3
        if importance:
            assert isinstance(importance[0], tuple)
            assert len(importance[0]) == 2  # (name, value)
