"""
Sprint risk scoring model.

Predicts the risk of sprint failure using a hybrid rule-based and ML approach.
"""

from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class SprintRiskScorer:
    """
    Scores sprint completion risk using a hybrid approach.

    Combines rule-based scoring with optional ML predictions
    for more accurate risk assessment.

    Example:
        >>> scorer = SprintRiskScorer()
        >>> risk = scorer.score(sprint_features)
        >>> print(f"Risk: {risk['score']}/100 ({risk['level']})")
    """

    # Default weights for rule-based scoring
    DEFAULT_WEIGHTS = {
        "completion_gap": 0.30,
        "velocity_ratio": 0.25,
        "blocked_ratio": 0.20,
        "scope_creep": 0.15,
        "urgency": 0.10,
    }

    # Risk level thresholds
    RISK_THRESHOLDS = {
        "low": 30,      # 0-30
        "medium": 60,   # 31-60
        "high": 100,    # 61-100
    }

    def __init__(
        self,
        mode: Literal["rule_based", "ml", "hybrid"] = "hybrid",
        weights: dict[str, float] | None = None,
    ):
        """
        Initialize the sprint risk scorer.

        Args:
            mode: Scoring mode (rule_based, ml, or hybrid)
            weights: Custom weights for rule-based scoring
        """
        self.mode = mode
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.ml_model_: LGBMClassifier | None = None
        self.calibrated_model_: CalibratedClassifierCV | None = None
        self.feature_names_: list[str] | None = None
        self.trained_at_: str | None = None

    def score(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate risk score for a sprint.

        Args:
            features: Sprint features dictionary

        Returns:
            Dictionary with score, level, factors, and recommendations
        """
        # Rule-based score
        rule_score, factors = self._calculate_rule_based_score(features)

        # ML score (if available and mode supports it)
        ml_score = None
        if self.mode in ["ml", "hybrid"] and self.ml_model_ is not None:
            ml_score = self._calculate_ml_score(features)

        # Final score
        if self.mode == "rule_based":
            final_score = rule_score
        elif self.mode == "ml" and ml_score is not None:
            final_score = ml_score
        elif self.mode == "hybrid" and ml_score is not None:
            # Weighted average: 60% ML, 40% rule-based
            final_score = ml_score * 0.6 + rule_score * 0.4
        else:
            final_score = rule_score

        # Clamp to 0-100
        final_score = max(0, min(100, final_score))

        # Determine risk level
        level = self._get_risk_level(final_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(features, factors, level)

        return {
            "score": round(final_score, 1),
            "level": level,
            "rule_based_score": round(rule_score, 1),
            "ml_score": round(ml_score, 1) if ml_score is not None else None,
            "factors": factors,
            "recommendations": recommendations,
            "sprint_id": features.get("sprint_id"),
            "sprint_name": features.get("sprint_name"),
        }

    def _calculate_rule_based_score(
        self,
        features: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        """Calculate rule-based risk score."""
        factors = {}

        # 1. Completion gap factor (behind schedule = high risk)
        completion_gap = features.get("completion_gap", 0)
        # Negative gap means behind schedule
        if completion_gap < -20:
            gap_factor = 1.0
        elif completion_gap < 0:
            gap_factor = abs(completion_gap) / 20
        else:
            gap_factor = max(0, -completion_gap / 50)  # Ahead of schedule reduces risk

        factors["completion_gap"] = {
            "value": completion_gap,
            "factor": gap_factor,
            "weight": self.weights["completion_gap"],
            "contribution": gap_factor * self.weights["completion_gap"],
        }

        # 2. Velocity ratio factor (low velocity = high risk)
        velocity_ratio = features.get("velocity_ratio", 1.0)
        if velocity_ratio < 0.5:
            velocity_factor = 1.0
        elif velocity_ratio < 1.0:
            velocity_factor = 1.0 - velocity_ratio
        else:
            velocity_factor = 0.0

        factors["velocity_ratio"] = {
            "value": velocity_ratio,
            "factor": velocity_factor,
            "weight": self.weights["velocity_ratio"],
            "contribution": velocity_factor * self.weights["velocity_ratio"],
        }

        # 3. Blocked ratio factor
        blocked_ratio = features.get("blocked_ratio", 0)
        blocked_factor = min(1.0, blocked_ratio * 5)  # 20% blocked = max risk

        factors["blocked_ratio"] = {
            "value": blocked_ratio,
            "factor": blocked_factor,
            "weight": self.weights["blocked_ratio"],
            "contribution": blocked_factor * self.weights["blocked_ratio"],
        }

        # 4. Scope creep factor
        scope_creep = features.get("scope_creep_ratio", 0)
        scope_factor = min(1.0, scope_creep * 2)  # 50% creep = max risk

        factors["scope_creep"] = {
            "value": scope_creep,
            "factor": scope_factor,
            "weight": self.weights["scope_creep"],
            "contribution": scope_factor * self.weights["scope_creep"],
        }

        # 5. Urgency factor (based on remaining time and work)
        days_remaining = features.get("days_remaining", 7)
        remaining_points = features.get("remaining_points", 0)
        total_points = features.get("total_points", 1)

        if days_remaining <= 2 and remaining_points > 0:
            urgency_factor = 1.0
        elif days_remaining <= 5:
            work_ratio = remaining_points / max(1, total_points)
            urgency_factor = work_ratio * (1 - days_remaining / 14)
        else:
            urgency_factor = 0.0

        factors["urgency"] = {
            "value": {"days_remaining": days_remaining, "remaining_points": remaining_points},
            "factor": urgency_factor,
            "weight": self.weights["urgency"],
            "contribution": urgency_factor * self.weights["urgency"],
        }

        # Calculate total score (0-100)
        total_contribution = sum(f["contribution"] for f in factors.values())
        score = total_contribution * 100

        return score, factors

    def _calculate_ml_score(self, features: dict[str, Any]) -> float | None:
        """Calculate ML-based risk score."""
        if self.ml_model_ is None:
            return None

        # Build feature vector
        feature_cols = self.feature_names_ or []
        X = pd.DataFrame([{col: features.get(col, 0) for col in feature_cols}])

        # Get probability of failure (class 0)
        if self.calibrated_model_ is not None:
            prob = self.calibrated_model_.predict_proba(X)[0]
        else:
            prob = self.ml_model_.predict_proba(X)[0]

        # Risk score = probability of NOT completing on time
        risk_probability = 1 - prob[1] if len(prob) > 1 else prob[0]

        return risk_probability * 100

    def _get_risk_level(self, score: float) -> str:
        """Determine risk level from score."""
        if score <= self.RISK_THRESHOLDS["low"]:
            return "low"
        elif score <= self.RISK_THRESHOLDS["medium"]:
            return "medium"
        else:
            return "high"

    def _generate_recommendations(
        self,
        features: dict[str, Any],
        factors: dict[str, Any],
        level: str,
    ) -> list[dict[str, str]]:
        """Generate actionable recommendations based on risk factors."""
        recommendations = []

        # Sort factors by contribution
        sorted_factors = sorted(
            factors.items(),
            key=lambda x: x[1]["contribution"],
            reverse=True,
        )

        for factor_name, factor_data in sorted_factors[:3]:
            if factor_data["contribution"] < 0.05:
                continue

            if factor_name == "completion_gap" and factor_data["factor"] > 0.3:
                recommendations.append({
                    "action": "Review and reduce scope",
                    "priority": "high" if factor_data["factor"] > 0.7 else "medium",
                    "rationale": f"Sprint is {abs(factor_data['value']):.0f}% behind schedule",
                })

            elif factor_name == "velocity_ratio" and factor_data["factor"] > 0.3:
                recommendations.append({
                    "action": "Investigate velocity drop",
                    "priority": "high" if factor_data["factor"] > 0.5 else "medium",
                    "rationale": f"Current velocity is {factor_data['value']:.0%} of historical average",
                })

            elif factor_name == "blocked_ratio" and factor_data["factor"] > 0.2:
                blocked_count = features.get("blocked_issues", 0)
                recommendations.append({
                    "action": "Unblock tickets immediately",
                    "priority": "high",
                    "rationale": f"{blocked_count} tickets are currently blocked",
                })

            elif factor_name == "scope_creep" and factor_data["factor"] > 0.2:
                recommendations.append({
                    "action": "Freeze scope for remainder of sprint",
                    "priority": "medium",
                    "rationale": f"Scope increased by {factor_data['value']:.0%} mid-sprint",
                })

            elif factor_name == "urgency" and factor_data["factor"] > 0.5:
                days = features.get("days_remaining", 0)
                points = features.get("remaining_points", 0)
                recommendations.append({
                    "action": "Focus on must-have items only",
                    "priority": "high",
                    "rationale": f"{points:.0f} points remaining with {days} days left",
                })

        # Add general recommendation if high risk
        if level == "high" and len(recommendations) < 3:
            recommendations.append({
                "action": "Consider sprint replanning meeting",
                "priority": "high",
                "rationale": "Overall sprint health is critical",
            })

        return recommendations

    def train_ml_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        calibrate: bool = True,
    ) -> dict[str, float]:
        """
        Train the ML component of the scorer.

        Args:
            X: Feature DataFrame
            y: Target (1 = completed on time, 0 = not)
            calibrate: Whether to calibrate probabilities

        Returns:
            Training metrics
        """
        logger.info(f"Training sprint risk ML model with {len(X)} samples")

        self.feature_names_ = list(X.columns)

        self.ml_model_ = LGBMClassifier(
            objective="binary",
            metric="auc",
            num_leaves=15,
            learning_rate=0.05,
            feature_fraction=0.7,
            n_estimators=100,
            verbose=-1,
            random_state=42,
        )

        self.ml_model_.fit(X, y)

        if calibrate and len(X) > 20:
            self.calibrated_model_ = CalibratedClassifierCV(
                self.ml_model_,
                method="isotonic",
                cv=3,
            )
            self.calibrated_model_.fit(X, y)

        # Evaluate
        y_pred = self.ml_model_.predict(X)
        y_prob = self.ml_model_.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
            "auc_roc": roc_auc_score(y, y_prob) if len(np.unique(y)) > 1 else 0,
        }

        self.trained_at_ = pd.Timestamp.now().isoformat()

        logger.info(f"Training complete. AUC-ROC: {metrics['auc_roc']:.3f}")

        return metrics

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "mode": self.mode,
            "weights": self.weights,
            "ml_model": self.ml_model_,
            "calibrated_model": self.calibrated_model_,
            "feature_names": self.feature_names_,
            "trained_at": self.trained_at_,
        }

        joblib.dump(model_data, path)
        logger.info(f"Sprint risk model saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "SprintRiskScorer":
        """Load model from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")

        model_data = joblib.load(path)

        instance = cls(
            mode=model_data["mode"],
            weights=model_data.get("weights"),
        )
        instance.ml_model_ = model_data.get("ml_model")
        instance.calibrated_model_ = model_data.get("calibrated_model")
        instance.feature_names_ = model_data.get("feature_names")
        instance.trained_at_ = model_data.get("trained_at")

        return instance
