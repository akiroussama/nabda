"""
Developer workload scoring model.

Analyzes developer workload and identifies overload/availability states.
"""

from typing import Any, Literal

import numpy as np
import pandas as pd
from loguru import logger


class WorkloadScorer:
    """
    Scores developer workload and identifies capacity issues.

    Uses a weighted multi-factor approach to assess current workload
    relative to team averages and historical patterns.

    Example:
        >>> scorer = WorkloadScorer()
        >>> result = scorer.score(developer_features)
        >>> print(f"Workload: {result['score']}/100 ({result['status']})")
    """

    # Default weights for workload factors
    DEFAULT_WEIGHTS = {
        "wip_points": 0.30,
        "hours_logged": 0.20,
        "unresolved_count": 0.20,
        "blocked_count": 0.15,
        "overdue_count": 0.15,
    }

    # Workload status thresholds
    THRESHOLDS = {
        "underloaded": 50,   # 0-50
        "optimal": 100,      # 51-100
        "high": 130,         # 101-130
        "overloaded": 200,   # 131+
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        *,
        overload_threshold: float = 1.3,
        underload_threshold: float = 0.5,
    ):
        """
        Initialize the workload scorer.

        Args:
            weights: Custom weights for scoring factors
            overload_threshold: Ratio above average for overload status
            underload_threshold: Ratio below average for underload status
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        self.overload_threshold = overload_threshold
        self.underload_threshold = underload_threshold
        self._team_baselines: dict[str, float] | None = None

    def set_team_baselines(self, developer_df: pd.DataFrame) -> None:
        """
        Calculate team baselines from developer data.

        Args:
            developer_df: DataFrame with developer metrics
        """
        if developer_df.empty:
            self._team_baselines = None
            return

        self._team_baselines = {
            "avg_wip_points": developer_df["wip_points"].mean() if "wip_points" in developer_df else 0,
            "avg_hours_logged": developer_df.get("hours_logged_30d", developer_df.get("hours_logged", pd.Series([0]))).mean(),
            "avg_unresolved": developer_df["unresolved_count"].mean() if "unresolved_count" in developer_df else 0,
            "avg_blocked": developer_df["blocked_count"].mean() if "blocked_count" in developer_df else 0,
            "avg_overdue": developer_df["overdue_count"].mean() if "overdue_count" in developer_df else 0,
            "team_size": len(developer_df),
        }

        logger.debug(f"Team baselines calculated: {self._team_baselines}")

    def score(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate workload score for a developer.

        Args:
            features: Developer features dictionary

        Returns:
            Dictionary with score, status, factors, and recommendations
        """
        # Calculate factor scores
        factors = self._calculate_factors(features)

        # Weighted score (0-200 scale, 100 = optimal)
        raw_score = sum(
            factors[name]["normalized"] * self.weights.get(name, 0)
            for name in factors
        )

        # Scale to 0-200
        score = raw_score * 200

        # Determine status
        status = self._get_status(score)

        # Generate recommendations
        recommendations = self._generate_recommendations(features, factors, status)

        return {
            "score": round(score, 1),
            "status": status,
            "relative_to_team": round(score / 100, 2),
            "factors": factors,
            "recommendations": recommendations,
            "developer_id": features.get("assignee_id"),
            "developer_name": features.get("pseudonym"),
        }

    def _calculate_factors(self, features: dict[str, Any]) -> dict[str, Any]:
        """Calculate normalized factor scores."""
        factors = {}

        # Get team baselines or use defaults
        baselines = self._team_baselines or {
            "avg_wip_points": 20,
            "avg_hours_logged": 40,
            "avg_unresolved": 5,
            "avg_blocked": 1,
            "avg_overdue": 1,
        }

        # WIP Points factor
        wip_points = features.get("wip_points", 0)
        avg_wip = baselines.get("avg_wip_points", 20) or 20
        wip_ratio = wip_points / avg_wip if avg_wip > 0 else 0

        factors["wip_points"] = {
            "value": wip_points,
            "team_average": round(avg_wip, 1),
            "ratio": round(wip_ratio, 2),
            "normalized": min(1.0, wip_ratio / 2),  # Cap at 200%
        }

        # Hours logged factor
        hours_key = next(
            (k for k in ["hours_logged_30d", "hours_logged"] if k in features),
            None
        )
        hours_logged = features.get(hours_key, 0) if hours_key else 0
        avg_hours = baselines.get("avg_hours_logged", 40) or 40
        hours_ratio = hours_logged / avg_hours if avg_hours > 0 else 0

        factors["hours_logged"] = {
            "value": hours_logged,
            "team_average": round(avg_hours, 1),
            "ratio": round(hours_ratio, 2),
            "normalized": min(1.0, hours_ratio / 2),
        }

        # Unresolved count factor
        unresolved = features.get("unresolved_count", 0)
        avg_unresolved = baselines.get("avg_unresolved", 5) or 5
        unresolved_ratio = unresolved / avg_unresolved if avg_unresolved > 0 else 0

        factors["unresolved_count"] = {
            "value": unresolved,
            "team_average": round(avg_unresolved, 1),
            "ratio": round(unresolved_ratio, 2),
            "normalized": min(1.0, unresolved_ratio / 2),
        }

        # Blocked count factor (higher weight for blockers)
        blocked = features.get("blocked_count", 0)
        avg_blocked = baselines.get("avg_blocked", 1) or 1
        blocked_ratio = blocked / avg_blocked if avg_blocked > 0 else 0

        factors["blocked_count"] = {
            "value": blocked,
            "team_average": round(avg_blocked, 1),
            "ratio": round(blocked_ratio, 2),
            "normalized": min(1.0, blocked_ratio / 2),
        }

        # Overdue count factor (high priority)
        overdue = features.get("overdue_count", 0)
        avg_overdue = baselines.get("avg_overdue", 1) or 1
        overdue_ratio = overdue / avg_overdue if avg_overdue > 0 else 0

        factors["overdue_count"] = {
            "value": overdue,
            "team_average": round(avg_overdue, 1),
            "ratio": round(overdue_ratio, 2),
            "normalized": min(1.0, overdue_ratio / 2),
        }

        return factors

    def _get_status(self, score: float) -> str:
        """Determine workload status from score."""
        if score <= self.THRESHOLDS["underloaded"]:
            return "underloaded"
        elif score <= self.THRESHOLDS["optimal"]:
            return "optimal"
        elif score <= self.THRESHOLDS["high"]:
            return "high"
        else:
            return "overloaded"

    def _generate_recommendations(
        self,
        features: dict[str, Any],
        factors: dict[str, Any],
        status: str,
    ) -> list[dict[str, str]]:
        """Generate actionable recommendations based on workload analysis."""
        recommendations = []

        if status == "overloaded":
            # Check specific factors
            if factors["blocked_count"]["value"] > 2:
                recommendations.append({
                    "action": "Prioritize unblocking tickets",
                    "priority": "high",
                    "rationale": f"{factors['blocked_count']['value']} blocked tickets require attention",
                })

            if factors["overdue_count"]["value"] > 0:
                recommendations.append({
                    "action": "Address overdue tickets",
                    "priority": "high",
                    "rationale": f"{factors['overdue_count']['value']} tickets are overdue",
                })

            if factors["wip_points"]["ratio"] > 1.5:
                recommendations.append({
                    "action": "Consider redistributing work",
                    "priority": "medium",
                    "rationale": f"WIP is {factors['wip_points']['ratio']:.0%} of team average",
                })

            recommendations.append({
                "action": "Review workload with manager",
                "priority": "high",
                "rationale": "Overall workload significantly exceeds team average",
            })

        elif status == "high":
            if factors["unresolved_count"]["ratio"] > 1.3:
                recommendations.append({
                    "action": "Focus on completing current tickets",
                    "priority": "medium",
                    "rationale": "High number of unresolved tickets",
                })

            recommendations.append({
                "action": "Avoid taking new assignments",
                "priority": "medium",
                "rationale": "Workload above optimal level",
            })

        elif status == "underloaded":
            recommendations.append({
                "action": "Capacity available for new work",
                "priority": "info",
                "rationale": f"Current workload at {factors['wip_points']['ratio']:.0%} of team average",
            })

            if features.get("completion_rate", 0) > 0.8:
                recommendations.append({
                    "action": "Consider taking on complex tickets",
                    "priority": "info",
                    "rationale": "Strong completion rate indicates capacity for challenging work",
                })

        return recommendations

    def score_team(self, developer_df: pd.DataFrame) -> dict[str, Any]:
        """
        Score entire team workload distribution.

        Args:
            developer_df: DataFrame with developer metrics

        Returns:
            Team-level workload analysis
        """
        if developer_df.empty:
            return {
                "team_size": 0,
                "distribution": {},
                "balance_score": 0,
                "recommendations": [],
            }

        # Set baselines from team data
        self.set_team_baselines(developer_df)

        # Score each developer
        individual_scores = []
        for _, row in developer_df.iterrows():
            score_result = self.score(row.to_dict())
            individual_scores.append({
                "developer_id": score_result["developer_id"],
                "developer_name": score_result["developer_name"],
                "score": score_result["score"],
                "status": score_result["status"],
            })

        # Calculate distribution
        status_counts = pd.DataFrame(individual_scores)["status"].value_counts().to_dict()

        # Calculate balance score (0-100, higher = more balanced)
        scores = [s["score"] for s in individual_scores]
        if len(scores) > 1:
            variance = np.var(scores)
            max_variance = 10000  # Maximum expected variance
            balance_score = max(0, 100 * (1 - variance / max_variance))
        else:
            balance_score = 100

        # Generate team recommendations
        team_recommendations = []

        overloaded_count = status_counts.get("overloaded", 0)
        underloaded_count = status_counts.get("underloaded", 0)

        if overloaded_count > 0 and underloaded_count > 0:
            team_recommendations.append({
                "action": "Redistribute work from overloaded to underloaded team members",
                "priority": "high",
                "rationale": f"{overloaded_count} overloaded, {underloaded_count} underloaded",
            })

        if overloaded_count / len(individual_scores) > 0.3:
            team_recommendations.append({
                "action": "Consider additional team capacity",
                "priority": "high",
                "rationale": f"{overloaded_count}/{len(individual_scores)} team members overloaded",
            })

        if balance_score < 50:
            team_recommendations.append({
                "action": "Improve workload distribution",
                "priority": "medium",
                "rationale": f"Workload balance score: {balance_score:.0f}/100",
            })

        return {
            "team_size": len(individual_scores),
            "individual_scores": individual_scores,
            "distribution": status_counts,
            "balance_score": round(balance_score, 1),
            "avg_workload": round(np.mean(scores), 1),
            "recommendations": team_recommendations,
        }

    def identify_reassignment_candidates(
        self,
        developer_df: pd.DataFrame,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Identify developers for work reassignment.

        Args:
            developer_df: DataFrame with developer metrics

        Returns:
            Dictionary with 'give' (overloaded) and 'receive' (available) lists
        """
        team_result = self.score_team(developer_df)

        give_work = []
        receive_work = []

        for score_data in team_result.get("individual_scores", []):
            if score_data["status"] == "overloaded":
                give_work.append({
                    "developer_id": score_data["developer_id"],
                    "developer_name": score_data["developer_name"],
                    "workload_score": score_data["score"],
                    "excess": round(score_data["score"] - 100, 1),
                })
            elif score_data["status"] in ["underloaded", "optimal"]:
                capacity = 100 - score_data["score"]
                if capacity > 20:  # At least 20% capacity
                    receive_work.append({
                        "developer_id": score_data["developer_id"],
                        "developer_name": score_data["developer_name"],
                        "workload_score": score_data["score"],
                        "available_capacity": round(capacity, 1),
                    })

        # Sort by excess/capacity
        give_work.sort(key=lambda x: x["excess"], reverse=True)
        receive_work.sort(key=lambda x: x["available_capacity"], reverse=True)

        return {
            "give": give_work,
            "receive": receive_work,
        }
