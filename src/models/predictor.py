"""
Prediction interface module.

Provides a unified interface for making predictions on new tickets, sprints, and developers.
"""

import sys
from pathlib import Path
from typing import Any

from loguru import logger

from src.features.pipeline import create_pipeline_from_settings
from src.models.sprint_risk import SprintRiskScorer
from src.models.ticket_estimator import TicketEstimator
from src.models.workload_scorer import WorkloadScorer


class Predictor:
    """
    Unified prediction interface for all ML models.

    Loads trained models and provides prediction methods for
    tickets, sprints, and developer workload.

    Example:
        >>> predictor = Predictor.from_model_dir("models/")
        >>> result = predictor.predict_ticket("PROJ-123")
        >>> print(f"Estimated: {result['predicted_hours']} hours")
    """

    def __init__(
        self,
        ticket_estimator: TicketEstimator | None = None,
        sprint_risk_scorer: SprintRiskScorer | None = None,
        workload_scorer: WorkloadScorer | None = None,
    ):
        """
        Initialize the predictor.

        Args:
            ticket_estimator: Trained ticket estimator
            sprint_risk_scorer: Trained sprint risk scorer
            workload_scorer: Initialized workload scorer
        """
        self._ticket_estimator = ticket_estimator
        self._sprint_risk_scorer = sprint_risk_scorer
        self._workload_scorer = workload_scorer
        self._pipeline = None

    @property
    def pipeline(self):
        """Get or create the feature pipeline."""
        if self._pipeline is None:
            self._pipeline = create_pipeline_from_settings()
        return self._pipeline

    @classmethod
    def from_model_dir(cls, model_dir: str | Path = "models") -> "Predictor":
        """
        Load predictor from model directory.

        Args:
            model_dir: Directory containing saved models

        Returns:
            Initialized Predictor instance
        """
        model_dir = Path(model_dir)

        ticket_estimator = None
        sprint_risk_scorer = None
        workload_scorer = None

        # Load ticket estimator
        ticket_path = model_dir / "ticket_estimator.pkl"
        if ticket_path.exists():
            ticket_estimator = TicketEstimator.load(ticket_path)
            logger.info("Loaded ticket estimator")

        # Load sprint risk scorer
        sprint_path = model_dir / "sprint_risk_scorer.pkl"
        if sprint_path.exists():
            sprint_risk_scorer = SprintRiskScorer.load(sprint_path)
            logger.info("Loaded sprint risk scorer")

        # Initialize workload scorer (no persistence)
        workload_scorer = WorkloadScorer()
        logger.info("Initialized workload scorer")

        return cls(
            ticket_estimator=ticket_estimator,
            sprint_risk_scorer=sprint_risk_scorer,
            workload_scorer=workload_scorer,
        )

    def predict_ticket(
        self,
        issue_key: str,
        *,
        return_confidence: bool = True,
    ) -> dict[str, Any]:
        """
        Predict duration for a ticket.

        Args:
            issue_key: Jira issue key (e.g., PROJ-123)
            return_confidence: Include confidence interval

        Returns:
            Prediction results
        """
        if self._ticket_estimator is None:
            return {"error": "Ticket estimator not loaded", "issue_key": issue_key}

        # Build features
        X = self.pipeline.build_prediction_features(issue_key)

        if X is None:
            return {"error": f"Issue {issue_key} not found", "issue_key": issue_key}

        # Predict
        result = self._ticket_estimator.predict_single(X, return_confidence=return_confidence)
        result["issue_key"] = issue_key

        return result

    def predict_sprint_risk(
        self,
        sprint_id: int,
    ) -> dict[str, Any]:
        """
        Predict risk score for a sprint.

        Args:
            sprint_id: Sprint ID

        Returns:
            Risk assessment results
        """
        if self._sprint_risk_scorer is None:
            return {"error": "Sprint risk scorer not loaded", "sprint_id": sprint_id}

        # Build features
        features = self.pipeline.build_sprint_features(sprint_id)

        if features is None:
            return {"error": f"Sprint {sprint_id} not found", "sprint_id": sprint_id}

        # Score
        result = self._sprint_risk_scorer.score(features)

        return result

    def predict_active_sprint_risk(
        self,
        board_id: int | None = None,
    ) -> dict[str, Any]:
        """
        Predict risk for the currently active sprint.

        Args:
            board_id: Optional board filter

        Returns:
            Risk assessment results
        """
        if self._sprint_risk_scorer is None:
            return {"error": "Sprint risk scorer not loaded"}

        # Get active sprint features directly
        from src.features.sprint_features import SprintFeatureExtractor
        from src.data.schema import get_connection

        conn = get_connection()
        extractor = SprintFeatureExtractor(conn)

        features = extractor.get_active_sprint_features(board_id)

        if features is None:
            return {"error": "No active sprint found", "board_id": board_id}

        # Score
        result = self._sprint_risk_scorer.score(features)

        return result

    def assess_developer_workload(
        self,
        developer_id: str | None = None,
        project_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Assess developer workload.

        Args:
            developer_id: Specific developer ID (or all if None)
            project_key: Optional project filter

        Returns:
            Workload assessment results
        """
        if self._workload_scorer is None:
            return {"error": "Workload scorer not loaded"}

        # Get developer features
        developer_df = self.pipeline.build_developer_features(project_key)

        if developer_df.empty:
            return {"error": "No developer data available"}

        # Set team baselines
        self._workload_scorer.set_team_baselines(developer_df)

        if developer_id:
            # Single developer
            dev_data = developer_df[developer_df["assignee_id"] == developer_id]
            if dev_data.empty:
                return {"error": f"Developer {developer_id} not found"}

            result = self._workload_scorer.score(dev_data.iloc[0].to_dict())
        else:
            # Team assessment
            result = self._workload_scorer.score_team(developer_df)

        return result

    def get_reassignment_suggestions(
        self,
        project_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Get work reassignment suggestions.

        Args:
            project_key: Optional project filter

        Returns:
            Reassignment suggestions
        """
        if self._workload_scorer is None:
            return {"error": "Workload scorer not loaded"}

        # Get developer features
        developer_df = self.pipeline.build_developer_features(project_key)

        if developer_df.empty:
            return {"error": "No developer data available"}

        # Set team baselines
        self._workload_scorer.set_team_baselines(developer_df)

        # Get reassignment candidates
        result = self._workload_scorer.identify_reassignment_candidates(developer_df)

        return result

    def batch_predict_tickets(
        self,
        issue_keys: list[str],
    ) -> list[dict[str, Any]]:
        """
        Predict durations for multiple tickets.

        Args:
            issue_keys: List of Jira issue keys

        Returns:
            List of prediction results
        """
        results = []
        for key in issue_keys:
            result = self.predict_ticket(key)
            results.append(result)
        return results

    def get_model_status(self) -> dict[str, bool]:
        """
        Check which models are loaded.

        Returns:
            Dictionary of model load status
        """
        return {
            "ticket_estimator": self._ticket_estimator is not None,
            "sprint_risk_scorer": self._sprint_risk_scorer is not None,
            "workload_scorer": self._workload_scorer is not None,
        }


def main():
    """CLI entry point for predictions."""
    import argparse

    parser = argparse.ArgumentParser(description="Jira AI Co-pilot Predictions")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Ticket prediction
    ticket_parser = subparsers.add_parser("ticket", help="Predict ticket duration")
    ticket_parser.add_argument("issue_key", help="Jira issue key (e.g., PROJ-123)")
    ticket_parser.add_argument("--model-dir", "-d", default="models", help="Model directory")

    # Sprint risk prediction
    sprint_parser = subparsers.add_parser("sprint", help="Assess sprint risk")
    sprint_parser.add_argument("sprint_id", type=int, nargs="?", help="Sprint ID (or active if omitted)")
    sprint_parser.add_argument("--board", "-b", type=int, help="Board ID for active sprint")
    sprint_parser.add_argument("--model-dir", "-d", default="models", help="Model directory")

    # Workload assessment
    workload_parser = subparsers.add_parser("workload", help="Assess developer workload")
    workload_parser.add_argument("--developer", "-u", help="Developer ID (or team if omitted)")
    workload_parser.add_argument("--project", "-p", help="Project key filter")
    workload_parser.add_argument("--model-dir", "-d", default="models", help="Model directory")

    # Status
    status_parser = subparsers.add_parser("status", help="Check model status")
    status_parser.add_argument("--model-dir", "-d", default="models", help="Model directory")

    args = parser.parse_args()

    if args.command == "ticket":
        predictor = Predictor.from_model_dir(args.model_dir)
        result = predictor.predict_ticket(args.issue_key)

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

    elif args.command == "sprint":
        predictor = Predictor.from_model_dir(args.model_dir)

        if args.sprint_id:
            result = predictor.predict_sprint_risk(args.sprint_id)
        else:
            result = predictor.predict_active_sprint_risk(args.board)

        if "error" in result:
            print(f"❌ {result['error']}", file=sys.stderr)
            return 1

        print(f"\n🎯 Sprint Risk Assessment")
        print(f"   Sprint: {result.get('sprint_name', 'N/A')} (ID: {result.get('sprint_id', 'N/A')})")
        print(f"   Risk Score: {result['score']}/100")
        print(f"   Risk Level: {result['level'].upper()}")

        if result.get("recommendations"):
            print("\n   Recommendations:")
            for rec in result["recommendations"]:
                priority_emoji = "🔴" if rec["priority"] == "high" else "🟡" if rec["priority"] == "medium" else "🔵"
                print(f"   {priority_emoji} {rec['action']}")
                print(f"      └─ {rec['rationale']}")

        return 0

    elif args.command == "workload":
        predictor = Predictor.from_model_dir(args.model_dir)
        result = predictor.assess_developer_workload(args.developer, args.project)

        if "error" in result:
            print(f"❌ {result['error']}", file=sys.stderr)
            return 1

        if args.developer:
            # Single developer
            print(f"\n👤 Workload Assessment: {result.get('developer_name', 'N/A')}")
            print(f"   Score: {result['score']}/200 ({result['status'].upper()})")
            print(f"   Relative to team: {result['relative_to_team']:.0%}")

            if result.get("recommendations"):
                print("\n   Recommendations:")
                for rec in result["recommendations"]:
                    print(f"   • {rec['action']}")
        else:
            # Team assessment
            print(f"\n👥 Team Workload Assessment")
            print(f"   Team Size: {result['team_size']}")
            print(f"   Balance Score: {result['balance_score']}/100")
            print(f"   Average Workload: {result['avg_workload']:.0f}/200")
            print(f"\n   Distribution:")
            for status, count in result.get("distribution", {}).items():
                print(f"   • {status}: {count}")

            if result.get("recommendations"):
                print("\n   Team Recommendations:")
                for rec in result["recommendations"]:
                    print(f"   • {rec['action']}")

        return 0

    elif args.command == "status":
        predictor = Predictor.from_model_dir(args.model_dir)
        status = predictor.get_model_status()

        print("\n📋 Model Status")
        for model, loaded in status.items():
            status_emoji = "✅" if loaded else "❌"
            print(f"   {status_emoji} {model}")

        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
