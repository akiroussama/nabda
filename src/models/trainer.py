"""
Model training orchestration module.

Handles training, evaluation, and persistence of ML models.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from loguru import logger

from src.features.pipeline import FeaturePipeline, create_pipeline_from_settings
from src.models.sprint_risk import SprintRiskScorer
from src.models.ticket_estimator import TicketEstimator
from src.models.workload_scorer import WorkloadScorer


class ModelTrainer:
    """
    Orchestrates training and evaluation of all ML models.

    Handles feature extraction, model training, cross-validation,
    and model persistence.

    Example:
        >>> trainer = ModelTrainer()
        >>> results = trainer.train_all()
        >>> trainer.save_models("models/")
    """

    MODEL_DIR = Path("models")

    def __init__(
        self,
        pipeline: FeaturePipeline | None = None,
        model_dir: str | Path | None = None,
    ):
        """
        Initialize the model trainer.

        Args:
            pipeline: Feature pipeline instance
            model_dir: Directory for model storage
        """
        self._pipeline = pipeline
        self._model_dir = Path(model_dir) if model_dir else self.MODEL_DIR

        # Model instances
        self.ticket_estimator: TicketEstimator | None = None
        self.sprint_risk_scorer: SprintRiskScorer | None = None
        self.workload_scorer: WorkloadScorer | None = None

        # Training results
        self._training_results: dict[str, Any] = {}

    @property
    def pipeline(self) -> FeaturePipeline:
        """Get or create the feature pipeline."""
        if self._pipeline is None:
            self._pipeline = create_pipeline_from_settings()
        return self._pipeline

    def train_ticket_estimator(
        self,
        project_key: str | None = None,
        *,
        model_type: Literal["baseline", "linear", "lightgbm"] = "lightgbm",
        cross_validate: bool = True,
        cv_folds: int = 5,
    ) -> dict[str, Any]:
        """
        Train the ticket duration estimation model.

        Args:
            project_key: Optional project filter
            model_type: Model type to train
            cross_validate: Whether to run cross-validation
            cv_folds: Number of CV folds

        Returns:
            Training results with metrics
        """
        logger.info(f"Training ticket estimator ({model_type})")

        # Build training set
        X, y = self.pipeline.build_ticket_training_set(project_key)

        if X.empty:
            logger.warning("No training data available for ticket estimator")
            return {"error": "No training data available", "samples": 0}

        logger.info(f"Training set: {len(X)} samples, {len(X.columns)} features")

        # Initialize model
        self.ticket_estimator = TicketEstimator(model_type=model_type)

        # Cross-validation
        cv_results = None
        if cross_validate and len(X) >= cv_folds * 10:
            logger.info(f"Running {cv_folds}-fold cross-validation")
            cv_results = self.ticket_estimator.cross_validate(X, y, n_splits=cv_folds)
            logger.info(f"CV MAE: {cv_results['mean_mae']:.2f} ± {cv_results['std_mae']:.2f}")

        # Train final model on full data
        train_metrics = self.ticket_estimator.train(X, y)

        results = {
            "model_type": model_type,
            "samples": len(X),
            "features": len(X.columns),
            "train_metrics": train_metrics,
            "cv_results": cv_results,
            "trained_at": datetime.now().isoformat(),
        }

        self._training_results["ticket_estimator"] = results

        logger.info(f"Ticket estimator training complete. MAE: {train_metrics['mae']:.2f} hours")

        return results

    def train_sprint_risk_scorer(
        self,
        board_id: int | None = None,
        *,
        mode: Literal["rule_based", "ml", "hybrid"] = "hybrid",
    ) -> dict[str, Any]:
        """
        Train the sprint risk scoring model.

        Args:
            board_id: Optional board filter
            mode: Scoring mode

        Returns:
            Training results with metrics
        """
        logger.info(f"Training sprint risk scorer ({mode})")

        # Initialize scorer
        self.sprint_risk_scorer = SprintRiskScorer(mode=mode)

        # If ML or hybrid, train ML component
        ml_metrics = None
        if mode in ["ml", "hybrid"]:
            X, y = self.pipeline.build_sprint_training_set(board_id)

            if X.empty:
                logger.warning("No training data for sprint risk ML component")
                if mode == "ml":
                    return {"error": "No training data available", "samples": 0}
                logger.info("Falling back to rule-based only")
            else:
                logger.info(f"Training ML component with {len(X)} sprints")
                ml_metrics = self.sprint_risk_scorer.train_ml_model(X, y)
                logger.info(f"ML AUC-ROC: {ml_metrics.get('auc_roc', 0):.3f}")

        results = {
            "mode": mode,
            "ml_metrics": ml_metrics,
            "trained_at": datetime.now().isoformat(),
        }

        self._training_results["sprint_risk_scorer"] = results

        logger.info("Sprint risk scorer training complete")

        return results

    def initialize_workload_scorer(
        self,
        project_key: str | None = None,
    ) -> dict[str, Any]:
        """
        Initialize the workload scorer with team baselines.

        Args:
            project_key: Optional project filter

        Returns:
            Initialization results
        """
        logger.info("Initializing workload scorer")

        self.workload_scorer = WorkloadScorer()

        # Get developer data for baselines
        developer_df = self.pipeline.build_developer_features(project_key)

        if developer_df.empty:
            logger.warning("No developer data for workload baselines")
            return {"error": "No developer data available", "team_size": 0}

        # Set team baselines
        self.workload_scorer.set_team_baselines(developer_df)

        results = {
            "team_size": len(developer_df),
            "initialized_at": datetime.now().isoformat(),
        }

        self._training_results["workload_scorer"] = results

        logger.info(f"Workload scorer initialized with {len(developer_df)} developers")

        return results

    def train_all(
        self,
        project_key: str | None = None,
        board_id: int | None = None,
    ) -> dict[str, Any]:
        """
        Train all models.

        Args:
            project_key: Optional project filter
            board_id: Optional board filter

        Returns:
            Combined training results
        """
        logger.info("Training all models")

        results = {
            "started_at": datetime.now().isoformat(),
        }

        # Train ticket estimator
        results["ticket_estimator"] = self.train_ticket_estimator(project_key)

        # Train sprint risk scorer
        results["sprint_risk_scorer"] = self.train_sprint_risk_scorer(board_id)

        # Initialize workload scorer
        results["workload_scorer"] = self.initialize_workload_scorer(project_key)

        results["completed_at"] = datetime.now().isoformat()

        logger.info("All models trained successfully")

        return results

    def save_models(self, model_dir: str | Path | None = None) -> dict[str, str]:
        """
        Save all trained models to disk.

        Args:
            model_dir: Directory for model storage

        Returns:
            Dictionary of model paths
        """
        save_dir = Path(model_dir) if model_dir else self._model_dir
        save_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save ticket estimator
        if self.ticket_estimator is not None:
            path = save_dir / "ticket_estimator.pkl"
            self.ticket_estimator.save(path)
            paths["ticket_estimator"] = str(path)

        # Save sprint risk scorer
        if self.sprint_risk_scorer is not None:
            path = save_dir / "sprint_risk_scorer.pkl"
            self.sprint_risk_scorer.save(path)
            paths["sprint_risk_scorer"] = str(path)

        # Save training metadata
        metadata = {
            "saved_at": datetime.now().isoformat(),
            "models": list(paths.keys()),
            "training_results": self._training_results,
        }

        metadata_path = save_dir / "training_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        paths["metadata"] = str(metadata_path)

        logger.info(f"Models saved to {save_dir}")

        return paths

    def load_models(self, model_dir: str | Path | None = None) -> dict[str, bool]:
        """
        Load all models from disk.

        Args:
            model_dir: Directory containing models

        Returns:
            Dictionary of load status per model
        """
        load_dir = Path(model_dir) if model_dir else self._model_dir
        status = {}

        # Load ticket estimator
        ticket_path = load_dir / "ticket_estimator.pkl"
        if ticket_path.exists():
            self.ticket_estimator = TicketEstimator.load(ticket_path)
            status["ticket_estimator"] = True
        else:
            status["ticket_estimator"] = False

        # Load sprint risk scorer
        sprint_path = load_dir / "sprint_risk_scorer.pkl"
        if sprint_path.exists():
            self.sprint_risk_scorer = SprintRiskScorer.load(sprint_path)
            status["sprint_risk_scorer"] = True
        else:
            status["sprint_risk_scorer"] = False

        # Initialize workload scorer (no persistence needed)
        self.workload_scorer = WorkloadScorer()
        status["workload_scorer"] = True

        logger.info(f"Models loaded from {load_dir}: {status}")

        return status

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about loaded models.

        Returns:
            Dictionary with model details
        """
        info = {}

        if self.ticket_estimator is not None:
            info["ticket_estimator"] = {
                "loaded": True,
                "model_type": self.ticket_estimator.model_type,
                "features": self.ticket_estimator.feature_names_,
                "metrics": self.ticket_estimator.metrics_,
                "trained_at": self.ticket_estimator.trained_at_,
            }
        else:
            info["ticket_estimator"] = {"loaded": False}

        if self.sprint_risk_scorer is not None:
            info["sprint_risk_scorer"] = {
                "loaded": True,
                "mode": self.sprint_risk_scorer.mode,
                "has_ml_model": self.sprint_risk_scorer.ml_model_ is not None,
                "trained_at": self.sprint_risk_scorer.trained_at_,
            }
        else:
            info["sprint_risk_scorer"] = {"loaded": False}

        if self.workload_scorer is not None:
            info["workload_scorer"] = {
                "loaded": True,
                "has_baselines": self.workload_scorer._team_baselines is not None,
            }
        else:
            info["workload_scorer"] = {"loaded": False}

        return info


def train_models(
    project_key: str | None = None,
    board_id: int | None = None,
    model_dir: str = "models",
) -> dict[str, Any]:
    """
    Train all models and save to disk.

    Args:
        project_key: Optional project filter
        board_id: Optional board filter
        model_dir: Directory for model storage

    Returns:
        Training results
    """
    trainer = ModelTrainer(model_dir=model_dir)

    # Train all models
    results = trainer.train_all(project_key, board_id)

    # Save models
    paths = trainer.save_models()
    results["saved_paths"] = paths

    return results


def main():
    """CLI entry point for model training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train Jira AI Co-pilot ML Models")
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--project", "-p", help="Project key filter")
    train_parser.add_argument("--board", "-b", type=int, help="Board ID filter")
    train_parser.add_argument("--model-dir", "-d", default="models", help="Model directory")
    train_parser.add_argument(
        "--model",
        "-m",
        choices=["ticket", "sprint", "workload", "all"],
        default="all",
        help="Which model to train",
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("--model-dir", "-d", default="models", help="Model directory")

    args = parser.parse_args()

    if args.command == "train":
        print("\n📊 Training Jira AI Co-pilot Models\n")

        trainer = ModelTrainer(model_dir=args.model_dir)

        if args.model == "all":
            results = trainer.train_all(args.project, args.board)
        elif args.model == "ticket":
            results = {"ticket_estimator": trainer.train_ticket_estimator(args.project)}
        elif args.model == "sprint":
            results = {"sprint_risk_scorer": trainer.train_sprint_risk_scorer(args.board)}
        elif args.model == "workload":
            results = {"workload_scorer": trainer.initialize_workload_scorer(args.project)}

        # Save models
        paths = trainer.save_models()

        print("\n✅ Training Complete\n")

        # Print results
        if "ticket_estimator" in results:
            te = results["ticket_estimator"]
            if "error" not in te:
                print(f"Ticket Estimator:")
                print(f"  Samples: {te.get('samples', 0)}")
                print(f"  MAE: {te.get('train_metrics', {}).get('mae', 0):.2f} hours")
                if te.get("cv_results"):
                    cv = te["cv_results"]
                    print(f"  CV MAE: {cv['mean_mae']:.2f} ± {cv['std_mae']:.2f}")

        if "sprint_risk_scorer" in results:
            sr = results["sprint_risk_scorer"]
            if "error" not in sr:
                print(f"\nSprint Risk Scorer:")
                print(f"  Mode: {sr.get('mode', 'N/A')}")
                if sr.get("ml_metrics"):
                    print(f"  AUC-ROC: {sr['ml_metrics'].get('auc_roc', 0):.3f}")

        if "workload_scorer" in results:
            ws = results["workload_scorer"]
            if "error" not in ws:
                print(f"\nWorkload Scorer:")
                print(f"  Team Size: {ws.get('team_size', 0)}")

        print(f"\n📁 Models saved to: {args.model_dir}/")

        return 0

    elif args.command == "info":
        print("\n📋 Model Information\n")

        trainer = ModelTrainer(model_dir=args.model_dir)
        status = trainer.load_models()
        info = trainer.get_model_info()

        for model_name, model_info in info.items():
            print(f"{model_name}:")
            if model_info.get("loaded"):
                for key, value in model_info.items():
                    if key != "loaded":
                        print(f"  {key}: {value}")
            else:
                print("  Not loaded")
            print()

        return 0

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
