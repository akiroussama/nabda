"""
Main CLI application using Typer.

Provides command-line interface for all Jira AI Co-pilot features.
"""

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from src.cli.formatters import (
    console,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info,
)

# Create main app
app = typer.Typer(
    name="jira-copilot",
    help="🤖 Jira AI Co-pilot - Intelligent project management assistant",
    rich_markup_mode="rich",
)

# Import command groups
from src.cli.commands import sync, analyze, predict, report


# Register command groups
app.add_typer(sync.app, name="sync", help="🔄 Synchronize data from Jira")
app.add_typer(analyze.app, name="analyze", help="📊 Analyze project data")
app.add_typer(predict.app, name="predict", help="🎯 Make predictions")
app.add_typer(report.app, name="report", help="📋 Generate reports")


@app.command()
def status(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed status"),
):
    """
    Show current status of Jira AI Co-pilot.

    Displays sync status, model status, and connection health.
    """
    import duckdb
    from src.models.trainer import ModelTrainer

    print_header("Jira AI Co-pilot Status")

    # Check database (use default path)
    db_path = Path("data/jira.duckdb")
    try:
        if not db_path.exists():
            print_warning("Database not found - run 'jira-copilot init' first")
        else:
            conn = duckdb.connect(str(db_path))

            # Get issue count
            issue_count = conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
            sprint_count = conn.execute("SELECT COUNT(*) FROM sprints").fetchone()[0]

            # Get last sync
            last_sync = conn.execute(
                "SELECT MAX(last_sync_timestamp) FROM sync_metadata"
            ).fetchone()[0]

            print_success("Database connected")
            print_info(f"  Issues: {issue_count}")
            print_info(f"  Sprints: {sprint_count}")
            if last_sync:
                print_info(f"  Last sync: {last_sync}")
            else:
                print_warning("  Never synced")

    except Exception as e:
        print_error(f"Database error: {e}")

    # Check models
    console.print()
    try:
        trainer = ModelTrainer()
        model_status = trainer.load_models()

        loaded_count = sum(1 for v in model_status.values() if v)
        total_count = len(model_status)

        if loaded_count == total_count:
            print_success(f"Models loaded ({loaded_count}/{total_count})")
        elif loaded_count > 0:
            print_warning(f"Some models loaded ({loaded_count}/{total_count})")
        else:
            print_warning("No models loaded - run 'jira-copilot train' first")

        if verbose:
            for model, is_loaded in model_status.items():
                status_text = "✓" if is_loaded else "✗"
                style = "green" if is_loaded else "red"
                console.print(f"  [{style}]{status_text}[/{style}] {model}")

    except Exception as e:
        print_warning(f"Models not available: {e}")

    # Check LLM (check environment variable directly)
    console.print()
    import os
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        print_success("LLM configured (Gemini)")
    else:
        print_warning("LLM not configured - set GOOGLE_API_KEY")


@app.command()
def train(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project key filter"),
    board: Optional[int] = typer.Option(None, "--board", "-b", help="Board ID filter"),
    model_dir: str = typer.Option("models", "--model-dir", "-d", help="Model directory"),
):
    """
    Train ML models on synchronized data.

    Trains ticket estimator, sprint risk scorer, and workload scorer.
    """
    from src.cli.formatters import create_sync_progress, print_model_metrics
    from src.models.trainer import ModelTrainer

    print_header("Training ML Models")

    trainer = ModelTrainer(model_dir=model_dir)

    with create_sync_progress() as progress:
        task = progress.add_task("Training models...", total=3)

        # Train ticket estimator
        progress.update(task, description="Training ticket estimator...")
        te_results = trainer.train_ticket_estimator(project)
        progress.advance(task)

        # Train sprint risk scorer
        progress.update(task, description="Training sprint risk scorer...")
        sr_results = trainer.train_sprint_risk_scorer(board)
        progress.advance(task)

        # Initialize workload scorer
        progress.update(task, description="Initializing workload scorer...")
        ws_results = trainer.initialize_workload_scorer(project)
        progress.advance(task)

    console.print()

    # Show results
    if "error" not in te_results:
        console.print("[bold]Ticket Estimator:[/bold]")
        print_info(f"  Samples: {te_results.get('samples', 0)}")
        if te_results.get("train_metrics"):
            print_info(f"  MAE: {te_results['train_metrics'].get('mae', 0):.2f} hours")
            if te_results.get("cv_results"):
                cv = te_results["cv_results"]
                print_info(f"  CV MAE: {cv['mean_mae']:.2f} ± {cv['std_mae']:.2f}")
    else:
        print_warning(f"Ticket Estimator: {te_results.get('error')}")

    console.print()

    if "error" not in sr_results:
        console.print("[bold]Sprint Risk Scorer:[/bold]")
        print_info(f"  Mode: {sr_results.get('mode', 'N/A')}")
        if sr_results.get("ml_metrics"):
            print_info(f"  AUC-ROC: {sr_results['ml_metrics'].get('auc_roc', 0):.3f}")
    else:
        print_warning(f"Sprint Risk Scorer: {sr_results.get('error')}")

    console.print()

    if "error" not in ws_results:
        console.print("[bold]Workload Scorer:[/bold]")
        print_info(f"  Team size: {ws_results.get('team_size', 0)}")
    else:
        print_warning(f"Workload Scorer: {ws_results.get('error')}")

    # Save models
    console.print()
    paths = trainer.save_models()
    print_success(f"Models saved to {model_dir}/")


@app.command()
def dashboard(
    port: int = typer.Option(8501, "--port", "-p", help="Dashboard port"),
    host: str = typer.Option("localhost", "--host", "-h", help="Dashboard host"),
):
    """
    Launch the Streamlit dashboard.

    Opens the interactive dashboard in your browser.
    """
    import subprocess

    print_header("Launching Dashboard")

    dashboard_path = Path(__file__).parent.parent / "dashboard" / "app.py"

    if not dashboard_path.exists():
        print_error(f"Dashboard not found at {dashboard_path}")
        raise typer.Exit(1)

    print_info(f"Starting dashboard at http://{host}:{port}")
    print_info("Press Ctrl+C to stop")

    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port", str(port),
            "--server.address", host,
        ])
    except KeyboardInterrupt:
        print_info("Dashboard stopped")


@app.command()
def init(
    force: bool = typer.Option(False, "--force", "-f", help="Force re-initialization"),
):
    """
    Initialize the Jira AI Co-pilot.

    Creates database schema and default configuration.
    """
    from src.data.schema import initialize_database
    from src.intelligence.prompts import create_default_templates
    from config.settings import get_settings
    from pathlib import Path

    print_header("Initializing Jira AI Co-pilot")

    # Initialize database (use default path, don't require full settings)
    try:
        db_path = Path("data/jira.duckdb")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = initialize_database(db_path)
        print_success("Database schema initialized")
    except Exception as e:
        print_error(f"Database initialization failed: {e}")
        raise typer.Exit(1)

    # Create prompt templates
    try:
        templates_dir = Path("prompts")
        create_default_templates(templates_dir)
        print_success("Prompt templates created")
    except Exception as e:
        print_warning(f"Template creation failed: {e}")

    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print_success("Models directory created")

    # Check .env file
    env_file = Path(".env")
    if not env_file.exists():
        print_warning(".env file not found - create one with your credentials")
        console.print("""
[dim]Required environment variables:
  JIRA_URL=https://your-instance.atlassian.net
  JIRA_EMAIL=your-email@example.com
  JIRA_API_TOKEN=your-api-token
  JIRA_PROJECT_KEY=YOUR_PROJECT
  GOOGLE_API_KEY=your-google-api-key[/dim]
""")
    else:
        print_success(".env file found")

    console.print()
    print_success("Initialization complete!")
    print_info("Next steps:")
    console.print("  1. Configure your .env file with Jira credentials")
    console.print("  2. Run 'jira-copilot sync full' to sync data")
    console.print("  3. Run 'jira-copilot train' to train models")
    console.print("  4. Run 'jira-copilot dashboard' to view insights")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V", help="Show version", is_eager=True),
):
    """
    🤖 Jira AI Co-pilot - Intelligent project management assistant.

    Use 'jira-copilot COMMAND --help' for more information on a command.
    """
    if version:
        console.print("Jira AI Co-pilot v0.1.0")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit(0)


def cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    cli()
