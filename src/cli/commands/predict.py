"""Predict commands for ML predictions."""

from datetime import datetime
from typing import Optional

import typer

from src.cli.formatters import (
    console,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info,
    create_prediction_panel,
    create_recommendations_panel,
    get_risk_emoji,
    get_risk_style,
)

app = typer.Typer(help="Make predictions using trained models")


@app.command()
def ticket(
    issue_key: str = typer.Argument(..., help="Issue key (e.g., PROJ-123)"),
    explain: bool = typer.Option(False, "--explain", "-e", help="Include AI explanation"),
    model_dir: str = typer.Option("models", "--model-dir", "-d", help="Model directory"),
):
    """
    Predict duration for a specific ticket.

    Uses the trained ticket estimator model.
    """
    from src.data.schema import get_connection
    from src.features.ticket_features import TicketFeatureExtractor
    from src.models.predictor import UnifiedPredictor

    print_header("Ticket Duration Prediction", issue_key)

    try:
        conn = get_connection()

        # Get issue
        issue = conn.execute(
            "SELECT * FROM issues WHERE key = ?", [issue_key]
        ).fetchone()

        if not issue:
            print_error(f"Issue {issue_key} not found")
            raise typer.Exit(1)

        # Extract features
        extractor = TicketFeatureExtractor(conn)
        features = extractor.extract_features(issue_key)

        # Predict
        predictor = UnifiedPredictor(model_dir=model_dir)
        prediction = predictor.predict_ticket(features)

        # Display prediction
        prediction["issue_key"] = issue_key
        console.print(create_prediction_panel(prediction))

        # Show feature importance if verbose
        if features.get("complexity_indicators"):
            console.print()
            console.print("[bold]Complexity Factors:[/bold]")
            for factor, value in features["complexity_indicators"].items():
                if value:
                    console.print(f"  - {factor}: {value}")

        # AI explanation
        if explain:
            console.print()
            print_info("Generating AI summary...")

            from src.intelligence.orchestrator import JiraIntelligence
            intel = JiraIntelligence()

            ticket_data = {
                "key": issue_key,
                "summary": issue[2],  # Assuming column order
                "description": issue[3],
                "issue_type": issue[4],
                "status": issue[5],
                "priority": issue[6],
            }

            summary = intel.summarize_ticket(ticket_data)

            console.print()
            console.print("[bold]AI Summary:[/bold]")
            console.print(summary.summary)

            if summary.next_action:
                console.print(f"\n[bold]Suggested Action:[/bold] {summary.next_action}")

    except Exception as e:
        print_error(f"Prediction failed: {e}")
        raise typer.Exit(1)


@app.command()
def risk(
    sprint_id: Optional[int] = typer.Option(None, "--sprint", "-s", help="Sprint ID (default: active)"),
    board: Optional[int] = typer.Option(None, "--board", "-b", help="Board ID"),
    explain: bool = typer.Option(False, "--explain", "-e", help="Include AI explanation"),
    model_dir: str = typer.Option("models", "--model-dir", "-d", help="Model directory"),
):
    """
    Predict risk for current or specified sprint.

    Returns risk score and contributing factors.
    """
    from src.data.schema import get_connection
    from src.features.sprint_features import SprintFeatureExtractor
    from src.models.predictor import UnifiedPredictor
    from config.settings import get_settings

    print_header("Sprint Risk Prediction")

    settings = get_settings()

    try:
        conn = get_connection()

        # Get sprint
        if sprint_id:
            sprint = conn.execute(
                "SELECT * FROM sprints WHERE sprint_id = ?", [sprint_id]
            ).fetchone()
        else:
            sprint = conn.execute("""
                SELECT * FROM sprints
                WHERE state = 'active'
                ORDER BY start_date DESC
                LIMIT 1
            """).fetchone()

        if not sprint:
            print_warning("No active sprint found")
            raise typer.Exit(1)

        sprint_id = sprint[0]
        sprint_name = sprint[1]

        print_info(f"Analyzing: {sprint_name}")
        console.print()

        # Extract features
        extractor = SprintFeatureExtractor(conn)
        features = extractor.extract_features(sprint_id)

        # Predict risk
        predictor = UnifiedPredictor(model_dir=model_dir)
        risk_score = predictor.predict_sprint_risk(features)

        # Display results
        level = risk_score.get("level", "unknown")
        score = risk_score.get("score", 0)

        console.print(f"[bold]Risk Score:[/bold] {get_risk_emoji(level)} ", end="")
        console.print(f"[{get_risk_style(level)}]{score:.0f}/100 ({level.upper()})[/{get_risk_style(level)}]")

        # Show factors
        if risk_score.get("factors"):
            console.print()
            console.print("[bold]Contributing Factors:[/bold]")

            factors = sorted(
                risk_score["factors"].items(),
                key=lambda x: x[1].get("contribution", 0),
                reverse=True
            )

            for factor, details in factors[:5]:
                contribution = details.get("contribution", 0) * 100
                console.print(f"  - {factor}: {contribution:.1f}%")

        # AI explanation
        if explain:
            console.print()
            print_info("Generating AI explanation...")

            from src.intelligence.orchestrator import JiraIntelligence
            intel = JiraIntelligence()

            explanation = intel.explain_sprint_risk(features, risk_score)

            console.print()
            console.print("[bold]Risk Summary:[/bold]")
            console.print(explanation.risk_summary)

            if explanation.main_concerns:
                console.print()
                console.print("[bold]Main Concerns:[/bold]")
                for concern in explanation.main_concerns:
                    console.print(f"  - {concern}")

            if explanation.recommended_actions:
                console.print()
                console.print(create_recommendations_panel(
                    explanation.recommended_actions,
                    "Recommended Actions"
                ))

    except Exception as e:
        print_error(f"Risk prediction failed: {e}")
        raise typer.Exit(1)


@app.command()
def priorities(
    sprint_id: Optional[int] = typer.Option(None, "--sprint", "-s", help="Sprint ID"),
    top: int = typer.Option(10, "--top", "-n", help="Number of top priorities"),
    model_dir: str = typer.Option("models", "--model-dir", "-d", help="Model directory"),
):
    """
    Get AI-suggested priority ranking for sprint.

    Suggests which tickets to focus on based on risk and impact.
    """
    from src.data.schema import get_connection
    from src.features.sprint_features import SprintFeatureExtractor
    from src.intelligence.orchestrator import JiraIntelligence
    from config.settings import get_settings

    print_header("AI Priority Suggestions")

    settings = get_settings()

    try:
        conn = get_connection()

        # Get sprint
        if sprint_id:
            sprint = conn.execute(
                "SELECT * FROM sprints WHERE sprint_id = ?", [sprint_id]
            ).fetchone()
        else:
            sprint = conn.execute("""
                SELECT * FROM sprints
                WHERE state = 'active'
                ORDER BY start_date DESC
                LIMIT 1
            """).fetchone()

        if not sprint:
            print_warning("No active sprint found")
            raise typer.Exit(1)

        # Extract sprint features
        extractor = SprintFeatureExtractor(conn)
        features = extractor.extract_features(sprint[0])

        # Get sprint tickets
        tickets = conn.execute("""
            SELECT key, summary, status, issue_type, priority, story_points
            FROM issues
            WHERE sprint_id = ? AND status NOT IN ('Done', 'Closed')
            ORDER BY
                CASE priority
                    WHEN 'Highest' THEN 1
                    WHEN 'High' THEN 2
                    WHEN 'Medium' THEN 3
                    ELSE 4
                END
        """, [sprint[0]]).fetchall()

        if not tickets:
            print_warning("No open tickets in sprint")
            return

        ticket_dicts = [
            {
                "key": t[0], "summary": t[1], "status": t[2],
                "issue_type": t[3], "priority": t[4], "story_points": t[5]
            }
            for t in tickets
        ]

        # Get AI suggestions
        intel = JiraIntelligence()
        suggestion = intel.suggest_priorities(features, ticket_dicts)

        # Display results
        console.print("[bold]Must Complete:[/bold]")
        for key in suggestion.must_complete[:top]:
            ticket = next((t for t in ticket_dicts if t["key"] == key), None)
            if ticket:
                console.print(f"  [green]●[/green] {key}: {ticket['summary'][:50]}")

        if suggestion.consider_deferring:
            console.print()
            console.print("[bold]Consider Deferring:[/bold]")
            for key in suggestion.consider_deferring[:5]:
                ticket = next((t for t in ticket_dicts if t["key"] == key), None)
                if ticket:
                    console.print(f"  [yellow]○[/yellow] {key}: {ticket['summary'][:50]}")

        console.print()
        console.print("[bold]Focus Recommendation:[/bold]")
        console.print(f"  {suggestion.focus_recommendation}")

        if suggestion.risk_if_unchanged:
            console.print()
            console.print("[bold]Risk if Unchanged:[/bold]")
            console.print(f"  [red]{suggestion.risk_if_unchanged}[/red]")

    except Exception as e:
        print_error(f"Priority suggestion failed: {e}")
        raise typer.Exit(1)


@app.command()
def batch(
    input_file: str = typer.Option(None, "--input", "-i", help="JSON file with issue keys"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project key"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file (JSON)"),
    model_dir: str = typer.Option("models", "--model-dir", "-d", help="Model directory"),
):
    """
    Batch predict durations for multiple tickets.

    Accepts a JSON file with issue keys or processes all open issues.
    """
    import json
    from src.data.schema import get_connection
    from src.features.ticket_features import TicketFeatureExtractor
    from src.models.predictor import UnifiedPredictor
    from config.settings import get_settings

    print_header("Batch Ticket Prediction")

    settings = get_settings()
    project_key = project or settings.jira.project_key

    try:
        conn = get_connection()
        extractor = TicketFeatureExtractor(conn)
        predictor = UnifiedPredictor(model_dir=model_dir)

        # Get issue keys
        if input_file:
            with open(input_file) as f:
                data = json.load(f)
                issue_keys = data.get("issues", data) if isinstance(data, dict) else data
        else:
            issues = conn.execute("""
                SELECT key FROM issues
                WHERE project_key = ? AND status NOT IN ('Done', 'Closed')
            """, [project_key]).fetchall()
            issue_keys = [i[0] for i in issues]

        if not issue_keys:
            print_warning("No issues to predict")
            return

        print_info(f"Processing {len(issue_keys)} issues...")

        from src.cli.formatters import create_sync_progress

        results = []

        with create_sync_progress() as progress:
            task = progress.add_task("Predicting...", total=len(issue_keys))

            for key in issue_keys:
                try:
                    features = extractor.extract_features(key)
                    prediction = predictor.predict_ticket(features)
                    prediction["issue_key"] = key
                    prediction["status"] = "success"
                    results.append(prediction)
                except Exception as e:
                    results.append({
                        "issue_key": key,
                        "status": "error",
                        "error": str(e)
                    })

                progress.advance(task)

        # Output results
        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print_success(f"Results saved to {output}")
        else:
            # Display summary
            successful = [r for r in results if r.get("status") == "success"]
            failed = [r for r in results if r.get("status") == "error"]

            console.print()
            print_success(f"Predicted: {len(successful)} issues")
            if failed:
                print_warning(f"Failed: {len(failed)} issues")

            # Show top 5 longest predictions
            if successful:
                console.print()
                console.print("[bold]Longest Predicted Durations:[/bold]")
                sorted_results = sorted(
                    successful,
                    key=lambda x: x.get("predicted_hours", 0),
                    reverse=True
                )

                for r in sorted_results[:5]:
                    hours = r.get("predicted_hours", 0)
                    console.print(f"  {r['issue_key']}: {hours:.1f} hours ({hours/8:.1f} days)")

    except Exception as e:
        print_error(f"Batch prediction failed: {e}")
        raise typer.Exit(1)
