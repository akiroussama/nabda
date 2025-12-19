"""Analyze commands for project analysis."""

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
    create_issues_table,
    create_sprint_summary_panel,
    create_developer_table,
    create_recommendations_panel,
)

app = typer.Typer(help="Analyze project data")


@app.command()
def sprint(
    sprint_id: Optional[int] = typer.Option(None, "--sprint", "-s", help="Sprint ID (default: active)"),
    board: Optional[int] = typer.Option(None, "--board", "-b", help="Board ID"),
    explain: bool = typer.Option(False, "--explain", "-e", help="Include AI explanation"),
):
    """
    Analyze current or specified sprint.

    Shows sprint health, risk factors, and recommendations.
    """
    from src.data.schema import get_connection
    from src.features.sprint_features import SprintFeatureExtractor
    from src.models.predictor import UnifiedPredictor
    from config.settings import get_settings

    print_header("Sprint Analysis")

    settings = get_settings()

    try:
        conn = get_connection()

        # Get sprint
        if sprint_id:
            sprint_data = conn.execute(
                "SELECT * FROM sprints WHERE sprint_id = ?", [sprint_id]
            ).fetchone()
        else:
            # Get active sprint
            sprint_data = conn.execute("""
                SELECT * FROM sprints
                WHERE state = 'active'
                ORDER BY start_date DESC
                LIMIT 1
            """).fetchone()

        if not sprint_data:
            print_warning("No active sprint found")
            raise typer.Exit(1)

        # Extract features
        extractor = SprintFeatureExtractor(conn)
        features = extractor.extract_features(sprint_data[0])  # sprint_id

        # Get risk prediction
        predictor = UnifiedPredictor(model_dir="models")
        risk_score = predictor.predict_sprint_risk(features)

        # Display sprint summary
        console.print(create_sprint_summary_panel(features, risk_score))

        # Get sprint issues
        issues = conn.execute("""
            SELECT key, issue_type, status, priority, summary, story_points, assignee
            FROM issues
            WHERE sprint_id = ?
            ORDER BY
                CASE priority
                    WHEN 'Highest' THEN 1
                    WHEN 'High' THEN 2
                    WHEN 'Medium' THEN 3
                    WHEN 'Low' THEN 4
                    ELSE 5
                END
        """, [sprint_data[0]]).fetchall()

        if issues:
            issue_dicts = [
                {
                    "key": i[0], "issue_type": i[1], "status": i[2],
                    "priority": i[3], "summary": i[4], "story_points": i[5],
                    "assignee": i[6]
                }
                for i in issues
            ]
            console.print()
            console.print(create_issues_table(issue_dicts, "Sprint Issues"))

        # AI explanation if requested
        if explain:
            console.print()
            print_info("Generating AI explanation...")

            from src.intelligence.orchestrator import JiraIntelligence
            intel = JiraIntelligence()
            explanation = intel.explain_sprint_risk(features, risk_score)

            console.print()
            console.print(create_recommendations_panel(
                explanation.recommended_actions,
                "AI Recommendations"
            ))

    except Exception as e:
        print_error(f"Analysis failed: {e}")
        raise typer.Exit(1)


@app.command()
def workload(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project key"),
    sprint_id: Optional[int] = typer.Option(None, "--sprint", "-s", help="Sprint ID"),
    explain: bool = typer.Option(False, "--explain", "-e", help="Include AI assessment"),
):
    """
    Analyze developer workload distribution.

    Shows WIP counts, blocked items, and workload balance.
    """
    from src.data.schema import get_connection
    from src.features.developer_features import DeveloperFeatureExtractor
    from src.models.predictor import UnifiedPredictor
    from config.settings import get_settings

    print_header("Developer Workload Analysis")

    settings = get_settings()
    project_key = project or settings.jira.project_key

    try:
        conn = get_connection()
        extractor = DeveloperFeatureExtractor(conn)
        predictor = UnifiedPredictor(model_dir="models")

        # Get developers with WIP
        developers = extractor.get_active_developers(project_key, sprint_id)

        if not developers:
            print_warning("No active developers found")
            return

        # Extract features and score workload
        developer_data = []
        for dev_id in developers:
            features = extractor.extract_features(dev_id, project_key, sprint_id)
            workload = predictor.assess_developer_workload(features)

            developer_data.append({
                "assignee_id": dev_id,
                "pseudonym": features.get("pseudonym", f"dev_{dev_id[:8]}"),
                "wip_count": features.get("wip_count", 0),
                "wip_points": features.get("wip_points", 0),
                "blocked_count": features.get("blocked_count", 0),
                "workload_relative": workload.get("relative_to_team", 1.0),
                "status": workload.get("status", "optimal"),
            })

        # Sort by workload
        developer_data.sort(key=lambda x: x["workload_relative"], reverse=True)

        console.print(create_developer_table(developer_data))

        # Show overloaded developers
        overloaded = [d for d in developer_data if d["status"] == "overloaded"]
        if overloaded:
            console.print()
            print_warning(f"{len(overloaded)} developer(s) overloaded:")
            for dev in overloaded:
                console.print(f"  - {dev['pseudonym']}: {dev['wip_points']} points WIP")

        # AI assessment if requested
        if explain and developer_data:
            console.print()
            print_info("Generating AI assessment...")

            from src.intelligence.orchestrator import JiraIntelligence
            intel = JiraIntelligence()

            # Assess most overloaded developer
            most_loaded = developer_data[0]
            assessment = intel.assess_developer_workload(most_loaded)

            console.print()
            console.print(f"[bold]Assessment for {most_loaded['pseudonym']}:[/bold]")
            console.print(assessment.summary)

            if assessment.recommendations:
                console.print()
                for rec in assessment.recommendations:
                    console.print(f"  - {rec}")

    except Exception as e:
        print_error(f"Workload analysis failed: {e}")
        raise typer.Exit(1)


@app.command()
def blockers(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project key"),
    sprint_id: Optional[int] = typer.Option(None, "--sprint", "-s", help="Sprint ID"),
):
    """
    Analyze blocked issues and dependencies.

    Shows blocked items, blockers, and aging analysis.
    """
    from src.data.schema import get_connection
    from config.settings import get_settings

    print_header("Blocker Analysis")

    settings = get_settings()
    project_key = project or settings.jira.project_key

    try:
        conn = get_connection()

        # Build query
        query = """
            SELECT
                key, summary, status, assignee, priority,
                blocked_since, blocked_by
            FROM issues
            WHERE is_blocked = true
        """
        params = []

        if project_key:
            query += " AND project_key = ?"
            params.append(project_key)

        if sprint_id:
            query += " AND sprint_id = ?"
            params.append(sprint_id)

        query += " ORDER BY blocked_since ASC"

        blocked = conn.execute(query, params).fetchall()

        if not blocked:
            print_success("No blocked issues found!")
            return

        from rich.table import Table
        from rich import box
        from datetime import datetime

        table = Table(title=f"Blocked Issues ({len(blocked)})", box=box.ROUNDED)
        table.add_column("Key", style="cyan")
        table.add_column("Summary", max_width=30)
        table.add_column("Assignee")
        table.add_column("Priority")
        table.add_column("Blocked Since")
        table.add_column("Days", justify="right")
        table.add_column("Blocked By")

        for issue in blocked:
            key, summary, status, assignee, priority, blocked_since, blocked_by = issue

            days_blocked = 0
            if blocked_since:
                delta = datetime.now() - datetime.fromisoformat(str(blocked_since))
                days_blocked = delta.days

            days_style = "green" if days_blocked < 2 else "yellow" if days_blocked < 5 else "red"

            table.add_row(
                key,
                (summary or "")[:30],
                assignee or "-",
                priority or "-",
                str(blocked_since)[:10] if blocked_since else "-",
                f"[{days_style}]{days_blocked}[/{days_style}]",
                blocked_by or "-",
            )

        console.print(table)

        # Summary
        console.print()
        total_days = sum(
            (datetime.now() - datetime.fromisoformat(str(b[5]))).days
            for b in blocked if b[5]
        )
        avg_days = total_days / len(blocked) if blocked else 0

        print_info(f"Total blocked: {len(blocked)} issues")
        print_info(f"Average block duration: {avg_days:.1f} days")

        long_blocked = [b for b in blocked if b[5] and (datetime.now() - datetime.fromisoformat(str(b[5]))).days > 5]
        if long_blocked:
            print_warning(f"Long-term blockers (>5 days): {len(long_blocked)}")

    except Exception as e:
        print_error(f"Blocker analysis failed: {e}")
        raise typer.Exit(1)


@app.command()
def velocity(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project key"),
    board: Optional[int] = typer.Option(None, "--board", "-b", help="Board ID"),
    sprints: int = typer.Option(5, "--sprints", "-n", help="Number of sprints to analyze"),
):
    """
    Analyze team velocity trends.

    Shows velocity history and commitment vs completion rates.
    """
    from src.data.schema import get_connection
    from config.settings import get_settings

    print_header("Velocity Analysis")

    settings = get_settings()

    try:
        conn = get_connection()

        # Get completed sprints
        sprint_data = conn.execute("""
            SELECT
                sprint_id, sprint_name, start_date, end_date,
                committed_points, completed_points
            FROM sprints
            WHERE state = 'closed'
            ORDER BY end_date DESC
            LIMIT ?
        """, [sprints]).fetchall()

        if not sprint_data:
            print_warning("No completed sprints found")
            return

        from rich.table import Table
        from rich import box

        table = Table(title=f"Velocity (Last {len(sprint_data)} Sprints)", box=box.ROUNDED)
        table.add_column("Sprint", style="cyan")
        table.add_column("Committed", justify="right")
        table.add_column("Completed", justify="right")
        table.add_column("Rate", justify="right")
        table.add_column("Delta", justify="right")

        total_committed = 0
        total_completed = 0

        for sprint in reversed(sprint_data):  # Oldest first
            sid, name, start, end, committed, completed = sprint
            committed = committed or 0
            completed = completed or 0

            total_committed += committed
            total_completed += completed

            rate = (completed / committed * 100) if committed > 0 else 0
            delta = completed - committed

            rate_style = "green" if rate >= 90 else "yellow" if rate >= 70 else "red"
            delta_style = "green" if delta >= 0 else "red"

            table.add_row(
                name[:25] if name else f"Sprint {sid}",
                str(committed),
                str(completed),
                f"[{rate_style}]{rate:.0f}%[/{rate_style}]",
                f"[{delta_style}]{delta:+d}[/{delta_style}]",
            )

        console.print(table)

        # Summary statistics
        console.print()
        avg_velocity = total_completed / len(sprint_data)
        completion_rate = (total_completed / total_committed * 100) if total_committed > 0 else 0

        print_info(f"Average velocity: {avg_velocity:.1f} points/sprint")
        print_info(f"Overall completion rate: {completion_rate:.1f}%")

        if completion_rate < 80:
            print_warning("Consider reducing sprint commitments")
        elif completion_rate > 100:
            print_info("Team consistently exceeds commitments - consider increasing scope")

    except Exception as e:
        print_error(f"Velocity analysis failed: {e}")
        raise typer.Exit(1)
