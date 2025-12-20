"""
CLI commands for Good Morning Dashboard.

Provides command-line interface for morning briefings and alerts.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from src.cli.formatters import (
    console,
    print_header,
    print_success,
    print_error,
    print_warning,
    print_info,
)

app = typer.Typer(
    name="morning",
    help="Your personalized morning briefing",
    rich_markup_mode="rich",
)


@app.command("brief")
def morning_brief(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Project key (uses default if not specified)"
    ),
    timeframe: str = typer.Option(
        "daily",
        "--timeframe",
        "-t",
        help="Timeframe: daily, weekly, or monthly",
    ),
    user_id: str = typer.Option(
        "default", "--user", "-u", help="User ID for personalization"
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output file (JSON format)"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
):
    """
    Generate your personalized morning briefing.

    Provides a summary of what changed, what needs attention, and recommendations.
    """
    import duckdb
    from src.features.delta_engine import DeltaEngine

    print_header("Good Morning Briefing")

    # Get database connection
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        print_error("Database not found. Run 'jira-copilot sync' first.")
        raise typer.Exit(1)

    conn = duckdb.connect(str(db_path), read_only=True)

    # Get project key
    if not project:
        # Get from most common project
        result = conn.execute("""
            SELECT project_key, COUNT(*) as cnt
            FROM issues
            WHERE project_key IS NOT NULL
            GROUP BY project_key
            ORDER BY cnt DESC
            LIMIT 1
        """).fetchone()
        if result:
            project = result[0]
        else:
            print_error("No projects found. Run 'jira-copilot sync' first.")
            raise typer.Exit(1)

    print_info(f"Project: {project}")
    print_info(f"Timeframe: {timeframe}")

    # Initialize Delta Engine
    engine = DeltaEngine(conn)

    # Get timeframe context
    context = engine.get_timeframe_context(timeframe)

    with console.status("Computing changes...", spinner="dots"):
        # Compute delta
        delta = engine.compute_delta(project, context)

        # Get attention items
        attention_items = engine.detect_attention_items(project, context)

        # Get comparison metrics
        comparison = engine.get_comparison_metrics(project, context)

        # Get sprint context
        sprint_info = engine.get_sprint_context(project)

    console.print()

    # Time greeting
    hour = datetime.now().hour
    if hour < 12:
        greeting = "Good Morning"
    elif hour < 17:
        greeting = "Good Afternoon"
    else:
        greeting = "Good Evening"

    # Display greeting panel
    console.print(Panel(
        f"[bold]{greeting}![/bold]\n"
        f"[dim]{datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}[/dim]",
        title="Your Briefing",
        border_style="blue",
    ))

    console.print()

    # Quick Stats Table
    stats_table = Table(title="Quick Stats", show_header=True, header_style="bold cyan")
    stats_table.add_column("Metric", style="dim")
    stats_table.add_column("Value", justify="right")
    stats_table.add_column("vs Previous", justify="right")

    trend_color = {"up": "green", "down": "red", "stable": "dim"}.get(comparison.trend, "dim")
    trend_arrow = {"up": "+", "down": "", "stable": ""}.get(comparison.trend, "")

    stats_table.add_row(
        "Completed",
        str(delta.tickets_completed),
        f"[{trend_color}]{trend_arrow}{comparison.velocity_change_percent:.0f}%[/{trend_color}]",
    )
    stats_table.add_row("Points", f"{delta.points_completed:.0f}", "-")
    stats_table.add_row("Active Blockers", str(delta.active_blockers), "-")
    stats_table.add_row("Created", str(delta.tickets_created), "-")
    stats_table.add_row(
        "Regressions",
        str(delta.regressions),
        "[red]Watch[/red]" if delta.regressions > 0 else "-",
    )

    console.print(stats_table)
    console.print()

    # Good News Section
    if delta.tickets_completed > 0:
        console.print("[bold green]The Good News[/bold green]")
        for ticket in delta.completed_tickets[:3]:
            console.print(
                f"  [green]+[/green] {ticket['ticket_key']}: {ticket['summary'][:60]}..."
            )
        console.print()

    # Concerns Section
    concerns = []
    if delta.active_blockers > 0:
        concerns.append(f"{delta.active_blockers} active blocker(s)")
    if delta.regressions > 0:
        concerns.append(f"{delta.regressions} regression(s)")
    if comparison.trend == "down" and comparison.velocity_change_percent < -10:
        concerns.append(f"Velocity down {abs(comparison.velocity_change_percent):.0f}%")

    if concerns:
        console.print("[bold yellow]The Concern[/bold yellow]")
        for concern in concerns:
            console.print(f"  [yellow]![/yellow] {concern}")
        console.print()

    # Attention Items Section
    critical_items = [
        item for item in attention_items
        if item.severity.value in ("critical", "high")
    ]

    if critical_items:
        console.print("[bold red]Items Needing Attention[/bold red]")
        for item in critical_items[:5]:
            severity_color = {
                "critical": "red",
                "high": "yellow",
                "medium": "cyan",
                "low": "dim",
            }.get(item.severity.value, "dim")

            console.print(
                f"  [{severity_color}]{item.severity.value.upper()}[/{severity_color}] "
                f"{item.ticket_key}: {item.evidence.description}"
            )
            if verbose and item.suggested_action:
                console.print(f"       [dim]Suggested: {item.suggested_action}[/dim]")
        console.print()

    # Sprint Context
    if sprint_info:
        console.print("[bold cyan]Sprint Context[/bold cyan]")
        console.print(f"  Sprint: {sprint_info.get('sprint_name', 'Unknown')}")
        console.print(
            f"  Day {sprint_info.get('days_elapsed', 0)} of "
            f"{sprint_info.get('total_days', 14)}"
        )
        if sprint_info.get("is_sprint_end"):
            console.print("  [yellow]Sprint ends soon![/yellow]")
        console.print()

    # Recommendation
    if critical_items:
        top_item = critical_items[0]
        recommendation = (
            f"Focus on unblocking {top_item.ticket_key} today. "
            f"{top_item.suggested_action}"
        )
    elif delta.active_blockers > 0:
        recommendation = "Review and prioritize unblocking the active blockers."
    elif delta.tickets_created > delta.tickets_completed:
        recommendation = (
            "More tickets created than completed. "
            "Consider reviewing incoming work."
        )
    else:
        recommendation = "Keep up the good work! Monitor progress and stay proactive."

    console.print(Panel(
        recommendation,
        title="My Top Recommendation",
        border_style="cyan",
    ))

    # Output to file if requested
    if output:
        output_data = {
            "generated_at": datetime.now().isoformat(),
            "project": project,
            "timeframe": timeframe,
            "delta": delta.to_dict(),
            "attention_items": [item.to_dict() for item in attention_items],
            "comparison": comparison.to_dict(),
            "sprint_info": sprint_info,
        }

        output_path = Path(output)
        output_path.write_text(json.dumps(output_data, indent=2, default=str))
        print_success(f"Briefing saved to {output}")


@app.command("attention")
def show_attention(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Project key"
    ),
    severity: Optional[str] = typer.Option(
        None, "--severity", "-s", help="Filter by severity: critical, high, medium, low"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum items to show"),
):
    """
    Show items needing attention.

    Lists tickets that need PM attention, ranked by severity and score.
    """
    import duckdb
    from src.features.delta_engine import DeltaEngine

    print_header("Attention Queue")

    # Get database connection
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        print_error("Database not found. Run 'jira-copilot sync' first.")
        raise typer.Exit(1)

    conn = duckdb.connect(str(db_path), read_only=True)

    # Get project key
    if not project:
        result = conn.execute("""
            SELECT project_key FROM issues
            WHERE project_key IS NOT NULL
            GROUP BY project_key
            ORDER BY COUNT(*) DESC
            LIMIT 1
        """).fetchone()
        if result:
            project = result[0]
        else:
            print_error("No projects found.")
            raise typer.Exit(1)

    # Initialize Delta Engine
    engine = DeltaEngine(conn)
    context = engine.get_timeframe_context("daily")

    with console.status("Analyzing attention items...", spinner="dots"):
        attention_items = engine.detect_attention_items(project, context)

    # Filter by severity if specified
    if severity:
        attention_items = [
            item for item in attention_items
            if item.severity.value == severity.lower()
        ]

    # Display results
    if not attention_items:
        print_success("No items requiring attention!")
        return

    console.print(f"\n[bold]Found {len(attention_items)} item(s) needing attention[/bold]\n")

    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Severity", width=10)
    table.add_column("Ticket", width=12)
    table.add_column("Reason", width=20)
    table.add_column("Evidence", width=40)
    table.add_column("Score", justify="right", width=8)

    for idx, item in enumerate(attention_items[:limit], 1):
        severity_color = {
            "critical": "red",
            "high": "yellow",
            "medium": "cyan",
            "low": "dim",
        }.get(item.severity.value, "dim")

        table.add_row(
            str(idx),
            f"[{severity_color}]{item.severity.value.upper()}[/{severity_color}]",
            item.ticket_key,
            item.reason.value.replace("_", " ").title(),
            item.evidence.description[:40],
            f"{item.attention_score:.1f}",
        )

    console.print(table)

    # Show summary
    console.print()
    critical_count = len([i for i in attention_items if i.severity.value == "critical"])
    high_count = len([i for i in attention_items if i.severity.value == "high"])

    if critical_count > 0:
        print_error(f"{critical_count} critical item(s) require immediate attention")
    if high_count > 0:
        print_warning(f"{high_count} high priority item(s) need attention today")


@app.command("delta")
def show_delta(
    project: Optional[str] = typer.Option(
        None, "--project", "-p", help="Project key"
    ),
    timeframe: str = typer.Option(
        "daily", "--timeframe", "-t", help="Timeframe: daily, weekly, or monthly"
    ),
):
    """
    Show what changed in a time period.

    Displays tickets completed, created, blockers, and status changes.
    """
    import duckdb
    from src.features.delta_engine import DeltaEngine

    print_header(f"{timeframe.title()} Delta")

    # Get database connection
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        print_error("Database not found. Run 'jira-copilot sync' first.")
        raise typer.Exit(1)

    conn = duckdb.connect(str(db_path), read_only=True)

    # Get project key
    if not project:
        result = conn.execute("""
            SELECT project_key FROM issues
            WHERE project_key IS NOT NULL
            GROUP BY project_key
            ORDER BY COUNT(*) DESC
            LIMIT 1
        """).fetchone()
        if result:
            project = result[0]
        else:
            print_error("No projects found.")
            raise typer.Exit(1)

    # Initialize Delta Engine
    engine = DeltaEngine(conn)
    context = engine.get_timeframe_context(timeframe)

    with console.status("Computing delta...", spinner="dots"):
        delta = engine.compute_delta(project, context)
        comparison = engine.get_comparison_metrics(project, context)

    console.print(f"\nProject: [bold]{project}[/bold]")
    console.print(f"Period: {context.period_start} to {context.period_end}\n")

    # Summary stats
    table = Table(title="Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    table.add_row("Tickets Completed", str(delta.tickets_completed))
    table.add_row("Points Completed", f"{delta.points_completed:.0f}")
    table.add_row("Tickets Created", str(delta.tickets_created))
    table.add_row("Points Added", f"{delta.points_added:.0f}")
    table.add_row("Active Blockers", str(delta.active_blockers))
    table.add_row("Status Transitions", str(delta.status_transitions))
    table.add_row("Regressions", str(delta.regressions))
    table.add_row("After-Hours Events", str(delta.after_hours_events))
    table.add_row("Weekend Events", str(delta.weekend_events))

    console.print(table)

    # Comparison
    console.print()
    trend_color = {"up": "green", "down": "red", "stable": "dim"}.get(comparison.trend, "dim")
    console.print(
        f"Velocity vs previous period: "
        f"[{trend_color}]{comparison.velocity_change_percent:+.1f}% ({comparison.trend})[/{trend_color}]"
    )

    # Show completed tickets
    if delta.completed_tickets:
        console.print("\n[bold]Completed Tickets:[/bold]")
        for ticket in delta.completed_tickets[:10]:
            console.print(
                f"  [green]+[/green] {ticket['ticket_key']}: "
                f"{ticket['summary'][:50]}..."
            )

    # Show blockers
    if delta.blocker_tickets:
        console.print("\n[bold red]Active Blockers:[/bold red]")
        for ticket in delta.blocker_tickets:
            console.print(
                f"  [red]![/red] {ticket['ticket_key']}: "
                f"{ticket['summary'][:50]}..."
            )
