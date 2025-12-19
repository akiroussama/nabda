"""
Rich formatting utilities for CLI output.

Provides consistent formatting for tables, panels, progress bars, and metrics.
"""

from datetime import datetime
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich import box


# Global console instance
console = Console()


def get_risk_style(level: str) -> str:
    """Get Rich style for risk level."""
    return {
        "low": "green",
        "medium": "yellow",
        "high": "red bold",
    }.get(level.lower(), "white")


def get_risk_emoji(level: str) -> str:
    """Get emoji for risk level."""
    return {
        "low": "🟢",
        "medium": "🟡",
        "high": "🔴",
    }.get(level.lower(), "⚪")


def get_priority_style(priority: str) -> str:
    """Get Rich style for priority."""
    return {
        "highest": "red bold",
        "high": "red",
        "medium": "yellow",
        "low": "green",
        "lowest": "dim",
    }.get(priority.lower(), "white")


def get_status_style(status: str) -> str:
    """Get Rich style for status."""
    status_lower = status.lower()
    if status_lower in ("done", "closed", "resolved"):
        return "green"
    elif status_lower in ("in progress", "in review", "testing"):
        return "cyan"
    elif status_lower in ("blocked", "on hold"):
        return "red"
    elif status_lower in ("to do", "open", "backlog"):
        return "dim"
    return "white"


def format_duration(hours: float) -> str:
    """Format hours as human-readable duration."""
    if hours < 1:
        return f"{int(hours * 60)}m"
    elif hours < 24:
        return f"{hours:.1f}h"
    else:
        days = hours / 24
        return f"{days:.1f}d"


def format_percentage(value: float, precision: int = 1) -> str:
    """Format value as percentage."""
    return f"{value:.{precision}f}%"


def print_header(title: str, subtitle: str | None = None) -> None:
    """Print a styled header."""
    text = Text(title, style="bold blue")
    if subtitle:
        text.append(f"\n{subtitle}", style="dim")
    console.print(Panel(text, box=box.ROUNDED))


def print_success(message: str) -> None:
    """Print success message."""
    console.print(f"[green]✓[/green] {message}")


def print_error(message: str) -> None:
    """Print error message."""
    console.print(f"[red]✗[/red] {message}")


def print_warning(message: str) -> None:
    """Print warning message."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def print_info(message: str) -> None:
    """Print info message."""
    console.print(f"[blue]ℹ[/blue] {message}")


def create_sync_progress() -> Progress:
    """Create progress bar for sync operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    )


def create_issues_table(issues: list[dict[str, Any]], title: str = "Issues") -> Table:
    """Create a formatted table of issues."""
    table = Table(title=title, box=box.ROUNDED, show_lines=True)

    table.add_column("Key", style="cyan", no_wrap=True)
    table.add_column("Type", style="dim")
    table.add_column("Status")
    table.add_column("Priority")
    table.add_column("Summary", max_width=40)
    table.add_column("Points", justify="right")
    table.add_column("Assignee")

    for issue in issues:
        status = issue.get("status", "")
        priority = issue.get("priority", "")

        table.add_row(
            issue.get("key", ""),
            issue.get("issue_type", ""),
            Text(status, style=get_status_style(status)),
            Text(priority, style=get_priority_style(priority)),
            (issue.get("summary", "") or "")[:40],
            str(issue.get("story_points", "-") or "-"),
            issue.get("assignee", "-") or "-",
        )

    return table


def create_sprint_summary_panel(sprint: dict[str, Any], risk_score: dict[str, Any] | None = None) -> Panel:
    """Create a panel with sprint summary."""
    risk_level = risk_score.get("level", "unknown") if risk_score else "unknown"
    risk_value = risk_score.get("score", 0) if risk_score else 0

    content = Text()
    content.append(f"Sprint: ", style="bold")
    content.append(f"{sprint.get('sprint_name', 'N/A')}\n")

    content.append(f"Progress: ", style="bold")
    content.append(f"{sprint.get('days_elapsed', 0)}/{sprint.get('total_days', 0)} days ")
    content.append(f"({sprint.get('progress_percent', 0):.0f}%)\n")

    content.append(f"Completion: ", style="bold")
    content.append(f"{sprint.get('completed_points', 0)}/{sprint.get('total_points', 0)} points ")
    content.append(f"({sprint.get('completion_rate', 0):.0f}%)\n")

    content.append(f"Remaining: ", style="bold")
    content.append(f"{sprint.get('remaining_points', 0)} points in {sprint.get('days_remaining', 0)} days\n")

    if risk_score:
        content.append(f"\nRisk: ", style="bold")
        content.append(f"{get_risk_emoji(risk_level)} ")
        content.append(f"{risk_value:.0f}/100 ", style=get_risk_style(risk_level))
        content.append(f"({risk_level.upper()})", style=get_risk_style(risk_level))

    return Panel(content, title="📊 Sprint Health", box=box.ROUNDED)


def create_developer_table(developers: list[dict[str, Any]], title: str = "Developer Workload") -> Table:
    """Create a formatted table of developer workload."""
    table = Table(title=title, box=box.ROUNDED)

    table.add_column("Developer", style="cyan")
    table.add_column("WIP", justify="right")
    table.add_column("Points", justify="right")
    table.add_column("Blocked", justify="right")
    table.add_column("Workload", justify="right")
    table.add_column("Status")

    for dev in developers:
        status = dev.get("status", "optimal")
        workload = dev.get("workload_relative", 1.0)

        status_style = {
            "underloaded": "dim",
            "optimal": "green",
            "high": "yellow",
            "overloaded": "red bold",
        }.get(status, "white")

        table.add_row(
            dev.get("pseudonym", dev.get("assignee_id", "Unknown")),
            str(dev.get("wip_count", 0)),
            str(dev.get("wip_points", 0)),
            str(dev.get("blocked_count", 0)),
            f"{workload:.0%}",
            Text(status.upper(), style=status_style),
        )

    return table


def create_prediction_panel(prediction: dict[str, Any]) -> Panel:
    """Create a panel with ticket prediction."""
    content = Text()

    content.append(f"Issue: ", style="bold")
    content.append(f"{prediction.get('issue_key', 'N/A')}\n\n")

    content.append(f"Estimated Duration:\n", style="bold")
    content.append(f"  {prediction.get('predicted_hours', 0):.1f} hours ")
    content.append(f"({prediction.get('predicted_days', 0):.1f} days)\n")

    if "confidence_interval" in prediction:
        ci = prediction["confidence_interval"]
        content.append(f"\nConfidence Interval:\n", style="bold")
        content.append(f"  {ci.get('lower_hours', 0):.1f} - {ci.get('upper_hours', 0):.1f} hours\n")

    content.append(f"\nModel: ", style="dim")
    content.append(f"{prediction.get('model_type', 'N/A')}", style="dim")

    return Panel(content, title="🎯 Duration Prediction", box=box.ROUNDED)


def create_recommendations_panel(recommendations: list[dict[str, str]], title: str = "Recommendations") -> Panel:
    """Create a panel with recommendations."""
    content = Text()

    for i, rec in enumerate(recommendations):
        priority = rec.get("priority", "medium")
        priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢", "info": "🔵"}.get(priority, "⚪")

        content.append(f"{priority_emoji} {rec.get('action', '')}\n")
        content.append(f"   {rec.get('rationale', '')}\n", style="dim")

        if i < len(recommendations) - 1:
            content.append("\n")

    return Panel(content, title=f"💡 {title}", box=box.ROUNDED)


def create_metrics_row(metrics: dict[str, Any]) -> Table:
    """Create a horizontal row of metrics."""
    table = Table(box=None, show_header=False, padding=(0, 2))

    for key in metrics:
        table.add_column(justify="center")

    values = []
    labels = []

    for key, value in metrics.items():
        if isinstance(value, float):
            values.append(f"{value:.1f}")
        else:
            values.append(str(value))
        labels.append(key.replace("_", " ").title())

    table.add_row(*[Text(v, style="bold cyan") for v in values])
    table.add_row(*[Text(l, style="dim") for l in labels])

    return table


def print_sync_results(results: dict[str, int]) -> None:
    """Print sync results in a formatted way."""
    table = Table(title="Sync Results", box=box.ROUNDED)

    table.add_column("Entity", style="cyan")
    table.add_column("Count", justify="right", style="green")

    for entity, count in results.items():
        table.add_row(entity.replace("_", " ").title(), str(count))

    console.print(table)


def print_model_metrics(metrics: dict[str, float], title: str = "Model Metrics") -> None:
    """Print model metrics in a formatted way."""
    table = Table(title=title, box=box.ROUNDED)

    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    for metric, value in metrics.items():
        formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
        table.add_row(metric.upper(), formatted)

    console.print(table)
