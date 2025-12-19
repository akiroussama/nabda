"""Sync commands for Jira data synchronization."""

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
    create_sync_progress,
    print_sync_results,
)

app = typer.Typer(help="Synchronize data from Jira")


@app.command()
def full(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project key to sync"),
    board: Optional[int] = typer.Option(None, "--board", "-b", help="Board ID to sync"),
    days: int = typer.Option(90, "--days", "-d", help="Days of history to sync"),
    batch_size: int = typer.Option(100, "--batch-size", help="Batch size for API requests"),
):
    """
    Perform full data synchronization from Jira.

    Syncs issues, sprints, changelogs, and worklogs.
    """
    from src.sync.client import JiraClient
    from src.sync.syncer import JiraSyncer
    from src.data.schema import get_connection, initialize_schema
    from config.settings import get_settings

    print_header("Full Jira Sync", f"Project: {project or 'All'} | Board: {board or 'All'}")

    settings = get_settings()

    try:
        # Initialize
        conn = get_connection()
        initialize_schema(conn)

        client = JiraClient(
            url=settings.jira.url,
            email=settings.jira.email,
            api_token=settings.jira.api_token,
        )

        syncer = JiraSyncer(client=client, conn=conn)

        results = {}

        with create_sync_progress() as progress:
            # Sync issues
            task = progress.add_task("Syncing issues...", total=None)
            issue_count = syncer.sync_issues(
                project_key=project or settings.jira.project_key,
                days_back=days,
                batch_size=batch_size,
            )
            results["issues"] = issue_count
            progress.update(task, completed=True)

            # Sync sprints
            if board or settings.jira.board_id:
                progress.update(task, description="Syncing sprints...")
                sprint_count = syncer.sync_sprints(
                    board_id=board or settings.jira.board_id,
                )
                results["sprints"] = sprint_count

            # Sync changelogs
            progress.update(task, description="Syncing changelogs...")
            changelog_count = syncer.sync_changelogs(
                project_key=project or settings.jira.project_key,
            )
            results["changelogs"] = changelog_count

            # Sync worklogs
            progress.update(task, description="Syncing worklogs...")
            worklog_count = syncer.sync_worklogs(
                project_key=project or settings.jira.project_key,
            )
            results["worklogs"] = worklog_count

        console.print()
        print_sync_results(results)
        print_success("Full sync completed successfully!")

    except Exception as e:
        print_error(f"Sync failed: {e}")
        raise typer.Exit(1)


@app.command()
def issues(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project key"),
    days: int = typer.Option(30, "--days", "-d", help="Days of history"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status"),
):
    """
    Sync issues only.

    Quick sync of issues without changelogs or worklogs.
    """
    from src.sync.client import JiraClient
    from src.sync.syncer import JiraSyncer
    from src.data.schema import get_connection
    from config.settings import get_settings

    print_header("Issue Sync")

    settings = get_settings()

    try:
        conn = get_connection()
        client = JiraClient(
            url=settings.jira.url,
            email=settings.jira.email,
            api_token=settings.jira.api_token,
        )

        syncer = JiraSyncer(client=client, conn=conn)

        with create_sync_progress() as progress:
            task = progress.add_task("Syncing issues...", total=None)
            count = syncer.sync_issues(
                project_key=project or settings.jira.project_key,
                days_back=days,
            )
            progress.update(task, completed=True)

        print_success(f"Synced {count} issues")

    except Exception as e:
        print_error(f"Issue sync failed: {e}")
        raise typer.Exit(1)


@app.command()
def sprints(
    board: Optional[int] = typer.Option(None, "--board", "-b", help="Board ID"),
    state: str = typer.Option("active,closed", "--state", "-s", help="Sprint states"),
):
    """
    Sync sprints only.

    Syncs sprint data from a specific board.
    """
    from src.sync.client import JiraClient
    from src.sync.syncer import JiraSyncer
    from src.data.schema import get_connection
    from config.settings import get_settings

    print_header("Sprint Sync")

    settings = get_settings()
    board_id = board or settings.jira.board_id

    if not board_id:
        print_error("Board ID required. Use --board or set JIRA_BOARD_ID")
        raise typer.Exit(1)

    try:
        conn = get_connection()
        client = JiraClient(
            url=settings.jira.url,
            email=settings.jira.email,
            api_token=settings.jira.api_token,
        )

        syncer = JiraSyncer(client=client, conn=conn)

        with create_sync_progress() as progress:
            task = progress.add_task("Syncing sprints...", total=None)
            count = syncer.sync_sprints(board_id=board_id, states=state.split(","))
            progress.update(task, completed=True)

        print_success(f"Synced {count} sprints")

    except Exception as e:
        print_error(f"Sprint sync failed: {e}")
        raise typer.Exit(1)


@app.command()
def status():
    """
    Show sync status and last sync times.
    """
    from src.data.schema import get_connection

    print_header("Sync Status")

    try:
        conn = get_connection()

        # Get sync metadata
        metadata = conn.execute("""
            SELECT entity_type, last_sync, record_count
            FROM sync_metadata
            ORDER BY entity_type
        """).fetchall()

        if not metadata:
            print_warning("No sync history found. Run 'sync full' first.")
            return

        from rich.table import Table
        from rich import box

        table = Table(title="Sync History", box=box.ROUNDED)
        table.add_column("Entity", style="cyan")
        table.add_column("Last Sync", style="green")
        table.add_column("Records", justify="right")

        for entity, last_sync, count in metadata:
            table.add_row(
                entity.title(),
                str(last_sync) if last_sync else "Never",
                str(count) if count else "-",
            )

        console.print(table)

    except Exception as e:
        print_error(f"Failed to get sync status: {e}")
        raise typer.Exit(1)
