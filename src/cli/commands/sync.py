"""Sync commands for Jira data synchronization."""

from pathlib import Path
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
    no_worklogs: bool = typer.Option(False, "--no-worklogs", help="Skip worklog sync"),
):
    """
    Perform full data synchronization from Jira.

    Syncs issues, sprints, changelogs, and worklogs.
    """
    from src.jira_client.sync import JiraSyncOrchestrator
    from src.jira_client.auth import create_jira_client_from_settings
    from src.data.schema import initialize_database
    from config.settings import get_settings

    settings = get_settings()

    project_key = project or settings.jira.project_key
    board_id = board or settings.jira.board_id

    print_header("Full Jira Sync", f"Project: {project_key} | Board: {board_id}")

    try:
        # Initialize database
        initialize_database(settings.database.full_path)

        # Create authenticator and orchestrator
        auth = create_jira_client_from_settings()

        orchestrator = JiraSyncOrchestrator(
            authenticator=auth,
            project_key=project_key,
            board_id=board_id,
            sync_worklogs=not no_worklogs,
        )

        with create_sync_progress() as progress:
            task = progress.add_task("Syncing from Jira...", total=None)
            results = orchestrator.full_sync()
            progress.update(task, completed=True)

        console.print()
        print_sync_results(results)
        print_success("Full sync completed successfully!")

    except Exception as e:
        print_error(f"Sync failed: {e}")
        raise typer.Exit(1)


@app.command()
def issues(
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Project key"),
    jql: Optional[str] = typer.Option(None, "--jql", help="Additional JQL filter"),
):
    """
    Sync issues only.

    Quick sync of issues without full orchestration.
    """
    from src.jira_client.fetcher import create_fetcher_from_settings
    from src.data.loader import create_loader_from_settings
    from src.data.schema import initialize_database
    from config.settings import get_settings

    print_header("Issue Sync")

    settings = get_settings()
    project_key = project or settings.jira.project_key

    try:
        # Initialize database
        initialize_database(settings.database.full_path)

        # Create fetcher and loader
        fetcher = create_fetcher_from_settings()
        loader = create_loader_from_settings()

        with create_sync_progress() as progress:
            task = progress.add_task("Syncing issues...", total=None)
            issues = fetcher.fetch_issues(project_key, jql_filter=jql, include_changelog=True)
            count = loader.load_issues(issues)
            loader.update_sync_metadata("issues", "success", count)
            progress.update(task, completed=True)

        print_success(f"Synced {count} issues")

    except Exception as e:
        print_error(f"Issue sync failed: {e}")
        raise typer.Exit(1)


@app.command()
def sprints(
    board: Optional[int] = typer.Option(None, "--board", "-b", help="Board ID"),
    state: str = typer.Option("active,closed,future", "--state", "-s", help="Sprint states"),
):
    """
    Sync sprints only.

    Syncs sprint data from a specific board.
    """
    from src.jira_client.fetcher import create_fetcher_from_settings
    from src.data.loader import create_loader_from_settings
    from src.data.schema import initialize_database
    from config.settings import get_settings

    print_header("Sprint Sync")

    settings = get_settings()
    board_id = board or settings.jira.board_id

    if not board_id:
        print_error("Board ID required. Use --board or set JIRA_BOARD_ID")
        raise typer.Exit(1)

    try:
        # Initialize database
        initialize_database(settings.database.full_path)

        # Create fetcher and loader
        fetcher = create_fetcher_from_settings()
        loader = create_loader_from_settings()

        total_count = 0
        states = [s.strip() for s in state.split(",")]

        with create_sync_progress() as progress:
            task = progress.add_task("Syncing sprints...", total=len(states))

            for sprint_state in states:
                sprints = fetcher.fetch_sprints(board_id, state=sprint_state)
                count = loader.load_sprints(sprints)
                total_count += count
                progress.advance(task)

            loader.update_sync_metadata("sprints", "success", total_count)

        print_success(f"Synced {total_count} sprints")

    except Exception as e:
        print_error(f"Sprint sync failed: {e}")
        raise typer.Exit(1)


@app.command()
def status():
    """
    Show sync status and last sync times.
    """
    import duckdb
    from rich.table import Table
    from rich import box

    print_header("Sync Status")

    db_path = Path("data/jira.duckdb")

    if not db_path.exists():
        print_warning("Database not found. Run 'jira-copilot init' first.")
        return

    try:
        conn = duckdb.connect(str(db_path))

        # Get sync metadata
        metadata = conn.execute("""
            SELECT entity_type, last_sync_timestamp, last_sync_status, records_synced
            FROM sync_metadata
            ORDER BY entity_type
        """).fetchall()

        if not metadata:
            print_warning("No sync history found. Run 'jira-copilot sync full' first.")
            return

        table = Table(title="Sync History", box=box.ROUNDED)
        table.add_column("Entity", style="cyan")
        table.add_column("Last Sync", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Records", justify="right")

        for entity, last_sync, sync_status, count in metadata:
            table.add_row(
                entity.title(),
                str(last_sync) if last_sync else "Never",
                sync_status or "-",
                str(count) if count else "-",
            )

        console.print(table)

        # Show entity counts
        console.print()
        issue_count = conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
        sprint_count = conn.execute("SELECT COUNT(*) FROM sprints").fetchone()[0]
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]

        print_info(f"Total issues: {issue_count}")
        print_info(f"Total sprints: {sprint_count}")
        print_info(f"Total users: {user_count}")

    except Exception as e:
        print_error(f"Failed to get sync status: {e}")
        raise typer.Exit(1)
