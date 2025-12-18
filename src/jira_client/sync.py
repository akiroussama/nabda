"""
Jira sync orchestrator module.

Orchestrates full and incremental synchronization of Jira data.
"""

import argparse
import sys
from datetime import datetime, timedelta
from typing import Literal

from loguru import logger

from src.data.loader import DataLoader, create_loader_from_settings
from src.data.schema import get_connection, initialize_database
from src.jira_client.auth import JiraAuthenticator, create_jira_client_from_settings
from src.jira_client.fetcher import JiraFetcher, create_fetcher_from_settings


class SyncError(Exception):
    """Raised when synchronization fails."""

    pass


class JiraSyncOrchestrator:
    """
    Orchestrates synchronization of Jira data.

    Handles both full and incremental sync modes, with proper
    error handling and metadata tracking.

    Example:
        >>> orchestrator = JiraSyncOrchestrator(auth, project_key, board_id)
        >>> orchestrator.full_sync()  # Initial sync
        >>> orchestrator.incremental_sync()  # Subsequent syncs
    """

    def __init__(
        self,
        authenticator: JiraAuthenticator,
        project_key: str,
        board_id: int,
        *,
        fetcher: JiraFetcher | None = None,
        loader: DataLoader | None = None,
        sync_worklogs: bool = True,
    ):
        """
        Initialize the sync orchestrator.

        Args:
            authenticator: Authenticated JiraAuthenticator
            project_key: Jira project key to sync
            board_id: Board ID for sprint data
            fetcher: Optional JiraFetcher (creates default if not provided)
            loader: Optional DataLoader (creates default if not provided)
            sync_worklogs: Whether to sync worklogs
        """
        self._auth = authenticator
        self._project_key = project_key
        self._board_id = board_id
        self._sync_worklogs = sync_worklogs

        # Initialize components
        self._fetcher = fetcher or create_fetcher_from_settings(authenticator)
        self._loader = loader or create_loader_from_settings()

    @property
    def project_key(self) -> str:
        """Get the project key being synced."""
        return self._project_key

    def full_sync(self) -> dict[str, int]:
        """
        Perform a full synchronization of all data.

        Fetches all issues, sprints, worklogs, and users.

        Returns:
            Dictionary with counts of synced entities
        """
        logger.info(f"Starting full sync for project {self._project_key}")
        start_time = datetime.now()

        results = {
            "issues": 0,
            "sprints": 0,
            "worklogs": 0,
            "users": 0,
        }

        try:
            # Sync users first (needed for pseudonymization mapping)
            logger.info("Syncing users...")
            users = self._fetcher.fetch_users(self._project_key)
            results["users"] = self._loader.load_users(users)
            self._loader.update_sync_metadata("users", "success", results["users"])

            # Sync sprints
            logger.info("Syncing sprints...")
            sprints = self._fetcher.fetch_sprints(self._board_id)
            results["sprints"] = self._loader.load_sprints(sprints)
            self._loader.update_sync_metadata("sprints", "success", results["sprints"])

            # Sync issues (most important)
            logger.info("Syncing issues...")
            issues = self._fetcher.fetch_issues(self._project_key, include_changelog=True)
            results["issues"] = self._loader.load_issues(issues)

            # Track the latest updated timestamp for incremental sync
            if issues:
                latest_updated = max(
                    i.get("fields", {}).get("updated", "") for i in issues
                )
                self._loader.update_sync_metadata(
                    "issues", "success", results["issues"], last_updated_value=latest_updated
                )
            else:
                self._loader.update_sync_metadata("issues", "success", 0)

            # Sync worklogs if enabled
            if self._sync_worklogs:
                logger.info("Syncing worklogs...")
                issue_keys = [i.get("key") for i in issues if i.get("key")]
                worklogs = self._fetcher.fetch_worklogs(issue_keys=issue_keys)
                results["worklogs"] = self._loader.load_worklogs(worklogs)
                self._loader.update_sync_metadata("worklogs", "success", results["worklogs"])

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Full sync completed in {duration:.1f}s: "
                f"{results['issues']} issues, {results['sprints']} sprints, "
                f"{results['worklogs']} worklogs, {results['users']} users"
            )

            return results

        except Exception as e:
            logger.error(f"Full sync failed: {e}")
            self._loader.update_sync_metadata("sync", "error", 0, str(e))
            raise SyncError(f"Full sync failed: {e}") from e

    def incremental_sync(self) -> dict[str, int]:
        """
        Perform an incremental synchronization.

        Only fetches data updated since the last sync.

        Returns:
            Dictionary with counts of synced entities
        """
        logger.info(f"Starting incremental sync for project {self._project_key}")
        start_time = datetime.now()

        results = {
            "issues": 0,
            "sprints": 0,
            "worklogs": 0,
            "users": 0,
        }

        try:
            # Get last sync timestamp for issues
            issues_metadata = self._loader.get_sync_metadata("issues")

            if not issues_metadata or not issues_metadata.get("last_updated_value"):
                logger.warning("No previous sync found, performing full sync instead")
                return self.full_sync()

            last_updated = issues_metadata["last_updated_value"]
            logger.info(f"Syncing changes since {last_updated}")

            # Build JQL for incremental sync
            jql_filter = f'updated >= "{last_updated}"'

            # Sync issues updated since last sync
            logger.info("Syncing updated issues...")
            issues = self._fetcher.fetch_issues(
                self._project_key,
                jql_filter=jql_filter,
                include_changelog=True,
            )
            results["issues"] = self._loader.load_issues(issues)

            # Track new latest timestamp
            if issues:
                latest_updated = max(
                    i.get("fields", {}).get("updated", "") for i in issues
                )
                self._loader.update_sync_metadata(
                    "issues", "success", results["issues"], last_updated_value=latest_updated
                )

            # Always sync active sprints
            logger.info("Syncing active sprints...")
            sprints = self._fetcher.fetch_sprints(self._board_id, state="active")
            # Also get recently closed sprints
            closed_sprints = self._fetcher.fetch_sprints(self._board_id, state="closed")
            # Filter to only recently closed (within last 7 days)
            recent_closed = []
            for s in closed_sprints:
                complete_date = s.get("completeDate")
                if complete_date:
                    try:
                        completed = datetime.fromisoformat(complete_date.replace("Z", "+00:00"))
                        if completed > datetime.now(completed.tzinfo) - timedelta(days=7):
                            recent_closed.append(s)
                    except (ValueError, TypeError):
                        pass

            all_sprints = sprints + recent_closed
            results["sprints"] = self._loader.load_sprints(all_sprints)
            self._loader.update_sync_metadata("sprints", "success", results["sprints"])

            # Sync worklogs for updated issues if enabled
            if self._sync_worklogs and issues:
                logger.info("Syncing worklogs for updated issues...")
                # Get worklogs metadata
                worklogs_metadata = self._loader.get_sync_metadata("worklogs")
                since_timestamp = None
                if worklogs_metadata and worklogs_metadata.get("last_sync_timestamp"):
                    since_timestamp = worklogs_metadata["last_sync_timestamp"]

                issue_keys = [i.get("key") for i in issues if i.get("key")]
                worklogs = self._fetcher.fetch_worklogs(
                    since_timestamp=since_timestamp,
                    issue_keys=issue_keys,
                )
                results["worklogs"] = self._loader.load_worklogs(worklogs)
                self._loader.update_sync_metadata("worklogs", "success", results["worklogs"])

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Incremental sync completed in {duration:.1f}s: "
                f"{results['issues']} issues, {results['sprints']} sprints, "
                f"{results['worklogs']} worklogs"
            )

            return results

        except Exception as e:
            logger.error(f"Incremental sync failed: {e}")
            self._loader.update_sync_metadata("sync", "error", 0, str(e))
            raise SyncError(f"Incremental sync failed: {e}") from e

    def sync(self, mode: Literal["full", "incremental", "auto"] = "auto") -> dict[str, int]:
        """
        Perform synchronization with specified mode.

        Args:
            mode: Sync mode ("full", "incremental", or "auto")
                  "auto" uses incremental if previous sync exists, else full

        Returns:
            Dictionary with counts of synced entities
        """
        if mode == "full":
            return self.full_sync()
        elif mode == "incremental":
            return self.incremental_sync()
        else:  # auto
            # Check if we have previous sync data
            metadata = self._loader.get_sync_metadata("issues")
            if metadata and metadata.get("last_sync_status") == "success":
                return self.incremental_sync()
            else:
                return self.full_sync()


def create_orchestrator_from_settings() -> JiraSyncOrchestrator:
    """
    Create a JiraSyncOrchestrator from application settings.

    Returns:
        Configured JiraSyncOrchestrator instance
    """
    from config.settings import get_settings

    settings = get_settings()

    # Initialize database
    initialize_database(settings.database.full_path)

    # Create authenticator
    auth = create_jira_client_from_settings()

    return JiraSyncOrchestrator(
        authenticator=auth,
        project_key=settings.jira.project_key,
        board_id=settings.jira.board_id,
        sync_worklogs=settings.enable_worklog_sync,
    )


def main():
    """CLI entry point for sync operations."""
    parser = argparse.ArgumentParser(description="Jira Data Sync")
    parser.add_argument(
        "--mode",
        choices=["full", "incremental", "auto"],
        default="auto",
        help="Sync mode (default: auto)",
    )
    parser.add_argument(
        "--project",
        help="Override project key from settings",
    )
    parser.add_argument(
        "--board",
        type=int,
        help="Override board ID from settings",
    )
    parser.add_argument(
        "--no-worklogs",
        action="store_true",
        help="Skip worklog synchronization",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger.remove()
    logger.add(sys.stderr, level=log_level)
    logger.add("logs/sync.log", rotation="10 MB", retention="7 days", level="DEBUG")

    try:
        from config.settings import get_settings

        settings = get_settings()

        # Initialize database
        initialize_database(settings.database.full_path)

        # Create authenticator
        auth = create_jira_client_from_settings()

        # Create orchestrator with optional overrides
        orchestrator = JiraSyncOrchestrator(
            authenticator=auth,
            project_key=args.project or settings.jira.project_key,
            board_id=args.board or settings.jira.board_id,
            sync_worklogs=not args.no_worklogs and settings.enable_worklog_sync,
        )

        # Run sync
        results = orchestrator.sync(mode=args.mode)

        print("\n✅ Sync completed successfully!")
        print(f"   Issues: {results['issues']}")
        print(f"   Sprints: {results['sprints']}")
        print(f"   Worklogs: {results['worklogs']}")
        print(f"   Users: {results['users']}")

        return 0

    except Exception as e:
        logger.exception("Sync failed")
        print(f"\n❌ Sync failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
