"""
Jira data fetcher module.

Provides methods to fetch issues, sprints, worklogs, and users from Jira
with automatic pagination and rate limiting.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from jira import JIRA
from jira.exceptions import JIRAError
from loguru import logger

from src.jira_client.auth import JiraAuthenticator
from src.jira_client.rate_limiter import RateLimiter, get_rate_limiter, rate_limited


class JiraFetchError(Exception):
    """Raised when fetching data from Jira fails."""

    pass


class JiraFetcher:
    """
    Fetches data from Jira with pagination and rate limiting.

    Handles fetching of issues (with changelog), sprints, worklogs,
    users, and boards.

    Example:
        >>> auth = JiraAuthenticator(url, email, token)
        >>> fetcher = JiraFetcher(auth)
        >>> issues = fetcher.fetch_issues("PROJ")
    """

    # Default batch size for pagination
    DEFAULT_BATCH_SIZE = 100

    # Fields to expand when fetching issues
    DEFAULT_EXPAND = ["changelog", "renderedFields"]

    # Issue fields to retrieve
    ISSUE_FIELDS = [
        "summary",
        "description",
        "issuetype",
        "status",
        "priority",
        "assignee",
        "reporter",
        "created",
        "updated",
        "resolutiondate",
        "components",
        "labels",
        "subtasks",
        "issuelinks",
        "attachment",
        "project",
        "customfield_10020",  # Sprint field
    ]

    def __init__(
        self,
        authenticator: JiraAuthenticator,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        rate_limiter: RateLimiter | None = None,
        custom_fields: dict[str, str] | None = None,
        raw_data_dir: Path | None = None,
    ):
        """
        Initialize the Jira fetcher.

        Args:
            authenticator: Authenticated JiraAuthenticator instance
            batch_size: Number of items per API request
            rate_limiter: Optional rate limiter (uses global if not provided)
            custom_fields: Mapping of field names to custom field IDs
            raw_data_dir: Directory to save raw JSON data (optional)
        """
        self._auth = authenticator
        self._jira = authenticator.client
        self._batch_size = min(batch_size, 100)  # Jira max is 100
        self._rate_limiter = rate_limiter or get_rate_limiter()
        self._custom_fields = custom_fields or {}
        self._raw_data_dir = raw_data_dir

        # Build fields list including custom fields
        self._fields = self.ISSUE_FIELDS + list(self._custom_fields.values())

    @property
    def jira(self) -> JIRA:
        """Get the JIRA client instance."""
        return self._jira

    @rate_limited(max_attempts=3)
    def fetch_issues(
        self,
        project_key: str,
        jql_filter: str | None = None,
        *,
        include_changelog: bool = True,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch all issues from a project with pagination.

        Args:
            project_key: Jira project key (e.g., "PROJ")
            jql_filter: Additional JQL filter to apply
            include_changelog: Whether to expand changelog
            max_results: Maximum number of issues to fetch (None for all)

        Returns:
            List of issue dictionaries with expanded changelog

        Raises:
            JiraFetchError: If fetching fails
        """
        # Build JQL query
        jql = f"project = {project_key}"
        if jql_filter:
            jql = f"{jql} AND ({jql_filter})"
        jql = f"{jql} ORDER BY created ASC"

        logger.info(f"Fetching issues with JQL: {jql}")

        all_issues: list[dict[str, Any]] = []
        expand = ",".join(self.DEFAULT_EXPAND) if include_changelog else ""

        try:
            # Use REST API directly for Jira Cloud compatibility
            # The search_issues method triggers deprecation warnings in Jira Cloud
            all_issues = self._fetch_issues_via_rest(
                jql=jql,
                expand=expand,
                max_results=max_results,
            )

            logger.info(f"Fetched {len(all_issues)} issues total")

            # Save raw data if directory specified
            if self._raw_data_dir:
                self._save_raw_data(all_issues, "issues")

            return all_issues

        except JIRAError as e:
            logger.error(f"Failed to fetch issues: {e.text}")
            raise JiraFetchError(f"Failed to fetch issues: {e.text}") from e

    def _fetch_issues_via_rest(
        self,
        jql: str,
        expand: str,
        max_results: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch issues using the enhanced_search_issues method.

        Uses the new Jira Cloud API /rest/api/3/search/jql endpoint.
        See: https://developer.atlassian.com/changelog/#CHANGE-2046
        """
        all_issues: list[dict[str, Any]] = []

        try:
            # Use enhanced_search_issues which handles the new API
            # It returns an iterator that handles pagination automatically
            issues_iterator = self._jira.enhanced_search_issues(
                jql,
                maxResults=max_results if max_results else False,  # False = all results
                expand=expand if expand else None,
                fields=self._fields,
            )

            count = 0
            for issue in issues_iterator:
                with self._rate_limiter.limit():
                    issue_dict = self._issue_to_dict(issue)
                    all_issues.append(issue_dict)
                    count += 1

                    if count % 100 == 0:
                        logger.debug(f"Fetched {count} issues so far...")

                    if max_results and count >= max_results:
                        break

            return all_issues

        except Exception as e:
            # Fallback to standard search_issues for Server/DC
            logger.warning(f"Enhanced search failed, falling back to standard search: {e}")
            return self._fetch_issues_fallback(jql, expand, max_results, 0)

    def _fetch_issues_fallback(
        self,
        jql: str,
        expand: str,
        max_results: int | None,
        start_at: int,
    ) -> list[dict[str, Any]]:
        """Fallback method using the standard search_issues (for Server/DC)."""
        import warnings

        all_issues: list[dict[str, Any]] = []

        # Suppress the deprecation warning since this is a fallback
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*deprecated.*")

            while True:
                with self._rate_limiter.limit():
                    batch = self._jira.search_issues(
                        jql,
                        startAt=start_at,
                        maxResults=self._batch_size,
                        expand=expand,
                        fields=self._fields,
                    )

                if not batch:
                    break

                for issue in batch:
                    issue_dict = self._issue_to_dict(issue)
                    all_issues.append(issue_dict)

                if len(batch) < self._batch_size:
                    break

                if max_results and len(all_issues) >= max_results:
                    all_issues = all_issues[:max_results]
                    break

                start_at += len(batch)

        return all_issues

    def _raw_issue_to_dict(self, issue_data: dict) -> dict[str, Any]:
        """Convert raw JSON issue data to our dictionary format."""
        fields = issue_data.get("fields", {})

        # Extract changelog if present
        changelog = []
        changelog_data = issue_data.get("changelog", {})
        histories = changelog_data.get("histories", [])

        for history in histories:
            author = history.get("author", {})
            history_entry = {
                "created": history.get("created"),
                "author": {
                    "accountId": author.get("accountId"),
                    "displayName": author.get("displayName"),
                }
                if author
                else None,
                "items": [
                    {
                        "field": item.get("field"),
                        "fromString": item.get("fromString"),
                        "toString": item.get("toString"),
                    }
                    for item in history.get("items", [])
                ],
            }
            changelog.append(history_entry)

        return {
            "key": issue_data.get("key"),
            "id": issue_data.get("id"),
            "self": issue_data.get("self"),
            "fields": fields,
            "changelog": {"histories": changelog},
        }

    def _issue_to_dict(self, issue) -> dict[str, Any]:
        """Convert a Jira issue object to a dictionary."""
        fields = issue.raw.get("fields", {})

        # Extract changelog if present
        changelog = []
        if hasattr(issue, "changelog") and issue.changelog:
            for history in issue.changelog.histories:
                history_entry = {
                    "created": history.created,
                    "author": {
                        "accountId": getattr(history.author, "accountId", None),
                        "displayName": getattr(history.author, "displayName", None),
                    }
                    if history.author
                    else None,
                    "items": [
                        {
                            "field": item.field,
                            "fromString": item.fromString,
                            "toString": item.toString,
                        }
                        for item in history.items
                    ],
                }
                changelog.append(history_entry)

        return {
            "key": issue.key,
            "id": issue.id,
            "self": issue.self,
            "fields": fields,
            "changelog": {"histories": changelog},
        }

    @rate_limited(max_attempts=3)
    def fetch_sprints(self, board_id: int, *, state: str | None = None) -> list[dict[str, Any]]:
        """
        Fetch all sprints for a board.

        Args:
            board_id: Jira board ID
            state: Filter by state ("active", "closed", "future", or None for all)

        Returns:
            List of sprint dictionaries (empty if board doesn't support sprints)
        """
        logger.info(f"Fetching sprints for board {board_id}")

        all_sprints: list[dict[str, Any]] = []
        start_at = 0

        try:
            while True:
                with self._rate_limiter.limit():
                    sprints = self._jira.sprints(
                        board_id,
                        startAt=start_at,
                        maxResults=self._batch_size,
                        state=state,
                    )

                if not sprints:
                    break

                for sprint in sprints:
                    sprint_dict = {
                        "id": sprint.id,
                        "name": sprint.name,
                        "state": sprint.state,
                        "startDate": getattr(sprint, "startDate", None),
                        "endDate": getattr(sprint, "endDate", None),
                        "completeDate": getattr(sprint, "completeDate", None),
                        "goal": getattr(sprint, "goal", None),
                        "boardId": board_id,
                    }
                    all_sprints.append(sprint_dict)

                if len(sprints) < self._batch_size:
                    break

                start_at += len(sprints)

            logger.info(f"Fetched {len(all_sprints)} sprints")

            if self._raw_data_dir:
                self._save_raw_data(all_sprints, "sprints")

            return all_sprints

        except JIRAError as e:
            # Check if the board doesn't support sprints (Kanban board)
            error_text = str(e.text).lower() if e.text else ""
            if (
                "does not support sprints" in error_text
                or "ne prend pas les sprints en charge" in error_text
                or "sprints are not supported" in error_text
                or e.status_code == 400
            ):
                logger.warning(
                    f"Board {board_id} does not support sprints (likely a Kanban board). "
                    "Skipping sprint sync."
                )
                return []
            logger.error(f"Failed to fetch sprints: {e.text}")
            raise JiraFetchError(f"Failed to fetch sprints: {e.text}") from e

    @rate_limited(max_attempts=3)
    def fetch_worklogs(
        self,
        since_timestamp: datetime | None = None,
        *,
        issue_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch worklogs, optionally filtered by update time.

        Args:
            since_timestamp: Only fetch worklogs updated since this time
            issue_keys: Specific issue keys to fetch worklogs for

        Returns:
            List of worklog dictionaries
        """
        logger.info(f"Fetching worklogs (since: {since_timestamp})")

        all_worklogs: list[dict[str, Any]] = []

        try:
            if issue_keys:
                # Fetch worklogs for specific issues
                for issue_key in issue_keys:
                    with self._rate_limiter.limit():
                        try:
                            worklogs = self._jira.worklogs(issue_key)
                            for wl in worklogs:
                                worklog_dict = self._worklog_to_dict(wl, issue_key)
                                if since_timestamp:
                                    updated = datetime.fromisoformat(
                                        worklog_dict["updated"].replace("Z", "+00:00")
                                    )
                                    if updated < since_timestamp:
                                        continue
                                all_worklogs.append(worklog_dict)
                        except JIRAError as e:
                            if e.status_code != 404:
                                logger.warning(f"Failed to fetch worklogs for {issue_key}: {e.text}")
            else:
                # Use updated worklogs API if available (Jira Cloud)
                # This endpoint returns worklogs updated since a given timestamp
                if since_timestamp:
                    since_unix = int(since_timestamp.timestamp() * 1000)
                    with self._rate_limiter.limit():
                        # Note: This uses the internal worklog IDs endpoint
                        updated_worklogs = self._jira._get_json(
                            f"worklog/updated?since={since_unix}"
                        )
                        worklog_ids = [w["worklogId"] for w in updated_worklogs.get("values", [])]

                        # Fetch each worklog
                        for wl_id in worklog_ids:
                            with self._rate_limiter.limit():
                                try:
                                    wl = self._jira.worklog(wl_id)
                                    all_worklogs.append(self._worklog_to_dict(wl, wl.issueId))
                                except JIRAError:
                                    pass

            logger.info(f"Fetched {len(all_worklogs)} worklogs")

            if self._raw_data_dir:
                self._save_raw_data(all_worklogs, "worklogs")

            return all_worklogs

        except JIRAError as e:
            logger.error(f"Failed to fetch worklogs: {e.text}")
            raise JiraFetchError(f"Failed to fetch worklogs: {e.text}") from e

    def _worklog_to_dict(self, worklog, issue_key: str) -> dict[str, Any]:
        """Convert a worklog object to a dictionary."""
        return {
            "id": worklog.id,
            "issueKey": issue_key,
            "author": {
                "accountId": getattr(worklog.author, "accountId", None),
                "displayName": getattr(worklog.author, "displayName", None),
            }
            if worklog.author
            else None,
            "timeSpentSeconds": worklog.timeSpentSeconds,
            "started": worklog.started,
            "created": worklog.created,
            "updated": worklog.updated,
            "comment": getattr(worklog, "comment", None),
        }

    @rate_limited(max_attempts=3)
    def fetch_users(self, project_key: str) -> list[dict[str, Any]]:
        """
        Fetch users assignable to a project.

        Args:
            project_key: Jira project key

        Returns:
            List of user dictionaries
        """
        logger.info(f"Fetching users for project {project_key}")

        all_users: list[dict[str, Any]] = []
        start_at = 0

        try:
            while True:
                with self._rate_limiter.limit():
                    users = self._jira.search_assignable_users_for_projects(
                        username="",
                        projectKeys=project_key,
                        startAt=start_at,
                        maxResults=self._batch_size,
                    )

                if not users:
                    break

                for user in users:
                    user_dict = {
                        "accountId": user.accountId,
                        "displayName": user.displayName,
                        "emailAddress": getattr(user, "emailAddress", None),
                        "active": getattr(user, "active", True),
                        "timeZone": getattr(user, "timeZone", None),
                    }
                    all_users.append(user_dict)

                if len(users) < self._batch_size:
                    break

                start_at += len(users)

            logger.info(f"Fetched {len(all_users)} users")

            if self._raw_data_dir:
                self._save_raw_data(all_users, "users")

            return all_users

        except JIRAError as e:
            logger.error(f"Failed to fetch users: {e.text}")
            raise JiraFetchError(f"Failed to fetch users: {e.text}") from e

    @rate_limited(max_attempts=3)
    def fetch_boards(self, project_key: str) -> list[dict[str, Any]]:
        """
        Fetch boards associated with a project.

        Args:
            project_key: Jira project key

        Returns:
            List of board dictionaries
        """
        logger.info(f"Fetching boards for project {project_key}")

        all_boards: list[dict[str, Any]] = []
        start_at = 0

        try:
            while True:
                with self._rate_limiter.limit():
                    boards = self._jira.boards(
                        projectKeyOrId=project_key,
                        startAt=start_at,
                        maxResults=self._batch_size,
                    )

                if not boards:
                    break

                for board in boards:
                    board_dict = {
                        "id": board.id,
                        "name": board.name,
                        "type": board.type,
                        "projectKey": project_key,
                    }
                    all_boards.append(board_dict)

                if len(boards) < self._batch_size:
                    break

                start_at += len(boards)

            logger.info(f"Fetched {len(all_boards)} boards")

            return all_boards

        except JIRAError as e:
            logger.error(f"Failed to fetch boards: {e.text}")
            raise JiraFetchError(f"Failed to fetch boards: {e.text}") from e

    def _save_raw_data(self, data: list[dict[str, Any]], entity_type: str) -> Path:
        """
        Save raw data to JSON file.

        Args:
            data: Data to save
            entity_type: Type of entity (issues, sprints, etc.)

        Returns:
            Path to saved file
        """
        if not self._raw_data_dir:
            raise ValueError("raw_data_dir not configured")

        self._raw_data_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{entity_type}_{timestamp}.json"
        filepath = self._raw_data_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved {len(data)} {entity_type} to {filepath}")
        return filepath


def create_fetcher_from_settings(
    authenticator: JiraAuthenticator | None = None,
) -> JiraFetcher:
    """
    Create a JiraFetcher from application settings.

    Args:
        authenticator: Optional authenticator (creates new one if not provided)

    Returns:
        Configured JiraFetcher instance
    """
    from pathlib import Path

    from config.settings import get_settings

    settings = get_settings()

    if authenticator is None:
        from src.jira_client.auth import create_jira_client_from_settings

        authenticator = create_jira_client_from_settings()

    custom_fields = {
        "story_points": settings.jira.story_points_field,
        "epic_link": settings.jira.epic_link_field,
        "sprint": settings.jira.sprint_field,
    }

    raw_data_dir = Path("data/raw")

    return JiraFetcher(
        authenticator=authenticator,
        custom_fields=custom_fields,
        raw_data_dir=raw_data_dir,
    )
