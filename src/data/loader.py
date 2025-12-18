"""
Data loader module for transforming and loading Jira data into DuckDB.

Handles parsing, transformation, and upsert operations for all Jira entities.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from loguru import logger


class DataLoaderError(Exception):
    """Raised when data loading fails."""

    pass


class DataLoader:
    """
    Loads Jira data from JSON into DuckDB.

    Handles transformation, custom field mapping, and idempotent upserts.

    Example:
        >>> loader = DataLoader(conn, custom_fields={"story_points": "customfield_10016"})
        >>> loader.load_issues(issues_data)
        >>> loader.load_sprints(sprints_data)
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        custom_fields: dict[str, str] | None = None,
        anonymize: bool = True,
        salt: str = "jira-copilot",
    ):
        """
        Initialize the data loader.

        Args:
            conn: DuckDB connection
            custom_fields: Mapping of logical field names to Jira custom field IDs
            anonymize: Whether to pseudonymize user identifiers
            salt: Salt for pseudonymization hash
        """
        self._conn = conn
        self._custom_fields = custom_fields or {}
        self._anonymize = anonymize
        self._salt = salt
        self._user_pseudonyms: dict[str, str] = {}

    def _pseudonymize(self, account_id: str | None) -> str | None:
        """
        Generate a pseudonym for a user account ID.

        Args:
            account_id: Jira account ID

        Returns:
            Pseudonymized identifier or None
        """
        if not account_id:
            return None

        if not self._anonymize:
            return account_id

        if account_id not in self._user_pseudonyms:
            hash_input = f"{self._salt}:{account_id}"
            hash_value = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
            self._user_pseudonyms[account_id] = f"user_{hash_value}"

        return self._user_pseudonyms[account_id]

    def _parse_timestamp(self, value: str | None) -> datetime | None:
        """Parse a Jira timestamp string to datetime."""
        if not value:
            return None

        try:
            # Handle various Jira timestamp formats
            if "T" in value:
                # ISO format with timezone
                value = value.replace("Z", "+00:00")
                if "." in value:
                    # Trim microseconds if too long
                    parts = value.split(".")
                    if len(parts) == 2:
                        tz_part = ""
                        if "+" in parts[1]:
                            micro, tz_part = parts[1].split("+")
                            tz_part = "+" + tz_part
                        elif "-" in parts[1][1:]:  # Skip first char to avoid negative sign confusion
                            idx = parts[1].rindex("-")
                            micro = parts[1][:idx]
                            tz_part = parts[1][idx:]
                        else:
                            micro = parts[1]
                        value = f"{parts[0]}.{micro[:6]}{tz_part}"
                return datetime.fromisoformat(value)
            else:
                # Simple date format
                return datetime.strptime(value, "%Y-%m-%d")
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to parse timestamp '{value}': {e}")
            return None

    def _get_custom_field(self, fields: dict, field_name: str) -> Any:
        """Get a custom field value from issue fields."""
        field_id = self._custom_fields.get(field_name)
        if field_id:
            return fields.get(field_id)
        return None

    def _calculate_cycle_time(self, changelog: list[dict]) -> float | None:
        """
        Calculate cycle time from changelog (first In Progress to Done).

        Returns:
            Cycle time in hours, or None if not calculable
        """
        in_progress_time = None
        done_time = None

        # Sort by timestamp
        sorted_changes = sorted(
            changelog, key=lambda x: x.get("changed_at") or x.get("created", "")
        )

        for change in sorted_changes:
            items = change.get("items", [])
            for item in items:
                if item.get("field") == "status":
                    to_value = item.get("to_value") or item.get("toString", "")
                    to_lower = to_value.lower()

                    # First time entering "in progress" state
                    if in_progress_time is None and any(
                        x in to_lower for x in ["progress", "development", "review", "testing"]
                    ):
                        timestamp = change.get("changed_at") or change.get("created")
                        in_progress_time = self._parse_timestamp(timestamp)

                    # Entering "done" state
                    if any(x in to_lower for x in ["done", "closed", "resolved"]):
                        timestamp = change.get("changed_at") or change.get("created")
                        done_time = self._parse_timestamp(timestamp)

        if in_progress_time and done_time:
            delta = done_time - in_progress_time
            return delta.total_seconds() / 3600  # Convert to hours

        return None

    def load_issues(self, issues: list[dict[str, Any]]) -> int:
        """
        Load issues into the database.

        Args:
            issues: List of issue dictionaries from Jira API

        Returns:
            Number of issues loaded
        """
        logger.info(f"Loading {len(issues)} issues")

        if not issues:
            return 0

        records = []
        changelog_records = []
        changelog_id = self._get_next_changelog_id()

        for issue in issues:
            fields = issue.get("fields", {})
            changelog = issue.get("changelog", {}).get("histories", [])

            # Extract sprint info
            sprint_field = self._get_custom_field(fields, "sprint")
            sprint_id = None
            sprint_name = None
            if sprint_field and isinstance(sprint_field, list) and len(sprint_field) > 0:
                # Get the most recent sprint
                sprint = sprint_field[-1]
                if isinstance(sprint, dict):
                    sprint_id = sprint.get("id")
                    sprint_name = sprint.get("name")
                elif hasattr(sprint, "id"):
                    sprint_id = sprint.id
                    sprint_name = sprint.name

            # Extract epic info
            epic_field = self._get_custom_field(fields, "epic_link")
            epic_key = epic_field if isinstance(epic_field, str) else None

            # Extract story points
            story_points = self._get_custom_field(fields, "story_points")
            if story_points is not None:
                try:
                    story_points = float(story_points)
                except (ValueError, TypeError):
                    story_points = None

            # Extract assignee/reporter
            assignee = fields.get("assignee") or {}
            reporter = fields.get("reporter") or {}

            # Extract components and labels
            components = [c.get("name") for c in fields.get("components", []) if c.get("name")]
            labels = fields.get("labels", [])

            # Calculate metrics
            created = self._parse_timestamp(fields.get("created"))
            resolved = self._parse_timestamp(fields.get("resolutiondate"))

            lead_time_hours = None
            if created and resolved:
                lead_time_hours = (resolved - created).total_seconds() / 3600

            cycle_time_hours = self._calculate_cycle_time(changelog)

            # Build record
            record = {
                "key": issue.get("key"),
                "id": issue.get("id"),
                "summary": fields.get("summary"),
                "description": (fields.get("description") or "")[:10000],  # Truncate long descriptions
                "issue_type": (fields.get("issuetype") or {}).get("name"),
                "status": (fields.get("status") or {}).get("name"),
                "priority": (fields.get("priority") or {}).get("name"),
                "assignee_id": self._pseudonymize(assignee.get("accountId")),
                "assignee_name": assignee.get("displayName") if not self._anonymize else None,
                "reporter_id": self._pseudonymize(reporter.get("accountId")),
                "reporter_name": reporter.get("displayName") if not self._anonymize else None,
                "created": created,
                "updated": self._parse_timestamp(fields.get("updated")),
                "resolved": resolved,
                "story_points": story_points,
                "original_estimate_seconds": fields.get("timeoriginalestimate"),
                "time_spent_seconds": fields.get("timespent"),
                "remaining_estimate_seconds": fields.get("timeestimate"),
                "sprint_id": sprint_id,
                "sprint_name": sprint_name,
                "epic_key": epic_key,
                "epic_name": None,  # Would need separate lookup
                "components": components,
                "labels": labels,
                "subtask_count": len(fields.get("subtasks", [])),
                "link_count": len(fields.get("issuelinks", [])),
                "attachment_count": len(fields.get("attachment", [])),
                "project_key": (fields.get("project") or {}).get("key"),
                "project_name": (fields.get("project") or {}).get("name"),
                "lead_time_hours": lead_time_hours,
                "cycle_time_hours": cycle_time_hours,
                "sync_timestamp": datetime.now(),
            }
            records.append(record)

            # Process changelog
            for history in changelog:
                for item in history.get("items", []):
                    author = history.get("author") or {}
                    changelog_records.append({
                        "id": changelog_id,
                        "issue_key": issue.get("key"),
                        "field": item.get("field"),
                        "from_value": item.get("fromString"),
                        "to_value": item.get("toString"),
                        "changed_at": self._parse_timestamp(history.get("created")),
                        "author_id": self._pseudonymize(author.get("accountId")),
                        "author_name": author.get("displayName") if not self._anonymize else None,
                    })
                    changelog_id += 1

        # Upsert issues
        df_issues = pd.DataFrame(records)
        self._upsert_dataframe("issues", df_issues, key_column="key")

        # Insert changelog (delete existing for these issues first)
        if changelog_records:
            issue_keys = [r["key"] for r in records]
            placeholders = ",".join(["?" for _ in issue_keys])
            self._conn.execute(
                f"DELETE FROM issue_changelog WHERE issue_key IN ({placeholders})",
                issue_keys,
            )

            df_changelog = pd.DataFrame(changelog_records)
            self._conn.execute("INSERT INTO issue_changelog SELECT * FROM df_changelog")

        logger.info(f"Loaded {len(records)} issues and {len(changelog_records)} changelog entries")
        return len(records)

    def load_sprints(self, sprints: list[dict[str, Any]]) -> int:
        """
        Load sprints into the database.

        Args:
            sprints: List of sprint dictionaries from Jira API

        Returns:
            Number of sprints loaded
        """
        logger.info(f"Loading {len(sprints)} sprints")

        if not sprints:
            return 0

        records = []
        for sprint in sprints:
            record = {
                "id": sprint.get("id"),
                "name": sprint.get("name"),
                "state": sprint.get("state"),
                "start_date": self._parse_timestamp(sprint.get("startDate")),
                "end_date": self._parse_timestamp(sprint.get("endDate")),
                "complete_date": self._parse_timestamp(sprint.get("completeDate")),
                "goal": sprint.get("goal"),
                "board_id": sprint.get("boardId"),
                "committed_points": None,  # Computed later
                "completed_points": None,
                "added_points": None,
                "removed_points": None,
                "completion_rate": None,
                "sync_timestamp": datetime.now(),
            }
            records.append(record)

        df = pd.DataFrame(records)
        self._upsert_dataframe("sprints", df, key_column="id")

        logger.info(f"Loaded {len(records)} sprints")
        return len(records)

    def load_worklogs(self, worklogs: list[dict[str, Any]]) -> int:
        """
        Load worklogs into the database.

        Args:
            worklogs: List of worklog dictionaries from Jira API

        Returns:
            Number of worklogs loaded
        """
        logger.info(f"Loading {len(worklogs)} worklogs")

        if not worklogs:
            return 0

        records = []
        for wl in worklogs:
            author = wl.get("author") or {}
            record = {
                "id": wl.get("id"),
                "issue_key": wl.get("issueKey"),
                "author_id": self._pseudonymize(author.get("accountId")),
                "author_name": author.get("displayName") if not self._anonymize else None,
                "time_spent_seconds": wl.get("timeSpentSeconds"),
                "started": self._parse_timestamp(wl.get("started")),
                "created": self._parse_timestamp(wl.get("created")),
                "updated": self._parse_timestamp(wl.get("updated")),
                "comment": wl.get("comment"),
            }
            records.append(record)

        df = pd.DataFrame(records)
        self._upsert_dataframe("worklogs", df, key_column="id")

        logger.info(f"Loaded {len(records)} worklogs")
        return len(records)

    def load_users(self, users: list[dict[str, Any]]) -> int:
        """
        Load users into the database with pseudonymization.

        Args:
            users: List of user dictionaries from Jira API

        Returns:
            Number of users loaded
        """
        logger.info(f"Loading {len(users)} users")

        if not users:
            return 0

        records = []
        for user in users:
            account_id = user.get("accountId")
            record = {
                "account_id": account_id,
                "display_name": user.get("displayName") if not self._anonymize else None,
                "email": user.get("emailAddress") if not self._anonymize else None,
                "pseudonym": self._pseudonymize(account_id),
                "active": user.get("active", True),
                "timezone": user.get("timeZone"),
                "sync_timestamp": datetime.now(),
            }
            records.append(record)

        df = pd.DataFrame(records)
        self._upsert_dataframe("users", df, key_column="account_id")

        logger.info(f"Loaded {len(records)} users")
        return len(records)

    def _upsert_dataframe(
        self, table_name: str, df: pd.DataFrame, key_column: str
    ) -> None:
        """
        Upsert a DataFrame into a table.

        Uses DELETE + INSERT for simplicity (DuckDB doesn't have native UPSERT).

        Args:
            table_name: Target table name
            df: DataFrame to upsert
            key_column: Primary key column name
        """
        if df.empty:
            return

        # Delete existing records with matching keys
        keys = df[key_column].tolist()
        if keys:
            placeholders = ",".join(["?" for _ in keys])
            self._conn.execute(
                f"DELETE FROM {table_name} WHERE {key_column} IN ({placeholders})",
                keys,
            )

        # Insert new records
        self._conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")

    def _get_next_changelog_id(self) -> int:
        """Get the next available changelog ID."""
        result = self._conn.execute(
            "SELECT COALESCE(MAX(id), 0) + 1 FROM issue_changelog"
        ).fetchone()
        return result[0] if result else 1

    def update_sync_metadata(
        self,
        entity_type: str,
        status: str,
        records_synced: int,
        error_message: str | None = None,
        last_updated_value: str | None = None,
    ) -> None:
        """
        Update sync metadata for an entity type.

        Args:
            entity_type: Type of entity (issues, sprints, etc.)
            status: Sync status (success, error)
            records_synced: Number of records synced
            error_message: Error message if failed
            last_updated_value: Last updated value for incremental sync
        """
        self._conn.execute("""
            INSERT OR REPLACE INTO sync_metadata
            (entity_type, last_sync_timestamp, last_sync_status, records_synced, error_message, last_updated_value)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            entity_type,
            datetime.now(),
            status,
            records_synced,
            error_message,
            last_updated_value,
        ])

    def get_sync_metadata(self, entity_type: str) -> dict[str, Any] | None:
        """
        Get sync metadata for an entity type.

        Args:
            entity_type: Type of entity

        Returns:
            Metadata dictionary or None
        """
        result = self._conn.execute(
            "SELECT * FROM sync_metadata WHERE entity_type = ?",
            [entity_type],
        ).fetchone()

        if result:
            columns = ["entity_type", "last_sync_timestamp", "last_sync_status",
                       "records_synced", "error_message", "last_updated_value"]
            return dict(zip(columns, result))
        return None


def create_loader_from_settings(
    conn: duckdb.DuckDBPyConnection | None = None,
) -> DataLoader:
    """
    Create a DataLoader from application settings.

    Args:
        conn: Optional DuckDB connection (creates new one if not provided)

    Returns:
        Configured DataLoader instance
    """
    from config.settings import get_settings
    from src.data.schema import get_connection

    settings = get_settings()

    if conn is None:
        conn = get_connection()

    custom_fields = {
        "story_points": settings.jira.story_points_field,
        "epic_link": settings.jira.epic_link_field,
        "sprint": settings.jira.sprint_field,
    }

    return DataLoader(
        conn=conn,
        custom_fields=custom_fields,
        anonymize=settings.anonymize_developers,
    )
