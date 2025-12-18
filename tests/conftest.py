"""
Pytest fixtures and configuration for Jira AI Co-pilot tests.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Generator

import duckdb
import pytest


@pytest.fixture
def test_db() -> Generator[duckdb.DuckDBPyConnection, None, None]:
    """Create an in-memory DuckDB database for testing."""
    conn = duckdb.connect(":memory:")

    # Initialize schema (will be implemented in src/data/schema.py)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS issues (
            key VARCHAR PRIMARY KEY,
            summary VARCHAR,
            description VARCHAR,
            issue_type VARCHAR,
            status VARCHAR,
            priority VARCHAR,
            assignee_id VARCHAR,
            assignee_name VARCHAR,
            reporter_id VARCHAR,
            reporter_name VARCHAR,
            created TIMESTAMP,
            updated TIMESTAMP,
            resolved TIMESTAMP,
            story_points FLOAT,
            sprint_id INTEGER,
            sprint_name VARCHAR,
            epic_key VARCHAR,
            components VARCHAR[],
            labels VARCHAR[],
            project_key VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS issue_changelog (
            id INTEGER PRIMARY KEY,
            issue_key VARCHAR,
            field VARCHAR,
            from_value VARCHAR,
            to_value VARCHAR,
            changed_at TIMESTAMP,
            author_id VARCHAR
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sprints (
            id INTEGER PRIMARY KEY,
            name VARCHAR,
            state VARCHAR,
            start_date TIMESTAMP,
            end_date TIMESTAMP,
            complete_date TIMESTAMP,
            goal VARCHAR,
            board_id INTEGER
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS worklogs (
            id INTEGER PRIMARY KEY,
            issue_key VARCHAR,
            author_id VARCHAR,
            time_spent_seconds INTEGER,
            started TIMESTAMP,
            created TIMESTAMP,
            updated TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            account_id VARCHAR PRIMARY KEY,
            display_name VARCHAR,
            email VARCHAR,
            pseudonym VARCHAR,
            active BOOLEAN
        )
    """)

    yield conn
    conn.close()


@pytest.fixture
def sample_issues() -> list[dict[str, Any]]:
    """Sample issue data for testing."""
    base_date = datetime.now() - timedelta(days=30)

    return [
        {
            "key": "PROJ-1",
            "fields": {
                "summary": "Implement user authentication",
                "description": "Add login and registration functionality",
                "issuetype": {"name": "Story"},
                "status": {"name": "Done"},
                "priority": {"name": "High"},
                "assignee": {
                    "accountId": "user-001",
                    "displayName": "John Doe",
                    "emailAddress": "john@example.com",
                },
                "reporter": {
                    "accountId": "user-002",
                    "displayName": "Jane Smith",
                },
                "created": (base_date).isoformat(),
                "updated": (base_date + timedelta(days=5)).isoformat(),
                "resolutiondate": (base_date + timedelta(days=5)).isoformat(),
                "customfield_10016": 5,  # story points
                "components": [{"name": "Backend"}],
                "labels": ["authentication", "security"],
            },
            "changelog": {
                "histories": [
                    {
                        "created": (base_date + timedelta(days=1)).isoformat(),
                        "author": {"accountId": "user-001"},
                        "items": [
                            {"field": "status", "fromString": "To Do", "toString": "In Progress"}
                        ],
                    },
                    {
                        "created": (base_date + timedelta(days=5)).isoformat(),
                        "author": {"accountId": "user-001"},
                        "items": [
                            {"field": "status", "fromString": "In Progress", "toString": "Done"}
                        ],
                    },
                ]
            },
        },
        {
            "key": "PROJ-2",
            "fields": {
                "summary": "Fix login bug",
                "description": "Users cannot login with special characters in password",
                "issuetype": {"name": "Bug"},
                "status": {"name": "In Progress"},
                "priority": {"name": "Highest"},
                "assignee": {
                    "accountId": "user-003",
                    "displayName": "Bob Wilson",
                    "emailAddress": "bob@example.com",
                },
                "reporter": {
                    "accountId": "user-002",
                    "displayName": "Jane Smith",
                },
                "created": (base_date + timedelta(days=10)).isoformat(),
                "updated": (base_date + timedelta(days=12)).isoformat(),
                "resolutiondate": None,
                "customfield_10016": 2,
                "components": [{"name": "Backend"}, {"name": "Security"}],
                "labels": ["bug", "urgent"],
            },
            "changelog": {
                "histories": [
                    {
                        "created": (base_date + timedelta(days=11)).isoformat(),
                        "author": {"accountId": "user-003"},
                        "items": [
                            {"field": "status", "fromString": "To Do", "toString": "In Progress"}
                        ],
                    }
                ]
            },
        },
    ]


@pytest.fixture
def sample_sprints() -> list[dict[str, Any]]:
    """Sample sprint data for testing."""
    base_date = datetime.now() - timedelta(days=14)

    return [
        {
            "id": 1,
            "name": "Sprint 1",
            "state": "closed",
            "startDate": (base_date - timedelta(days=14)).isoformat(),
            "endDate": base_date.isoformat(),
            "completeDate": base_date.isoformat(),
            "goal": "Complete authentication module",
        },
        {
            "id": 2,
            "name": "Sprint 2",
            "state": "active",
            "startDate": base_date.isoformat(),
            "endDate": (base_date + timedelta(days=14)).isoformat(),
            "completeDate": None,
            "goal": "Bug fixes and performance improvements",
        },
    ]


@pytest.fixture
def sample_worklogs() -> list[dict[str, Any]]:
    """Sample worklog data for testing."""
    base_date = datetime.now() - timedelta(days=7)

    return [
        {
            "id": 1,
            "issueKey": "PROJ-1",
            "author": {"accountId": "user-001"},
            "timeSpentSeconds": 14400,  # 4 hours
            "started": base_date.isoformat(),
            "created": base_date.isoformat(),
            "updated": base_date.isoformat(),
        },
        {
            "id": 2,
            "issueKey": "PROJ-1",
            "author": {"accountId": "user-001"},
            "timeSpentSeconds": 7200,  # 2 hours
            "started": (base_date + timedelta(days=1)).isoformat(),
            "created": (base_date + timedelta(days=1)).isoformat(),
            "updated": (base_date + timedelta(days=1)).isoformat(),
        },
    ]


@pytest.fixture
def mock_jira_response() -> dict[str, Any]:
    """Mock Jira API response for connection test."""
    return {
        "self": "https://example.atlassian.net/rest/api/3/myself",
        "accountId": "test-user-id",
        "displayName": "Test User",
        "emailAddress": "test@example.com",
        "active": True,
    }


@pytest.fixture
def temp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directory structure."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "raw").mkdir()
    (data_dir / "exports").mkdir()
    return data_dir


@pytest.fixture
def temp_models_dir(tmp_path: Path) -> Path:
    """Create temporary models directory."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir
