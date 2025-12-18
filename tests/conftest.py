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


@pytest.fixture
def populated_test_db(test_db) -> duckdb.DuckDBPyConnection:
    """Test database populated with sample data for feature extraction."""
    base_date = datetime.now() - timedelta(days=30)

    # Add more columns needed for feature extraction
    test_db.execute("ALTER TABLE issues ADD COLUMN IF NOT EXISTS cycle_time_hours FLOAT")
    test_db.execute("ALTER TABLE issues ADD COLUMN IF NOT EXISTS component_primary VARCHAR")

    # Insert sample issues
    issues = [
        ("PROJ-1", "Implement auth", "Story", "Done", "High", "user-001",
         base_date, base_date + timedelta(days=5), base_date + timedelta(days=5),
         5.0, 1, "Sprint 1", "Backend", "PROJ", 96.0),
        ("PROJ-2", "Fix login bug", "Bug", "In Progress", "Highest", "user-002",
         base_date + timedelta(days=10), base_date + timedelta(days=15), None,
         2.0, 2, "Sprint 2", "Frontend", "PROJ", None),
        ("PROJ-3", "Add tests", "Task", "Done", "Medium", "user-001",
         base_date + timedelta(days=2), base_date + timedelta(days=8), base_date + timedelta(days=8),
         3.0, 1, "Sprint 1", "Backend", "PROJ", 144.0),
        ("PROJ-4", "Refactor code", "Story", "To Do", "Low", "user-003",
         base_date + timedelta(days=20), base_date + timedelta(days=25), None,
         8.0, 2, "Sprint 2", "Backend", "PROJ", None),
        ("PROJ-5", "Write docs", "Task", "Done", "Low", "user-002",
         base_date + timedelta(days=5), base_date + timedelta(days=7), base_date + timedelta(days=7),
         1.0, 1, "Sprint 1", "Documentation", "PROJ", 48.0),
    ]

    for issue in issues:
        test_db.execute("""
            INSERT INTO issues (key, summary, issue_type, status, priority, assignee_id,
                              created, updated, resolved, story_points, sprint_id, sprint_name,
                              component_primary, project_key, cycle_time_hours)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, issue)

    # Insert sprints
    sprints = [
        (1, "Sprint 1", "closed", base_date - timedelta(days=14), base_date,
         base_date, "Complete auth", 1),
        (2, "Sprint 2", "active", base_date, base_date + timedelta(days=14),
         None, "Bug fixes", 1),
    ]

    for sprint in sprints:
        test_db.execute("""
            INSERT INTO sprints (id, name, state, start_date, end_date, complete_date, goal, board_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, sprint)

    # Insert users
    users = [
        ("user-001", "John Doe", "john@example.com", "dev_001", True),
        ("user-002", "Jane Smith", "jane@example.com", "dev_002", True),
        ("user-003", "Bob Wilson", "bob@example.com", "dev_003", True),
    ]

    for user in users:
        test_db.execute("""
            INSERT INTO users (account_id, display_name, email, pseudonym, active)
            VALUES (?, ?, ?, ?, ?)
        """, user)

    # Insert worklogs
    worklogs = [
        (1, "PROJ-1", "user-001", 14400, base_date + timedelta(days=1), base_date + timedelta(days=1), base_date + timedelta(days=1)),
        (2, "PROJ-1", "user-001", 7200, base_date + timedelta(days=2), base_date + timedelta(days=2), base_date + timedelta(days=2)),
        (3, "PROJ-3", "user-001", 10800, base_date + timedelta(days=5), base_date + timedelta(days=5), base_date + timedelta(days=5)),
    ]

    for wl in worklogs:
        test_db.execute("""
            INSERT INTO worklogs (id, issue_key, author_id, time_spent_seconds, started, created, updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, wl)

    return test_db


@pytest.fixture
def sample_ticket_features() -> dict[str, Any]:
    """Sample ticket features for model testing."""
    return {
        "issue_key": "PROJ-123",
        "issue_type": "Story",
        "priority": "High",
        "story_points": 5.0,
        "component_primary": "Backend",
        "description_length": 250,
        "summary_length": 45,
        "has_epic": True,
        "labels_count": 2,
        "components_count": 1,
        "assignee_completed_30d": 15,
        "assignee_avg_cycle_time": 48.5,
        "project_avg_cycle_time": 72.0,
        "type_avg_cycle_time": 60.0,
        "created_day_of_week": 2,
        "created_hour": 10,
        "sprint_day_created": 3,
    }


@pytest.fixture
def sample_sprint_features() -> dict[str, Any]:
    """Sample sprint features for model testing."""
    return {
        "sprint_id": 1,
        "sprint_name": "Sprint 1",
        "sprint_state": "active",
        "board_id": 1,
        "total_days": 14,
        "days_elapsed": 7,
        "days_remaining": 7,
        "progress_percent": 50.0,
        "total_issues": 10,
        "completed_issues": 4,
        "remaining_issues": 6,
        "blocked_issues": 1,
        "in_progress_issues": 3,
        "todo_issues": 2,
        "total_points": 30,
        "completed_points": 12,
        "remaining_points": 18,
        "blocked_points": 3,
        "completion_rate": 40.0,
        "expected_completion_rate": 50.0,
        "completion_gap": -10.0,
        "avg_historical_velocity": 25.0,
        "velocity_ratio": 0.48,
        "blocked_ratio": 0.1,
        "urgency_factor": 1.2,
        "unique_assignees": 3,
        "scope_creep_ratio": 0.05,
    }


@pytest.fixture
def sample_developer_features() -> dict[str, Any]:
    """Sample developer features for workload scoring."""
    return {
        "assignee_id": "user-001",
        "pseudonym": "dev_001",
        "wip_count": 4,
        "wip_points": 15,
        "unresolved_count": 6,
        "blocked_count": 1,
        "overdue_count": 0,
        "completed_count_30d": 8,
        "completed_points_30d": 25,
        "hours_logged_30d": 45,
        "completion_rate": 0.75,
    }
