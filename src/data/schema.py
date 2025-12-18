"""
DuckDB schema definition and initialization.

Defines the database schema optimized for analytical queries on Jira data.
"""

from pathlib import Path

import duckdb
from loguru import logger

# SQL statements for creating tables
CREATE_ISSUES_TABLE = """
CREATE TABLE IF NOT EXISTS issues (
    -- Primary identification
    key VARCHAR PRIMARY KEY,
    id VARCHAR UNIQUE,

    -- Basic fields
    summary VARCHAR,
    description VARCHAR,
    issue_type VARCHAR,
    status VARCHAR,
    priority VARCHAR,

    -- People
    assignee_id VARCHAR,
    assignee_name VARCHAR,
    reporter_id VARCHAR,
    reporter_name VARCHAR,

    -- Timestamps
    created TIMESTAMP,
    updated TIMESTAMP,
    resolved TIMESTAMP,

    -- Estimation and tracking
    story_points FLOAT,
    original_estimate_seconds INTEGER,
    time_spent_seconds INTEGER,
    remaining_estimate_seconds INTEGER,

    -- Sprint and epic
    sprint_id INTEGER,
    sprint_name VARCHAR,
    epic_key VARCHAR,
    epic_name VARCHAR,

    -- Categories
    components VARCHAR[],
    labels VARCHAR[],

    -- Relationships
    subtask_count INTEGER DEFAULT 0,
    link_count INTEGER DEFAULT 0,
    attachment_count INTEGER DEFAULT 0,

    -- Project
    project_key VARCHAR,
    project_name VARCHAR,

    -- Derived metrics (computed during load)
    lead_time_hours FLOAT,
    cycle_time_hours FLOAT,

    -- Metadata
    sync_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_ISSUE_CHANGELOG_TABLE = """
CREATE TABLE IF NOT EXISTS issue_changelog (
    id INTEGER PRIMARY KEY,
    issue_key VARCHAR NOT NULL,

    -- Change details
    field VARCHAR NOT NULL,
    from_value VARCHAR,
    to_value VARCHAR,

    -- Timestamp
    changed_at TIMESTAMP NOT NULL,

    -- Author
    author_id VARCHAR,
    author_name VARCHAR,

    -- Foreign key relationship (logical, not enforced in DuckDB)
    FOREIGN KEY (issue_key) REFERENCES issues(key)
)
"""

CREATE_SPRINTS_TABLE = """
CREATE TABLE IF NOT EXISTS sprints (
    id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    state VARCHAR,

    -- Dates
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    complete_date TIMESTAMP,

    -- Details
    goal VARCHAR,
    board_id INTEGER,

    -- Computed metrics
    committed_points FLOAT,
    completed_points FLOAT,
    added_points FLOAT,
    removed_points FLOAT,
    completion_rate FLOAT,

    -- Metadata
    sync_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_WORKLOGS_TABLE = """
CREATE TABLE IF NOT EXISTS worklogs (
    id INTEGER PRIMARY KEY,
    issue_key VARCHAR NOT NULL,

    -- Author
    author_id VARCHAR,
    author_name VARCHAR,

    -- Time tracking
    time_spent_seconds INTEGER NOT NULL,

    -- Timestamps
    started TIMESTAMP,
    created TIMESTAMP,
    updated TIMESTAMP,

    -- Comment
    comment VARCHAR,

    -- Foreign key relationship
    FOREIGN KEY (issue_key) REFERENCES issues(key)
)
"""

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    account_id VARCHAR PRIMARY KEY,
    display_name VARCHAR,
    email VARCHAR,

    -- Pseudonymized identifier (for privacy)
    pseudonym VARCHAR UNIQUE,

    -- Status
    active BOOLEAN DEFAULT TRUE,
    timezone VARCHAR,

    -- Metadata
    sync_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_SYNC_METADATA_TABLE = """
CREATE TABLE IF NOT EXISTS sync_metadata (
    entity_type VARCHAR PRIMARY KEY,
    last_sync_timestamp TIMESTAMP,
    last_sync_status VARCHAR,
    records_synced INTEGER,
    error_message VARCHAR,

    -- Incremental sync tracking
    last_updated_value VARCHAR
)
"""

CREATE_ALERTS_TABLE = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY,

    -- Alert details
    alert_type VARCHAR NOT NULL,
    severity VARCHAR NOT NULL,
    message VARCHAR NOT NULL,
    details VARCHAR,

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged_at TIMESTAMP,
    resolved_at TIMESTAMP,

    -- Context
    sprint_id INTEGER,
    issue_key VARCHAR,
    user_id VARCHAR,

    -- Suggested actions (JSON)
    suggested_actions VARCHAR
)
"""

CREATE_ML_FEATURES_TABLE = """
CREATE TABLE IF NOT EXISTS ml_features (
    id INTEGER PRIMARY KEY,
    issue_key VARCHAR NOT NULL,

    -- Feature set identifier
    feature_set VARCHAR NOT NULL,
    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Features stored as JSON for flexibility
    features VARCHAR NOT NULL,

    -- Target variable (if training data)
    target_value FLOAT,

    UNIQUE(issue_key, feature_set)
)
"""

# Index creation statements
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_issues_assignee ON issues(assignee_id)",
    "CREATE INDEX IF NOT EXISTS idx_issues_sprint ON issues(sprint_id)",
    "CREATE INDEX IF NOT EXISTS idx_issues_project ON issues(project_key)",
    "CREATE INDEX IF NOT EXISTS idx_issues_status ON issues(status)",
    "CREATE INDEX IF NOT EXISTS idx_issues_created ON issues(created)",
    "CREATE INDEX IF NOT EXISTS idx_issues_updated ON issues(updated)",
    "CREATE INDEX IF NOT EXISTS idx_issues_resolved ON issues(resolved)",
    "CREATE INDEX IF NOT EXISTS idx_changelog_issue ON issue_changelog(issue_key)",
    "CREATE INDEX IF NOT EXISTS idx_changelog_field ON issue_changelog(field)",
    "CREATE INDEX IF NOT EXISTS idx_changelog_timestamp ON issue_changelog(changed_at)",
    "CREATE INDEX IF NOT EXISTS idx_worklogs_issue ON worklogs(issue_key)",
    "CREATE INDEX IF NOT EXISTS idx_worklogs_author ON worklogs(author_id)",
    "CREATE INDEX IF NOT EXISTS idx_worklogs_started ON worklogs(started)",
    "CREATE INDEX IF NOT EXISTS idx_sprints_state ON sprints(state)",
    "CREATE INDEX IF NOT EXISTS idx_sprints_board ON sprints(board_id)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity)",
    "CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at)",
]


def initialize_database(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    """
    Initialize the DuckDB database with the required schema.

    Creates all tables and indexes if they don't exist.

    Args:
        db_path: Path to the DuckDB database file

    Returns:
        DuckDB connection object
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing database at {db_path}")

    conn = duckdb.connect(str(db_path))

    # Create tables
    tables = [
        ("issues", CREATE_ISSUES_TABLE),
        ("issue_changelog", CREATE_ISSUE_CHANGELOG_TABLE),
        ("sprints", CREATE_SPRINTS_TABLE),
        ("worklogs", CREATE_WORKLOGS_TABLE),
        ("users", CREATE_USERS_TABLE),
        ("sync_metadata", CREATE_SYNC_METADATA_TABLE),
        ("alerts", CREATE_ALERTS_TABLE),
        ("ml_features", CREATE_ML_FEATURES_TABLE),
    ]

    for table_name, create_sql in tables:
        try:
            conn.execute(create_sql)
            logger.debug(f"Created/verified table: {table_name}")
        except Exception as e:
            logger.error(f"Failed to create table {table_name}: {e}")
            raise

    # Create indexes
    for index_sql in CREATE_INDEXES:
        try:
            conn.execute(index_sql)
        except Exception as e:
            logger.warning(f"Failed to create index: {e}")

    logger.info("Database initialization complete")
    return conn


def get_connection(db_path: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """
    Get a connection to the database.

    Args:
        db_path: Path to database (uses settings default if not provided)

    Returns:
        DuckDB connection object
    """
    if db_path is None:
        from config.settings import get_settings

        settings = get_settings()
        db_path = settings.database.full_path

    return duckdb.connect(str(db_path))


def drop_all_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Drop all tables in the database.

    WARNING: This is destructive and should only be used for testing/reset.

    Args:
        conn: DuckDB connection object
    """
    logger.warning("Dropping all tables!")

    tables = [
        "ml_features",
        "alerts",
        "sync_metadata",
        "worklogs",
        "issue_changelog",
        "sprints",
        "users",
        "issues",
    ]

    for table in tables:
        try:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            logger.debug(f"Dropped table: {table}")
        except Exception as e:
            logger.warning(f"Failed to drop table {table}: {e}")


def get_table_stats(conn: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """
    Get row counts for all tables.

    Args:
        conn: DuckDB connection object

    Returns:
        Dictionary mapping table names to row counts
    """
    tables = [
        "issues",
        "issue_changelog",
        "sprints",
        "worklogs",
        "users",
        "sync_metadata",
        "alerts",
        "ml_features",
    ]

    stats = {}
    for table in tables:
        try:
            result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            stats[table] = result[0] if result else 0
        except Exception:
            stats[table] = 0

    return stats
