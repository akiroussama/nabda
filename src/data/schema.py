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
    author_name VARCHAR
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
    comment VARCHAR
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

# ============================================================================
# GOOD MORNING DASHBOARD TABLES
# ============================================================================

CREATE_DAILY_BRIEFINGS_TABLE = """
CREATE TABLE IF NOT EXISTS daily_briefings (
    id INTEGER PRIMARY KEY,
    user_id VARCHAR NOT NULL DEFAULT 'default',
    project_key VARCHAR NOT NULL,

    briefing_date DATE NOT NULL,
    timeframe VARCHAR NOT NULL DEFAULT 'daily',  -- 'daily', 'weekly', 'monthly'

    -- The generated content
    narrative_summary TEXT,
    key_highlights VARCHAR,     -- JSON: [{type, message, evidence, severity}]
    recommendations VARCHAR,    -- JSON: [{id, action, reason, evidence}]

    -- Metrics snapshot
    metrics_snapshot VARCHAR,   -- JSON with metrics at time of briefing

    -- Generation metadata
    model_used VARCHAR,
    tokens_used INTEGER,
    generation_time_ms INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(user_id, project_key, briefing_date, timeframe)
)
"""

CREATE_RECOMMENDATION_ACTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS recommendation_actions (
    id INTEGER PRIMARY KEY,
    recommendation_id VARCHAR NOT NULL,
    briefing_id INTEGER NOT NULL,

    -- The recommendation
    recommendation_text TEXT,
    recommendation_type VARCHAR,  -- 'nudge', 'escalate', 'investigate', 'celebrate'
    ticket_keys VARCHAR[],        -- Tickets involved

    -- User response
    action_taken VARCHAR DEFAULT 'pending',  -- 'completed', 'snoozed', 'dismissed', 'pending'
    action_taken_at TIMESTAMP,
    snooze_until DATE,
    user_notes TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_STAKEHOLDER_QUERIES_TABLE = """
CREATE TABLE IF NOT EXISTS stakeholder_queries (
    id INTEGER PRIMARY KEY,
    project_key VARCHAR NOT NULL,

    -- Who asked
    asker_name VARCHAR,           -- "CEO", "CTO", "VP Product", etc.
    asker_role VARCHAR,           -- 'executive', 'manager', 'stakeholder'

    -- What they asked
    query_topic VARCHAR,          -- 'timeline', 'budget', 'blockers', 'team', 'scope'
    query_text TEXT,

    -- When
    asked_at TIMESTAMP NOT NULL,

    -- For pattern detection
    query_month INTEGER,
    query_week INTEGER,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_DAILY_DELTAS_TABLE = """
CREATE TABLE IF NOT EXISTS daily_deltas (
    id INTEGER PRIMARY KEY,
    project_key VARCHAR NOT NULL,
    delta_date DATE NOT NULL,

    -- What changed (counts)
    tickets_created INTEGER DEFAULT 0,
    tickets_completed INTEGER DEFAULT 0,
    tickets_reopened INTEGER DEFAULT 0,
    points_completed FLOAT DEFAULT 0,
    points_added FLOAT DEFAULT 0,
    points_removed FLOAT DEFAULT 0,

    -- Blockers
    new_blockers INTEGER DEFAULT 0,
    resolved_blockers INTEGER DEFAULT 0,
    active_blockers INTEGER DEFAULT 0,

    -- People signals
    after_hours_events INTEGER DEFAULT 0,
    weekend_events INTEGER DEFAULT 0,

    -- Status transitions
    status_transitions INTEGER DEFAULT 0,
    regressions INTEGER DEFAULT 0,  -- Done -> back to In Progress

    -- Lists (for drill-down, stored as JSON)
    completed_ticket_keys VARCHAR,   -- JSON array
    created_ticket_keys VARCHAR,     -- JSON array
    blocker_ticket_keys VARCHAR,     -- JSON array

    computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(project_key, delta_date)
)
"""

CREATE_ATTENTION_QUEUE_TABLE = """
CREATE TABLE IF NOT EXISTS attention_queue (
    id INTEGER PRIMARY KEY,
    project_key VARCHAR NOT NULL,
    issue_key VARCHAR NOT NULL,

    -- Why it needs attention
    attention_reason VARCHAR NOT NULL,
    -- 'silent_blocker', 'status_churn', 'overdue', 'scope_creep',
    -- 'no_assignee', 'blocked_too_long', 'approaching_deadline',
    -- 'after_hours_spike', 'regression'

    severity VARCHAR NOT NULL,  -- 'critical', 'high', 'medium', 'low'

    -- Evidence
    evidence_summary TEXT,
    days_in_state INTEGER,
    status_changes_last_week INTEGER,

    -- Scoring
    attention_score FLOAT,  -- For ranking

    -- Draft action
    suggested_action TEXT,
    draft_message TEXT,

    -- Lifecycle
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP,
    resolved_by VARCHAR,  -- 'user_action', 'auto_resolved', 'dismissed'

    -- Prevent duplicates
    UNIQUE(project_key, issue_key, attention_reason)
)
"""

CREATE_PREDICTION_SNAPSHOTS_TABLE = """
CREATE TABLE IF NOT EXISTS prediction_snapshots (
    id INTEGER PRIMARY KEY,
    project_key VARCHAR NOT NULL,

    -- What we're predicting
    target_type VARCHAR NOT NULL,    -- 'sprint', 'epic', 'milestone'
    target_id VARCHAR NOT NULL,
    target_name VARCHAR,

    snapshot_date DATE NOT NULL,

    -- Prediction results
    target_date DATE,
    probability_on_time FLOAT,
    p50_completion_date DATE,  -- Median
    p75_completion_date DATE,  -- Conservative
    p90_completion_date DATE,  -- Safe

    -- Risk factors (JSON)
    risk_factors VARCHAR,  -- [{factor, impact_days, confidence}]

    -- Simulation metadata
    simulations_run INTEGER,
    confidence_level VARCHAR,  -- 'high', 'medium', 'low'

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(project_key, target_type, target_id, snapshot_date)
)
"""

# Index creation statements
CREATE_INDEXES = [
    # Core tables
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
    # Good Morning Dashboard tables
    "CREATE INDEX IF NOT EXISTS idx_briefings_date ON daily_briefings(briefing_date)",
    "CREATE INDEX IF NOT EXISTS idx_briefings_project ON daily_briefings(project_key)",
    "CREATE INDEX IF NOT EXISTS idx_briefings_user_project ON daily_briefings(user_id, project_key)",
    "CREATE INDEX IF NOT EXISTS idx_rec_actions_briefing ON recommendation_actions(briefing_id)",
    "CREATE INDEX IF NOT EXISTS idx_rec_actions_status ON recommendation_actions(action_taken)",
    "CREATE INDEX IF NOT EXISTS idx_stakeholder_project ON stakeholder_queries(project_key)",
    "CREATE INDEX IF NOT EXISTS idx_stakeholder_asked ON stakeholder_queries(asked_at)",
    "CREATE INDEX IF NOT EXISTS idx_deltas_date ON daily_deltas(delta_date)",
    "CREATE INDEX IF NOT EXISTS idx_deltas_project ON daily_deltas(project_key)",
    "CREATE INDEX IF NOT EXISTS idx_attention_project ON attention_queue(project_key)",
    "CREATE INDEX IF NOT EXISTS idx_attention_severity ON attention_queue(severity)",
    "CREATE INDEX IF NOT EXISTS idx_attention_score ON attention_queue(attention_score)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_project ON prediction_snapshots(project_key)",
    "CREATE INDEX IF NOT EXISTS idx_predictions_date ON prediction_snapshots(snapshot_date)",
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
        # Good Morning Dashboard tables
        ("daily_briefings", CREATE_DAILY_BRIEFINGS_TABLE),
        ("recommendation_actions", CREATE_RECOMMENDATION_ACTIONS_TABLE),
        ("stakeholder_queries", CREATE_STAKEHOLDER_QUERIES_TABLE),
        ("daily_deltas", CREATE_DAILY_DELTAS_TABLE),
        ("attention_queue", CREATE_ATTENTION_QUEUE_TABLE),
        ("prediction_snapshots", CREATE_PREDICTION_SNAPSHOTS_TABLE),
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
        # Good Morning Dashboard tables (drop first due to FK)
        "prediction_snapshots",
        "attention_queue",
        "daily_deltas",
        "stakeholder_queries",
        "recommendation_actions",
        "daily_briefings",
        # Core tables
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
        # Good Morning Dashboard tables
        "daily_briefings",
        "recommendation_actions",
        "stakeholder_queries",
        "daily_deltas",
        "attention_queue",
        "prediction_snapshots",
    ]

    stats = {}
    for table in tables:
        try:
            result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            stats[table] = result[0] if result else 0
        except Exception:
            stats[table] = 0

    return stats
