"""Overview page - Global statistics and key metrics."""

import streamlit as st
import duckdb
from pathlib import Path

st.set_page_config(page_title="Overview", page_icon="🏠", layout="wide")

def get_connection():
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        st.error("Database not found. Run 'jira-copilot sync full' first.")
        st.stop()
    return duckdb.connect(str(db_path), read_only=True)

st.title("🏠 Project Overview")

try:
    conn = get_connection()

    # Global Stats
    st.header("📊 Global Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_issues = conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
        st.metric("Total Issues", total_issues)

    with col2:
        total_sprints = conn.execute("SELECT COUNT(*) FROM sprints").fetchone()[0]
        st.metric("Total Sprints", total_sprints)

    with col3:
        active_sprint = conn.execute(
            "SELECT name FROM sprints WHERE state = 'active' LIMIT 1"
        ).fetchone()
        st.metric("Active Sprint", active_sprint[0] if active_sprint else "None")

    with col4:
        total_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        st.metric("Team Members", total_users)

    # Status Distribution
    st.header("📈 Issue Status Distribution")

    status_data = conn.execute("""
        SELECT status, COUNT(*) as count
        FROM issues
        GROUP BY status
        ORDER BY count DESC
    """).fetchall()

    if status_data:
        import pandas as pd
        df = pd.DataFrame(status_data, columns=["Status", "Count"])

        col1, col2 = st.columns([2, 1])
        with col1:
            st.bar_chart(df.set_index("Status"))
        with col2:
            st.dataframe(df, hide_index=True, use_container_width=True)

    # Sprint Summary
    st.header("🏃 Sprint Summary")

    sprints = conn.execute("""
        SELECT
            s.name,
            s.state,
            s.start_date,
            s.end_date,
            COUNT(i.key) as issue_count,
            SUM(CASE WHEN i.status IN ('Done', 'Closed', 'Resolved') THEN 1 ELSE 0 END) as done_count
        FROM sprints s
        LEFT JOIN issues i ON s.id = i.sprint_id
        GROUP BY s.id, s.name, s.state, s.start_date, s.end_date
        ORDER BY s.start_date DESC
        LIMIT 5
    """).fetchall()

    if sprints:
        import pandas as pd
        df = pd.DataFrame(sprints, columns=["Sprint", "State", "Start", "End", "Issues", "Done"])
        df["Completion"] = (df["Done"] / df["Issues"] * 100).fillna(0).round(1).astype(str) + "%"
        st.dataframe(df[["Sprint", "State", "Issues", "Done", "Completion"]], hide_index=True, use_container_width=True)

    # Recent Activity
    st.header("🕐 Recent Activity")

    recent = conn.execute("""
        SELECT key, summary, status, updated
        FROM issues
        ORDER BY updated DESC
        LIMIT 10
    """).fetchall()

    if recent:
        import pandas as pd
        df = pd.DataFrame(recent, columns=["Key", "Summary", "Status", "Updated"])
        df["Summary"] = df["Summary"].str[:50] + "..."
        st.dataframe(df, hide_index=True, use_container_width=True)

except Exception as e:
    st.error(f"Error: {e}")
