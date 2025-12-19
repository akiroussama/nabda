"""Tickets page - Full list of issues with filtering."""

import streamlit as st
import duckdb
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Tickets", page_icon="📋", layout="wide")

def get_connection():
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        st.error("Database not found. Run 'jira-copilot sync full' first.")
        st.stop()
    return duckdb.connect(str(db_path), read_only=True)

st.title("📋 All Tickets")

try:
    conn = get_connection()

    # Filters
    st.sidebar.header("🔍 Filters")

    # Status filter
    statuses = conn.execute("SELECT DISTINCT status FROM issues WHERE status IS NOT NULL ORDER BY status").fetchall()
    status_options = ["All"] + [s[0] for s in statuses]
    selected_status = st.sidebar.selectbox("Status", status_options)

    # Sprint filter
    sprints = conn.execute("SELECT DISTINCT id, name FROM sprints ORDER BY start_date DESC").fetchall()
    sprint_options = {"All": None, "Backlog (No Sprint)": -1}
    sprint_options.update({s[1]: s[0] for s in sprints})
    selected_sprint = st.sidebar.selectbox("Sprint", list(sprint_options.keys()))

    # Type filter
    types = conn.execute("SELECT DISTINCT issue_type FROM issues WHERE issue_type IS NOT NULL ORDER BY issue_type").fetchall()
    type_options = ["All"] + [t[0] for t in types]
    selected_type = st.sidebar.selectbox("Type", type_options)

    # Priority filter
    priorities = conn.execute("SELECT DISTINCT priority FROM issues WHERE priority IS NOT NULL ORDER BY priority").fetchall()
    priority_options = ["All"] + [p[0] for p in priorities]
    selected_priority = st.sidebar.selectbox("Priority", priority_options)

    # Build query
    conditions = []
    params = []

    if selected_status != "All":
        conditions.append("status = ?")
        params.append(selected_status)

    if selected_sprint != "All":
        sprint_id = sprint_options[selected_sprint]
        if sprint_id == -1:
            conditions.append("sprint_id IS NULL")
        else:
            conditions.append("sprint_id = ?")
            params.append(sprint_id)

    if selected_type != "All":
        conditions.append("issue_type = ?")
        params.append(selected_type)

    if selected_priority != "All":
        conditions.append("priority = ?")
        params.append(selected_priority)

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Fetch issues
    query = f"""
        SELECT
            i.key,
            i.summary,
            i.issue_type,
            i.status,
            i.priority,
            i.story_points,
            COALESCE(un.display_name, i.assignee_id, 'Unassigned') as assignee,
            i.sprint_name,
            i.created,
            i.updated
        FROM issues i
        LEFT JOIN user_names un ON i.assignee_id = un.pseudonym
        WHERE {where_clause}
        ORDER BY i.updated DESC
    """

    issues = conn.execute(query, params).fetchall()

    # Display count
    st.info(f"📊 Showing {len(issues)} tickets")

    if issues:
        df = pd.DataFrame(issues, columns=[
            "Key", "Summary", "Type", "Status", "Priority",
            "Points", "Assignee", "Sprint", "Created", "Updated"
        ])

        # Truncate summary
        df["Summary"] = df["Summary"].str[:60]

        # Format dates
        df["Created"] = pd.to_datetime(df["Created"]).dt.strftime("%Y-%m-%d")
        df["Updated"] = pd.to_datetime(df["Updated"]).dt.strftime("%Y-%m-%d")

        # Display table
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Key": st.column_config.TextColumn("Key", width="small"),
                "Points": st.column_config.NumberColumn("SP", width="small"),
            }
        )

        # Export option
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            "tickets.csv",
            "text/csv",
            key="download-csv"
        )
    else:
        st.warning("No tickets found with the selected filters.")

except Exception as e:
    st.error(f"Error: {e}")
