"""Sprints page - Sprint overview and details."""

import streamlit as st
import duckdb
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Sprints", page_icon="🏃", layout="wide")

def get_connection():
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        st.error("Database not found. Run 'jira-copilot sync full' first.")
        st.stop()
    return duckdb.connect(str(db_path), read_only=True)

st.title("🏃 Sprints")

try:
    conn = get_connection()

    # Sprint List
    st.header("📅 All Sprints")

    sprints = conn.execute("""
        SELECT
            s.id,
            s.name,
            s.state,
            s.start_date,
            s.end_date,
            COUNT(i.key) as total_issues,
            SUM(CASE WHEN i.status IN ('Done', 'Closed', 'Resolved') THEN 1 ELSE 0 END) as done_issues,
            COALESCE(SUM(i.story_points), 0) as total_points,
            COALESCE(SUM(CASE WHEN i.status IN ('Done', 'Closed', 'Resolved') THEN i.story_points ELSE 0 END), 0) as done_points
        FROM sprints s
        LEFT JOIN issues i ON s.id = i.sprint_id
        GROUP BY s.id, s.name, s.state, s.start_date, s.end_date
        ORDER BY s.start_date DESC
    """).fetchall()

    if sprints:
        df = pd.DataFrame(sprints, columns=[
            "ID", "Name", "State", "Start", "End",
            "Issues", "Done", "Points", "Done Points"
        ])

        # Calculate completion rate
        df["Completion %"] = (df["Done"] / df["Issues"] * 100).fillna(0).round(1)

        # State emoji
        state_emoji = {"active": "🟢", "closed": "✅", "future": "⏳"}
        df["State"] = df["State"].apply(lambda x: f"{state_emoji.get(x, '❓')} {x}")

        # Format dates
        df["Start"] = pd.to_datetime(df["Start"]).dt.strftime("%Y-%m-%d")
        df["End"] = pd.to_datetime(df["End"]).dt.strftime("%Y-%m-%d")

        st.dataframe(
            df[["Name", "State", "Start", "End", "Issues", "Done", "Completion %", "Points", "Done Points"]],
            hide_index=True,
            use_container_width=True
        )

    # Sprint Details
    st.header("🔍 Sprint Details")

    sprint_names = {s[1]: s[0] for s in sprints}
    selected_sprint = st.selectbox("Select a Sprint", list(sprint_names.keys()))
    sprint_id = sprint_names[selected_sprint]

    # Sprint Issues
    st.subheader(f"📋 Issues in {selected_sprint}")

    sprint_issues = conn.execute("""
        SELECT
            i.key,
            i.summary,
            i.issue_type,
            i.status,
            i.priority,
            i.story_points,
            COALESCE(un.display_name, i.assignee_id, 'Unassigned') as assignee
        FROM issues i
        LEFT JOIN user_names un ON i.assignee_id = un.pseudonym
        WHERE i.sprint_id = ?
        ORDER BY
            CASE i.status
                WHEN 'Done' THEN 4
                WHEN 'In Progress' THEN 1
                WHEN 'To Do' THEN 2
                ELSE 3
            END,
            i.priority
    """, [sprint_id]).fetchall()

    if sprint_issues:
        df = pd.DataFrame(sprint_issues, columns=[
            "Key", "Summary", "Type", "Status", "Priority", "Points", "Assignee"
        ])
        df["Summary"] = df["Summary"].str[:50]

        # Status color
        def status_color(status):
            colors = {
                "Done": "🟢", "Closed": "🟢", "Resolved": "🟢",
                "In Progress": "🔵",
                "To Do": "⚪",
                "Blocked": "🔴"
            }
            return f"{colors.get(status, '⚫')} {status}"

        df["Status"] = df["Status"].apply(status_color)

        st.dataframe(df, hide_index=True, use_container_width=True)

        # Status breakdown
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Status Breakdown")
            status_counts = conn.execute("""
                SELECT status, COUNT(*) as count
                FROM issues
                WHERE sprint_id = ?
                GROUP BY status
                ORDER BY count DESC
            """, [sprint_id]).fetchall()

            if status_counts:
                status_df = pd.DataFrame(status_counts, columns=["Status", "Count"])
                st.bar_chart(status_df.set_index("Status"))

        with col2:
            st.subheader("👥 By Assignee")
            assignee_counts = conn.execute("""
                SELECT COALESCE(un.display_name, i.assignee_id, 'Unassigned') as assignee, COUNT(*) as count
                FROM issues i
                LEFT JOIN user_names un ON i.assignee_id = un.pseudonym
                WHERE i.sprint_id = ?
                GROUP BY COALESCE(un.display_name, i.assignee_id, 'Unassigned')
                ORDER BY count DESC
            """, [sprint_id]).fetchall()

            if assignee_counts:
                assignee_df = pd.DataFrame(assignee_counts, columns=["Assignee", "Count"])
                st.bar_chart(assignee_df.set_index("Assignee"))
    else:
        st.info("No issues in this sprint.")

except Exception as e:
    st.error(f"Error: {e}")
