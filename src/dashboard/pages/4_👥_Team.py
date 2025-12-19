"""Team page - Developer workload and statistics."""

import streamlit as st
import duckdb
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Team", page_icon="👥", layout="wide")

def get_connection():
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        st.error("Database not found. Run 'jira-copilot sync full' first.")
        st.stop()
    return duckdb.connect(str(db_path), read_only=True)

st.title("👥 Team Overview")

try:
    conn = get_connection()

    # Team Members
    st.header("👤 Team Members")

    users = conn.execute("""
        SELECT
            COALESCE(display_name, pseudonym) as name,
            COALESCE(NULLIF(email, ''), '-') as email,
            CASE WHEN active THEN '✅ Active' ELSE '❌ Inactive' END as status
        FROM users
        ORDER BY name
    """).fetchall()

    if users:
        df = pd.DataFrame(users, columns=["Name", "Email", "Status"])
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("No team members found in database.")

    # Workload by Assignee
    st.header("📊 Current Workload")

    workload = conn.execute("""
        SELECT
            COALESCE(un.display_name, i.assignee_id, 'Unassigned') as assignee,
            COUNT(*) as total_issues,
            SUM(CASE WHEN i.status NOT IN ('Done', 'Closed', 'Resolved') THEN 1 ELSE 0 END) as open_issues,
            SUM(CASE WHEN i.status = 'In Progress' THEN 1 ELSE 0 END) as in_progress,
            COALESCE(SUM(CASE WHEN i.status NOT IN ('Done', 'Closed', 'Resolved') THEN i.story_points ELSE 0 END), 0) as open_points,
            SUM(CASE WHEN i.status IN ('Done', 'Closed', 'Resolved') THEN 1 ELSE 0 END) as done_issues
        FROM issues i
        LEFT JOIN user_names un ON i.assignee_id = un.pseudonym
        GROUP BY COALESCE(un.display_name, i.assignee_id, 'Unassigned')
        ORDER BY open_issues DESC
    """).fetchall()

    if workload:
        df = pd.DataFrame(workload, columns=[
            "Assignee", "Total", "Open", "In Progress", "Open Points", "Done"
        ])

        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(df, hide_index=True, use_container_width=True)

        with col2:
            st.subheader("Open Issues by Person")
            chart_df = df[["Assignee", "Open"]].set_index("Assignee")
            st.bar_chart(chart_df)

    # Developer Labels (simulated developers)
    st.header("🏷️ Developer Labels Distribution")

    # Extract developer labels using subquery for UNNEST
    dev_labels = conn.execute("""
        SELECT label, COUNT(*) as count
        FROM (
            SELECT UNNEST(labels) as label
            FROM issues
            WHERE labels IS NOT NULL
        )
        WHERE label LIKE 'dev-%'
        GROUP BY label
        ORDER BY count DESC
    """).fetchall()

    if dev_labels:
        df = pd.DataFrame(dev_labels, columns=["Developer", "Issues"])
        df["Developer"] = df["Developer"].str.replace("dev-", "").str.title()

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(df, hide_index=True, use_container_width=True)

        with col2:
            st.bar_chart(df.set_index("Developer"))

    # Story Points Labels Distribution
    st.header("📏 Story Points Distribution")

    sp_labels = conn.execute("""
        SELECT label, COUNT(*) as count
        FROM (
            SELECT UNNEST(labels) as label
            FROM issues
            WHERE labels IS NOT NULL
        )
        WHERE label LIKE 'sp-%'
        GROUP BY label
        ORDER BY label
    """).fetchall()

    if sp_labels:
        df = pd.DataFrame(sp_labels, columns=["Points", "Count"])
        df["Points"] = df["Points"].str.replace("sp-", "")

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(df, hide_index=True, use_container_width=True)

        with col2:
            st.bar_chart(df.set_index("Points"))

except Exception as e:
    st.error(f"Error: {e}")
