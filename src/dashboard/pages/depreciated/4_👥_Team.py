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

    # Team Members (6 Famous Tech Developers)
    st.header("🌟 Team Members")
    st.caption("Famous tech developers workload")

    # Get workload from actual issue assignments
    dev_workload = conn.execute("""
        SELECT
            u.display_name as developer,
            u.email,
            COUNT(i.key) as total,
            SUM(CASE WHEN i.status NOT IN ('Done', 'Closed', 'Resolved') THEN 1 ELSE 0 END) as open,
            SUM(CASE WHEN i.status = 'In Progress' THEN 1 ELSE 0 END) as in_progress,
            SUM(CASE WHEN i.status IN ('Done', 'Closed', 'Resolved') THEN 1 ELSE 0 END) as done
        FROM users u
        LEFT JOIN issues i ON i.assignee_id = u.pseudonym
        GROUP BY u.display_name, u.email
        ORDER BY total DESC
    """).fetchall()

    if dev_workload:
        df = pd.DataFrame(dev_workload, columns=["Developer", "Email", "Total", "Open", "In Progress", "Done"])

        col1, col2 = st.columns([2, 1])

        with col1:
            st.dataframe(df, hide_index=True, width="stretch")

        with col2:
            st.subheader("📊 Workload by Developer")
            chart_df = df[["Developer", "Open", "Done"]].set_index("Developer")
            st.bar_chart(chart_df)
    else:
        st.info("No team members found.")

    # Frontend vs Backend Distribution
    st.header("🎨 Frontend vs Backend")

    category_dist = conn.execute("""
        WITH category_issues AS (
            SELECT
                i.key,
                i.status,
                UNNEST(i.labels) as label
            FROM issues i
            WHERE i.labels IS NOT NULL
        )
        SELECT
            label as category,
            COUNT(*) as total,
            SUM(CASE WHEN status NOT IN ('Done', 'Closed', 'Resolved') THEN 1 ELSE 0 END) as open,
            SUM(CASE WHEN status IN ('Done', 'Closed', 'Resolved') THEN 1 ELSE 0 END) as done
        FROM category_issues
        WHERE label IN ('frontend', 'backend')
        GROUP BY label
        ORDER BY total DESC
    """).fetchall()

    if category_dist:
        df = pd.DataFrame(category_dist, columns=["Category", "Total", "Open", "Done"])
        df["Category"] = df["Category"].str.title()

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(df, hide_index=True, width="stretch")

        with col2:
            chart_df = df[["Category", "Open", "Done"]].set_index("Category")
            st.bar_chart(chart_df)

    # Issue Types Distribution
    st.header("📋 Issue Types")

    type_dist = conn.execute("""
        WITH type_issues AS (
            SELECT
                i.key,
                i.status,
                UNNEST(i.labels) as label
            FROM issues i
            WHERE i.labels IS NOT NULL
        )
        SELECT
            REPLACE(label, 'type-', '') as type,
            COUNT(*) as total,
            SUM(CASE WHEN status NOT IN ('Done', 'Closed', 'Resolved') THEN 1 ELSE 0 END) as open,
            SUM(CASE WHEN status IN ('Done', 'Closed', 'Resolved') THEN 1 ELSE 0 END) as done
        FROM type_issues
        WHERE label LIKE 'type-%'
        GROUP BY label
        ORDER BY total DESC
    """).fetchall()

    if type_dist:
        df = pd.DataFrame(type_dist, columns=["Type", "Total", "Open", "Done"])
        df["Type"] = df["Type"].str.title()

        col1, col2 = st.columns(2)

        with col1:
            st.dataframe(df, hide_index=True, width="stretch")

        with col2:
            chart_df = df[["Type", "Total"]].set_index("Type")
            st.bar_chart(chart_df)

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
            st.dataframe(df, hide_index=True, width="stretch")

        with col2:
            st.bar_chart(df.set_index("Points"))

except Exception as e:
    st.error(f"Error: {e}")
