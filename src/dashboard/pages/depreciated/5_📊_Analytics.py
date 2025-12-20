"""Analytics page - Charts and trends."""

import streamlit as st
import duckdb
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

def get_connection():
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        st.error("Database not found. Run 'jira-copilot sync full' first.")
        st.stop()
    return duckdb.connect(str(db_path), read_only=True)

st.title("📊 Analytics")

try:
    conn = get_connection()

    # Velocity Trend
    st.header("🚀 Velocity by Sprint")

    velocity = conn.execute("""
        SELECT
            s.name as sprint,
            COALESCE(SUM(CASE WHEN i.status IN ('Done', 'Closed', 'Resolved') THEN i.story_points ELSE 0 END), 0) as completed_points,
            COALESCE(SUM(i.story_points), 0) as total_points
        FROM sprints s
        LEFT JOIN issues i ON s.id = i.sprint_id
        WHERE s.state IN ('closed', 'active')
        GROUP BY s.id, s.name, s.start_date
        ORDER BY s.start_date ASC
    """).fetchall()

    if velocity:
        df = pd.DataFrame(velocity, columns=["Sprint", "Completed", "Committed"])
        df["Completion %"] = (df["Completed"] / df["Committed"] * 100).fillna(0).round(1)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.line_chart(df.set_index("Sprint")[["Completed", "Committed"]])

        with col2:
            st.dataframe(df, hide_index=True, width="stretch")

    # Issue Types Distribution
    st.header("🏷️ Issue Types")

    types = conn.execute("""
        SELECT
            issue_type,
            COUNT(*) as count,
            COALESCE(AVG(story_points), 0) as avg_points
        FROM issues
        GROUP BY issue_type
        ORDER BY count DESC
    """).fetchall()

    if types:
        df = pd.DataFrame(types, columns=["Type", "Count", "Avg Points"])
        df["Avg Points"] = df["Avg Points"].round(1)

        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(df.set_index("Type")["Count"])

        with col2:
            st.dataframe(df, hide_index=True, width="stretch")

    # Priority Distribution
    st.header("🎯 Priority Distribution")

    priorities = conn.execute("""
        SELECT
            priority,
            COUNT(*) as count,
            SUM(CASE WHEN status IN ('Done', 'Closed', 'Resolved') THEN 1 ELSE 0 END) as done
        FROM issues
        WHERE priority IS NOT NULL
        GROUP BY priority
        ORDER BY
            CASE priority
                WHEN 'Highest' THEN 1
                WHEN 'High' THEN 2
                WHEN 'Medium' THEN 3
                WHEN 'Low' THEN 4
                WHEN 'Lowest' THEN 5
                ELSE 6
            END
    """).fetchall()

    if priorities:
        df = pd.DataFrame(priorities, columns=["Priority", "Total", "Done"])
        df["Open"] = df["Total"] - df["Done"]

        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(df.set_index("Priority")[["Done", "Open"]])

        with col2:
            st.dataframe(df, hide_index=True, width="stretch")

    # Created vs Resolved Over Time
    st.header("📈 Created vs Resolved (Last 30 Days)")

    timeline = conn.execute("""
        WITH dates AS (
            SELECT DISTINCT DATE_TRUNC('day', created) as date FROM issues
            WHERE created >= CURRENT_DATE - INTERVAL '30' DAY
            UNION
            SELECT DISTINCT DATE_TRUNC('day', resolved) as date FROM issues
            WHERE resolved >= CURRENT_DATE - INTERVAL '30' DAY
        )
        SELECT
            d.date,
            COUNT(CASE WHEN DATE_TRUNC('day', i.created) = d.date THEN 1 END) as created,
            COUNT(CASE WHEN DATE_TRUNC('day', i.resolved) = d.date THEN 1 END) as resolved
        FROM dates d
        LEFT JOIN issues i ON DATE_TRUNC('day', i.created) = d.date
                           OR DATE_TRUNC('day', i.resolved) = d.date
        WHERE d.date IS NOT NULL
        GROUP BY d.date
        ORDER BY d.date
    """).fetchall()

    if timeline:
        df = pd.DataFrame(timeline, columns=["Date", "Created", "Resolved"])
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%m-%d")
        st.line_chart(df.set_index("Date"))

    # Epic Progress
    st.header("📚 Epic Progress")

    epics = conn.execute("""
        SELECT
            COALESCE(epic_name, 'No Epic') as epic,
            COUNT(*) as total,
            SUM(CASE WHEN status IN ('Done', 'Closed', 'Resolved') THEN 1 ELSE 0 END) as done,
            COALESCE(SUM(story_points), 0) as total_points
        FROM issues
        GROUP BY epic_name
        ORDER BY total DESC
        LIMIT 10
    """).fetchall()

    if epics:
        df = pd.DataFrame(epics, columns=["Epic", "Total", "Done", "Points"])
        df["Progress %"] = (df["Done"] / df["Total"] * 100).fillna(0).round(1)
        df["Epic"] = df["Epic"].str[:30]

        st.dataframe(df, hide_index=True, width="stretch")

        # Progress bars
        for _, row in df.iterrows():
            progress = row["Done"] / row["Total"] if row["Total"] > 0 else 0
            st.progress(min(progress, 1.0), text=f"{row['Epic']}: {row['Progress %']}%")

except Exception as e:
    st.error(f"Error: {e}")
