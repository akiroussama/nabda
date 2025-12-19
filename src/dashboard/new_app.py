"""
Jira AI Co-pilot Dashboard - Clean Multi-Page Version

Launch with: streamlit run src/dashboard/new_app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Jira AI Co-pilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Jira AI Co-pilot")

st.markdown("""
Welcome to the Jira AI Co-pilot Dashboard!

### 📁 Pages

Use the sidebar to navigate between pages:

- **🏠 Overview** - Global statistics and key metrics
- **📋 Tickets** - Full list of issues with filtering
- **🏃 Sprints** - Sprint overview and details
- **👥 Team** - Developer workload and statistics
- **📊 Analytics** - Charts and trends

### 🚀 Quick Start

1. Make sure you've synced data: `jira-copilot sync full`
2. Navigate to the page you want using the sidebar
3. Use filters to narrow down your view

### 📈 Current Data

""")

# Show quick stats
import duckdb
from pathlib import Path

db_path = Path("data/jira.duckdb")
if db_path.exists():
    try:
        conn = duckdb.connect(str(db_path), read_only=True)

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            count = conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
            st.metric("Total Issues", count)

        with col2:
            count = conn.execute("SELECT COUNT(*) FROM sprints").fetchone()[0]
            st.metric("Total Sprints", count)

        with col3:
            count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            st.metric("Team Members", count)

        with col4:
            active = conn.execute("SELECT name FROM sprints WHERE state = 'active' LIMIT 1").fetchone()
            st.metric("Active Sprint", active[0] if active else "None")

        conn.close()
    except Exception as e:
        st.error(f"Error loading data: {e}")
else:
    st.warning("⚠️ No data found. Run `jira-copilot sync full` to sync data from Jira.")
