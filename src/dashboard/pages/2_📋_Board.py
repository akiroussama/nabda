"""
📋 Kanban Board - Visual Work Management
Interactive Kanban board with swimlanes, filters, and visual cards.
"""

import streamlit as st
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime, timedelta

st.set_page_config(page_title="Board", page_icon="📋", layout="wide")

# Premium Kanban CSS - LIGHT MODE ADAPTED
st.markdown("""
<style>
    .kanban-column {
        background: #f1f5f9;
        border-radius: 12px;
        padding: 16px;
        min-height: 500px;
        border: 1px solid #e2e8f0;
    }

    .column-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 12px;
        border-radius: 8px;
        margin-bottom: 16px;
        background: #ffffff;
        border: 1px solid #e2e8f0;
    }
    .header-todo { border-left: 4px solid #3498db; }
    .header-progress { border-left: 4px solid #f39c12; }
    .header-blocked { border-left: 4px solid #e74c3c; }
    .header-done { border-left: 4px solid #27ae60; }

    .column-count {
        background: #e2e8f0;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 12px;
        color: #64748b;
        font-weight: 600;
    }

    .kanban-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 12px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
        cursor: pointer;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }
    .kanban-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-color: #6366f1;
    }

    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .card-key {
        font-size: 11px;
        color: #6366f1;
        font-weight: 600;
    }

    .card-points {
        background: #e0e7ff;
        color: #4338ca;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 700;
    }

    .card-summary {
        font-size: 14px;
        color: #1e293b;
        font-weight: 500;
        line-height: 1.4;
        margin-bottom: 12px;
    }

    .card-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .card-meta {
        display: flex;
        gap: 8px;
        align-items: center;
    }

    .type-badge {
        font-size: 10px;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .type-bug { background: #fee2e2; color: #991b1b; }
    .type-story { background: #dcfce7; color: #166534; }
    .type-task { background: #e0f2fe; color: #075985; }
    .type-epic { background: #f3e8ff; color: #6b21a8; }
    .type-improvement { background: #ffedd5; color: #9a3412; }

    .priority-badge {
        font-size: 10px;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .priority-highest { background: #ef4444; color: white; }
    .priority-high { background: #f97316; color: white; }
    .priority-medium { background: #eab308; color: #fff; }
    .priority-low { background: #3b82f6; color: white; }
    .priority-lowest { background: #94a3b8; color: white; }

    .card-avatar {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 700;
        color: white;
    }

    .filter-bar {
        background: #ffffff;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    .wip-warning {
        background: #fef2f2;
        border: 1px solid #ef4444;
        padding: 8px 12px;
        border-radius: 8px;
        font-size: 12px;
        color: #b91c1c;
        margin-top: 8px;
    }

    .swimlane-header {
        background: #f8fafc;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 20px 0 12px 0;
        border-left: 4px solid #6366f1;
        border: 1px solid #e2e8f0;
        border-left-width: 4px;
    }
</style>
""", unsafe_allow_html=True)


def get_connection():
    db_path = Path("data/jira.duckdb")
    return duckdb.connect(str(db_path), read_only=True) if db_path.exists() else None


def get_avatar_color(name: str) -> str:
    colors = ['#6366f1', '#8b5cf6', '#d946ef', '#ec4899', '#f43f5e', '#f97316', '#eab308', '#10b981']
    return colors[hash(name or '') % len(colors)]


def render_card(issue: pd.Series):
    """Render a single Kanban card."""
    name = issue['assignee_name'] or 'Unassigned'
    initials = ''.join([n[0].upper() for n in name.split()[:2]]) if name != 'Unassigned' else '?'
    avatar_color = get_avatar_color(name)

    type_class = issue['issue_type'].lower().replace('-', '').replace(' ', '')
    if type_class not in ['bug', 'story', 'task', 'epic', 'improvement']:
        type_class = 'task'

    priority = (issue['priority'] or 'medium').lower()
    if priority not in ['highest', 'high', 'medium', 'low', 'lowest']:
        priority = 'medium'

    points = issue.get('story_points', 0) or 0
    points_html = f'<span class="card-points">{int(points)} pts</span>' if points > 0 else ''

    # Important: Do not indent the HTML string to avoid Markdown code block rendering
    return f"""<div class="kanban-card">
    <div class="card-header">
        <span class="card-key">{issue['key']}</span>
        {points_html}
    </div>
    <div class="card-summary">{issue['summary'][:80]}{'...' if len(str(issue['summary'])) > 80 else ''}</div>
    <div class="card-footer">
        <div class="card-meta">
            <span class="type-badge type-{type_class}">{issue['issue_type']}</span>
            <span class="priority-badge priority-{priority}">{issue['priority'] or 'Medium'}</span>
        </div>
        <div class="card-avatar" style="background: {avatar_color};">{initials}</div>
    </div>
</div>"""


def main():
    st.markdown("# 📋 Kanban Board")
    st.markdown("*Visual work management with real-time updates*")

    conn = get_connection()
    if not conn:
        st.error("Database not found. Please sync data first.")
        st.stop()

    # ========== FILTERS ==========
    st.markdown('<div class="filter-bar">', unsafe_allow_html=True)

    f1, f2, f3, f4, f5 = st.columns([2, 1, 1, 1, 1])

    with f1:
        search = st.text_input("🔍 Search", placeholder="Search issues by key or summary...")

    with f2:
        assignees = conn.execute("""
            SELECT DISTINCT COALESCE(assignee_name, 'Unassigned') as name
            FROM issues
            ORDER BY name
        """).fetchdf()['name'].tolist()
        selected_assignee = st.selectbox("👤 Assignee", ['All'] + assignees)

    with f3:
        types = conn.execute("SELECT DISTINCT issue_type FROM issues WHERE issue_type IS NOT NULL").fetchdf()['issue_type'].tolist()
        selected_type = st.selectbox("📋 Type", ['All'] + types)

    with f4:
        priorities = ['All', 'Highest', 'High', 'Medium', 'Low', 'Lowest']
        selected_priority = st.selectbox("🎯 Priority", priorities)

    with f5:
        sprints = conn.execute("SELECT DISTINCT name FROM sprints ORDER BY id DESC").fetchdf()['name'].tolist()
        selected_sprint = st.selectbox("🏃 Sprint", ['All'] + sprints)

    st.markdown('</div>', unsafe_allow_html=True)

    # View mode
    col_view, col_wip = st.columns([2, 1])
    with col_view:
        view_mode = st.radio("View", ['Standard', 'Swimlanes (by Assignee)', 'Swimlanes (by Type)'], horizontal=True)
    with col_wip:
        wip_limit = st.number_input("WIP Limit (In Progress)", min_value=0, max_value=50, value=0, help="Set to 0 for no limit")

    # Build query
    query = """
        SELECT key, summary, status, priority, issue_type,
               COALESCE(assignee_name, 'Unassigned') as assignee_name,
               story_points, sprint_name
        FROM issues
        WHERE 1=1
    """

    if search:
        query += f" AND (summary ILIKE '%{search}%' OR key ILIKE '%{search}%')"
    if selected_assignee != 'All':
        query += f" AND COALESCE(assignee_name, 'Unassigned') = '{selected_assignee}'"
    if selected_type != 'All':
        query += f" AND issue_type = '{selected_type}'"
    if selected_priority != 'All':
        query += f" AND priority = '{selected_priority}'"
    if selected_sprint != 'All':
        query += f" AND sprint_name = '{selected_sprint}'"

    issues = conn.execute(query).fetchdf()

    if issues.empty:
        st.info("No issues found matching your criteria.")
        st.stop()

    # Define columns (using French status names from Jira)
    columns = [
        ('To Do', ['À faire'], 'header-todo'),
        ('In Progress', ['En cours'], 'header-progress'),
        ('Done', ['Terminé(e)'], 'header-done'),
    ]

    if view_mode == 'Standard':
        # ========== STANDARD VIEW ==========
        cols = st.columns(3)

        for idx, (col_name, status_list, header_class) in enumerate(columns):
            with cols[idx]:
                col_issues = issues[issues['status'].isin(status_list)]
                count = len(col_issues)
                points = col_issues['story_points'].sum() if 'story_points' in col_issues else 0

                st.markdown(f"""
                <div class="column-header {header_class}">
                    <span style="color: #fff; font-weight: 600;">{col_name}</span>
                    <span class="column-count">{count} issues • {int(points or 0)} pts</span>
                </div>
                """, unsafe_allow_html=True)

                # WIP limit warning
                if wip_limit > 0 and col_name == 'In Progress' and count > wip_limit:
                    st.markdown(f'<div class="wip-warning">⚠️ WIP limit exceeded! ({count}/{wip_limit})</div>', unsafe_allow_html=True)

                st.markdown('<div class="kanban-column">', unsafe_allow_html=True)

                for _, issue in col_issues.iterrows():
                    st.markdown(render_card(issue), unsafe_allow_html=True)

                if count == 0:
                    st.markdown('<p style="color: #8892b0; text-align: center; padding: 40px;">No issues</p>', unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

    elif view_mode.startswith('Swimlanes (by Assignee)'):
        # ========== SWIMLANE BY ASSIGNEE ==========
        assignee_groups = issues.groupby('assignee_name')

        for assignee, group_issues in assignee_groups:
            avatar_color = get_avatar_color(assignee)
            initials = ''.join([n[0].upper() for n in str(assignee).split()[:2]])

            total_pts = group_issues['story_points'].sum() or 0
            st.markdown(f"""
            <div class="swimlane-header">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <div class="card-avatar" style="background: {avatar_color}; width: 36px; height: 36px; font-size: 14px;">{initials}</div>
                    <div>
                        <div style="color: #fff; font-weight: 600; font-size: 16px;">{assignee}</div>
                        <div style="color: #8892b0; font-size: 12px;">{len(group_issues)} issues • {int(total_pts)} points</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(3)
            for idx, (col_name, status_list, _) in enumerate(columns):
                with cols[idx]:
                    col_issues = group_issues[group_issues['status'].isin(status_list)]
                    for _, issue in col_issues.iterrows():
                        st.markdown(render_card(issue), unsafe_allow_html=True)
                    if len(col_issues) == 0:
                        st.markdown('<p style="color: #8892b055; text-align: center; font-size: 12px;">-</p>', unsafe_allow_html=True)

    else:
        # ========== SWIMLANE BY TYPE ==========
        type_groups = issues.groupby('issue_type')

        type_icons = {'Bug': '🐛', 'Story': '📖', 'Task': '✅', 'Epic': '🎯', 'Sub-task': '📌', 'Improvement': '⬆️'}

        for issue_type, group_issues in type_groups:
            icon = type_icons.get(issue_type, '📋')
            total_pts = group_issues['story_points'].sum() or 0

            st.markdown(f"""
            <div class="swimlane-header">
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 24px;">{icon}</span>
                    <div>
                        <div style="color: #fff; font-weight: 600; font-size: 16px;">{issue_type}</div>
                        <div style="color: #8892b0; font-size: 12px;">{len(group_issues)} issues • {int(total_pts)} points</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            cols = st.columns(3)
            for idx, (col_name, status_list, _) in enumerate(columns):
                with cols[idx]:
                    col_issues = group_issues[group_issues['status'].isin(status_list)]
                    for _, issue in col_issues.iterrows():
                        st.markdown(render_card(issue), unsafe_allow_html=True)
                    if len(col_issues) == 0:
                        st.markdown('<p style="color: #8892b055; text-align: center; font-size: 12px;">-</p>', unsafe_allow_html=True)

    # ========== SUMMARY FOOTER ==========
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Total Displayed", len(issues))
    with m2:
        st.metric("To Do", len(issues[issues['status'] == 'À faire']))
    with m3:
        st.metric("In Progress", len(issues[issues['status'] == 'En cours']))
    with m4:
        total_pts = issues['story_points'].sum() or 0
        st.metric("Total Points", f"{int(total_pts)}")

    conn.close()


if __name__ == "__main__":
    main()
