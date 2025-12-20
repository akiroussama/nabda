"""
📊 Project Overview - Command Center
Advanced project dashboard with real-time metrics, interactive charts, and activity feed.
"""

import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta

st.set_page_config(page_title="Overview", page_icon="📊", layout="wide")

# Premium CSS - Light Mode
st.markdown("""
<style>
    .overview-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .kpi-box {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
    }
    .kpi-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border-color: #cbd5e1;
    }

    .kpi-value {
        font-size: 32px;
        font-weight: 800;
        color: #1e293b;
        margin: 0;
    }

    .kpi-label {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .kpi-delta {
        font-size: 14px;
        margin-top: 4px;
    }
    .delta-up { color: #16a34a; }
    .delta-down { color: #dc2626; }
    .delta-neutral { color: #94a3b8; }

    .activity-item {
        display: flex;
        align-items: center;
        padding: 12px;
        border-bottom: 1px solid #f1f5f9;
        transition: background 0.2s;
    }
    .activity-item:hover { background: #f8fafc; }

    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        margin-right: 12px;
        font-size: 14px;
        color: white; /* Keep text white as background is colored */
    }

    .component-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        margin: 4px;
        background: #eff6ff;
        color: #3b82f6;
        border: 1px solid #dbeafe;
    }

    .status-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }
    .status-todo { background: #f1f5f9; color: #475569; }
    .status-progress { background: #fff7ed; color: #ea580c; }
    .status-done { background: #f0fdf4; color: #16a34a; }
    .status-blocked { background: #fef2f2; color: #dc2626; }

    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }
    
    h1, h2, h3 {
        color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)


def get_connection():
    db_path = Path("data/jira.duckdb")
    return duckdb.connect(str(db_path), read_only=True) if db_path.exists() else None


def get_avatar_color(name: str) -> str:
    """Generate consistent color for avatar based on name."""
    colors = ['#6366f1', '#8b5cf6', '#d946ef', '#ec4899', '#f43f5e', '#f97316', '#eab308', '#10b981']
    return colors[hash(name) % len(colors)]


def format_time_ago(dt) -> str:
    """Format datetime as 'time ago' string."""
    if pd.isna(dt):
        return "Unknown"

    try:
        if isinstance(dt, str):
            dt = pd.to_datetime(dt)

        now = datetime.now()
        if dt.tzinfo:
            dt = dt.replace(tzinfo=None)

        diff = now - dt

        if diff.days > 30:
            return f"{diff.days // 30}mo ago"
        elif diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}m ago"
        else:
            return "Just now"
    except:
        return "Unknown"


def create_donut_chart(df: pd.DataFrame, values_col: str, names_col: str, title: str) -> go.Figure:
    """Create a premium donut chart (Light)."""
    colors = ['#6366f1', '#8b5cf6', '#d946ef', '#3b82f6', '#10b981', '#f43f5e', '#f59e0b', '#22c55e']

    fig = go.Figure(data=[go.Pie(
        labels=df[names_col],
        values=df[values_col],
        hole=0.6,
        marker=dict(colors=colors[:len(df)]),
        textinfo='percent',
        textfont=dict(size=12, color='white'), # Keep white inside slices if colored
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])

    # Add center text
    total = df[values_col].sum()
    fig.add_annotation(
        text=f"<b>{total}</b><br>Total",
        x=0.5, y=0.5,
        font=dict(size=20, color='#1e293b'),
        showarrow=False
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#64748b')),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5,
            font=dict(color='#64748b')
        ),
        height=300,
        margin=dict(t=40, b=60, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, color: str = '#6366f1') -> go.Figure:
    """Create a premium bar chart (Light)."""
    fig = go.Figure(data=[go.Bar(
        x=df[x_col],
        y=df[y_col],
        marker=dict(
            color=color,
            line=dict(color=color, width=1)
        ),
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    )])

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#64748b')),
        xaxis=dict(
            title='',
            tickfont=dict(color='#64748b'),
            gridcolor='rgba(0,0,0,0.05)'
        ),
        yaxis=dict(
            title='',
            tickfont=dict(color='#64748b'),
            gridcolor='rgba(0,0,0,0.05)'
        ),
        height=300,
        margin=dict(t=40, b=40, l=40, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def main():
    st.markdown("# 📊 Project Overview")
    st.markdown("*Real-time project health and activity dashboard*")

    conn = get_connection()
    if not conn:
        st.error("Database not found. Please sync data first.")
        st.stop()

    # ========== TOP KPIs ==========
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    # Get KPI data
    total_issues = conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
    open_issues = conn.execute("SELECT COUNT(*) FROM issues WHERE status != 'Terminé(e)'").fetchone()[0]
    blocked = 0  # No 'Blocked' status in this Jira instance
    in_progress = conn.execute("SELECT COUNT(*) FROM issues WHERE status = 'En cours'").fetchone()[0]
    completed_today = conn.execute(f"""
        SELECT COUNT(*) FROM issues
        WHERE DATE(resolved) = CURRENT_DATE
    """).fetchone()[0]

    # Get week-over-week changes
    created_this_week = conn.execute("""
        SELECT COUNT(*) FROM issues WHERE created >= CURRENT_DATE - INTERVAL '7 days'
    """).fetchone()[0]
    created_last_week = conn.execute("""
        SELECT COUNT(*) FROM issues
        WHERE created >= CURRENT_DATE - INTERVAL '14 days'
        AND created < CURRENT_DATE - INTERVAL '7 days'
    """).fetchone()[0]

    with kpi1:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Total Issues</div>
            <div class="kpi-value">{total_issues:,}</div>
            <div class="kpi-delta delta-{'up' if created_this_week > 0 else 'neutral'}">
                +{created_this_week} this week
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Open Issues</div>
            <div class="kpi-value">{open_issues:,}</div>
            <div class="kpi-delta delta-neutral">{open_issues/max(total_issues,1)*100:.0f}% of total</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">In Progress</div>
            <div class="kpi-value">{in_progress}</div>
            <div class="kpi-delta delta-up">Active work</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Blocked</div>
            <div class="kpi-value" style="color: {'#dc2626' if blocked > 0 else '#16a34a'};">{blocked}</div>
            <div class="kpi-delta delta-{'down' if blocked > 0 else 'up'}">
                {'Needs attention' if blocked > 0 else 'All clear'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi5:
        st.markdown(f"""
        <div class="kpi-box">
            <div class="kpi-label">Done Today</div>
            <div class="kpi-value" style="color: #16a34a;">{completed_today}</div>
            <div class="kpi-delta delta-up">Keep it up!</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ========== ROW 1: Status & Priority ==========
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Status Distribution")
        status_df = conn.execute("""
            SELECT status, COUNT(*) as count
            FROM issues
            GROUP BY status
            ORDER BY count DESC
        """).fetchdf()

        if not status_df.empty:
            fig = create_donut_chart(status_df, 'count', 'status', '')
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Priority Breakdown")
        priority_df = conn.execute("""
            SELECT priority, COUNT(*) as count
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
                END
        """).fetchdf()

        if not priority_df.empty:
            # Color by priority
            priority_colors = {
                'Highest': '#ef4444',
                'High': '#f97316',
                'Medium': '#eab308',
                'Low': '#3b82f6',
                'Lowest': '#94a3b8'
            }
            colors = [priority_colors.get(p, '#6366f1') for p in priority_df['priority']]

            fig = go.Figure(data=[go.Bar(
                x=priority_df['priority'],
                y=priority_df['count'],
                marker_color=colors,
                text=priority_df['count'],
                textposition='auto',
                textfont=dict(color='white')
            )])
            fig.update_layout(
                height=300,
                margin=dict(t=20, b=40, l=40, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(tickfont=dict(color='#64748b')),
                yaxis=dict(tickfont=dict(color='#64748b'), gridcolor='rgba(0,0,0,0.05)')
            )
            st.plotly_chart(fig, use_container_width=True)

    # ========== ROW 2: Activity & Types ==========
    col3, col4 = st.columns([3, 2])

    with col3:
        st.markdown("### Recent Activity")

        recent = conn.execute("""
            SELECT key, summary, status, issue_type, assignee_name, updated
            FROM issues
            ORDER BY updated DESC
            LIMIT 8
        """).fetchdf()

        if not recent.empty:
            for _, row in recent.iterrows():
                name = row['assignee_name'] or 'Unassigned'
                initials = ''.join([n[0].upper() for n in name.split()[:2]]) if name != 'Unassigned' else '?'
                color = get_avatar_color(name)
                time_ago = format_time_ago(row['updated'])

                status_class = 'todo'
                if row['status'] == 'En cours':
                    status_class = 'progress'
                elif row['status'] == 'Terminé(e)':
                    status_class = 'done'

                st.markdown(f"""
                <div class="activity-item">
                    <div class="avatar" style="background: {color}; color: white;">{initials}</div>
                    <div style="flex: 1;">
                        <div style="color: #1e293b; font-weight: 500;">{row['summary'][:60]}{'...' if len(row['summary']) > 60 else ''}</div>
                        <div style="color: #64748b; font-size: 12px;">
                            <span class="status-pill status-{status_class}">{row['status']}</span>
                            {row['key']} • {row['issue_type']} • {name} • {time_ago}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col4:
        st.markdown("### Issue Types")

        types_df = conn.execute("""
            SELECT issue_type, COUNT(*) as count
            FROM issues
            GROUP BY issue_type
            ORDER BY count DESC
        """).fetchdf()

        if not types_df.empty:
            type_icons = {
                'Bug': '🐛',
                'Story': '📖',
                'Task': '✅',
                'Epic': '🎯',
                'Sub-task': '📌',
                'Improvement': '⬆️'
            }

            for _, row in types_df.iterrows():
                icon = type_icons.get(row['issue_type'], '📋')
                pct = row['count'] / total_issues * 100
                st.markdown(f"""
                <div style="margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                        <span style="color: #1e293b;">{icon} {row['issue_type']}</span>
                        <span style="color: #64748b;">{row['count']} ({pct:.0f}%)</span>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 4px; height: 8px; overflow: hidden;">
                        <div style="background: linear-gradient(90deg, #6366f1, #8b5cf6); width: {pct}%; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # ========== ROW 3: Team & Components ==========
    col5, col6 = st.columns(2)

    with col5:
        st.markdown("### Team Workload")

        workload_df = conn.execute("""
            SELECT
                COALESCE(assignee_name, 'Unassigned') as name,
                COUNT(*) as total,
                SUM(CASE WHEN status != 'Terminé(e)' THEN 1 ELSE 0 END) as open,
                SUM(CASE WHEN status = 'En cours' THEN 1 ELSE 0 END) as in_progress
            FROM issues
            GROUP BY COALESCE(assignee_name, 'Unassigned')
            ORDER BY total DESC
            LIMIT 6
        """).fetchdf()

        if not workload_df.empty:
            fig = go.Figure()

            fig.add_trace(go.Bar(
                y=workload_df['name'],
                x=workload_df['in_progress'],
                name='In Progress',
                orientation='h',
                marker_color='#f59e0b'
            ))

            fig.add_trace(go.Bar(
                y=workload_df['name'],
                x=workload_df['open'] - workload_df['in_progress'],
                name='Open',
                orientation='h',
                marker_color='#6366f1'
            ))

            fig.update_layout(
                barmode='stack',
                height=250,
                margin=dict(t=20, b=20, l=100, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', y=-0.1, font=dict(color='#64748b')),
                xaxis=dict(tickfont=dict(color='#64748b'), gridcolor='rgba(0,0,0,0.05)'),
                yaxis=dict(tickfont=dict(color='#64748b'))
            )
            st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.markdown("### Component Distribution")

        # Get components (they're stored as arrays) - use subquery for UNNEST
        components_df = conn.execute("""
            SELECT component, COUNT(*) as count
            FROM (
                SELECT UNNEST(components) as component
                FROM issues
                WHERE components IS NOT NULL
            )
            GROUP BY component
            ORDER BY count DESC
            LIMIT 10
        """).fetchdf()

        if not components_df.empty:
            st.markdown('<div style="padding: 10px;">', unsafe_allow_html=True)
            for _, row in components_df.iterrows():
                st.markdown(f"""
                <span class="component-tag">
                    {row['component']} <b>({row['count']})</b>
                </span>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No components assigned to issues yet.")

    # ========== ROW 4: Sprint & Velocity ==========
    st.markdown("---")
    col7, col8 = st.columns(2)

    with col7:
        st.markdown("### Current Sprint Progress")

        sprint = conn.execute("""
            SELECT id, name, state FROM sprints WHERE state = 'active' LIMIT 1
        """).fetchone()

        if sprint:
            sprint_id, sprint_name, _ = sprint

            sprint_issues = conn.execute(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'Terminé(e)' THEN 1 ELSE 0 END) as done,
                    SUM(CASE WHEN status = 'En cours' THEN 1 ELSE 0 END) as in_progress,
                    SUM(COALESCE(story_points, 0)) as total_points,
                    SUM(CASE WHEN status = 'Terminé(e)' THEN COALESCE(story_points, 0) ELSE 0 END) as done_points
                FROM issues
                WHERE sprint_id = {sprint_id}
            """).fetchone()

            total, done, in_prog, total_pts, done_pts = sprint_issues
            pct = (done / max(total, 1)) * 100
            pts_pct = (done_pts / max(total_pts, 1)) * 100

            st.markdown(f"**{sprint_name}**")

            # Progress bar
            st.markdown(f"""
            <div style="margin: 16px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #64748b;">Issues: {done}/{total}</span>
                    <span style="color: #16a34a; font-weight: 600;">{pct:.0f}%</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 8px; height: 16px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #22c55e, #4ade80); width: {pct}%; height: 100%;"></div>
                </div>
            </div>

            <div style="margin: 16px 0;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="color: #64748b;">Points: {done_pts:.0f}/{total_pts:.0f}</span>
                    <span style="color: #6366f1; font-weight: 600;">{pts_pct:.0f}%</span>
                </div>
                <div style="background: #e2e8f0; border-radius: 8px; height: 16px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #6366f1, #8b5cf6); width: {pts_pct}%; height: 100%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Sprint stats
            st.markdown(f"""
            <div style="display: flex; gap: 16px; margin-top: 16px;">
                <div style="flex: 1; text-align: center; padding: 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;">
                    <div style="color: #64748b; font-size: 11px;">TODO</div>
                    <div style="color: #3b82f6; font-size: 20px; font-weight: 700;">{total - done - in_prog}</div>
                </div>
                <div style="flex: 1; text-align: center; padding: 12px; background: #fff7ed; border: 1px solid #ffedd5; border-radius: 8px;">
                    <div style="color: #9a3412; font-size: 11px;">IN PROGRESS</div>
                    <div style="color: #ea580c; font-size: 20px; font-weight: 700;">{in_prog}</div>
                </div>
                <div style="flex: 1; text-align: center; padding: 12px; background: #f0fdf4; border: 1px solid #dcfce7; border-radius: 8px;">
                    <div style="color: #166534; font-size: 11px;">DONE</div>
                    <div style="color: #16a34a; font-size: 20px; font-weight: 700;">{done}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No active sprint found.")

    with col8:
        st.markdown("### Weekly Velocity Trend")

        # Get velocity data for last 8 weeks
        velocity_df = conn.execute("""
            SELECT
                DATE_TRUNC('week', resolved) as week,
                COUNT(*) as issues_completed,
                SUM(COALESCE(story_points, 0)) as points_completed
            FROM issues
            WHERE resolved IS NOT NULL
            AND resolved >= CURRENT_DATE - INTERVAL '56 days'
            GROUP BY DATE_TRUNC('week', resolved)
            ORDER BY week
        """).fetchdf()

        if not velocity_df.empty and len(velocity_df) > 1:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=velocity_df['week'],
                y=velocity_df['points_completed'],
                mode='lines+markers+text',
                name='Story Points',
                line=dict(color='#6366f1', width=3),
                marker=dict(size=10),
                fill='tozeroy',
                fillcolor='rgba(99, 102, 241, 0.2)',
                text=[f"{int(v)}" for v in velocity_df['points_completed']],
                textposition='top center',
                textfont=dict(color='#6366f1')
            ))

            # Add trend line
            if len(velocity_df) >= 3:
                z = np.polyfit(range(len(velocity_df)), velocity_df['points_completed'].values, 1)
                p = np.poly1d(z)
                trend_values = p(range(len(velocity_df)))

                fig.add_trace(go.Scatter(
                    x=velocity_df['week'],
                    y=trend_values,
                    mode='lines',
                    name='Trend',
                    line=dict(color='#ef4444', width=2, dash='dash')
                ))

            fig.update_layout(
                height=280,
                margin=dict(t=20, b=40, l=40, r=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', y=-0.15, font=dict(color='#64748b')),
                xaxis=dict(tickfont=dict(color='#64748b'), gridcolor='rgba(0,0,0,0.05)'),
                yaxis=dict(tickfont=dict(color='#64748b'), gridcolor='rgba(0,0,0,0.05)', title='Points')
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough velocity data. Complete more issues to see trends.")

    conn.close()


if __name__ == "__main__":
    main()
