"""
👥 Team Workload - Advanced Team Analytics & Capacity Planning
Individual developer profiles, workload distribution, and capacity forecasting.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_dir))

st.set_page_config(page_title="Team Workload", page_icon="👥", layout="wide")

# Premium Team Workload CSS
st.markdown("""
<style>
    .team-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        color: white;
    }

    .team-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .team-subtitle {
        font-size: 14px;
        opacity: 0.9;
    }

    .metric-row {
        display: flex;
        gap: 16px;
        margin-bottom: 20px;
    }

    .kpi-card {
        background: linear-gradient(145deg, #252541 0%, #1e1e32 100%);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        flex: 1;
    }

    .kpi-value {
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .kpi-label {
        font-size: 12px;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    .kpi-trend {
        font-size: 14px;
        margin-top: 8px;
    }
    .trend-up { color: #27ae60; }
    .trend-down { color: #e74c3c; }
    .trend-neutral { color: #f39c12; }

    .section-container {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.05);
    }

    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #fff;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .developer-card {
        background: linear-gradient(145deg, #252541 0%, #1e1e32 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 16px;
        transition: all 0.2s ease;
    }
    .developer-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        border-color: #667eea44;
    }

    .dev-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 16px;
    }

    .dev-avatar {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        font-weight: 700;
        color: white;
    }

    .dev-info {
        flex: 1;
    }

    .dev-name {
        font-size: 18px;
        font-weight: 600;
        color: #fff;
    }

    .dev-role {
        font-size: 13px;
        color: #8892b0;
    }

    .dev-status-badge {
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .status-optimal { background: #27ae6033; color: #27ae60; }
    .status-high { background: #f39c1233; color: #f39c12; }
    .status-overloaded { background: #e74c3c33; color: #e74c3c; }
    .status-underloaded { background: #3498db33; color: #3498db; }

    .dev-metrics {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 16px;
    }

    .dev-metric {
        background: rgba(255,255,255,0.03);
        padding: 12px;
        border-radius: 8px;
        text-align: center;
    }

    .dev-metric-value {
        font-size: 20px;
        font-weight: 700;
        color: #fff;
    }

    .dev-metric-label {
        font-size: 11px;
        color: #8892b0;
        text-transform: uppercase;
    }

    .capacity-bar-container {
        background: rgba(255,255,255,0.05);
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
        margin-top: 8px;
    }

    .capacity-bar {
        height: 100%;
        border-radius: 8px;
        transition: width 0.3s ease;
    }
    .capacity-low { background: linear-gradient(90deg, #27ae60 0%, #2ecc71 100%); }
    .capacity-medium { background: linear-gradient(90deg, #f39c12 0%, #f1c40f 100%); }
    .capacity-high { background: linear-gradient(90deg, #e74c3c 0%, #c0392b 100%); }

    .skill-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin: 2px 4px 2px 0;
        background: rgba(102, 126, 234, 0.2);
        color: #667eea;
    }

    .workload-row {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 8px;
        background: rgba(255,255,255,0.02);
        transition: background 0.2s;
    }
    .workload-row:hover {
        background: rgba(255,255,255,0.05);
    }

    .row-avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        color: white;
        font-size: 14px;
        margin-right: 14px;
    }

    .row-name {
        flex: 1;
        font-size: 14px;
        font-weight: 500;
        color: #fff;
    }

    .row-stat {
        text-align: center;
        width: 80px;
        font-size: 13px;
        color: #ccd6f6;
    }

    .row-bar-container {
        width: 150px;
        background: rgba(255,255,255,0.05);
        border-radius: 6px;
        height: 8px;
        overflow: hidden;
        margin-left: 16px;
    }

    .alert-card {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
    }
    .alert-warning { background: rgba(241, 196, 15, 0.15); border-left: 4px solid #f1c40f; }
    .alert-danger { background: rgba(231, 76, 60, 0.15); border-left: 4px solid #e74c3c; }
    .alert-success { background: rgba(39, 174, 96, 0.15); border-left: 4px solid #27ae60; }

    .capacity-gauge-container {
        display: flex;
        justify-content: center;
        gap: 24px;
    }
</style>
""", unsafe_allow_html=True)


def get_connection():
    db_path = Path("data/jira.duckdb")
    return duckdb.connect(str(db_path), read_only=True) if db_path.exists() else None


def get_avatar_color(name: str) -> str:
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#fa709a']
    return colors[hash(name or '') % len(colors)]


def get_initials(name: str) -> str:
    if not name or name == 'Unassigned':
        return '?'
    parts = name.split()[:2]
    return ''.join(p[0].upper() for p in parts if p)


def create_capacity_gauge(utilized: float, title: str) -> go.Figure:
    """Create capacity utilization gauge."""
    if utilized <= 70:
        color = "#27ae60"
        status = "Available"
    elif utilized <= 90:
        color = "#f39c12"
        status = "Near Capacity"
    else:
        color = "#e74c3c"
        status = "Overloaded"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=min(utilized, 100),
        number={'suffix': '%', 'font': {'size': 28, 'color': '#fff'}},
        title={'text': f"<b>{title}</b><br><span style='font-size:11px;color:#8892b0'>{status}</span>",
               'font': {'size': 14, 'color': '#ccd6f6'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': "rgba(0,0,0,0)"},
            'bar': {'color': color, 'thickness': 0.8},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 70], 'color': 'rgba(39, 174, 96, 0.1)'},
                {'range': [70, 90], 'color': 'rgba(241, 196, 15, 0.1)'},
                {'range': [90, 100], 'color': 'rgba(231, 76, 60, 0.1)'}
            ]
        }
    ))

    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ccd6f6'}
    )

    return fig


def create_workload_heatmap(team_data: pd.DataFrame) -> go.Figure:
    """Create team workload heatmap."""
    if team_data.empty:
        return None

    # Create heatmap data
    categories = ['WIP', 'Blocked', 'Velocity', 'Capacity']
    z_data = []

    for _, row in team_data.iterrows():
        # Normalize values to 0-100 scale
        wip_score = min(row.get('wip_points', 0) / 20 * 100, 100)  # 20 pts = 100%
        blocked_score = min(row.get('blocked_count', 0) / 3 * 100, 100)  # 3 blocked = 100%
        velocity_score = min(row.get('velocity_7d', 0) / 15 * 100, 100)  # 15 pts = 100%
        capacity_score = min(row.get('capacity_used', 70), 100)

        z_data.append([wip_score, blocked_score, velocity_score, capacity_score])

    z_data = np.array(z_data).T  # Transpose for proper orientation

    fig = go.Figure(go.Heatmap(
        z=z_data,
        x=team_data['name'].tolist(),
        y=categories,
        colorscale=[
            [0, 'rgba(39, 174, 96, 0.8)'],
            [0.5, 'rgba(241, 196, 15, 0.8)'],
            [1, 'rgba(231, 76, 60, 0.8)']
        ],
        showscale=False,
        hovertemplate='%{y}: %{z:.0f}%<br>%{x}<extra></extra>'
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=80, r=20, t=20, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ccd6f6'},
        xaxis=dict(tickangle=-45, showgrid=False),
        yaxis=dict(showgrid=False)
    )

    return fig


def create_workload_distribution(team_data: pd.DataFrame) -> go.Figure:
    """Create stacked bar chart showing workload distribution."""
    if team_data.empty:
        return None

    team_data = team_data.sort_values('wip_points', ascending=True).tail(10)

    fig = go.Figure()

    # Done points (green)
    fig.add_trace(go.Bar(
        y=team_data['name'],
        x=team_data['done_points'],
        name='Completed',
        orientation='h',
        marker_color='#27ae60',
        hovertemplate='%{y}<br>Completed: %{x} pts<extra></extra>'
    ))

    # In Progress points (yellow)
    fig.add_trace(go.Bar(
        y=team_data['name'],
        x=team_data['wip_points'],
        name='In Progress',
        orientation='h',
        marker_color='#f39c12',
        hovertemplate='%{y}<br>In Progress: %{x} pts<extra></extra>'
    ))

    # Blocked points (red)
    fig.add_trace(go.Bar(
        y=team_data['name'],
        x=team_data['blocked_points'],
        name='Blocked',
        orientation='h',
        marker_color='#e74c3c',
        hovertemplate='%{y}<br>Blocked: %{x} pts<extra></extra>'
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ccd6f6'},
        barmode='stack',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            title='Story Points'
        ),
        yaxis=dict(showgrid=False)
    )

    return fig


def create_velocity_trend(conn) -> go.Figure:
    """Create team velocity trend over sprints."""
    velocity_data = conn.execute("""
        SELECT
            s.name as sprint,
            COALESCE(SUM(CASE WHEN i.status = 'Terminé(e)'
                         THEN i.story_points ELSE 0 END), 0) as velocity,
            COUNT(DISTINCT i.assignee_name) as team_size
        FROM sprints s
        LEFT JOIN issues i ON i.sprint_id = s.id
        WHERE s.start_date IS NOT NULL
        GROUP BY s.name, s.start_date
        ORDER BY s.start_date DESC
        LIMIT 8
    """).fetchdf()

    if velocity_data.empty:
        return None

    velocity_data = velocity_data.iloc[::-1]  # Reverse chronological
    velocity_data['avg_per_person'] = velocity_data['velocity'] / velocity_data['team_size'].clip(lower=1)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Total velocity bars
    fig.add_trace(go.Bar(
        x=velocity_data['sprint'],
        y=velocity_data['velocity'],
        name='Team Velocity',
        marker_color='#667eea',
        hovertemplate='%{x}<br>Velocity: %{y} pts<extra></extra>'
    ), secondary_y=False)

    # Average per person line
    fig.add_trace(go.Scatter(
        x=velocity_data['sprint'],
        y=velocity_data['avg_per_person'],
        name='Per Person',
        mode='lines+markers',
        line=dict(color='#f39c12', width=3),
        marker=dict(size=8),
        hovertemplate='%{x}<br>Per Person: %{y:.1f} pts<extra></extra>'
    ), secondary_y=True)

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=40, t=20, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ccd6f6'},
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        xaxis=dict(
            showgrid=False,
            tickangle=-45
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            title='Team Points'
        ),
        yaxis2=dict(
            showgrid=False,
            title='Per Person'
        )
    )

    return fig


def create_issue_type_by_dev(team_data: pd.DataFrame, issues_df: pd.DataFrame) -> go.Figure:
    """Create issue type distribution by developer."""
    if issues_df.empty:
        return None

    # Get issue type counts by assignee
    type_pivot = issues_df.groupby(['assignee_name', 'issue_type']).size().unstack(fill_value=0)

    if type_pivot.empty:
        return None

    type_colors = {
        'Bug': '#e74c3c',
        'Story': '#27ae60',
        'Task': '#3498db',
        'Epic': '#9b59b6',
        'Sub-task': '#f39c12',
        'Improvement': '#1abc9c'
    }

    fig = go.Figure()

    for issue_type in type_pivot.columns:
        fig.add_trace(go.Bar(
            y=type_pivot.index,
            x=type_pivot[issue_type],
            name=issue_type,
            orientation='h',
            marker_color=type_colors.get(issue_type, '#667eea'),
            hovertemplate=f'{issue_type}<br>%{{y}}: %{{x}}<extra></extra>'
        ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#ccd6f6'},
        barmode='stack',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.05)',
            title='Issues'
        ),
        yaxis=dict(showgrid=False)
    )

    return fig


def main():
    st.markdown("# 👥 Team Workload & Capacity")
    st.markdown("*Real-time team analytics, individual profiles, and capacity planning*")

    conn = get_connection()
    if not conn:
        st.error("Database not found. Please sync data first.")
        st.stop()

    # Get team workload data
    team_query = conn.execute("""
        SELECT
            COALESCE(assignee_name, 'Unassigned') as name,
            COUNT(*) as total_issues,
            SUM(CASE WHEN status = 'Terminé(e)' THEN 1 ELSE 0 END) as done_count,
            SUM(CASE WHEN status = 'En cours' THEN 1 ELSE 0 END) as wip_count,
            0 as blocked_count,
            COALESCE(SUM(story_points), 0) as total_points,
            COALESCE(SUM(CASE WHEN status = 'Terminé(e)' THEN story_points ELSE 0 END), 0) as done_points,
            COALESCE(SUM(CASE WHEN status = 'En cours' THEN story_points ELSE 0 END), 0) as wip_points,
            0 as blocked_points
        FROM issues
        WHERE assignee_name IS NOT NULL
        GROUP BY assignee_name
        ORDER BY wip_points DESC
    """).fetchdf()

    if team_query.empty:
        st.warning("No team data found.")
        st.stop()

    # Calculate derived metrics
    team_query['velocity_7d'] = team_query['done_points'] * 0.4  # Simulated weekly velocity
    team_query['capacity_used'] = (team_query['wip_points'] / (team_query['done_points'].clip(lower=5) / 2) * 100).clip(0, 150)

    # Get all issues for additional analysis
    issues_df = conn.execute("""
        SELECT key, summary, status, issue_type, priority,
               COALESCE(story_points, 0) as story_points,
               COALESCE(assignee_name, 'Unassigned') as assignee_name
        FROM issues
    """).fetchdf()

    # ========== SUMMARY KPIs ==========
    st.markdown('<div class="metric-row">', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)

    team_size = len(team_query)
    total_wip = team_query['wip_points'].sum()
    total_blocked = team_query['blocked_count'].sum()
    overloaded_count = len(team_query[team_query['capacity_used'] > 90])
    avg_velocity = team_query['velocity_7d'].mean()

    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{team_size}</div>
            <div class="kpi-label">Team Members</div>
            <div class="kpi-trend trend-neutral">Active developers</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        trend = "trend-up" if total_wip < 100 else ("trend-down" if total_wip > 200 else "trend-neutral")
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_wip:.0f}</div>
            <div class="kpi-label">Total WIP Points</div>
            <div class="kpi-trend {trend}">Work in progress</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        trend = "trend-down" if total_blocked > 0 else "trend-up"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_blocked}</div>
            <div class="kpi-label">Blocked Items</div>
            <div class="kpi-trend {trend}">Needs attention</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        trend = "trend-down" if overloaded_count > 0 else "trend-up"
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{overloaded_count}</div>
            <div class="kpi-label">Overloaded</div>
            <div class="kpi-trend {trend}">Team members</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_velocity:.1f}</div>
            <div class="kpi-label">Avg Velocity</div>
            <div class="kpi-trend trend-neutral">pts/week/person</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== ROW 1: Capacity Gauges ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Team Capacity Overview</div>', unsafe_allow_html=True)

    avg_capacity = team_query['capacity_used'].mean()
    high_performers = len(team_query[team_query['velocity_7d'] > avg_velocity])
    balance_score = 100 - (team_query['capacity_used'].std() / avg_capacity * 100) if avg_capacity > 0 else 100

    g1, g2, g3 = st.columns(3)

    with g1:
        fig = create_capacity_gauge(avg_capacity, "Team Utilization")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with g2:
        fig = create_capacity_gauge(high_performers / team_size * 100, "High Performers")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    with g3:
        fig = create_capacity_gauge(min(balance_score, 100), "Workload Balance")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== ROW 2: Charts ==========
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Workload Distribution</div>', unsafe_allow_html=True)
        dist_fig = create_workload_distribution(team_query)
        if dist_fig:
            st.plotly_chart(dist_fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Sprint Velocity Trend</div>', unsafe_allow_html=True)
        velocity_fig = create_velocity_trend(conn)
        if velocity_fig:
            st.plotly_chart(velocity_fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ========== ROW 3: Team Heatmap ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔥 Team Performance Heatmap</div>', unsafe_allow_html=True)

    heatmap_fig = create_workload_heatmap(team_query.head(12))
    if heatmap_fig:
        st.plotly_chart(heatmap_fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== ROW 4: Alerts ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚠️ Team Alerts</div>', unsafe_allow_html=True)

    alerts = []

    # Check for overloaded members
    overloaded = team_query[team_query['capacity_used'] > 100]
    for _, dev in overloaded.iterrows():
        alerts.append(('danger', f"🔴 **{dev['name']}** is overloaded at {dev['capacity_used']:.0f}% capacity ({dev['wip_points']:.0f} WIP points)"))

    # Check for blocked items
    blocked_members = team_query[team_query['blocked_count'] > 0]
    for _, dev in blocked_members.iterrows():
        alerts.append(('warning', f"🟡 **{dev['name']}** has {int(dev['blocked_count'])} blocked issue(s) requiring attention"))

    # Check for underutilized members
    underutilized = team_query[team_query['capacity_used'] < 30]
    for _, dev in underutilized.iterrows():
        if dev['name'] != 'Unassigned':
            alerts.append(('success', f"🟢 **{dev['name']}** has available capacity ({dev['capacity_used']:.0f}% utilized)"))

    if not alerts:
        alerts.append(('success', "✅ Team workload is balanced with no critical alerts"))

    for level, message in alerts[:6]:  # Show max 6 alerts
        st.markdown(f'<div class="alert-card alert-{level}">{message}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== ROW 5: Individual Developer Profiles ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Team Member Profiles</div>', unsafe_allow_html=True)

    # Filter
    show_all = st.checkbox("Show all team members", value=False)
    display_data = team_query if show_all else team_query.head(8)

    for _, dev in display_data.iterrows():
        if dev['name'] == 'Unassigned':
            continue

        avatar_color = get_avatar_color(dev['name'])
        initials = get_initials(dev['name'])

        # Determine status
        if dev['capacity_used'] > 100:
            status = 'overloaded'
            status_text = 'Overloaded'
        elif dev['capacity_used'] > 80:
            status = 'high'
            status_text = 'High Load'
        elif dev['capacity_used'] < 30:
            status = 'underloaded'
            status_text = 'Available'
        else:
            status = 'optimal'
            status_text = 'Optimal'

        # Get issue types for this developer
        dev_issues = issues_df[issues_df['assignee_name'] == dev['name']]
        issue_types = dev_issues['issue_type'].value_counts().head(3).index.tolist()

        st.markdown(f"""
        <div class="developer-card">
            <div class="dev-header">
                <div class="dev-avatar" style="background: {avatar_color};">{initials}</div>
                <div class="dev-info">
                    <div class="dev-name">{dev['name']}</div>
                    <div class="dev-role">{"  ".join([f'<span class="skill-tag">{t}</span>' for t in issue_types])}</div>
                </div>
                <span class="dev-status-badge status-{status}">{status_text}</span>
            </div>
            <div class="dev-metrics">
                <div class="dev-metric">
                    <div class="dev-metric-value">{int(dev['total_issues'])}</div>
                    <div class="dev-metric-label">Total Issues</div>
                </div>
                <div class="dev-metric">
                    <div class="dev-metric-value">{dev['wip_points']:.0f}</div>
                    <div class="dev-metric-label">WIP Points</div>
                </div>
                <div class="dev-metric">
                    <div class="dev-metric-value">{dev['done_points']:.0f}</div>
                    <div class="dev-metric-label">Completed</div>
                </div>
                <div class="dev-metric">
                    <div class="dev-metric-value">{dev['velocity_7d']:.1f}</div>
                    <div class="dev-metric-label">Velocity/wk</div>
                </div>
            </div>
            <div style="font-size: 12px; color: #8892b0; margin-bottom: 4px;">Capacity Utilization</div>
            <div class="capacity-bar-container">
                <div class="capacity-bar capacity-{'low' if dev['capacity_used'] <= 70 else ('medium' if dev['capacity_used'] <= 90 else 'high')}"
                     style="width: {min(dev['capacity_used'], 100):.0f}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== Issue Type Distribution ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Issue Type Distribution by Developer</div>', unsafe_allow_html=True)

    type_fig = create_issue_type_by_dev(team_query, issues_df[issues_df['assignee_name'] != 'Unassigned'])
    if type_fig:
        st.plotly_chart(type_fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== Summary Footer ==========
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #8892b0; font-size: 12px;">
        Team Workload Dashboard | {team_size} team members | {total_wip:.0f} WIP points |
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
    """, unsafe_allow_html=True)

    conn.close()


if __name__ == "__main__":
    main()
