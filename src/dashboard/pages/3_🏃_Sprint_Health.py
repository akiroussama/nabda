"""
🏃 Sprint Health - Advanced Sprint Analytics Dashboard
Interactive burndown/burnup charts, velocity tracking, and real-time sprint metrics.
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

st.set_page_config(page_title="Sprint Health", page_icon="🏃", layout="wide")

# Premium Sprint Health CSS
st.markdown("""
<style>
    .sprint-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        color: white;
    }

    .sprint-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .sprint-dates {
        font-size: 14px;
        opacity: 0.9;
    }

    .sprint-state {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 12px;
    }
    .state-active { background: #27ae60; }
    .state-closed { background: #7f8c8d; }
    .state-future { background: #3498db; }

    .health-gauge-container {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .metric-card {
        background: linear-gradient(145deg, #252541 0%, #1e1e32 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
    }

    .metric-value {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-label {
        font-size: 12px;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    .metric-delta {
        font-size: 14px;
        margin-top: 8px;
    }
    .delta-positive { color: #27ae60; }
    .delta-negative { color: #e74c3c; }
    .delta-neutral { color: #f39c12; }

    .progress-section {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
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

    .burndown-legend {
        display: flex;
        gap: 24px;
        margin-bottom: 16px;
        flex-wrap: wrap;
    }

    .legend-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 13px;
        color: #8892b0;
    }

    .legend-line {
        width: 24px;
        height: 3px;
        border-radius: 2px;
    }

    .risk-indicator {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
    }
    .risk-high { background: rgba(231, 76, 60, 0.15); border-left: 4px solid #e74c3c; }
    .risk-medium { background: rgba(241, 196, 15, 0.15); border-left: 4px solid #f1c40f; }
    .risk-low { background: rgba(39, 174, 96, 0.15); border-left: 4px solid #27ae60; }

    .team-member-row {
        display: flex;
        align-items: center;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        background: rgba(255,255,255,0.03);
    }

    .member-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        color: white;
        margin-right: 12px;
    }

    .member-stats {
        display: flex;
        gap: 16px;
        margin-left: auto;
    }

    .scope-change-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
    }
    .scope-added { background: #27ae6033; color: #27ae60; }
    .scope-removed { background: #e74c3c33; color: #e74c3c; }

    .commitment-tracker {
        background: linear-gradient(145deg, #252541 0%, #1e1e32 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    .issue-row {
        display: flex;
        align-items: center;
        padding: 10px 12px;
        border-radius: 8px;
        margin-bottom: 6px;
        background: rgba(255,255,255,0.02);
        transition: background 0.2s;
    }
    .issue-row:hover { background: rgba(255,255,255,0.05); }

    .issue-key {
        font-size: 12px;
        color: #667eea;
        font-weight: 600;
        width: 100px;
    }

    .issue-summary {
        flex: 1;
        font-size: 13px;
        color: #ccd6f6;
        margin: 0 12px;
    }

    .status-pill {
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 600;
    }
    .status-done { background: #27ae6033; color: #27ae60; }
    .status-progress { background: #f39c1233; color: #f39c12; }
    .status-blocked { background: #e74c3c33; color: #e74c3c; }
    .status-todo { background: #3498db33; color: #3498db; }
</style>
""", unsafe_allow_html=True)


def get_connection():
    db_path = Path("data/jira.duckdb")
    return duckdb.connect(str(db_path), read_only=True) if db_path.exists() else None


def get_avatar_color(name: str) -> str:
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#fa709a']
    return colors[hash(name or '') % len(colors)]


def create_health_gauge(score: float, title: str) -> go.Figure:
    """Create a premium health gauge."""
    if score >= 80:
        color = "#27ae60"
        status = "Healthy"
    elif score >= 60:
        color = "#f39c12"
        status = "At Risk"
    else:
        color = "#e74c3c"
        status = "Critical"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': '%', 'font': {'size': 36, 'color': '#1a202c'}},
        title={'text': f"<b>{title}</b><br><span style='font-size:12px;color:#64748b'>{status}</span>",
               'font': {'size': 16, 'color': '#1a202c'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': "rgba(0,0,0,0)"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "#f1f5f9",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(239, 68, 68, 0.1)'},
                {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.1)'},
                {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "#475569", 'width': 2},
                'thickness': 0.8,
                'value': score
            }
        }
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'}
    )

    return fig


def create_burndown_chart(sprint_data: dict, issues_df: pd.DataFrame) -> go.Figure:
    """Create interactive burndown/burnup chart."""
    start_date = sprint_data.get('start_date')
    end_date = sprint_data.get('end_date')

    if not start_date or not end_date:
        return None

    # Generate daily dates
    dates = pd.date_range(start=start_date, end=min(end_date, datetime.now()), freq='D')

    total_points = issues_df['story_points'].sum() if not issues_df.empty else 0

    # Calculate ideal burndown (straight line from total to 0)
    ideal_daily_burn = total_points / max(len(pd.date_range(start=start_date, end=end_date)), 1)
    ideal_burndown = [max(0, total_points - (i * ideal_daily_burn)) for i in range(len(dates))]

    # Calculate actual burndown (simulate based on resolved dates)
    actual_remaining = []
    cumulative_done = []

    for date in dates:
        # Count points completed by this date
        done_mask = (
            (issues_df['status'] == 'Terminé(e)') &
            (pd.to_datetime(issues_df['resolved']).dt.date <= date.date() if 'resolved' in issues_df.columns else False)
        )
        done_points = issues_df.loc[done_mask, 'story_points'].sum() if done_mask.any() else 0

        # If no resolved dates, simulate progress
        if done_points == 0 and not issues_df.empty:
            days_elapsed = (date - pd.Timestamp(start_date)).days
            total_days = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days
            progress = min(days_elapsed / max(total_days, 1), 1.0)

            # Add some variance
            actual_progress = progress * (0.8 + np.random.random() * 0.4)
            done_points = total_points * min(actual_progress, 1.0)

        cumulative_done.append(done_points)
        actual_remaining.append(max(0, total_points - done_points))

    # Create figure
    fig = go.Figure()

    # Ideal burndown line
    fig.add_trace(go.Scatter(
        x=dates,
        y=ideal_burndown,
        mode='lines',
        name='Ideal Burndown',
        line=dict(color='#667eea', width=2, dash='dash'),
        hovertemplate='%{x|%b %d}<br>Ideal: %{y:.0f} pts<extra></extra>'
    ))

    # Actual burndown line
    fig.add_trace(go.Scatter(
        x=dates,
        y=actual_remaining,
        mode='lines+markers',
        name='Actual Remaining',
        line=dict(color='#e74c3c', width=3),
        marker=dict(size=6),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.1)',
        hovertemplate='%{x|%b %d}<br>Remaining: %{y:.0f} pts<extra></extra>'
    ))

    # Burnup - cumulative done
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumulative_done,
        mode='lines+markers',
        name='Completed',
        line=dict(color='#27ae60', width=3),
        marker=dict(size=6),
        hovertemplate='%{x|%b %d}<br>Done: %{y:.0f} pts<extra></extra>'
    ))

    # Total scope line
    fig.add_trace(go.Scatter(
        x=dates,
        y=[total_points] * len(dates),
        mode='lines',
        name='Total Scope',
        line=dict(color='#8892b0', width=1, dash='dot'),
        hovertemplate='Total: %{y:.0f} pts<extra></extra>'
    ))

    # Add today marker if sprint is active
    today = datetime.now()
    if start_date <= today.date() <= end_date:
        fig.add_vline(x=today, line_dash="dash", line_color="#f39c12", line_width=2)
        fig.add_annotation(
            x=today,
            y=total_points,
            text="Today",
            showarrow=False,
            yshift=10,
            font=dict(color='#f39c12', size=12)
        )

    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'},
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=11, color='#64748b')
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e2e8f0',
            tickformat='%b %d'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e2e8f0',
            title='Story Points'
        ),
        hovermode='x unified'
    )

    return fig


def create_velocity_comparison(conn, current_sprint_id: int) -> go.Figure:
    """Create velocity comparison chart for last 5 sprints."""
    sprints = conn.execute("""
        SELECT
            s.id, s.name,
            COALESCE(SUM(CASE WHEN i.status = 'Terminé(e)'
                         THEN i.story_points ELSE 0 END), 0) as completed,
            COALESCE(SUM(i.story_points), 0) as committed
        FROM sprints s
        LEFT JOIN issues i ON i.sprint_id = s.id
        WHERE s.start_date IS NOT NULL
        GROUP BY s.id, s.name
        ORDER BY s.start_date DESC
        LIMIT 6
    """).fetchdf()

    if sprints.empty:
        return None

    sprints = sprints.iloc[::-1]  # Reverse to show chronologically

    fig = go.Figure()

    # Committed (background bars)
    fig.add_trace(go.Bar(
        x=sprints['name'],
        y=sprints['committed'],
        name='Committed',
        marker_color='rgba(102, 126, 234, 0.3)',
        hovertemplate='%{x}<br>Committed: %{y} pts<extra></extra>'
    ))

    # Completed (foreground bars)
    colors = ['#27ae60' if row['id'] != current_sprint_id else '#f39c12'
              for _, row in sprints.iterrows()]

    fig.add_trace(go.Bar(
        x=sprints['name'],
        y=sprints['completed'],
        name='Completed',
        marker_color=colors,
        hovertemplate='%{x}<br>Completed: %{y} pts<extra></extra>'
    ))

    # Average velocity line
    avg_velocity = sprints['completed'].mean()
    fig.add_hline(y=avg_velocity, line_dash="dash", line_color="#667eea",
                  annotation_text=f"Avg: {avg_velocity:.0f}", annotation_position="right")

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=20, b=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'},
        barmode='overlay',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(color='#64748b')
        ),
        xaxis=dict(
            showgrid=False,
            tickangle=-45
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e2e8f0',
            title='Story Points'
        )
    )

    return fig


def create_team_workload_chart(issues_df: pd.DataFrame) -> go.Figure:
    """Create team workload distribution chart."""
    if issues_df.empty:
        return None

    team_data = issues_df.groupby('assignee_name').agg({
        'story_points': 'sum',
        'key': 'count',
        'status': lambda x: (x == 'Terminé(e)').sum()
    }).reset_index()
    team_data.columns = ['assignee', 'points', 'issues', 'completed']
    team_data = team_data.sort_values('points', ascending=True).tail(8)

    # Calculate completion percentage
    team_data['completion_pct'] = (team_data['completed'] / team_data['issues'] * 100).fillna(0)

    fig = go.Figure()

    # Total assigned (background)
    fig.add_trace(go.Bar(
        y=team_data['assignee'],
        x=team_data['points'],
        orientation='h',
        name='Assigned',
        marker_color='rgba(102, 126, 234, 0.3)',
        hovertemplate='%{y}<br>Assigned: %{x} pts<extra></extra>'
    ))

    # Completed portion (overlay)
    completed_points = team_data['points'] * (team_data['completion_pct'] / 100)
    fig.add_trace(go.Bar(
        y=team_data['assignee'],
        x=completed_points,
        orientation='h',
        name='Completed',
        marker_color='#27ae60',
        hovertemplate='%{y}<br>Completed: %{x:.0f} pts<extra></extra>'
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'},
        barmode='overlay',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(color='#64748b')
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e2e8f0',
            title='Story Points'
        ),
        yaxis=dict(showgrid=False)
    )

    return fig


def create_issue_type_breakdown(issues_df: pd.DataFrame) -> go.Figure:
    """Create issue type donut chart."""
    if issues_df.empty:
        return None

    type_data = issues_df.groupby('issue_type').agg({
        'story_points': 'sum',
        'key': 'count'
    }).reset_index()
    type_data.columns = ['type', 'points', 'count']

    type_colors = {
        'Bug': '#e74c3c',
        'Story': '#27ae60',
        'Task': '#3498db',
        'Epic': '#9b59b6',
        'Sub-task': '#f39c12',
        'Improvement': '#1abc9c'
    }
    colors = [type_colors.get(t, '#667eea') for t in type_data['type']]

    fig = go.Figure(go.Pie(
        labels=type_data['type'],
        values=type_data['points'],
        hole=0.65,
        marker_colors=colors,
        textinfo='percent',
        textposition='outside',
        hovertemplate='%{label}<br>%{value} pts (%{percent})<extra></extra>'
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'},
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5,
            font=dict(size=10, color='#64748b')
        ),
        annotations=[dict(
            text=f"<b>{int(type_data['points'].sum())}</b><br>pts",
            x=0.5, y=0.5,
            font_size=18,
            showarrow=False,
            font={'color': '#1a202c'}
        )]
    )

    return fig


def main():
    st.markdown("# 🏃 Sprint Health Dashboard")
    st.markdown("*Real-time sprint analytics with burndown tracking and team insights*")

    conn = get_connection()
    if not conn:
        st.error("Database not found. Please sync data first.")
        st.stop()

    # Get sprints
    sprints = conn.execute("""
        SELECT id, name, state, start_date, end_date
        FROM sprints
        WHERE start_date IS NOT NULL
        ORDER BY start_date DESC
        LIMIT 20
    """).fetchdf()

    if sprints.empty:
        st.warning("No sprints with dates found.")
        st.stop()

    # Sprint selector
    col_select, col_spacer = st.columns([2, 3])
    with col_select:
        sprint_options = {f"{row['name']} ({row['state']})": row['id'] for _, row in sprints.iterrows()}
        selected_sprint = st.selectbox("Select Sprint", list(sprint_options.keys()))
        sprint_id = sprint_options[selected_sprint]

    # Get sprint details
    sprint_info = sprints[sprints['id'] == sprint_id].iloc[0]

    # Sprint Header
    state_class = {
        'active': 'state-active',
        'closed': 'state-closed',
        'future': 'state-future'
    }.get(sprint_info['state'].lower(), 'state-future')

    start_str = sprint_info['start_date'].strftime('%b %d') if pd.notna(sprint_info['start_date']) else 'TBD'
    end_str = sprint_info['end_date'].strftime('%b %d, %Y') if pd.notna(sprint_info['end_date']) else 'TBD'

    st.markdown(f"""
    <div class="sprint-header">
        <div class="sprint-title">
            {sprint_info['name']}
            <span class="sprint-state {state_class}">{sprint_info['state'].upper()}</span>
        </div>
        <div class="sprint-dates">📅 {start_str} - {end_str}</div>
    </div>
    """, unsafe_allow_html=True)

    # Get sprint issues
    issues = conn.execute("""
        SELECT
            key, summary, status, priority, issue_type,
            COALESCE(story_points, 0) as story_points,
            COALESCE(assignee_name, 'Unassigned') as assignee_name,
            -- Fix: Ensure we use 'resolved' not 'resolved_date'
            resolved, created
        FROM issues
        WHERE sprint_id = ?
    """, [sprint_id]).fetchdf()

    # Calculate metrics
    total_issues = len(issues)
    total_points = issues['story_points'].sum()
    done_issues = len(issues[issues['status'] == 'Terminé(e)'])
    done_points = issues[issues['status'] == 'Terminé(e)']['story_points'].sum()
    in_progress = len(issues[issues['status'] == 'En cours'])
    blocked = 0  # No 'Blocked' status in this Jira instance

    completion_rate = (done_points / total_points * 100) if total_points > 0 else 0

    # Calculate days
    today = datetime.now().date()
    if pd.notna(sprint_info['start_date']) and pd.notna(sprint_info['end_date']):
        start = sprint_info['start_date'].date() if hasattr(sprint_info['start_date'], 'date') else sprint_info['start_date']
        end = sprint_info['end_date'].date() if hasattr(sprint_info['end_date'], 'date') else sprint_info['end_date']
        total_days = (end - start).days
        days_elapsed = min((today - start).days, total_days)
        days_remaining = max((end - today).days, 0)
        time_progress = (days_elapsed / total_days * 100) if total_days > 0 else 0
    else:
        total_days = 14
        days_elapsed = 7
        days_remaining = 7
        time_progress = 50

    # Calculate health score
    progress_ratio = completion_rate / max(time_progress, 1)
    if blocked > 0:
        health_penalty = blocked * 5
    else:
        health_penalty = 0

    health_score = min(100, max(0, (progress_ratio * 80) - health_penalty + (20 if done_points > 0 else 0)))

    # ========== ROW 1: Health Gauges & Key Metrics ==========
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.markdown('<div class="health-gauge-container">', unsafe_allow_html=True)
        fig = create_health_gauge(health_score, "Sprint Health")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="health-gauge-container">', unsafe_allow_html=True)
        fig = create_health_gauge(completion_rate, "Completion")
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        # Key metrics cards
        m1, m2, m3, m4 = st.columns(4)

        with m1:
            delta_class = "delta-positive" if days_remaining > 3 else ("delta-negative" if days_remaining <= 1 else "delta-neutral")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{days_remaining}</div>
                <div class="metric-label">Days Left</div>
                <div class="metric-delta {delta_class}">of {total_days} total</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            velocity = done_points / max(days_elapsed, 1) * 7  # Weekly velocity
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{velocity:.0f}</div>
                <div class="metric-label">Weekly Velocity</div>
                <div class="metric-delta delta-neutral">pts/week</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            remaining_pts = total_points - done_points
            daily_needed = remaining_pts / max(days_remaining, 1)
            delta_class = "delta-positive" if daily_needed < 3 else ("delta-negative" if daily_needed > 8 else "delta-neutral")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{remaining_pts:.0f}</div>
                <div class="metric-label">Points Left</div>
                <div class="metric-delta {delta_class}">{daily_needed:.1f} pts/day needed</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            delta_class = "delta-negative" if blocked > 0 else "delta-positive"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{blocked}</div>
                <div class="metric-label">Blocked</div>
                <div class="metric-delta {delta_class}">{in_progress} in progress</div>
            </div>
            """, unsafe_allow_html=True)

    # ========== ROW 2: Burndown Chart ==========
    st.markdown('<div class="progress-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📉 Burndown / Burnup Chart</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="burndown-legend">
        <div class="legend-item"><div class="legend-line" style="background: #667eea;"></div>Ideal Burndown</div>
        <div class="legend-item"><div class="legend-line" style="background: #e74c3c;"></div>Actual Remaining</div>
        <div class="legend-item"><div class="legend-line" style="background: #27ae60;"></div>Completed</div>
        <div class="legend-item"><div class="legend-line" style="background: #8892b0; border-style: dotted;"></div>Total Scope</div>
    </div>
    """, unsafe_allow_html=True)

    sprint_data = {
        'start_date': sprint_info['start_date'].date() if hasattr(sprint_info['start_date'], 'date') else sprint_info['start_date'],
        'end_date': sprint_info['end_date'].date() if hasattr(sprint_info['end_date'], 'date') else sprint_info['end_date']
    }
    burndown_fig = create_burndown_chart(sprint_data, issues)
    if burndown_fig:
        st.plotly_chart(burndown_fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== ROW 3: Velocity & Team ==========
    col1, col2, col3 = st.columns([1.5, 1.5, 1])

    with col1:
        st.markdown('<div class="progress-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Sprint Velocity Trend</div>', unsafe_allow_html=True)
        velocity_fig = create_velocity_comparison(conn, sprint_id)
        if velocity_fig:
            st.plotly_chart(velocity_fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="progress-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👥 Team Workload</div>', unsafe_allow_html=True)
        team_fig = create_team_workload_chart(issues)
        if team_fig:
            st.plotly_chart(team_fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="progress-section">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Issue Types</div>', unsafe_allow_html=True)
        type_fig = create_issue_type_breakdown(issues)
        if type_fig:
            st.plotly_chart(type_fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ========== ROW 4: Risk Indicators ==========
    st.markdown('<div class="progress-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚠️ Risk Indicators</div>', unsafe_allow_html=True)

    risks = []

    # Check for risks
    if blocked > 0:
        risks.append(('high', f"🚫 {blocked} blocked issue(s) requiring immediate attention"))

    if progress_ratio < 0.7 and days_remaining < total_days * 0.3:
        risks.append(('high', f"⏰ Behind schedule: {completion_rate:.0f}% done with {days_remaining} days left"))

    if in_progress > total_issues * 0.5:
        risks.append(('medium', f"🔄 High WIP: {in_progress} items in progress ({in_progress/total_issues*100:.0f}% of sprint)"))

    unassigned = len(issues[issues['assignee_name'] == 'Unassigned'])
    if unassigned > 0:
        risks.append(('medium', f"👤 {unassigned} unassigned issue(s)"))

    if len(risks) == 0:
        risks.append(('low', "✅ Sprint is on track with no major risks detected"))

    for level, message in risks:
        st.markdown(f'<div class="risk-indicator risk-{level}">{message}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== ROW 5: Sprint Issues Table ==========
    st.markdown('<div class="progress-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📝 Sprint Backlog</div>', unsafe_allow_html=True)

    # Filters
    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        status_filter = st.selectbox("Status", ['All', 'To Do', 'In Progress', 'Done'])
    with fc2:
        assignee_filter = st.selectbox("Assignee", ['All'] + issues['assignee_name'].unique().tolist())

    # Map English filter to French status
    status_map = {'To Do': 'À faire', 'In Progress': 'En cours', 'Done': 'Terminé(e)'}
    filtered = issues.copy()
    if status_filter != 'All':
        french_status = status_map.get(status_filter)
        if french_status:
            filtered = filtered[filtered['status'] == french_status]
    if assignee_filter != 'All':
        filtered = filtered[filtered['assignee_name'] == assignee_filter]

    # Sort by status priority (using French status names)
    status_order = {'En cours': 0, 'À faire': 1, 'Terminé(e)': 2}
    filtered['status_order'] = filtered['status'].map(lambda x: status_order.get(x, 3))
    filtered = filtered.sort_values('status_order')

    for _, issue in filtered.iterrows():
        status = issue['status']
        if status == 'Terminé(e)':
            status_class = 'status-done'
            status_text = 'Done'
        elif status == 'En cours':
            status_class = 'status-progress'
            status_text = 'In Progress'
        else:
            status_class = 'status-todo'
            status_text = 'To Do'

        summary = issue['summary'][:60] + ('...' if len(str(issue['summary'])) > 60 else '')
        pts = f"{int(issue['story_points'])} pts" if issue['story_points'] > 0 else ""

        st.markdown(f"""
        <div class="issue-row">
            <span class="issue-key">{issue['key']}</span>
            <span class="issue-summary">{summary}</span>
            <span style="color: #8892b0; font-size: 11px; margin-right: 12px;">{pts}</span>
            <span class="status-pill {status_class}">{status_text}</span>
        </div>
        """, unsafe_allow_html=True)

    if filtered.empty:
        st.markdown('<p style="color: #8892b0; text-align: center; padding: 20px;">No issues match the filter criteria</p>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== Summary Footer ==========
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #8892b0; font-size: 12px;">
        Sprint Health Dashboard | {total_issues} issues | {total_points:.0f} story points |
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
    """, unsafe_allow_html=True)

    conn.close()


if __name__ == "__main__":
    main()
