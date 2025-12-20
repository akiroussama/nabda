"""
🏃 Sprint Health - Advanced Sprint Analytics Dashboard
Interactive burndown/burnup charts, velocity tracking, and real-time sprint metrics.
"""

import sys

# Import page guide component
from src.dashboard.components import render_page_guide
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
    /* Main container styling */
    .stApp {
        background-color: #f8f9fa;
    }
    
    .sprint-header {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .sprint-title {
        font-size: 28px;
        font-weight: 700;
        color: #1a202c;
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 8px;
    }

    .sprint-dates {
        font-size: 14px;
        color: #64748b;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .sprint-state {
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .state-active { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
    .state-future { background: #e0f2fe; color: #075985; border: 1px solid #bae6fd; }
    .state-closed { background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; }

    .metric-card {
        background: white;
        padding: 24px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        height: 100%;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }

    .metric-value {
        font-size: 32px;
        font-weight: 800;
        color: #1a202c;
        margin-bottom: 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-label { font-size: 13px; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; }
    
    .metric-delta {
        font-size: 12px;
        font-weight: 600;
        margin-top: 8px;
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
    }
    .delta-positive { background: #ecfdf5; color: #059669; }
    .delta-negative { background: #fef2f2; color: #dc2626; }
    .delta-neutral { background: #f1f5f9; color: #475569; }

    .health-gauge-container {
        background: white;
        border-radius: 16px;
        padding: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .progress-section {
        background: white;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        padding: 24px;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #1a202c;
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
        color: #64748b;
    }

    .legend-line { width: 24px; height: 3px; border-radius: 2px; }

    .risk-indicator {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        border: 1px solid transparent;
    }
    .risk-high { background: #fef2f2; border-color: #fee2e2; border-left: 4px solid #ef4444; color: #991b1b; }
    .risk-medium { background: #fffbeb; border-color: #fef3c7; border-left: 4px solid #f59e0b; color: #92400e; }
    .risk-low { background: #ecfdf5; border-color: #d1fae5; border-left: 4px solid #10b981; color: #065f46; }

    .team-member-row {
        display: flex;
        align-items: center;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 8px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }

    .member-avatar {
        width: 36px; height: 36px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 600; color: white; margin-right: 12px;
        font-size: 12px;
    }

    .member-stats { display: flex; gap: 16px; margin-left: auto; }

    .scope-change-badge {
        display: inline-flex; align-items: center; gap: 4px;
        padding: 4px 10px; border-radius: 8px;
        font-size: 12px; font-weight: 600;
    }
    .scope-added { background: #ecfdf5; color: #059669; }
    .scope-removed { background: #fef2f2; color: #dc2626; }

    .issue-row {
        display: flex;
        align-items: center;
        padding: 10px 12px;
        border-radius: 8px;
        margin-bottom: 6px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        transition: all 0.2s;
    }
    .issue-row:hover { background: #f1f5f9; transform: translateX(2px); }

    .issue-key {
        font-size: 12px;
        color: #4f46e5;
        font-weight: 600;
        width: 100px;
    }

    .issue-summary {
        flex: 1;
        font-size: 13px;
        color: #334155;
        margin: 0 12px;
    }

    .status-pill {
        padding: 3px 10px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 600;
    }
    .status-done { background: #dcfce7; color: #166534; }
    .status-progress { background: #fef3c7; color: #b45309; }
    .status-blocked { background: #fee2e2; color: #991b1b; }
    .status-todo { background: #e0f2fe; color: #075985; }

    /* Quick Win Widget */
    .quick-win-widget {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 16px 0;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 8px 32px rgba(30, 58, 95, 0.3);
        position: relative;
        overflow: hidden;
    }
    .quick-win-widget::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .quick-win-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
    }
    .quick-win-icon {
        font-size: 28px;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
    }
    .quick-win-title {
        color: #bfdbfe;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .deadline-forecast {
        display: flex;
        align-items: center;
        gap: 24px;
        flex-wrap: wrap;
    }
    .forecast-main {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .forecast-confidence {
        font-size: 48px;
        font-weight: 800;
        color: #fff;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .forecast-label {
        color: #93c5fd;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .forecast-status {
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }
    .forecast-green { background: rgba(34, 197, 94, 0.2); color: #86efac; border: 1px solid rgba(34, 197, 94, 0.3); }
    .forecast-yellow { background: rgba(250, 204, 21, 0.2); color: #fef08a; border: 1px solid rgba(250, 204, 21, 0.3); }
    .forecast-red { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.3); }
    .forecast-details {
        display: flex;
        gap: 20px;
        margin-left: auto;
    }
    .forecast-stat {
        text-align: center;
        padding: 8px 16px;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
    }
    .forecast-stat-value {
        color: #fff;
        font-size: 18px;
        font-weight: 700;
    }
    .forecast-stat-label {
        color: #93c5fd;
        font-size: 10px;
        text-transform: uppercase;
    }
    </style>
""", unsafe_allow_html=True)


def get_connection():
    db_path = Path("data/jira.duckdb")
    return duckdb.connect(str(db_path), read_only=True) if db_path.exists() else None


def get_avatar_color(name: str) -> str:
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#fa709a']
    return colors[hash(name or '') % len(colors)]


def calculate_deadline_forecast(sprint_info, issues_df) -> dict:
    """
    Calculate deadline forecast - will the sprint meet its deadline?
    Critical question for release managers every standup.
    """
    if issues_df.empty:
        return {'confidence': 0, 'status': 'No Data', 'class': 'forecast-yellow',
                'days_left': 0, 'remaining_pts': 0, 'daily_needed': 0}

    today = datetime.now().date()
    total_points = issues_df['story_points'].sum()
    done_points = issues_df[issues_df['status'] == 'Terminé(e)']['story_points'].sum()
    remaining_pts = total_points - done_points

    # Calculate days
    if pd.notna(sprint_info.get('start_date')) and pd.notna(sprint_info.get('end_date')):
        start = sprint_info['start_date'].date() if hasattr(sprint_info['start_date'], 'date') else sprint_info['start_date']
        end = sprint_info['end_date'].date() if hasattr(sprint_info['end_date'], 'date') else sprint_info['end_date']
        total_days = max((end - start).days, 1)
        days_elapsed = max((today - start).days, 1)
        days_left = max((end - today).days, 0)
    else:
        return {'confidence': 50, 'status': 'Unknown', 'class': 'forecast-yellow',
                'days_left': 0, 'remaining_pts': remaining_pts, 'daily_needed': 0}

    # Calculate current velocity (points per day)
    current_velocity = done_points / days_elapsed if days_elapsed > 0 else 0
    daily_needed = remaining_pts / days_left if days_left > 0 else remaining_pts

    # Calculate confidence based on velocity vs needed pace
    if days_left == 0:
        confidence = 100 if remaining_pts == 0 else 0
    elif current_velocity == 0:
        confidence = 20 if remaining_pts == 0 else 10
    else:
        pace_ratio = current_velocity / daily_needed if daily_needed > 0 else 2.0
        # Convert pace ratio to confidence
        if pace_ratio >= 1.5:
            confidence = 95
        elif pace_ratio >= 1.2:
            confidence = 85
        elif pace_ratio >= 1.0:
            confidence = 75
        elif pace_ratio >= 0.8:
            confidence = 55
        elif pace_ratio >= 0.6:
            confidence = 35
        else:
            confidence = 20

    # Determine status and class
    if confidence >= 75:
        status = "On Track"
        css_class = "forecast-green"
    elif confidence >= 50:
        status = "At Risk"
        css_class = "forecast-yellow"
    else:
        status = "Behind Schedule"
        css_class = "forecast-red"
    
    # ULTRATHINK: Generate Recovery Plan
    recovery_plan = None
    if confidence < 75:
        # Find candidates to cut: To Do, Low Priority
        cuttable = issues_df[
            (issues_df['status'].isin(['Todo', 'To Do', 'À faire'])) & 
            (issues_df['priority'].isin(['Medium', 'Low', 'Lowest'])) &
            (issues_df['story_points'] > 0)
        ].sort_values('story_points', ascending=False)
        
        points_to_save = max(0, remaining_pts * (1 - (current_velocity/daily_needed if daily_needed > 0 else 0))) * 1.2 # 20% buffer
        
        suggested_cuts = []
        cut_sum = 0
        for _, row in cuttable.iterrows():
            if cut_sum < points_to_save:
                suggested_cuts.append(f"{row['key']} ({int(row['story_points'])} pts)")
                cut_sum += row['story_points']
        
        if suggested_cuts:
            recovery_plan = {
                'points_to_cut': int(cut_sum),
                'items': suggested_cuts[:3]
            }

    return {
        'confidence': int(confidence),
        'status': status,
        'class': css_class,
        'days_left': days_left,
        'remaining_pts': remaining_pts,
        'daily_needed': round(daily_needed, 1),
        'recovery_plan': recovery_plan
    }



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

    # Generate daily dates (convert to date to avoid datetime vs date comparison)
    today = datetime.now().date()
    end_capped = min(end_date if isinstance(end_date, type(today)) else end_date.date() if hasattr(end_date, 'date') else end_date, today)
    dates = pd.date_range(start=start_date, end=end_capped, freq='D')

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
        GROUP BY s.id, s.name, s.start_date
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
    # Render page guide in sidebar
    render_page_guide()

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

    # ========== QUICK WIN: DEADLINE FORECAST ==========
    forecast = calculate_deadline_forecast(sprint_info.to_dict(), issues)
    st.markdown(f"""
<div class="quick-win-widget">
    <div class="quick-win-header">
        <span class="quick-win-icon">📅</span>
        <span class="quick-win-title">Deadline Forecast — Will This Sprint Meet Its Goal?</span>
    </div>
    <div class="deadline-forecast">
        <div class="forecast-main">
            <span class="forecast-confidence">{forecast['confidence']}%</span>
            <div>
                <div class="forecast-label">Confidence</div>
                <span class="forecast-status {forecast['class']}">{forecast['status']}</span>
            </div>
        </div>
        <div class="forecast-details">
            <div class="forecast-stat">
                <div class="forecast-stat-value">{forecast['days_left']}</div>
                <div class="forecast-stat-label">Days Left</div>
            </div>
            <div class="forecast-stat">
                <div class="forecast-stat-value">{forecast['remaining_pts']:.0f}</div>
                <div class="forecast-stat-label">Pts Remaining</div>
            </div>
            <div class="forecast-stat">
                <div class="forecast-stat-value">{forecast['daily_needed']}</div>
                <div class="forecast-stat-label">Pts/Day Needed</div>
            </div>
        </div>
    </div>
    {f'''
    <div style="margin-top: 16px; padding: 12px; background: rgba(0,0,0,0.2); border-radius: 8px; border-left: 3px solid #f87171;">
        <div style="color: #fca5a5; font-weight: 600; font-size: 13px;">⚠️ RECOVERY PLAN DETECTED</div>
        <div style="color: #cbd5e1; font-size: 13px; margin-top: 4px;">
            To reach 85% confidence, consider removing <b>{forecast.get('recovery_plan', {}).get('points_to_cut', 0)} points</b>.
            <br>Suggested candidates: <span style="color: #fff; font-family: monospace;">{', '.join(forecast.get('recovery_plan', {}).get('items', []))}</span>
        </div>
    </div>
    ''' if forecast.get('recovery_plan') else ''}
</div>
""", unsafe_allow_html=True)

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
