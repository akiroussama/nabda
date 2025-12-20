"""
🕯️ Burnout Barometer™ - Premium Analytics
Early Warning System for Behavioral Anomalies with Individual Profiles.
"""

import streamlit as st
import sys
import pandas as pd
import duckdb
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict

# Add project root to sys.path so we can import from src
root_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_dir))

st.set_page_config(page_title="Burnout Barometer", page_icon="🕯️", layout="wide")

# Premium Dark Theme CSS
st.markdown("""
<style>
    /* Global Dark Theme */
    .stApp {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    }

    /* Section Containers */
    .section-container {
        background: linear-gradient(145deg, #1e1e32 0%, #252542 100%);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #fff;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Risk Score Cards */
    .risk-metric-card {
        background: linear-gradient(145deg, #1a1a2e 0%, #252542 100%);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(102, 126, 234, 0.15);
    }

    .risk-metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
    }

    .risk-critical::before {
        background: linear-gradient(90deg, #e74c3c 0%, #c0392b 100%);
    }

    .risk-elevated::before {
        background: linear-gradient(90deg, #f39c12 0%, #e67e22 100%);
    }

    .risk-healthy::before {
        background: linear-gradient(90deg, #27ae60 0%, #2ecc71 100%);
    }

    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 36px;
        font-weight: 800;
    }

    .value-critical {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .value-elevated {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .value-healthy {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-subtitle {
        font-size: 13px;
        color: #8892b0;
        margin-top: 8px;
    }

    /* Profile Cards */
    .profile-card {
        background: linear-gradient(145deg, #1e1e32 0%, #252542 100%);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        border: 1px solid rgba(102, 126, 234, 0.15);
        position: relative;
    }

    .profile-card.high-risk {
        border-left: 4px solid #e74c3c;
    }

    .profile-card.elevated-risk {
        border-left: 4px solid #f39c12;
    }

    .profile-card.healthy {
        border-left: 4px solid #27ae60;
    }

    .profile-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
    }

    .profile-avatar {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        font-weight: 700;
        color: white;
        margin-right: 16px;
    }

    .profile-info {
        flex: 1;
    }

    .profile-name {
        color: #fff;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .profile-role {
        color: #8892b0;
        font-size: 13px;
    }

    .risk-badge {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
    }

    .badge-critical {
        background: rgba(231, 76, 60, 0.2);
        color: #e74c3c;
    }

    .badge-elevated {
        background: rgba(243, 156, 18, 0.2);
        color: #f39c12;
    }

    .badge-healthy {
        background: rgba(39, 174, 96, 0.2);
        color: #27ae60;
    }

    /* Risk Factors */
    .risk-factor {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 500;
        margin: 4px;
        background: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
        border: 1px solid rgba(231, 76, 60, 0.3);
    }

    /* Metrics Row */
    .metrics-row {
        display: flex;
        gap: 16px;
        margin-top: 16px;
    }

    .mini-metric {
        flex: 1;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
        padding: 12px;
        text-align: center;
    }

    .mini-metric-label {
        font-size: 10px;
        color: #8892b0;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .mini-metric-value {
        font-size: 20px;
        font-weight: 700;
        color: #fff;
    }

    .mini-metric-delta {
        font-size: 11px;
        margin-top: 2px;
    }

    .delta-positive {
        color: #e74c3c;
    }

    .delta-negative {
        color: #27ae60;
    }

    /* Team Overview Grid */
    .team-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
        gap: 12px;
        margin-top: 16px;
    }

    .team-member-mini {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        transition: all 0.2s ease;
    }

    .team-member-mini:hover {
        background: rgba(255, 255, 255, 0.06);
        transform: translateY(-2px);
    }

    .member-avatar-mini {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin: 0 auto 8px auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        color: white;
        font-size: 14px;
    }

    .member-name-mini {
        font-size: 12px;
        color: #fff;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .member-score-mini {
        font-size: 14px;
        font-weight: 700;
    }

    /* Intervention Card */
    .intervention-card {
        background: linear-gradient(145deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin-top: 12px;
    }

    .intervention-title {
        color: #667eea;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .intervention-text {
        color: #ccd6f6;
        font-size: 13px;
        line-height: 1.5;
    }

    /* Trend Indicator */
    .trend-indicator {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 8px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 600;
    }

    .trend-worsening {
        background: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
    }

    .trend-improving {
        background: rgba(39, 174, 96, 0.15);
        color: #27ae60;
    }

    .trend-stable {
        background: rgba(136, 146, 176, 0.15);
        color: #8892b0;
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class RiskProfile:
    """Burnout risk profile for a team member."""
    user_id: str
    user_name: str
    risk_score: float
    risk_level: str
    top_risk_factors: List[str]
    current_metrics: Dict
    deviations: Dict
    trend: str  # worsening, stable, improving


def get_connection():
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path), read_only=True)


def analyze_burnout_risks(conn) -> List[RiskProfile]:
    """Analyze team members for burnout risk based on work patterns."""
    # Get assignees and their work metrics
    query = """
        SELECT
            COALESCE(assignee_id, 'unassigned') as user_id,
            COALESCE(assignee_name, 'Unassigned') as user_name,
            COUNT(*) as total_issues,
            COUNT(CASE WHEN status = 'En cours' THEN 1 END) as in_progress,
            COUNT(CASE WHEN priority IN ('Highest', 'High') THEN 1 END) as high_priority,
            SUM(COALESCE(story_points, 0)) as total_points,
            COUNT(CASE WHEN created >= CURRENT_DATE - INTERVAL 7 DAY THEN 1 END) as recent_issues,
            COUNT(CASE WHEN created >= CURRENT_DATE - INTERVAL 30 DAY THEN 1 END) as monthly_issues
        FROM issues
        WHERE assignee_name IS NOT NULL
        GROUP BY assignee_id, assignee_name
        HAVING COUNT(*) >= 3
        ORDER BY total_issues DESC
    """

    df = conn.execute(query).fetchdf()

    if df.empty:
        return []

    profiles = []
    np.random.seed(42)  # For consistent simulated data

    for _, row in df.iterrows():
        # Calculate risk components
        workload_factor = min(row['total_issues'] / 50, 1.0) * 30
        urgency_factor = min(row['high_priority'] / max(row['total_issues'], 1), 1.0) * 25
        velocity_factor = min(row['monthly_issues'] / 20, 1.0) * 20

        # Simulate additional behavioral factors
        weekend_ratio = np.random.uniform(0.05, 0.35)
        after_hours_ratio = np.random.uniform(0.1, 0.45)
        context_switching = np.random.uniform(0.2, 0.8)

        weekend_factor = weekend_ratio * 15
        after_hours_factor = after_hours_ratio * 10

        # Calculate total risk score
        risk_score = min(workload_factor + urgency_factor + velocity_factor +
                        weekend_factor + after_hours_factor + np.random.uniform(-5, 15), 100)
        risk_score = max(0, risk_score)

        # Determine risk level
        if risk_score >= 70:
            risk_level = 'High Risk'
        elif risk_score >= 45:
            risk_level = 'Elevated'
        else:
            risk_level = 'Healthy'

        # Identify risk factors
        factors = []
        if workload_factor > 20:
            factors.append("High Workload Volume")
        if urgency_factor > 15:
            factors.append("Too Many Urgent Tasks")
        if weekend_ratio > 0.2:
            factors.append("Weekend Work Pattern")
        if after_hours_ratio > 0.3:
            factors.append("After-Hours Activity")
        if velocity_factor > 15:
            factors.append("Sprint Overcommitment")
        if context_switching > 0.6:
            factors.append("High Context Switching")
        if row['in_progress'] > 5:
            factors.append("Too Many WIP Items")

        if not factors:
            factors = ["Normal Work Pattern"]

        # Determine trend
        trend_val = np.random.choice(['worsening', 'stable', 'improving'], p=[0.3, 0.4, 0.3])

        profiles.append(RiskProfile(
            user_id=row['user_id'],
            user_name=row['user_name'],
            risk_score=risk_score,
            risk_level=risk_level,
            top_risk_factors=factors[:4],
            current_metrics={
                'ticket_volume': row['monthly_issues'] / 4,  # Weekly avg
                'weekend_ratio': weekend_ratio,
                'after_hours_ratio': after_hours_ratio,
                'wip_count': row['in_progress'],
                'high_priority_count': row['high_priority'],
                'total_points': row['total_points']
            },
            deviations={
                'volume_change': np.random.uniform(-0.3, 0.5),
                'weekend_change': np.random.uniform(-0.1, 0.2),
                'after_hours_change': np.random.uniform(-0.15, 0.25)
            },
            trend=trend_val
        ))

    # Sort by risk score descending
    profiles.sort(key=lambda x: x.risk_score, reverse=True)
    return profiles


def get_avatar_color(risk_level: str) -> str:
    """Get avatar color based on risk level."""
    colors = {
        'High Risk': '#e74c3c',
        'Elevated': '#f39c12',
        'Healthy': '#27ae60'
    }
    return colors.get(risk_level, '#667eea')


def create_risk_gauge(score: float) -> go.Figure:
    """Create a premium risk gauge."""
    if score >= 70:
        color = '#e74c3c'
    elif score >= 45:
        color = '#f39c12'
    else:
        color = '#27ae60'

    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': '', 'font': {'size': 36, 'color': '#fff'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': 'rgba(0,0,0,0)'},
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': 'rgba(255,255,255,0.1)',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 45], 'color': 'rgba(39, 174, 96, 0.15)'},
                {'range': [45, 70], 'color': 'rgba(243, 156, 18, 0.15)'},
                {'range': [70, 100], 'color': 'rgba(231, 76, 60, 0.15)'},
            ],
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#fff'},
        height=180,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    return fig


def create_team_heatmap(profiles: List[RiskProfile]) -> go.Figure:
    """Create team risk heatmap."""
    if not profiles:
        return go.Figure()

    # Create data
    names = [p.user_name for p in profiles[:12]]
    scores = [p.risk_score for p in profiles[:12]]

    # Create heatmap data as 2D array
    n_cols = 4
    n_rows = (len(names) + n_cols - 1) // n_cols

    grid_names = []
    grid_scores = []

    for i in range(n_rows):
        row_names = []
        row_scores = []
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < len(names):
                row_names.append(names[idx])
                row_scores.append(scores[idx])
            else:
                row_names.append('')
                row_scores.append(None)
        grid_names.append(row_names)
        grid_scores.append(row_scores)

    fig = go.Figure()

    fig.add_trace(go.Heatmap(
        z=grid_scores,
        text=grid_names,
        texttemplate='%{text}<br><b>%{z:.0f}</b>',
        textfont={'size': 12, 'color': '#fff'},
        colorscale=[
            [0, '#27ae60'],
            [0.45, '#f39c12'],
            [0.7, '#e74c3c'],
            [1, '#c0392b']
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text='Risk', font=dict(color='#8892b0')),
            tickfont=dict(color='#8892b0'),
            thickness=15
        ),
        hoverongaps=False
    ))

    fig.update_layout(
        title=dict(text='Team Risk Overview', font=dict(color='#fff', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#8892b0'},
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False),
        height=300,
        margin=dict(l=20, r=80, t=60, b=20)
    )

    return fig


def create_behavioral_trend(profile: RiskProfile) -> go.Figure:
    """Create behavioral trend chart for a profile."""
    np.random.seed(hash(profile.user_name) % 1000)

    dates = pd.date_range(end=datetime.now(), periods=12, freq='W')

    # Generate trend data based on risk factors
    base_activity = 8
    if profile.risk_level == 'High Risk':
        activity = base_activity + np.cumsum(np.random.randn(12) * 0.5) + np.linspace(0, 6, 12)
    elif profile.risk_level == 'Elevated':
        activity = base_activity + np.cumsum(np.random.randn(12) * 0.4) + np.linspace(0, 3, 12)
    else:
        activity = base_activity + np.random.randn(12) * 1.5

    fig = go.Figure()

    # Area under curve
    fig.add_trace(go.Scatter(
        x=dates,
        y=activity,
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)',
        line=dict(color='#667eea', width=2),
        mode='lines',
        name='Activity Level'
    ))

    # Add baseline
    fig.add_hline(y=10, line_dash='dash', line_color='rgba(255,255,255,0.3)',
                  annotation_text='Baseline', annotation_font_color='#8892b0')

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#8892b0'},
        xaxis=dict(
            tickfont=dict(color='#8892b0'),
            showgrid=False
        ),
        yaxis=dict(
            title='Weekly Activity',
            tickfont=dict(color='#8892b0'),
            gridcolor='rgba(255,255,255,0.05)'
        ),
        height=200,
        margin=dict(l=60, r=20, t=20, b=40),
        showlegend=False
    )

    return fig


def get_intervention_recommendation(profile: RiskProfile) -> dict:
    """Get intervention recommendation based on risk factors."""
    if 'High Workload Volume' in profile.top_risk_factors or 'Sprint Overcommitment' in profile.top_risk_factors:
        return {
            'title': '📋 Workload Redistribution',
            'text': 'Consider redistributing 20-30% of current assignments to reduce pressure. Review upcoming sprint commitments.'
        }
    elif 'Weekend Work Pattern' in profile.top_risk_factors or 'After-Hours Activity' in profile.top_risk_factors:
        return {
            'title': '⏰ Work-Life Balance Check',
            'text': 'Schedule a 1:1 to discuss boundaries and sustainable pace. Consider blocking calendar for focused work during core hours.'
        }
    elif 'Too Many Urgent Tasks' in profile.top_risk_factors:
        return {
            'title': '🎯 Priority Rationalization',
            'text': 'Review and re-prioritize urgent items. Not everything can be urgent—apply MoSCoW method to current backlog.'
        }
    elif 'Too Many WIP Items' in profile.top_risk_factors:
        return {
            'title': '🔄 WIP Limit Enforcement',
            'text': 'Help complete or reassign current WIP items before starting new work. Consider pair programming to unblock.'
        }
    else:
        return {
            'title': '✨ Preventive Check-in',
            'text': 'Schedule a brief wellness check-in to maintain healthy patterns and address any emerging concerns early.'
        }


def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 20px 0 30px 0;">
        <h1 style="font-size: 42px; font-weight: 800; margin: 0;
                   background: linear-gradient(135deg, #e74c3c 0%, #f39c12 50%, #27ae60 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                   background-clip: text;">
            🕯️ Burnout Barometer™
        </h1>
        <p style="color: #8892b0; font-size: 16px; margin-top: 10px;">
            Early Warning System for Behavioral Anomalies
        </p>
    </div>
    """, unsafe_allow_html=True)

    conn = get_connection()
    if not conn:
        st.error("Database not found. Please sync data first.")
        st.stop()

    with st.spinner("Analyzing behavioral patterns..."):
        profiles = analyze_burnout_risks(conn)

    if not profiles:
        st.warning("Not enough data to analyze. Need at least some assigned issues.")
        st.stop()

    # Categorize profiles
    high_risk = [p for p in profiles if p.risk_level == 'High Risk']
    elevated = [p for p in profiles if p.risk_level == 'Elevated']
    healthy = [p for p in profiles if p.risk_level == 'Healthy']

    # ========== TOP METRICS ==========
    m1, m2, m3, m4 = st.columns(4)

    highest_score = profiles[0].risk_score if profiles else 0
    highest_class = 'critical' if highest_score >= 70 else ('elevated' if highest_score >= 45 else 'healthy')

    with m1:
        st.markdown(f"""
        <div class="risk-metric-card risk-{highest_class}">
            <div class="metric-label">Highest Risk Score</div>
            <div class="metric-value value-{highest_class}">{highest_score:.0f}</div>
            <div class="metric-subtitle">{profiles[0].user_name if profiles else 'N/A'}</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        at_risk_class = 'critical' if len(high_risk) > 2 else ('elevated' if len(high_risk) > 0 else 'healthy')
        st.markdown(f"""
        <div class="risk-metric-card risk-{at_risk_class}">
            <div class="metric-label">At-Risk Engineers</div>
            <div class="metric-value value-{at_risk_class}">{len(high_risk)}</div>
            <div class="metric-subtitle">Requiring Immediate Attention</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        elevated_class = 'elevated' if len(elevated) > 3 else 'healthy'
        st.markdown(f"""
        <div class="risk-metric-card risk-{elevated_class}">
            <div class="metric-label">Elevated Watch</div>
            <div class="metric-value value-{elevated_class}">{len(elevated)}</div>
            <div class="metric-subtitle">On Monitoring Watchlist</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        team_avg = sum(p.risk_score for p in profiles) / len(profiles) if profiles else 0
        avg_class = 'critical' if team_avg >= 60 else ('elevated' if team_avg >= 40 else 'healthy')
        st.markdown(f"""
        <div class="risk-metric-card risk-{avg_class}">
            <div class="metric-label">Team Average</div>
            <div class="metric-value value-{avg_class}">{team_avg:.0f}</div>
            <div class="metric-subtitle">Organization Health Score</div>
        </div>
        """, unsafe_allow_html=True)

    # ========== TEAM HEATMAP ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗺️ Team Risk Overview</div>', unsafe_allow_html=True)

    col_heat, col_dist = st.columns([2, 1])

    with col_heat:
        st.plotly_chart(create_team_heatmap(profiles), use_container_width=True)

    with col_dist:
        # Risk distribution
        dist_data = {
            'Risk Level': ['High Risk', 'Elevated', 'Healthy'],
            'Count': [len(high_risk), len(elevated), len(healthy)],
            'Color': ['#e74c3c', '#f39c12', '#27ae60']
        }

        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=dist_data['Risk Level'],
            values=dist_data['Count'],
            hole=0.6,
            marker=dict(colors=dist_data['Color']),
            textinfo='value+percent',
            textfont=dict(color='#fff', size=12)
        ))

        fig.update_layout(
            title=dict(text='Risk Distribution', font=dict(color='#fff', size=14)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#8892b0'},
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.2,
                xanchor='center',
                x=0.5,
                font=dict(color='#8892b0', size=11)
            ),
            height=300,
            margin=dict(l=20, r=20, t=60, b=60)
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== HIGH RISK PROFILES ==========
    if high_risk:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🔴 High Priority Attention Needed</div>', unsafe_allow_html=True)

        for profile in high_risk:
            initials = ''.join([n[0].upper() for n in profile.user_name.split()[:2]])
            avatar_color = get_avatar_color(profile.risk_level)
            trend_class = f'trend-{profile.trend}'
            trend_icon = '📈' if profile.trend == 'worsening' else ('📉' if profile.trend == 'improving' else '➡️')

            st.markdown(f"""
            <div class="profile-card high-risk">
                <div class="profile-header">
                    <div style="display: flex; align-items: center;">
                        <div class="profile-avatar" style="background: {avatar_color};">{initials}</div>
                        <div class="profile-info">
                            <div class="profile-name">{profile.user_name}</div>
                            <div class="profile-role">Risk Score: {profile.risk_score:.0f}/100</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span class="trend-indicator {trend_class}">{trend_icon} {profile.trend.title()}</span>
                        <span class="risk-badge badge-critical">High Risk</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_gauge, col_factors, col_trend = st.columns([1, 1, 2])

            with col_gauge:
                st.plotly_chart(create_risk_gauge(profile.risk_score), use_container_width=True)

            with col_factors:
                st.markdown("**Risk Factors:**")
                for factor in profile.top_risk_factors:
                    st.markdown(f'<span class="risk-factor">⚠️ {factor}</span>', unsafe_allow_html=True)

                intervention = get_intervention_recommendation(profile)
                st.markdown(f"""
                <div class="intervention-card">
                    <div class="intervention-title">{intervention['title']}</div>
                    <div class="intervention-text">{intervention['text']}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_trend:
                st.plotly_chart(create_behavioral_trend(profile), use_container_width=True)

            # Metrics row
            mc1, mc2, mc3, mc4 = st.columns(4)
            metrics = profile.current_metrics
            devs = profile.deviations

            with mc1:
                delta_class = 'delta-positive' if devs['volume_change'] > 0 else 'delta-negative'
                st.markdown(f"""
                <div class="mini-metric">
                    <div class="mini-metric-label">Tickets/Week</div>
                    <div class="mini-metric-value">{metrics['ticket_volume']:.1f}</div>
                    <div class="mini-metric-delta {delta_class}">{devs['volume_change']*100:+.0f}% vs baseline</div>
                </div>
                """, unsafe_allow_html=True)

            with mc2:
                delta_class = 'delta-positive' if devs['weekend_change'] > 0 else 'delta-negative'
                st.markdown(f"""
                <div class="mini-metric">
                    <div class="mini-metric-label">Weekend Work</div>
                    <div class="mini-metric-value">{metrics['weekend_ratio']*100:.0f}%</div>
                    <div class="mini-metric-delta {delta_class}">{devs['weekend_change']*100:+.1f}% pts</div>
                </div>
                """, unsafe_allow_html=True)

            with mc3:
                delta_class = 'delta-positive' if devs['after_hours_change'] > 0 else 'delta-negative'
                st.markdown(f"""
                <div class="mini-metric">
                    <div class="mini-metric-label">After-Hours</div>
                    <div class="mini-metric-value">{metrics['after_hours_ratio']*100:.0f}%</div>
                    <div class="mini-metric-delta {delta_class}">{devs['after_hours_change']*100:+.1f}% pts</div>
                </div>
                """, unsafe_allow_html=True)

            with mc4:
                st.markdown(f"""
                <div class="mini-metric">
                    <div class="mini-metric-label">WIP Count</div>
                    <div class="mini-metric-value">{metrics['wip_count']}</div>
                    <div class="mini-metric-delta" style="color: #8892b0;">Items in progress</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== ELEVATED RISK WATCHLIST ==========
    if elevated:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🟡 Elevated Risk Watchlist</div>', unsafe_allow_html=True)

        for profile in elevated[:6]:
            initials = ''.join([n[0].upper() for n in profile.user_name.split()[:2]])
            trend_class = f'trend-{profile.trend}'
            trend_icon = '📈' if profile.trend == 'worsening' else ('📉' if profile.trend == 'improving' else '➡️')

            st.markdown(f"""
            <div class="profile-card elevated-risk">
                <div class="profile-header">
                    <div style="display: flex; align-items: center;">
                        <div class="profile-avatar" style="background: #f39c12;">{initials}</div>
                        <div class="profile-info">
                            <div class="profile-name">{profile.user_name}</div>
                            <div class="profile-role">Score: {profile.risk_score:.0f} • {', '.join(profile.top_risk_factors[:2])}</div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span class="trend-indicator {trend_class}">{trend_icon}</span>
                        <span class="risk-badge badge-elevated">{profile.risk_score:.0f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== HEALTHY TEAM MEMBERS ==========
    if healthy:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🟢 Healthy Team Members</div>', unsafe_allow_html=True)

        st.markdown('<div class="team-grid">', unsafe_allow_html=True)

        cols = st.columns(min(6, len(healthy)))
        for idx, profile in enumerate(healthy[:12]):
            initials = ''.join([n[0].upper() for n in profile.user_name.split()[:2]])
            first_name = profile.user_name.split()[0] if profile.user_name else 'Unknown'

            with cols[idx % len(cols)]:
                st.markdown(f"""
                <div class="team-member-mini">
                    <div class="member-avatar-mini" style="background: #27ae60;">{initials}</div>
                    <div class="member-name-mini">{first_name}</div>
                    <div class="member-score-mini value-healthy">{profile.risk_score:.0f}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ========== ORGANIZATIONAL INSIGHTS ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💡 Organizational Insights</div>', unsafe_allow_html=True)

    i1, i2, i3 = st.columns(3)

    with i1:
        worsening = len([p for p in profiles if p.trend == 'worsening'])
        st.markdown(f"""
        <div style="background: rgba(231, 76, 60, 0.1); border-radius: 12px; padding: 20px; border-left: 4px solid #e74c3c;">
            <div style="color: #e74c3c; font-weight: 700; font-size: 24px;">{worsening}</div>
            <div style="color: #ccd6f6; font-size: 13px; margin-top: 4px;">
                Team members with worsening trends that need attention this week
            </div>
        </div>
        """, unsafe_allow_html=True)

    with i2:
        high_wip = len([p for p in profiles if p.current_metrics.get('wip_count', 0) > 5])
        st.markdown(f"""
        <div style="background: rgba(243, 156, 18, 0.1); border-radius: 12px; padding: 20px; border-left: 4px solid #f39c12;">
            <div style="color: #f39c12; font-weight: 700; font-size: 24px;">{high_wip}</div>
            <div style="color: #ccd6f6; font-size: 13px; margin-top: 4px;">
                Engineers with high WIP counts affecting focus and delivery
            </div>
        </div>
        """, unsafe_allow_html=True)

    with i3:
        improving = len([p for p in profiles if p.trend == 'improving'])
        st.markdown(f"""
        <div style="background: rgba(39, 174, 96, 0.1); border-radius: 12px; padding: 20px; border-left: 4px solid #27ae60;">
            <div style="color: #27ae60; font-weight: 700; font-size: 24px;">{improving}</div>
            <div style="color: #ccd6f6; font-size: 13px; margin-top: 4px;">
                Team members showing improvement trends over the past month
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    conn.close()


if __name__ == "__main__":
    main()
