"""
🏆 Executive Cockpit - The Holy Trinity™
Advanced C-Level Dashboard with Real-Time Intelligence.

Features:
- Interactive Gauge Charts
- Trend Sparklines
- Traffic Light System
- AI-Powered Action Cards
- Risk Heat Matrix
"""

import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple

from src.intelligence.classifier import WorkClassifier
from src.features.strategic_alignment import StrategicAlignmentAnalyzer
from src.features.burnout_models import BurnoutAnalyzer
from src.features.delivery_forecast import DeliveryForecaster

st.set_page_config(page_title="Executive Cockpit", page_icon="🏆", layout="wide")

# Premium CSS
st.markdown("""
<style>
    /* Global Light Theme */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Premium Cards */
    .exec-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
    }

    .metric-giant {
        font-size: 56px;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #64748b;
        text-align: center;
        margin-bottom: 8px;
    }

    .metric-sublabel {
        font-size: 14px;
        color: #059669;
        text-align: center;
        margin-top: 8px;
        font-weight: 600;
    }

    /* Traffic Lights */
    .traffic-light {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .light-red { background: #ef4444; color: #ef4444; }
    .light-amber { background: #f59e0b; color: #f59e0b; }
    .light-green { background: #22c55e; color: #22c55e; }

    /* Action Cards */
    .action-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
        border: 1px solid #e2e8f0;
        border-left: 4px solid;
        transition: transform 0.2s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .action-card:hover { transform: translateX(8px); }
    .action-critical { border-left-color: #ef4444; }
    .action-warning { border-left-color: #f59e0b; }
    .action-info { border-left-color: #3b82f6; }

    /* Pulse Animation */
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.7; transform: scale(1.05); }
        100% { opacity: 1; transform: scale(1); }
    }
    .pulse { animation: pulse 2s infinite; }

    /* Score Badge */
    .score-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
    }
    .score-excellent { background: #dcfce7; color: #166534; }
    .score-good { background: #d1fae5; color: #065f46; }
    .score-warning { background: #fef3c7; color: #92400e; }
    .score-danger { background: #fee2e2; color: #991b1b; }

    /* Executive Summary */
    .exec-summary {
        background: white;
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }
    .exec-summary h3 {
        color: #1a202c;
        margin-bottom: 16px;
        font-weight: 700;
    }

    /* Quick Win Widget */
    .quick-win-widget {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    .quick-win-title {
        font-size: 14px;
        opacity: 0.9;
        margin-bottom: 4px;
    }
    .quick-win-value {
        font-size: 32px;
        font-weight: 800;
    }
    .quick-win-action {
        background: rgba(255,255,255,0.2);
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
    }
    .quick-win-status {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .quick-win-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
</style>
""", unsafe_allow_html=True)


def get_connection():
    db_path = Path("data/jira.duckdb")
    return duckdb.connect(str(db_path), read_only=True) if db_path.exists() else None


@st.cache_resource
def get_classifier():
    return WorkClassifier()


def create_gauge_chart(value: float, title: str, max_val: float = 100,
                       thresholds: List[float] = [30, 70]) -> go.Figure:
    """Create a premium gauge chart."""
    if value <= thresholds[0]:
        color = "#ef4444"
    elif value <= thresholds[1]:
        color = "#f59e0b"
    else:
        color = "#22c55e"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 14, 'color': '#64748b'}},
        number={'font': {'size': 40, 'color': color}, 'suffix': '%'},
        gauge={
            'axis': {'range': [0, max_val], 'tickcolor': '#64748b'},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': '#f1f5f9',
            'borderwidth': 0,
            'steps': [
                {'range': [0, thresholds[0]], 'color': 'rgba(239, 68, 68, 0.2)'},
                {'range': [thresholds[0], thresholds[1]], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [thresholds[1], max_val], 'color': 'rgba(34, 197, 94, 0.2)'}
            ],
            'threshold': {
                'line': {'color': '#1a202c', 'width': 2},
                'thickness': 0.8,
                'value': value
            }
        }
    ))
    fig.update_layout(
        height=200,
        margin=dict(t=40, b=20, l=20, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'}
    )
    return fig


def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convert hex color to rgba format."""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def create_sparkline(data: List[float], title: str, color: str = "#667eea") -> go.Figure:
    """Create a mini sparkline chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=data,
        mode='lines+markers',
        line=dict(color=color, width=2),
        marker=dict(size=4),
        fill='tozeroy',
        fillcolor=hex_to_rgba(color, 0.2)
    ))
    fig.update_layout(
        height=60,
        margin=dict(t=5, b=5, l=5, r=5),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig


def create_risk_heatmap(data: Dict[str, Dict[str, float]]) -> go.Figure:
    """Create a risk heat matrix."""
    categories = list(data.keys())
    dimensions = ['Impact', 'Likelihood', 'Velocity']

    z_data = [[data[cat].get(dim, 0) for dim in dimensions] for cat in categories]

    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=dimensions,
        y=categories,
        colorscale=[
            [0, '#2ed573'],
            [0.5, '#ffa502'],
            [1, '#ff4757']
        ],
        showscale=True,
        colorbar=dict(title='Risk', tickvals=[0, 50, 100], ticktext=['Low', 'Med', 'High'])
    ))
    fig.update_layout(
        height=250,
        margin=dict(t=20, b=20, l=100, r=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#64748b')
    )
    return fig


def get_trend_data(conn, days: int = 7) -> Dict[str, List[float]]:
    """Get trend data for the last N days."""
    trends = {'created': [], 'resolved': [], 'velocity': []}

    for i in range(days, 0, -1):
        day = datetime.now() - timedelta(days=i)
        day_str = day.strftime('%Y-%m-%d')

        # Created per day
        created = conn.execute(f"""
            SELECT COUNT(*) FROM issues
            WHERE DATE(created) = '{day_str}'
        """).fetchone()[0]
        trends['created'].append(created)

        # Resolved per day
        resolved = conn.execute(f"""
            SELECT COUNT(*) FROM issues
            WHERE DATE(resolved) = '{day_str}'
        """).fetchone()[0]
        trends['resolved'].append(resolved)

        # Velocity (story points completed)
        points = conn.execute(f"""
            SELECT COALESCE(SUM(story_points), 0) FROM issues
            WHERE DATE(resolved) = '{day_str}'
        """).fetchone()[0]
        trends['velocity'].append(float(points))

    return trends


def calculate_health_score(strategy_score: float, burnout_risk: float, delivery_prob: float) -> Tuple[int, str]:
    """Calculate overall health score."""
    # Normalize inputs (0-100 scale)
    strat = max(0, 100 - strategy_score * 100)  # Lower drift = higher score
    burn = max(0, 100 - burnout_risk * 100)  # Lower risk = higher score
    deliv = delivery_prob * 100

    score = int((strat * 0.3) + (burn * 0.3) + (deliv * 0.4))

    if score >= 80:
        grade = "A"
    elif score >= 65:
        grade = "B"
    elif score >= 50:
        grade = "C"
    else:
        grade = "D"

    return score, grade


def calculate_release_readiness(conn) -> Dict:
    """Calculate release readiness score for quick win widget."""
    # Get active sprint data
    sprint = conn.execute("""
        SELECT id, name FROM sprints
        WHERE state = 'active' ORDER BY start_date DESC LIMIT 1
    """).fetchone()

    if not sprint:
        sprint = conn.execute("""
            SELECT id, name FROM sprints ORDER BY start_date DESC LIMIT 1
        """).fetchone()

    if not sprint:
        return {'score': 0, 'status': 'No Sprint', 'blockers': 0, 'action': 'Create a sprint'}

    sprint_id, sprint_name = sprint

    # Get sprint metrics
    metrics = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status IN ('Done', 'Terminé(e)', 'Closed') THEN 1 ELSE 0 END) as done,
            SUM(CASE WHEN status IN ('Blocked', 'Bloqué') OR priority = 'Highest' THEN 1 ELSE 0 END) as blockers,
            SUM(CASE WHEN status IN ('In Progress', 'En cours') THEN 1 ELSE 0 END) as wip
        FROM issues WHERE sprint_id = ?
    """, [sprint_id]).fetchone()

    total, done, blockers, wip = metrics or (0, 0, 0, 0)

    if total == 0:
        return {'score': 0, 'status': 'Empty Sprint', 'blockers': 0, 'action': 'Add items to sprint'}

    # Calculate readiness score
    completion_pct = (done / total) * 100
    blocker_penalty = min(30, blockers * 10)
    wip_penalty = max(0, (wip - 5) * 2) if wip > 5 else 0

    score = max(0, min(100, completion_pct - blocker_penalty - wip_penalty))

    # Determine status and action
    if score >= 80 and blockers == 0:
        status = '🟢 Ready to Ship'
        action = 'Proceed with release'
    elif score >= 60:
        status = '🟡 Almost Ready'
        action = f'Clear {blockers} blocker(s)' if blockers > 0 else 'Complete remaining items'
    else:
        status = '🔴 Not Ready'
        action = f'{total - done} items remaining'

    return {
        'score': int(score),
        'status': status,
        'blockers': blockers,
        'action': action,
        'sprint_name': sprint_name,
        'done': done,
        'total': total
    }


def main():
    # Header
    col_title, col_date = st.columns([3, 1])
    with col_title:
        st.markdown("# 🏆 Executive Cockpit")
        st.markdown("*Real-time Engineering Organization Health*")
    with col_date:
        st.markdown(f"""
        <div style="text-align: right; padding: 20px;">
            <div style="font-size: 12px; color: #64748b;">LAST UPDATED</div>
            <div style="font-size: 18px; color: #4f46e5; font-weight: 600;">{datetime.now().strftime('%H:%M:%S')}</div>
            <div style="font-size: 14px; color: #64748b;">{datetime.now().strftime('%B %d, %Y')}</div>
        </div>
        """, unsafe_allow_html=True)

    conn = get_connection()
    if not conn:
        st.error("⚠️ Database not found. Please sync data first.")
        st.stop()

    # ========== QUICK WIN: Release Readiness ==========
    release_data = calculate_release_readiness(conn)
    indicator_color = '#22c55e' if release_data['score'] >= 80 else '#f59e0b' if release_data['score'] >= 60 else '#ef4444'

    st.markdown(f"""
    <div class="quick-win-widget">
        <div>
            <div class="quick-win-title">⚡ RELEASE READINESS</div>
            <div class="quick-win-value">{release_data['score']}%</div>
        </div>
        <div class="quick-win-status">
            <div class="quick-win-indicator" style="background: {indicator_color};"></div>
            <span>{release_data['status']}</span>
        </div>
        <div>
            <div style="font-size: 12px; opacity: 0.8;">{release_data.get('done', 0)}/{release_data.get('total', 0)} done • {release_data['blockers']} blockers</div>
            <div class="quick-win-action">→ {release_data['action']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("🔄 Aggregating Intelligence..."):
        df_tickets = conn.execute("SELECT * FROM issues").fetchdf()
        df_completed = conn.execute("SELECT * FROM issues WHERE status = 'Terminé(e)'").fetchdf()
        df_users = conn.execute("SELECT * FROM users WHERE active = true").fetchdf()

        try:
            df_worklogs = conn.execute("SELECT * FROM worklogs").fetchdf()
        except:
            df_worklogs = pd.DataFrame()

        # Compute metrics
        classifier = get_classifier()
        strat_analyzer = StrategicAlignmentAnalyzer(classifier)

        try:
            recent = df_tickets[pd.to_datetime(df_tickets['created'], utc=True) >=
                               (pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=90))]
        except:
            recent = df_tickets

        strat_res = strat_analyzer.calculate_alignment(
            recent,
            {"New Value": 0.7, "Maintenance": 0.3},
            team_size=len(df_users)
        )

        burn_analyzer = BurnoutAnalyzer()
        burn_res = burn_analyzer.analyze_team_risks(df_tickets, df_worklogs, df_users)
        high_risk = len([p for p in burn_res if p.risk_level == 'High Risk'])
        burnout_pct = high_risk / max(len(df_users), 1)

        forecaster = DeliveryForecaster()
        del_params = forecaster.analyze_historical_performance(df_completed, pd.DataFrame())
        del_res = forecaster.run_simulation(
            remaining_backlog_items=len(df_tickets[df_tickets['status'] != 'Terminé(e)']),
            target_date=datetime.now() + timedelta(days=90),
            historical_params=del_params
        )

        # Get trends
        trends = get_trend_data(conn)

        # Calculate health score
        health_score, health_grade = calculate_health_score(
            strat_res.drift_velocity,
            burnout_pct,
            del_res.target_date_prob
        )

    # ========== HEALTH SCORE BANNER ==========
    grade_colors = {'A': '#2ed573', 'B': '#7bed9f', 'C': '#ffa502', 'D': '#ff4757'}
    st.markdown(f"""
    <div class="exec-summary">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h3>🎯 Organization Health Score</h3>
                <p style="color: #64748b; margin: 0;">Composite score based on Strategy, Team Health, and Delivery Metrics</p>
            </div>
            <div style="text-align: center;">
                <div class="metric-giant" style="background: {grade_colors[health_grade]}; -webkit-background-clip: text;">
                    {health_score}
                </div>
                <span class="score-badge score-{'excellent' if health_grade == 'A' else 'good' if health_grade == 'B' else 'warning' if health_grade == 'C' else 'danger'}">
                    Grade {health_grade}
                </span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ========== THE HOLY TRINITY - GAUGE CHARTS ==========
    st.markdown("### 📊 The Holy Trinity™")

    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown('<div class="exec-card">', unsafe_allow_html=True)
        strategy_score = max(0, 100 - strat_res.drift_velocity * 100)
        fig = create_gauge_chart(strategy_score, "Strategic Alignment", thresholds=[40, 70])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""
        <div style="text-align: center;">
            <div class="metric-sublabel">💰 ${strat_res.total_drift_cost:,.0f} drift cost</div>
            <div style="color: #64748b; font-size: 12px;">{strat_res.shadow_work_percentage*100:.1f}% shadow work detected</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with g2:
        st.markdown('<div class="exec-card">', unsafe_allow_html=True)
        team_health = max(0, 100 - burnout_pct * 100)
        fig = create_gauge_chart(team_health, "Team Health", thresholds=[50, 80])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""
        <div style="text-align: center;">
            <div class="metric-sublabel">🚨 {high_risk} high-risk engineers</div>
            <div style="color: #64748b; font-size: 12px;">{len(df_users)} active team members</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with g3:
        st.markdown('<div class="exec-card">', unsafe_allow_html=True)
        fig = create_gauge_chart(del_res.target_date_prob * 100, "Delivery Confidence", thresholds=[50, 80])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""
        <div style="text-align: center;">
            <div class="metric-sublabel">📅 P85: {del_res.p85_date.strftime('%b %d')}</div>
            <div style="color: #64748b; font-size: 12px;">Bias factor: {del_params['estimation_bias_mean']:.1f}x</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ========== TREND SPARKLINES ==========
    st.markdown("### 📈 7-Day Trends")

    t1, t2, t3, t4 = st.columns(4)

    with t1:
        st.markdown('<div class="exec-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Issues Created</div>', unsafe_allow_html=True)
        st.plotly_chart(create_sparkline(trends['created'], "Created", "#667eea"), use_container_width=True)
        delta = sum(trends['created'][-3:]) - sum(trends['created'][:3])
        st.markdown(f'<div class="metric-sublabel">{"↑" if delta > 0 else "↓"} {abs(delta)} vs prior week</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        st.markdown('<div class="exec-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Issues Resolved</div>', unsafe_allow_html=True)
        st.plotly_chart(create_sparkline(trends['resolved'], "Resolved", "#2ed573"), use_container_width=True)
        delta = sum(trends['resolved'][-3:]) - sum(trends['resolved'][:3])
        st.markdown(f'<div class="metric-sublabel">{"↑" if delta > 0 else "↓"} {abs(delta)} vs prior week</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with t3:
        st.markdown('<div class="exec-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Velocity (Points)</div>', unsafe_allow_html=True)
        st.plotly_chart(create_sparkline(trends['velocity'], "Velocity", "#ffa502"), use_container_width=True)
        avg_vel = sum(trends['velocity']) / len(trends['velocity'])
        st.markdown(f'<div class="metric-sublabel">{avg_vel:.0f} pts/day avg</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with t4:
        # Net flow (resolved - created)
        net_flow = [r - c for r, c in zip(trends['resolved'], trends['created'])]
        st.markdown('<div class="exec-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Net Flow</div>', unsafe_allow_html=True)
        color = "#2ed573" if sum(net_flow) >= 0 else "#ff4757"
        st.plotly_chart(create_sparkline(net_flow, "Net", color), use_container_width=True)
        st.markdown(f'<div class="metric-sublabel">{"Burning down" if sum(net_flow) >= 0 else "Backlog growing"}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ========== RISK MATRIX & ACTIONS ==========
    col_risk, col_actions = st.columns([1, 1])

    with col_risk:
        st.markdown("### 🔥 Risk Heat Matrix")

        risk_data = {
            'Delivery': {
                'Impact': (1 - del_res.target_date_prob) * 100,
                'Likelihood': (1 - del_res.target_date_prob) * 80,
                'Velocity': del_params['velocity_std'] / max(del_params['velocity_mean'], 1) * 100
            },
            'Team Health': {
                'Impact': burnout_pct * 100,
                'Likelihood': high_risk * 20,
                'Velocity': min(high_risk * 15, 100)
            },
            'Strategy': {
                'Impact': strat_res.drift_velocity * 100,
                'Likelihood': strat_res.shadow_work_percentage * 100,
                'Velocity': strat_res.drift_velocity * 80
            },
            'Technical': {
                'Impact': 40,  # Placeholder
                'Likelihood': 35,
                'Velocity': 30
            }
        }

        fig = create_risk_heatmap(risk_data)
        st.plotly_chart(fig, use_container_width=True)

    with col_actions:
        st.markdown("### ⚡ Priority Actions")

        actions = []

        # Generate actions based on metrics
        if high_risk > 0:
            actions.append({
                'priority': 'critical',
                'title': f'🚨 {high_risk} Engineers at Burnout Risk',
                'action': 'Schedule 1:1s immediately. Review workload distribution.',
                'impact': 'High attrition risk'
            })

        if del_res.target_date_prob < 0.5:
            actions.append({
                'priority': 'critical',
                'title': '📅 Delivery at Risk',
                'action': f'Cut scope by {int((1-del_res.target_date_prob)*30)}% or extend deadline.',
                'impact': f'Only {del_res.target_date_prob*100:.0f}% confidence'
            })

        if strat_res.shadow_work_percentage > 0.15:
            actions.append({
                'priority': 'warning',
                'title': '👻 High Shadow Work Detected',
                'action': 'Review sprint planning. Align tickets with OKRs.',
                'impact': f'${strat_res.total_drift_cost:,.0f} potential waste'
            })

        if del_params['estimation_bias_mean'] > 1.5:
            actions.append({
                'priority': 'warning',
                'title': '📊 Estimation Accuracy Issue',
                'action': f'Apply {del_params["estimation_bias_mean"]:.1f}x buffer to estimates.',
                'impact': 'Historical underestimation pattern'
            })

        # Add default action if none
        if not actions:
            actions.append({
                'priority': 'info',
                'title': '✅ Organization Healthy',
                'action': 'Continue monitoring. Consider stretch goals.',
                'impact': 'All metrics within acceptable ranges'
            })

        for action in actions[:4]:
            st.markdown(f"""
            <div class="action-card action-{action['priority']}">
                <strong style="color: #1a202c;">{action['title']}</strong>
                <p style="color: #64748b; margin: 8px 0 4px 0; font-size: 14px;">{action['action']}</p>
                <span style="color: #059669; font-size: 12px;">Impact: {action['impact']}</span>
            </div>
            """, unsafe_allow_html=True)

    # ========== EXECUTIVE QUESTIONS ==========
    st.markdown("---")
    st.markdown("### 💡 Executive Q&A")

    q1, q2, q3 = st.columns(3)

    with q1:
        status = "🟢" if strategy_score > 70 else "🟡" if strategy_score > 40 else "🔴"
        st.markdown(f"""
        <div class="exec-card">
            <h4 style="color: #1a202c;">{status} Are we building the right things?</h4>
            <p style="color: #64748b;">
                {"Yes - " if strategy_score > 70 else "Partially - " if strategy_score > 40 else "No - "}
                {strat_res.shadow_work_percentage*100:.0f}% of work is unplanned.
                ${strat_res.total_drift_cost:,.0f} spent on non-strategic work.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with q2:
        status = "🟢" if team_health > 80 else "🟡" if team_health > 50 else "🔴"
        st.markdown(f"""
        <div class="exec-card">
            <h4 style="color: #1a202c;">{status} Is the team sustainable?</h4>
            <p style="color: #64748b;">
                {"Yes - " if team_health > 80 else "Caution - " if team_health > 50 else "No - "}
                {high_risk} of {len(df_users)} engineers showing burnout signals.
                {"Workload is balanced." if high_risk == 0 else "Immediate intervention needed."}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with q3:
        prob = del_res.target_date_prob * 100
        status = "🟢" if prob > 80 else "🟡" if prob > 50 else "🔴"
        st.markdown(f"""
        <div class="exec-card">
            <h4 style="color: #1a202c;">{status} Will we deliver on time?</h4>
            <p style="color: #64748b;">
                {"Likely - " if prob > 80 else "Uncertain - " if prob > 50 else "Unlikely - "}
                {prob:.0f}% confidence for 90-day target.
                P85 delivery: {del_res.p85_date.strftime('%B %d, %Y')}.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ========== FOOTER METRICS ==========
    st.markdown("---")

    m1, m2, m3, m4, m5 = st.columns(5)

    total_issues = len(df_tickets)
    open_issues = len(df_tickets[df_tickets['status'] != 'Terminé(e)'])
    bugs = len(df_tickets[df_tickets['issue_type'] == 'Bug'])
    avg_cycle = df_completed['cycle_time_hours'].mean() if 'cycle_time_hours' in df_completed.columns else 0

    with m1:
        st.metric("Total Issues", f"{total_issues:,}")
    with m2:
        st.metric("Open Issues", f"{open_issues:,}", delta=f"-{len(df_completed)} resolved")
    with m3:
        st.metric("Active Bugs", f"{bugs:,}")
    with m4:
        st.metric("Avg Cycle Time", f"{avg_cycle:.0f}h" if avg_cycle else "N/A")
    with m5:
        st.metric("Team Size", len(df_users))

    conn.close()


if __name__ == "__main__":
    main()
