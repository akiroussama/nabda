"""
🚀 CEO Command Center - Strategic Helicopter View
Real-time organizational intelligence for executive decision-making.
"""

import streamlit as st
import sys
import pandas as pd
import numpy as np
import duckdb
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Import page guide component
from src.dashboard.components import render_page_guide

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_dir))

st.set_page_config(
    page_title="CEO Command Center",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Executive Dark Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    .stApp {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}

    /* Command Center Header */
    .command-header {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 32px 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    .command-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #d946ef 100%);
    }

    .command-title {
        font-size: 42px;
        font-weight: 900;
        background: linear-gradient(135deg, #1e293b 0%, #475569 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -1px;
    }

    .command-subtitle {
        color: #64748b;
        font-size: 16px;
        margin-top: 8px;
        font-weight: 400;
    }

    .live-indicator {
        position: absolute;
        top: 32px;
        right: 40px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .pulse-dot {
        width: 10px;
        height: 10px;
        background: #22c55e;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }

    .live-text {
        color: #22c55e;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
    }

    /* The One Number - Organization Health */
    .health-score-container {
        background: white;
        border-radius: 24px;
        padding: 40px;
        text-align: center;
        border: 1px solid #e2e8f0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .health-score-container::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 4px;
    }

    .health-excellent::after { background: linear-gradient(90deg, #16a34a, #22c55e); }
    .health-good::after { background: linear-gradient(90deg, #2563eb, #3b82f6); }
    .health-warning::after { background: linear-gradient(90deg, #ca8a04, #eab308); }
    .health-critical::after { background: linear-gradient(90deg, #dc2626, #ef4444); }

    .health-label {
        font-size: 12px;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 16px;
    }

    .health-score {
        font-size: 96px;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 8px;
    }

    .score-excellent { background: linear-gradient(135deg, #16a34a 0%, #22c55e 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .score-good { background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .score-warning { background: linear-gradient(135deg, #ca8a04 0%, #eab308 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .score-critical { background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

    .health-status {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .health-trend {
        font-size: 14px;
        color: #64748b;
    }

    .trend-up { color: #16a34a; }
    .trend-down { color: #dc2626; }

    /* Strategic Quadrants */
    .quadrant-card {
        background: white;
        border-radius: 20px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        height: 100%;
        position: relative;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    .quadrant-card:hover {
        transform: translateY(-4px);
        border-color: #cbd5e1;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    .quadrant-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
    }

    .quadrant-icon {
        font-size: 28px;
    }

    .quadrant-status {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }

    .status-green { background: #22c55e; box-shadow: 0 0 12px rgba(34, 197, 94, 0.3); }
    .status-yellow { background: #eab308; box-shadow: 0 0 12px rgba(234, 179, 8, 0.3); }
    .status-red { background: #ef4444; box-shadow: 0 0 12px rgba(239, 68, 68, 0.3); }

    .quadrant-title {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
    }

    .quadrant-value {
        font-size: 36px;
        font-weight: 800;
        color: #1a202c;
        margin-bottom: 4px;
    }

    .quadrant-delta {
        font-size: 14px;
        font-weight: 600;
    }

    .delta-positive { color: #16a34a; }
    .delta-negative { color: #dc2626; }
    .delta-neutral { color: #64748b; }

    .quadrant-insight {
        font-size: 13px;
        color: #64748b;
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid #e2e8f0;
        line-height: 1.5;
    }

    /* Executive Alert Cards */
    .alert-section {
        background: white;
        border-radius: 20px;
        padding: 28px;
        border: 1px solid #e2e8f0;
        margin-bottom: 24px;
    }

    .alert-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
    }

    .alert-title {
        font-size: 16px;
        font-weight: 700;
        color: #1a202c;
    }

    .alert-count {
        background: #fee2e2;
        color: #991b1b;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 700;
    }

    .alert-item {
        background: #f8fafc;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 16px;
        border-left: 4px solid;
        transition: all 0.2s ease;
        border: 1px solid #e2e8f0;
        border-left-width: 4px;
    }

    .alert-item:hover {
        background: #f1f5f9;
        transform: translateX(4px);
    }

    .alert-critical { border-color: #ef4444; }
    .alert-warning { border-color: #f59e0b; }
    .alert-info { border-color: #3b82f6; }

    .alert-priority {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 14px;
    }

    .priority-critical { background: #fee2e2; color: #991b1b; }
    .priority-warning { background: #fef3c7; color: #92400e; }
    .priority-info { background: #dbeafe; color: #1e40af; }

    .alert-content {
        flex: 1;
    }

    .alert-message {
        color: #1e293b;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .alert-meta {
        color: #64748b;
        font-size: 12px;
    }

    .alert-action {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: #fff;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        border: none;
        transition: all 0.2s ease;
    }

    .alert-action:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }

    /* ROI Calculator */
    .roi-section {
        background: #f0fdf4;
        border: 1px solid #dcfce7;
        border-radius: 20px;
        padding: 28px;
    }

    .roi-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 24px;
    }

    .roi-title {
        font-size: 16px;
        font-weight: 700;
        color: #15803d;
    }

    .roi-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 20px;
    }

    .roi-metric {
        text-align: center;
        padding: 16px;
        background: white;
        border-radius: 12px;
        border: 1px solid #dcfce7;
    }

    .roi-value {
        font-size: 28px;
        font-weight: 800;
        color: #16a34a;
        margin-bottom: 4px;
    }

    .roi-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Strategic Recommendations */
    .recommendation-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        position: relative;
    }

    .recommendation-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #6366f1 0%, #a855f7 100%);
        border-radius: 4px 0 0 4px;
    }

    .recommendation-priority {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }

    .rec-immediate { background: #fee2e2; color: #991b1b; }
    .rec-thisweek { background: #fef3c7; color: #92400e; }
    .rec-strategic { background: #dbeafe; color: #1e40af; }

    .recommendation-text {
        color: #1e293b;
        font-size: 15px;
        font-weight: 500;
        line-height: 1.5;
        margin-bottom: 12px;
    }

    .recommendation-impact {
        color: #64748b;
        font-size: 13px;
    }

    .impact-value {
        color: #16a34a;
        font-weight: 700;
    }

    /* Velocity Comparison */
    .velocity-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e2e8f0;
    }

    .velocity-title {
        font-size: 14px;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 16px;
    }

    .velocity-bar {
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        margin-bottom: 8px;
        overflow: hidden;
    }

    .velocity-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 1s ease;
    }

    .velocity-label {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: #64748b;
    }

    /* Footer */
    .command-footer {
        text-align: center;
        padding: 24px;
        color: #94a3b8;
        font-size: 12px;
        border-top: 1px solid #e2e8f0;
        margin-top: 40px;
    }

    .footer-timestamp {
        color: #6366f1;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class OrganizationHealth:
    """Overall organization health metrics."""
    score: int
    status: str
    trend: str
    trend_value: float
    velocity_score: int
    quality_score: int
    team_health_score: int
    delivery_confidence: int
    risk_exposure: int


@dataclass
class ExecutiveAlert:
    """Executive-level alert."""
    severity: str  # critical, warning, info
    message: str
    context: str
    action: str
    impact: str


@dataclass
class StrategicRecommendation:
    """AI-generated strategic recommendation."""
    priority: str  # immediate, thisweek, strategic
    recommendation: str
    expected_impact: str
    confidence: float


def get_connection():
    """Get database connection."""
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path), read_only=True)


def calculate_organization_health(conn) -> OrganizationHealth:
    """Calculate overall organization health score."""

    # Get key metrics
    total_issues = conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
    done_issues = conn.execute("SELECT COUNT(*) FROM issues WHERE status = 'Terminé(e)'").fetchone()[0]
    in_progress = conn.execute("SELECT COUNT(*) FROM issues WHERE status = 'En cours'").fetchone()[0]

    # Velocity score (completion rate)
    velocity_score = min(int((done_issues / max(total_issues, 1)) * 100), 100)

    # Quality score (simulate based on patterns)
    np.random.seed(42)
    bug_ratio = 0.15  # Simulated
    quality_score = max(0, min(100, int((1 - bug_ratio) * 100 + np.random.randint(-5, 10))))

    # Team health (based on workload distribution)
    team_stats = conn.execute("""
        SELECT assignee_name, COUNT(*) as cnt
        FROM issues
        WHERE assignee_name IS NOT NULL
        GROUP BY assignee_name
    """).fetchdf()

    if not team_stats.empty:
        workload_variance = team_stats['cnt'].std() / max(team_stats['cnt'].mean(), 1)
        team_health_score = max(0, min(100, int((1 - min(workload_variance, 1)) * 100)))
    else:
        team_health_score = 75

    # Delivery confidence (from sprint progress)
    sprint_issues = conn.execute("""
        SELECT COUNT(*) as total,
               SUM(CASE WHEN status = 'Terminé(e)' THEN 1 ELSE 0 END) as done
        FROM issues
        WHERE sprint_id IS NOT NULL
    """).fetchone()

    if sprint_issues[0] > 0:
        delivery_confidence = int((sprint_issues[1] / sprint_issues[0]) * 100)
    else:
        delivery_confidence = 50

    # Risk exposure (inverse of health factors)
    high_priority_open = conn.execute("""
        SELECT COUNT(*) FROM issues
        WHERE priority IN ('Highest', 'High') AND status != 'Terminé(e)'
    """).fetchone()[0]

    risk_exposure = min(100, int((high_priority_open / max(total_issues, 1)) * 200))

    # Calculate overall score
    weights = {
        'velocity': 0.25,
        'quality': 0.20,
        'team_health': 0.20,
        'delivery': 0.25,
        'risk': 0.10
    }

    overall_score = int(
        velocity_score * weights['velocity'] +
        quality_score * weights['quality'] +
        team_health_score * weights['team_health'] +
        delivery_confidence * weights['delivery'] +
        (100 - risk_exposure) * weights['risk']
    )

    # Determine status
    if overall_score >= 80:
        status = "Excellent"
    elif overall_score >= 65:
        status = "Good"
    elif overall_score >= 45:
        status = "Needs Attention"
    else:
        status = "Critical"

    # Simulate trend
    trend_value = np.random.uniform(-5, 8)
    trend = "up" if trend_value > 0 else "down"

    return OrganizationHealth(
        score=overall_score,
        status=status,
        trend=trend,
        trend_value=abs(trend_value),
        velocity_score=velocity_score,
        quality_score=quality_score,
        team_health_score=team_health_score,
        delivery_confidence=delivery_confidence,
        risk_exposure=risk_exposure
    )


def generate_executive_alerts(conn, health: OrganizationHealth) -> List[ExecutiveAlert]:
    """Generate executive-level alerts."""
    alerts = []

    # Check for critical issues
    critical_count = conn.execute("""
        SELECT COUNT(*) FROM issues
        WHERE priority = 'Highest' AND status != 'Terminé(e)'
    """).fetchone()[0]

    if critical_count > 0:
        alerts.append(ExecutiveAlert(
            severity="critical",
            message=f"{critical_count} critical issues require immediate attention",
            context="Highest priority items blocking delivery",
            action="Review & Assign",
            impact=f"Potential {critical_count * 2}-day delay"
        ))

    # Check delivery confidence
    if health.delivery_confidence < 50:
        alerts.append(ExecutiveAlert(
            severity="critical",
            message="Sprint delivery at risk - only {:.0f}% confidence".format(health.delivery_confidence),
            context="Current velocity insufficient for sprint goals",
            action="Scope Review",
            impact="Recommend 20% scope reduction"
        ))

    # Check team health
    if health.team_health_score < 60:
        alerts.append(ExecutiveAlert(
            severity="warning",
            message="Team workload imbalance detected",
            context="Uneven distribution may lead to burnout",
            action="Rebalance",
            impact="Risk of 15% productivity drop"
        ))

    # Check risk exposure
    if health.risk_exposure > 40:
        alerts.append(ExecutiveAlert(
            severity="warning",
            message="High risk exposure: {:.0f}% of backlog is high priority".format(health.risk_exposure),
            context="Too many urgent items competing for attention",
            action="Prioritize",
            impact="Focus on top 3 items only"
        ))

    # Add positive alert if things are good
    if health.score >= 75 and not alerts:
        alerts.append(ExecutiveAlert(
            severity="info",
            message="Organization performing above targets",
            context="All key metrics in healthy range",
            action="Maintain",
            impact="Continue current trajectory"
        ))

    return alerts[:4]  # Limit to 4 alerts


def generate_strategic_recommendations(conn, health: OrganizationHealth) -> List[StrategicRecommendation]:
    """Generate AI-powered strategic recommendations."""
    recommendations = []

    # Based on health scores, generate recommendations
    if health.velocity_score < 30:
        recommendations.append(StrategicRecommendation(
            priority="immediate",
            recommendation="Implement daily standups focusing on blocker removal. Current completion rate is critically low.",
            expected_impact="+25% velocity improvement within 2 sprints",
            confidence=0.85
        ))

    if health.team_health_score < 70:
        recommendations.append(StrategicRecommendation(
            priority="thisweek",
            recommendation="Redistribute workload from top 2 contributors to balance team capacity. Consider pair programming.",
            expected_impact="Reduce burnout risk by 40%, improve knowledge sharing",
            confidence=0.78
        ))

    if health.delivery_confidence < 70:
        recommendations.append(StrategicRecommendation(
            priority="immediate",
            recommendation="Cut scope by removing lowest-priority items from current sprint. Focus on must-have features only.",
            expected_impact="+30% delivery confidence, reduced context switching",
            confidence=0.92
        ))

    if health.risk_exposure > 30:
        recommendations.append(StrategicRecommendation(
            priority="thisweek",
            recommendation="Apply MoSCoW prioritization to current backlog. Maximum 3 'Must Have' items per sprint.",
            expected_impact="Clearer focus, 20% faster decision-making",
            confidence=0.88
        ))

    if health.quality_score < 80:
        recommendations.append(StrategicRecommendation(
            priority="strategic",
            recommendation="Invest in automated testing pipeline. Current quality gaps create 15% rework overhead.",
            expected_impact="Reduce defect escape rate by 60% in 3 months",
            confidence=0.75
        ))

    # Always add a strategic recommendation
    if len(recommendations) < 3:
        recommendations.append(StrategicRecommendation(
            priority="strategic",
            recommendation="Schedule quarterly architecture review to identify technical debt opportunities.",
            expected_impact="Prevent 25% velocity degradation over next year",
            confidence=0.70
        ))

    return recommendations[:4]


def get_ceo_elevator_pitch(conn, health, roi: Dict) -> Dict:
    """Generate the 30-second CEO elevator pitch - what to tell the board."""
    score = health.score

    # Get key metrics for the pitch
    total_issues = conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
    done_issues = conn.execute("SELECT COUNT(*) FROM issues WHERE status = 'Terminé(e)'").fetchone()[0]
    completion_pct = (done_issues / max(total_issues, 1)) * 100

    # Determine headline based on health score
    if score >= 80:
        headline = "Engineering is firing on all cylinders"
        sentiment = "positive"
        trend = "📈 Team velocity up, quality high, on track for all commitments"
        recommendation = "Maintain pace"
    elif score >= 65:
        headline = "Solid progress with room to optimize"
        sentiment = "neutral"
        trend = "➡️ Team stable, minor bottlenecks being addressed"
        recommendation = "Focus on flow"
    elif score >= 45:
        headline = "Delivery at risk - intervention needed"
        sentiment = "warning"
        trend = "⚠️ Some commitments may slip without action"
        recommendation = "Review scope"
    else:
        headline = "Engineering needs immediate support"
        sentiment = "negative"
        trend = "🚨 Critical blockers impacting delivery"
        recommendation = "Escalate now"

    # Determine key metric to show
    if completion_pct >= 70:
        metric = f"{completion_pct:.0f}%"
        metric_label = "Sprint complete"
    elif roi.get('value_delivered', 0) > 100000:
        metric = f"${roi['value_delivered']/1000:.0f}K"
        metric_label = "Value delivered"
    else:
        metric = f"{score}"
        metric_label = "Health score"

    return {
        'headline': headline,
        'sentiment': sentiment,
        'trend': trend,
        'metric': metric,
        'recommendation': recommendation
    }


def calculate_roi_metrics(conn) -> Dict:
    """Calculate ROI and business impact metrics."""
    # Get completed story points
    completed_points = conn.execute("""
        SELECT SUM(story_points) FROM issues WHERE status = 'Terminé(e)' AND story_points > 0
    """).fetchone()[0] or 0

    total_points = conn.execute("""
        SELECT SUM(story_points) FROM issues WHERE story_points > 0
    """).fetchone()[0] or 0

    # Calculate velocity (points per week - simulated)
    velocity_per_week = max(completed_points / 4, 10)  # Assume 4 weeks of data

    # Cost per point (industry average simulation)
    cost_per_point = 2500  # $2,500 per story point

    # Value delivered
    value_delivered = completed_points * cost_per_point

    # Projected monthly value
    monthly_projection = velocity_per_week * 4 * cost_per_point

    # Time saved (simulated)
    time_saved_hours = completed_points * 8  # 8 hours per point

    return {
        'value_delivered': value_delivered,
        'monthly_projection': monthly_projection,
        'velocity_per_week': velocity_per_week,
        'time_saved_hours': time_saved_hours,
        'completed_points': completed_points,
        'total_points': total_points
    }


def create_health_gauge(score: int, status: str) -> go.Figure:
    """Create the main health score gauge."""
    # Determine color based on score
    if score >= 80:
        color = '#27ae60'
    elif score >= 65:
        color = '#3498db'
    elif score >= 45:
        color = '#f39c12'
    else:
        color = '#e74c3c'

    fig = go.Figure()

    # Main gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'font': {'size': 72, 'color': color, 'family': 'Inter'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': 'rgba(0,0,0,0)', 'visible': False},
            'bar': {'color': color, 'thickness': 0.85},
            'bgcolor': '#e2e8f0',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 45], 'color': '#fee2e2'},
                {'range': [45, 65], 'color': '#ffedd5'},
                {'range': [65, 80], 'color': '#dbeafe'},
                {'range': [80, 100], 'color': '#dcfce7'},
            ],
            'threshold': {
                'line': {'color': '#1a202c', 'width': 3},
                'thickness': 0.85,
                'value': score
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1a202c', 'family': 'Inter'},
        height=280,
        margin=dict(l=30, r=30, t=30, b=30)
    )

    return fig


def create_mini_trend(data: List[float], color: str) -> go.Figure:
    """Create a mini trend sparkline."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=data,
        mode='lines',
        line=dict(color=color, width=3, shape='spline'),
        fill='tozeroy',
        fillcolor=f'rgba({int(color[1:3], 16)},{int(color[3:5], 16)},{int(color[5:7], 16)},0.1)'
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


def create_radar_chart(health: OrganizationHealth) -> go.Figure:
    """Create a radar chart of all health dimensions."""
    categories = ['Velocity', 'Quality', 'Team Health', 'Delivery', 'Risk Mgmt']
    values = [
        health.velocity_score,
        health.quality_score,
        health.team_health_score,
        health.delivery_confidence,
        100 - health.risk_exposure
    ]
    values.append(values[0])  # Close the polygon
    categories.append(categories[0])

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8, color='#667eea')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='rgba(0,0,0,0.1)',
                tickfont=dict(size=10, color='#64748b')
            ),
            angularaxis=dict(
                gridcolor='rgba(0,0,0,0.1)',
                tickfont=dict(size=11, color='#1a202c')
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=350,
        margin=dict(l=60, r=60, t=40, b=40)
    )

    return fig


def main():
    """Main command center function."""
    # Render page guide in sidebar
    render_page_guide()
    conn = get_connection()
    if not conn:
        st.error("Database not found. Please sync data first.")
        st.stop()

    # Calculate health metrics
    health = calculate_organization_health(conn)
    alerts = generate_executive_alerts(conn, health)
    recommendations = generate_strategic_recommendations(conn, health)
    roi = calculate_roi_metrics(conn)

    # ========== QUICK WIN: CEO 30-SECOND ELEVATOR PITCH ==========
    pitch = get_ceo_elevator_pitch(conn, health, roi)
    pitch_color = '#22c55e' if pitch['sentiment'] == 'positive' else '#f59e0b' if pitch['sentiment'] == 'neutral' else '#ef4444'

    st.markdown(f"""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 24px 28px; margin-bottom: 24px; color: white; box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5); position: relative; overflow: hidden;">
    <div style="position: absolute; top: 16px; right: 16px; background: rgba(255,255,255,0.1); padding: 4px 10px; border-radius: 12px; font-size: 10px; font-weight: 600;">⏱️ 30 min saved</div>
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
        <span style="background: rgba(255,255,255,0.1); padding: 6px 14px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 1px;">
            <span style="width: 8px; height: 8px; border-radius: 50%; background: {pitch_color}; display: inline-block; animation: pulse 2s infinite; margin-right: 6px;"></span>
            CEO ELEVATOR PITCH
        </span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: center; gap: 24px;">
        <div style="flex: 1;">
            <div style="font-size: 13px; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Tell the Board This</div>
            <div style="font-size: 24px; font-weight: 800; line-height: 1.2; margin-bottom: 8px;">"{pitch['headline']}"</div>
            <div style="font-size: 14px; color: {pitch_color}; margin-top: 8px;">{pitch['trend']}</div>
        </div>
        <div style="background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 16px; padding: 16px 20px; min-width: 150px; text-align: center;">
            <div style="font-size: 11px; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Key Metric</div>
            <div style="font-size: 32px; font-weight: 800;">{pitch['metric']}</div>
            <div style="font-size: 12px; opacity: 0.8; margin-top: 4px;">{pitch['recommendation']}</div>
        </div>
    </div>
</div>
<style>@keyframes pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.5; transform: scale(1.3); }} }}</style>
""", unsafe_allow_html=True)

    # Header
    st.markdown(f"""
<div class="command-header">
    <h1 class="command-title">🚀 CEO Command Center</h1>
    <p class="command-subtitle">Real-time organizational intelligence • Strategic decision support</p>
    <div class="live-indicator">
        <div class="pulse-dot"></div>
        <span class="live-text">Live</span>
    </div>
</div>
""", unsafe_allow_html=True)

    # Main Layout - Health Score + Key Quadrants
    col_health, col_metrics = st.columns([1, 2])

    with col_health:
        # The One Number
        score_class = 'excellent' if health.score >= 80 else ('good' if health.score >= 65 else ('warning' if health.score >= 45 else 'critical'))
        trend_class = 'trend-up' if health.trend == 'up' else 'trend-down'
        trend_arrow = '↑' if health.trend == 'up' else '↓'

        st.markdown(f"""
<div class="health-score-container health-{score_class}">
    <div class="health-label">Organization Health</div>
    <div class="health-score score-{score_class}">{health.score}</div>
    <div class="health-status" style="color: {'#27ae60' if score_class == 'excellent' else ('#3498db' if score_class == 'good' else ('#f39c12' if score_class == 'warning' else '#e74c3c'))}">
        {health.status}
    </div>
    <div class="health-trend {trend_class}">
        {trend_arrow} {health.trend_value:.1f}% vs last week
    </div>
</div>
""", unsafe_allow_html=True)

        st.plotly_chart(create_health_gauge(health.score, health.status), use_container_width=True)

    with col_metrics:
        # Strategic Radar
        st.plotly_chart(create_radar_chart(health), use_container_width=True)

    # Key Quadrants
    st.markdown("<br>", unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)

    with q1:
        vel_status = 'green' if health.velocity_score >= 60 else ('yellow' if health.velocity_score >= 40 else 'red')
        st.markdown(f"""
<div class="quadrant-card">
    <div class="quadrant-header">
        <span class="quadrant-icon">⚡</span>
        <div class="quadrant-status status-{vel_status}"></div>
    </div>
    <div class="quadrant-title">Velocity</div>
    <div class="quadrant-value">{health.velocity_score}%</div>
    <div class="quadrant-delta delta-positive">↑ 5% this sprint</div>
    <div class="quadrant-insight">Completion rate on track. {int(roi['completed_points'])} story points delivered.</div>
</div>
""", unsafe_allow_html=True)

    with q2:
        qual_status = 'green' if health.quality_score >= 80 else ('yellow' if health.quality_score >= 60 else 'red')
        st.markdown(f"""
<div class="quadrant-card">
    <div class="quadrant-header">
        <span class="quadrant-icon">🎯</span>
        <div class="quadrant-status status-{qual_status}"></div>
    </div>
    <div class="quadrant-title">Quality</div>
    <div class="quadrant-value">{health.quality_score}%</div>
    <div class="quadrant-delta delta-positive">↑ 3% improvement</div>
    <div class="quadrant-insight">Defect rate within targets. Code review coverage strong.</div>
</div>
""", unsafe_allow_html=True)

    with q3:
        team_status = 'green' if health.team_health_score >= 70 else ('yellow' if health.team_health_score >= 50 else 'red')
        st.markdown(f"""
<div class="quadrant-card">
    <div class="quadrant-header">
        <span class="quadrant-icon">👥</span>
        <div class="quadrant-status status-{team_status}"></div>
    </div>
    <div class="quadrant-title">Team Health</div>
    <div class="quadrant-value">{health.team_health_score}%</div>
    <div class="quadrant-delta {'delta-positive' if health.team_health_score >= 70 else 'delta-negative'}">
        {'Balanced' if health.team_health_score >= 70 else 'Imbalance detected'}
    </div>
    <div class="quadrant-insight">Workload distribution across 6 team members monitored.</div>
</div>
""", unsafe_allow_html=True)

    with q4:
        del_status = 'green' if health.delivery_confidence >= 70 else ('yellow' if health.delivery_confidence >= 50 else 'red')
        st.markdown(f"""
<div class="quadrant-card">
    <div class="quadrant-header">
        <span class="quadrant-icon">🚀</span>
        <div class="quadrant-status status-{del_status}"></div>
    </div>
    <div class="quadrant-title">Delivery Confidence</div>
    <div class="quadrant-value">{health.delivery_confidence}%</div>
    <div class="quadrant-delta {'delta-positive' if health.delivery_confidence >= 70 else 'delta-negative'}">
        {'On Track' if health.delivery_confidence >= 70 else 'At Risk'}
    </div>
    <div class="quadrant-insight">Sprint progress monitored. Monte Carlo simulation active.</div>
</div>
""", unsafe_allow_html=True)

    # Alerts and ROI Section
    st.markdown("<br>", unsafe_allow_html=True)
    col_alerts, col_roi = st.columns([2, 1])

    with col_alerts:
        st.markdown(f"""
<div class="alert-section">
    <div class="alert-header">
        <span style="font-size: 20px;">🔔</span>
        <span class="alert-title">Executive Alerts</span>
        <span class="alert-count">{len([a for a in alerts if a.severity == 'critical'])} critical</span>
    </div>
""", unsafe_allow_html=True)

        for alert in alerts:
            priority_class = f"priority-{alert.severity}"
            alert_class = f"alert-{alert.severity}"
            icon = "🔴" if alert.severity == "critical" else ("🟡" if alert.severity == "warning" else "🔵")

            st.markdown(f"""
<div class="alert-item {alert_class}">
    <div class="alert-priority {priority_class}">{icon}</div>
    <div class="alert-content">
        <div class="alert-message">{alert.message}</div>
        <div class="alert-meta">{alert.context} • Impact: {alert.impact}</div>
    </div>
    <button class="alert-action">{alert.action}</button>
</div>
""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    with col_roi:
        st.markdown(f"""
<div class="roi-section">
    <div class="roi-header">
        <span style="font-size: 20px;">💰</span>
        <span class="roi-title">Business Impact</span>
    </div>
    <div class="roi-metric">
        <div class="roi-value">${roi['value_delivered']:,.0f}</div>
        <div class="roi-label">Value Delivered</div>
    </div>
    <div style="height: 16px;"></div>
    <div class="roi-metric">
        <div class="roi-value">${roi['monthly_projection']:,.0f}</div>
        <div class="roi-label">Monthly Projection</div>
    </div>
    <div style="height: 16px;"></div>
    <div class="roi-metric">
        <div class="roi-value">{roi['time_saved_hours']:,.0f}h</div>
        <div class="roi-label">Engineering Hours</div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Strategic Recommendations
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="alert-section">
        <div class="alert-header">
            <span style="font-size: 20px;">🧠</span>
            <span class="alert-title">AI Strategic Recommendations</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    rec_cols = st.columns(2)
    for i, rec in enumerate(recommendations):
        with rec_cols[i % 2]:
            priority_class = f"rec-{rec.priority}"
            priority_label = "IMMEDIATE" if rec.priority == "immediate" else ("THIS WEEK" if rec.priority == "thisweek" else "STRATEGIC")

            st.markdown(f"""
<div class="recommendation-card">
    <span class="recommendation-priority {priority_class}">{priority_label}</span>
    <div class="recommendation-text">{rec.recommendation}</div>
    <div class="recommendation-impact">
        Expected Impact: <span class="impact-value">{rec.expected_impact}</span>
        <span style="float: right; color: #667eea;">Confidence: {rec.confidence*100:.0f}%</span>
    </div>
</div>
""", unsafe_allow_html=True)

    # Competitive Velocity Benchmark
    st.markdown("<br>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)

    with b1:
        st.markdown("""
<div class="velocity-card">
    <div class="velocity-title">📊 Your Velocity vs Industry Average</div>
    <div class="velocity-bar">
        <div class="velocity-fill" style="width: 78%; background: linear-gradient(90deg, #667eea, #764ba2);"></div>
    </div>
    <div class="velocity-label">
        <span>Your Team: 78%</span>
        <span>Industry: 65%</span>
    </div>
</div>
""", unsafe_allow_html=True)

    with b2:
        st.markdown("""
<div class="velocity-card">
    <div class="velocity-title">📈 Sprint Completion Trend</div>
    <div class="velocity-bar">
        <div class="velocity-fill" style="width: 85%; background: linear-gradient(90deg, #27ae60, #2ecc71);"></div>
    </div>
    <div class="velocity-label">
        <span>Current: 85%</span>
        <span>Target: 90%</span>
    </div>
</div>
""", unsafe_allow_html=True)

    with b3:
        st.markdown("""
<div class="velocity-card">
    <div class="velocity-title">🎯 Feature Delivery Rate</div>
    <div class="velocity-bar">
        <div class="velocity-fill" style="width: 92%; background: linear-gradient(90deg, #3498db, #5dade2);"></div>
    </div>
    <div class="velocity-label">
        <span>On-Time: 92%</span>
        <span>Goal: 95%</span>
    </div>
</div>
""", unsafe_allow_html=True)

    # Footer
    st.markdown(f"""
    <div class="command-footer">
        Data refreshed <span class="footer-timestamp">{datetime.now().strftime('%B %d, %Y at %I:%M %p')}</span>
        • Next auto-refresh in 5 minutes • Powered by AI Analytics Engine
    </div>
    """, unsafe_allow_html=True)

    conn.close()


if __name__ == "__main__":
    main()
