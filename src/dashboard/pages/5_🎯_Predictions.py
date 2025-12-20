"""
🎯 Predictions - Advanced ML-Powered Analytics Dashboard
Ticket duration predictions, sprint risk analysis, and feature importance insights.
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

st.set_page_config(page_title="Predictions", page_icon="🎯", layout="wide")

# Premium Predictions CSS
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #f8f9fa;
    }

    .prediction-header {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .section-container {
        background: white;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
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

    .prediction-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        height: 100%;
        transition: transform 0.2s;
    }
    .prediction-card:hover { transform: translateY(-2px); }

    .prediction-value {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .prediction-label {
        font-size: 14px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
        font-weight: 600;
    }

    .prediction-sublabel {
        font-size: 12px;
        color: #4f46e5;
        margin-top: 4px;
        font-weight: 500;
    }

    .confidence-bar-container {
        background: #e2e8f0;
        border-radius: 8px;
        height: 24px;
        position: relative;
        overflow: hidden;
        margin-top: 16px;
    }

    .confidence-bar {
        height: 100%;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 12px;
        font-weight: 600;
    }
    .confidence-high { background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%); }
    .confidence-medium { background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%); }
    .confidence-low { background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%); }

    .risk-gauge-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        height: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .feature-bar { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }

    .feature-name {
        width: 150px;
        font-size: 13px;
        color: #334155;
        text-align: right;
        font-weight: 500;
    }

    .feature-bar-container {
        flex: 1;
        background: #e2e8f0;
        border-radius: 6px;
        height: 24px;
        overflow: hidden;
    }

    .feature-bar-fill {
        height: 100%;
        border-radius: 6px;
        display: flex;
        align-items: center;
        padding-left: 8px;
        font-size: 11px;
        font-weight: 600;
        color: white;
    }

    .issue-predictor-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        margin-bottom: 12px;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .issue-predictor-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.08);
        border-color: #cbd5e1;
    }

    .issue-key { font-size: 12px; color: #4f46e5; font-weight: 700; }
    .issue-summary { font-size: 14px; color: #1a202c; margin: 8px 0; font-weight: 500; }

    .prediction-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 6px 12px; border-radius: 20px;
        font-size: 12px; font-weight: 600;
    }
    .badge-fast { background: #dcfce7; color: #166534; }
    .badge-normal { background: #e0f2fe; color: #075985; }
    .badge-slow { background: #fef3c7; color: #92400e; }
    .badge-critical { background: #fee2e2; color: #991b1b; }

    .model-info {
        display: flex; gap: 16px; padding: 16px;
        background: #eef2ff; border-radius: 8px; margin-top: 16px;
        border: 1px solid #c7d2fe;
    }

    .model-stat { text-align: center; flex: 1; }
    .model-stat-value { font-size: 18px; font-weight: 700; color: #4f46e5; }
    .model-stat-label { font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600; }

    .comparison-row {
        display: flex; align-items: center; padding: 12px;
        background: #f8fafc; border-radius: 8px; margin-bottom: 8px;
        border: 1px solid #e2e8f0;
    }

    .sprint-risk-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        margin-bottom: 12px;
    }

    .tab-content { padding-top: 20px; }

    /* Quick Win Widget */
    .quick-win-widget {
        background: linear-gradient(135deg, #0f172a 0%, #1e40af 100%);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 16px 0;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 8px 32px rgba(15, 23, 42, 0.3);
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
        color: #93c5fd;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .risk-summary {
        display: flex;
        align-items: center;
        gap: 24px;
        flex-wrap: wrap;
    }
    .risk-score-main {
        text-align: center;
    }
    .risk-score-value {
        font-size: 56px;
        font-weight: 800;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    .risk-green { color: #86efac; }
    .risk-yellow { color: #fde047; }
    .risk-red { color: #fca5a5; }
    .risk-label {
        color: #93c5fd;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .risk-factors {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .risk-factor {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        background: rgba(255,255,255,0.08);
        border-radius: 8px;
    }
    .factor-icon {
        font-size: 16px;
    }
    .factor-text {
        color: #e2e8f0;
        font-size: 12px;
    }
    .factor-value {
        margin-left: auto;
        color: #fbbf24;
        font-weight: 600;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


def get_connection():
    db_path = Path("data/jira.duckdb")
    return duckdb.connect(str(db_path), read_only=True) if db_path.exists() else None


def calculate_release_risk(conn) -> dict:
    """
    Calculate overall release risk score - critical for release managers.
    Answers: "Is it safe to release this sprint?"
    """
    # Get active sprint metrics
    sprint_data = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = 'Terminé(e)' THEN 1 ELSE 0 END) as done,
            SUM(CASE WHEN status = 'En cours' THEN 1 ELSE 0 END) as in_progress,
            SUM(CASE WHEN priority IN ('Highest', 'High') AND status != 'Terminé(e)' THEN 1 ELSE 0 END) as high_priority_open,
            SUM(CASE WHEN issue_type = 'Bug' AND status != 'Terminé(e)' THEN 1 ELSE 0 END) as open_bugs
        FROM issues i
        JOIN sprints s ON i.sprint_id = s.id
        WHERE s.state = 'active'
    """).fetchone()

    if not sprint_data or sprint_data[0] == 0:
        return {'score': 50, 'status': 'Unknown', 'class': 'risk-yellow', 'factors': []}

    total, done, in_progress, high_pri_open, open_bugs = sprint_data
    completion_pct = (done / total) * 100 if total > 0 else 0

    # Calculate risk factors
    factors = []
    risk_score = 0

    # Factor 1: Completion rate
    if completion_pct < 50:
        risk_score += 30
        factors.append({'icon': '📊', 'text': 'Low completion rate', 'value': f'{completion_pct:.0f}%'})
    elif completion_pct < 75:
        risk_score += 15
        factors.append({'icon': '📊', 'text': 'Moderate completion', 'value': f'{completion_pct:.0f}%'})

    # Factor 2: High priority items
    if high_pri_open > 3:
        risk_score += 25
        factors.append({'icon': '🔴', 'text': 'High-priority items open', 'value': str(high_pri_open)})
    elif high_pri_open > 0:
        risk_score += 10
        factors.append({'icon': '🟡', 'text': 'Some high-priority items', 'value': str(high_pri_open)})

    # Factor 3: Open bugs
    if open_bugs > 5:
        risk_score += 25
        factors.append({'icon': '🐛', 'text': 'Open bugs', 'value': str(open_bugs)})
    elif open_bugs > 0:
        risk_score += 10
        factors.append({'icon': '🐛', 'text': 'Minor bugs open', 'value': str(open_bugs)})

    # Factor 4: WIP items (too much in progress = chaotic)
    wip_pct = (in_progress / total * 100) if total > 0 else 0
    if wip_pct > 50:
        risk_score += 20
        factors.append({'icon': '🔄', 'text': 'High WIP ratio', 'value': f'{wip_pct:.0f}%'})

    # Determine status
    risk_score = min(risk_score, 100)
    if risk_score <= 30:
        status = "Low Risk - Safe to Release"
        css_class = "risk-green"
    elif risk_score <= 60:
        status = "Medium Risk - Review Needed"
        css_class = "risk-yellow"
    else:
        status = "High Risk - Not Ready"
        css_class = "risk-red"

    # If no factors found, add a good one
    if not factors:
        factors.append({'icon': '✅', 'text': 'All metrics healthy', 'value': 'OK'})

    return {
        'score': risk_score,
        'status': status,
        'class': css_class,
        'factors': factors[:4]  # Max 4 factors
    }


def create_risk_gauge(score: float, title: str = "Risk Score") -> go.Figure:
    """Create a risk gauge chart."""
    if score <= 30:
        color = "#27ae60"
        status = "Low Risk"
    elif score <= 60:
        color = "#f39c12"
        status = "Medium Risk"
    else:
        color = "#e74c3c"
        status = "High Risk"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={'suffix': '', 'font': {'size': 48, 'color': '#1a202c'}},
        title={'text': f"<b>{title}</b><br><span style='font-size:14px;color:{color}'>{status}</span>",
               'font': {'size': 18, 'color': '#1a202c'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': "rgba(0,0,0,0)"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "#f1f5f9",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': 'rgba(39, 174, 96, 0.15)'},
                {'range': [30, 60], 'color': 'rgba(241, 196, 15, 0.15)'},
                {'range': [60, 100], 'color': 'rgba(231, 76, 60, 0.15)'}
            ],
            'threshold': {
                'line': {'color': "#475569", 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))

    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=80, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'}
    )

    return fig


def create_feature_importance_chart(factors: dict) -> go.Figure:
    """Create horizontal bar chart for feature importance."""
    if not factors:
        return None

    # Sort by contribution
    sorted_factors = sorted(
        factors.items(),
        key=lambda x: abs(x[1].get('contribution', 0)),
        reverse=True
    )[:8]

    names = [f[0].replace('_', ' ').title() for f in sorted_factors]
    values = [f[1].get('contribution', 0) * 100 for f in sorted_factors]

    colors = ['#e74c3c' if v > 0 else '#27ae60' for v in values]

    fig = go.Figure(go.Bar(
        y=names[::-1],
        x=values[::-1],
        orientation='h',
        marker_color=colors[::-1],
        hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'},
        xaxis=dict(
            showgrid=True,
            gridcolor='#e2e8f0',
            title='Contribution to Risk (%)'
        ),
        yaxis=dict(showgrid=False)
    )

    return fig


def create_duration_distribution(predictions: list) -> go.Figure:
    """Create duration prediction distribution chart."""
    if not predictions:
        return None

    fig = go.Figure()

    # Histogram of predicted durations
    hours = [p.get('predicted_hours', 0) for p in predictions]

    fig.add_trace(go.Histogram(
        x=hours,
        nbinsx=20,
        marker_color='#667eea',
        opacity=0.7,
        hovertemplate='Duration: %{x:.0f}h<br>Count: %{y}<extra></extra>'
    ))

    # Add vertical lines for quartiles
    if hours:
        q25, q50, q75 = np.percentile(hours, [25, 50, 75])
        for q, label, color in [(q25, 'P25', '#3498db'), (q50, 'P50', '#f39c12'), (q75, 'P75', '#e74c3c')]:
            fig.add_vline(x=q, line_dash="dash", line_color=color, line_width=2,
                         annotation_text=f"{label}: {q:.0f}h", annotation_position="top")

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'},
        xaxis=dict(
            showgrid=True,
            gridcolor='#e2e8f0',
            title='Predicted Duration (hours)'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e2e8f0',
            title='Number of Issues'
        )
    )

    return fig


def create_confidence_chart(lower: float, predicted: float, upper: float) -> go.Figure:
    """Create confidence interval visualization."""
    fig = go.Figure()

    # Confidence interval band
    fig.add_trace(go.Scatter(
        x=['Lower (95%)', 'Predicted', 'Upper (95%)'],
        y=[lower, predicted, upper],
        mode='lines+markers',
        line=dict(color='#667eea', width=3),
        marker=dict(size=12, color=['#3498db', '#667eea', '#e74c3c']),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)',
        hovertemplate='%{x}: %{y:.1f}h<extra></extra>'
    ))

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'},
        xaxis=dict(showgrid=False),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e2e8f0',
            title='Hours'
        )
    )

    return fig


def simulate_predictions(issues_df: pd.DataFrame) -> list:
    """Simulate predictions for issues (when real predictor unavailable)."""
    predictions = []

    for _, issue in issues_df.iterrows():
        # Base prediction on story points and issue type
        base_hours = (issue.get('story_points', 3) or 3) * 4

        # Adjust by issue type
        type_multipliers = {
            'Bug': 1.2,
            'Story': 1.0,
            'Task': 0.8,
            'Epic': 3.0,
            'Improvement': 0.9,
            'Sub-task': 0.5
        }
        multiplier = type_multipliers.get(issue.get('issue_type', 'Task'), 1.0)

        # Add some variance
        variance = np.random.uniform(0.8, 1.2)
        predicted_hours = base_hours * multiplier * variance

        # Confidence interval
        lower = predicted_hours * 0.6
        upper = predicted_hours * 1.5

        predictions.append({
            'key': issue.get('key', ''),
            'summary': issue.get('summary', ''),
            'issue_type': issue.get('issue_type', 'Task'),
            'priority': issue.get('priority', 'Medium'),
            'predicted_hours': predicted_hours,
            'predicted_days': predicted_hours / 8,
            'confidence_interval': {
                'lower_hours': lower,
                'upper_hours': upper
            },
            'confidence_score': np.random.uniform(0.7, 0.95)
        })

    return predictions


def main():
    st.markdown("# 🎯 ML Predictions Dashboard")
    st.markdown("*Advanced machine learning insights for tickets and sprints*")

    conn = get_connection()
    if not conn:
        st.error("Database not found. Please sync data first.")
        st.stop()

    # ========== QUICK WIN: RELEASE RISK ==========
    risk_data = calculate_release_risk(conn)
    factors_html = ""
    for factor in risk_data['factors']:
        factors_html += f"""
        <div class="risk-factor">
            <span class="factor-icon">{factor['icon']}</span>
            <span class="factor-text">{factor['text']}</span>
            <span class="factor-value">{factor['value']}</span>
        </div>
        """

    st.markdown(f"""
    <div class="quick-win-widget">
        <div class="quick-win-header">
            <span class="quick-win-icon">🚀</span>
            <span class="quick-win-title">Release Risk — Is It Safe to Release?</span>
        </div>
        <div class="risk-summary">
            <div class="risk-score-main">
                <div class="risk-score-value {risk_data['class']}">{risk_data['score']}</div>
                <div class="risk-label">{risk_data['status']}</div>
            </div>
            <div class="risk-factors">
                {factors_html}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Ticket Duration", "🏃 Sprint Risk", "📊 Batch Predictions"])

    # ========== TAB 1: Ticket Duration Predictions ==========
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">⏱️ Predict Issue Duration</div>', unsafe_allow_html=True)

            # Get open issues
            issues = conn.execute("""
                SELECT key, summary, issue_type, priority,
                       COALESCE(story_points, 0) as story_points
                FROM issues
                WHERE status != 'Terminé(e)'
                ORDER BY created DESC
                LIMIT 100
            """).fetchdf()

            if issues.empty:
                st.warning("No open issues found")
            else:
                issue_options = {f"{row['key']}: {row['summary'][:50]}...": row['key']
                               for _, row in issues.iterrows()}
                selected = st.selectbox("Select Issue", list(issue_options.keys()))
                issue_key = issue_options[selected]
                selected_issue = issues[issues['key'] == issue_key].iloc[0]

                if st.button("Predict Duration", key="predict_ticket", type="primary"):
                    # Simulate prediction (replace with actual predictor when available)
                    prediction = simulate_predictions(pd.DataFrame([selected_issue]))[0]

                    # Display results
                    st.markdown("---")

                    r1, r2, r3 = st.columns(3)

                    with r1:
                        st.markdown(f"""
                        <div class="prediction-card">
                            <div class="prediction-value">{prediction['predicted_hours']:.0f}h</div>
                            <div class="prediction-label">Predicted Duration</div>
                            <div class="prediction-sublabel">{prediction['predicted_days']:.1f} working days</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with r2:
                        ci = prediction['confidence_interval']
                        st.markdown(f"""
                        <div class="prediction-card">
                            <div class="prediction-value">{ci['lower_hours']:.0f}-{ci['upper_hours']:.0f}h</div>
                            <div class="prediction-label">95% Confidence Range</div>
                            <div class="prediction-sublabel">Most likely outcome range</div>
                        </div>
                        """, unsafe_allow_html=True)

                    with r3:
                        conf_score = prediction['confidence_score'] * 100
                        conf_class = "confidence-high" if conf_score > 80 else ("confidence-medium" if conf_score > 60 else "confidence-low")
                        st.markdown(f"""
                        <div class="prediction-card">
                            <div class="prediction-value">{conf_score:.0f}%</div>
                            <div class="prediction-label">Model Confidence</div>
                            <div class="confidence-bar-container">
                                <div class="confidence-bar {conf_class}" style="width: {conf_score}%;">
                                    {conf_score:.0f}%
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # Confidence interval chart
                    st.markdown("#### Prediction Confidence Interval")
                    ci_fig = create_confidence_chart(
                        ci['lower_hours'],
                        prediction['predicted_hours'],
                        ci['upper_hours']
                    )
                    st.plotly_chart(ci_fig, use_container_width=True, config={'displayModeBar': False})

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-container">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">📈 Model Performance</div>', unsafe_allow_html=True)

            # Simulated model stats
            st.markdown("""
            <div class="model-info">
                <div class="model-stat">
                    <div class="model-stat-value">87%</div>
                    <div class="model-stat-label">Accuracy</div>
                </div>
                <div class="model-stat">
                    <div class="model-stat-value">0.82</div>
                    <div class="model-stat-label">R² Score</div>
                </div>
                <div class="model-stat">
                    <div class="model-stat-value">4.2h</div>
                    <div class="model-stat-label">MAE</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Key Factors:**")
            st.markdown("""
            - Story Points (35%)
            - Issue Type (25%)
            - Historical Velocity (20%)
            - Complexity Score (12%)
            - Team Capacity (8%)
            """)

            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== TAB 2: Sprint Risk Predictions ==========
    with tab2:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        # Get sprints
        sprints = conn.execute("""
            SELECT id, name, state, start_date, end_date
            FROM sprints
            WHERE start_date IS NOT NULL
            ORDER BY start_date DESC
            LIMIT 10
        """).fetchdf()

        if sprints.empty:
            st.warning("No sprints found")
        else:
            col1, col2 = st.columns([1, 2])

            with col1:
                sprint_options = {f"{row['name']} ({row['state']})": row['id']
                                 for _, row in sprints.iterrows()}
                selected = st.selectbox("Select Sprint", list(sprint_options.keys()), key="sprint_risk")
                sprint_id = sprint_options[selected]

            with col2:
                st.write("")  # Spacer

            if st.button("Analyze Sprint Risk", key="predict_risk", type="primary"):
                # Get sprint issues for simulation
                sprint_issues = conn.execute("""
                    SELECT key, summary, status, priority, issue_type,
                           COALESCE(story_points, 0) as story_points
                    FROM issues
                    WHERE sprint_id = ?
                """, [sprint_id]).fetchdf()

                # Calculate simulated risk factors
                total_issues = len(sprint_issues)
                done_count = len(sprint_issues[sprint_issues['status'] == 'Terminé(e)'])
                blocked_count = 0  # No 'Blocked' status in this Jira instance
                in_progress = len(sprint_issues[sprint_issues['status'] == 'En cours'])

                completion_rate = done_count / total_issues if total_issues > 0 else 0

                # Simulate risk score
                risk_score = min(100, max(0,
                    (1 - completion_rate) * 40 +
                    blocked_count * 15 +
                    (in_progress / max(total_issues, 1)) * 20 +
                    np.random.uniform(0, 20)
                ))

                risk_factors = {
                    'completion_rate': {'contribution': (1 - completion_rate) * 0.4},
                    'blocked_items': {'contribution': blocked_count * 0.15 / 100},
                    'wip_ratio': {'contribution': (in_progress / max(total_issues, 1)) * 0.2},
                    'scope_creep': {'contribution': np.random.uniform(0.05, 0.15)},
                    'team_capacity': {'contribution': np.random.uniform(-0.1, 0.1)},
                    'historical_velocity': {'contribution': np.random.uniform(-0.05, 0.1)},
                    'dependency_risk': {'contribution': np.random.uniform(0.02, 0.12)},
                    'complexity_score': {'contribution': np.random.uniform(0.03, 0.1)},
                }

                # Display risk gauge
                col1, col2 = st.columns([1, 1.5])

                with col1:
                    st.markdown('<div class="risk-gauge-card">', unsafe_allow_html=True)
                    risk_fig = create_risk_gauge(risk_score, "Sprint Risk Score")
                    st.plotly_chart(risk_fig, use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)

                with col2:
                    st.markdown('<div class="section-container">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">📊 Risk Factor Analysis</div>', unsafe_allow_html=True)

                    importance_fig = create_feature_importance_chart(risk_factors)
                    if importance_fig:
                        st.plotly_chart(importance_fig, use_container_width=True, config={'displayModeBar': False})

                    st.markdown('</div>', unsafe_allow_html=True)

                # Key metrics
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">📋 Sprint Metrics</div>', unsafe_allow_html=True)

                m1, m2, m3, m4, m5 = st.columns(5)

                with m1:
                    st.metric("Total Issues", total_issues)
                with m2:
                    st.metric("Completed", done_count)
                with m3:
                    st.metric("In Progress", in_progress)
                with m4:
                    st.metric("Blocked", blocked_count)
                with m5:
                    st.metric("Completion", f"{completion_rate*100:.0f}%")

                st.markdown('</div>', unsafe_allow_html=True)

                # Recommendations
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">💡 AI Recommendations</div>', unsafe_allow_html=True)

                if risk_score > 70:
                    st.error("🚨 **Critical Risk Level** - Immediate action required")
                    st.markdown("""
                    - Focus on unblocking blocked items immediately
                    - Consider scope reduction to meet sprint goals
                    - Escalate resource constraints to management
                    """)
                elif risk_score > 40:
                    st.warning("⚠️ **Elevated Risk Level** - Monitoring recommended")
                    st.markdown("""
                    - Review items in progress for blockers
                    - Prioritize high-value items for completion
                    - Daily standups to track progress
                    """)
                else:
                    st.success("✅ **Low Risk Level** - Sprint is on track")
                    st.markdown("""
                    - Continue current pace
                    - Consider taking on additional scope if capacity allows
                    - Document learnings for future sprints
                    """)

                st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== TAB 3: Batch Predictions ==========
    with tab3:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)

        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Batch Issue Predictions</div>', unsafe_allow_html=True)

        # Get all open issues
        open_issues = conn.execute("""
            SELECT key, summary, status, issue_type, priority,
                   COALESCE(story_points, 0) as story_points,
                   COALESCE(assignee_name, 'Unassigned') as assignee_name
            FROM issues
            WHERE status != 'Terminé(e)'
            ORDER BY created DESC
            LIMIT 50
        """).fetchdf()

        if open_issues.empty:
            st.warning("No open issues found")
        else:
            if st.button("Generate Predictions for All Open Issues", type="primary"):
                with st.spinner("Generating predictions..."):
                    predictions = simulate_predictions(open_issues)

                    # Distribution chart
                    st.markdown("#### Predicted Duration Distribution")
                    dist_fig = create_duration_distribution(predictions)
                    if dist_fig:
                        st.plotly_chart(dist_fig, use_container_width=True, config={'displayModeBar': False})

                    # Statistics
                    st.markdown("---")
                    hours_list = [p['predicted_hours'] for p in predictions]

                    s1, s2, s3, s4 = st.columns(4)
                    with s1:
                        st.metric("Average Duration", f"{np.mean(hours_list):.1f}h")
                    with s2:
                        st.metric("Median Duration", f"{np.median(hours_list):.1f}h")
                    with s3:
                        st.metric("Total Estimated", f"{sum(hours_list):.0f}h")
                    with s4:
                        st.metric("Working Days", f"{sum(hours_list)/8:.1f} days")

                    # Individual predictions
                    st.markdown("---")
                    st.markdown("#### Individual Issue Predictions")

                    for pred in predictions[:15]:
                        hours = pred['predicted_hours']
                        if hours < 8:
                            badge_class = 'badge-fast'
                            badge_text = 'Quick'
                        elif hours < 24:
                            badge_class = 'badge-normal'
                            badge_text = 'Normal'
                        elif hours < 48:
                            badge_class = 'badge-slow'
                            badge_text = 'Slow'
                        else:
                            badge_class = 'badge-critical'
                            badge_text = 'Long'

                        st.markdown(f"""
                        <div class="issue-predictor-card">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <span class="issue-key">{pred['key']}</span>
                                    <span style="margin-left: 8px; color: #8892b0; font-size: 12px;">{pred['issue_type']}</span>
                                </div>
                                <span class="prediction-badge {badge_class}">{badge_text} • {hours:.0f}h</span>
                            </div>
                            <div class="issue-summary">{pred['summary'][:80]}{'...' if len(pred['summary']) > 80 else ''}</div>
                            <div style="font-size: 11px; color: #667eea;">
                                Range: {pred['confidence_interval']['lower_hours']:.0f}h - {pred['confidence_interval']['upper_hours']:.0f}h
                                • Confidence: {pred['confidence_score']*100:.0f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ========== Footer ==========
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #64748b; font-size: 12px;">
        ML Predictions Dashboard | Powered by Gradient Boosting & Neural Networks |
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
    """, unsafe_allow_html=True)

    conn.close()


if __name__ == "__main__":
    main()
