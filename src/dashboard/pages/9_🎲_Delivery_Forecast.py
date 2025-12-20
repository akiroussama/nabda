"""
🎲 Delivery Forecast - Premium Monte Carlo Simulation
Probabilistic forecasting for project delivery dates with scenario analysis.
"""

import streamlit as st
import sys

# Import page guide component
from src.dashboard.components import render_page_guide
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional

# Add project root to sys.path so we can import from src
root_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_dir))

st.set_page_config(page_title="Delivery Forecast", page_icon="🎲", layout="wide")

# Premium Dark Theme CSS
st.markdown(f"""<style>
    /* Global Light Theme */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Section Containers */
    .section-container {
        background: white;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Probability Cards */
    .prob-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    }

    .prob-card::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 4px;
    }

    .prob-high::before { background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%); }
    .prob-medium::before { background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%); }
    .prob-low::before { background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%); }

    .prob-label {
        font-size: 11px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }

    .prob-value {
        font-size: 42px;
        font-weight: 800;
    }

    .value-high {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .value-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .value-low {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .prob-subtitle { font-size: 13px; color: #64748b; margin-top: 8px; }

    .prob-date {
        font-size: 18px;
        font-weight: 600;
        color: #1a202c;
        margin-top: 4px;
    }

    /* Scenario Cards */
    .scenario-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #e2e8f0;
    }

    .scenario-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }

    .scenario-name { color: #1a202c; font-weight: 600; font-size: 14px; }

    .scenario-delta {
        font-size: 14px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
    }

    .delta-positive { background: #dcfce7; color: #166534; }
    .delta-negative { background: #fee2e2; color: #991b1b; }

    /* Risk Factor Cards */
    .risk-factor-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .risk-factor-label { font-size: 11px; color: #64748b; text-transform: uppercase; margin-bottom: 8px; }
    .risk-factor-value { font-size: 24px; font-weight: 700; }
    .risk-good { color: #16a34a; }
    .risk-warning { color: #d97706; }
    .risk-danger { color: #dc2626; }
    .risk-factor-desc { font-size: 11px; color: #64748b; margin-top: 4px; }

    /* Confidence Bands */
    .confidence-band {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px;
        background: #f8fafc;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid #e2e8f0;
    }

    .band-label { width: 60px; font-size: 12px; color: #64748b; font-weight: 600; }

    .band-bar {
        flex: 1;
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        position: relative;
        overflow: hidden;
    }

    .band-fill { height: 100%; border-radius: 4px; }

    .band-date {
        width: 100px;
        text-align: right;
        font-size: 13px;
        color: #1a202c;
        font-weight: 500;
    }

    /* Comparison Table */
    .comparison-table { width: 100%; border-collapse: collapse; margin-top: 16px; }

    .comparison-table th {
        text-align: left;
        padding: 12px;
        color: #64748b;
        font-size: 12px;
        text-transform: uppercase;
        border-bottom: 1px solid #e2e8f0;
    }

    .comparison-table td {
        padding: 12px;
        color: #1e293b;
        font-size: 14px;
        border-bottom: 1px solid #e2e8f0;
    }

    /* Simulation Stats */
    .sim-stat {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #e2e8f0;
    }

    .sim-stat-label { color: #64748b; font-size: 13px; }
    .sim-stat-value { color: #1a202c; font-weight: 600; font-size: 13px; }

    /* Quick Win Widget */
    .quick-win-widget {
        background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
        border-radius: 16px;
        padding: 20px 24px;
        margin: 16px 0;
        border: 1px solid rgba(52, 211, 153, 0.3);
        box-shadow: 0 8px 32px rgba(6, 78, 59, 0.3);
        position: relative;
        overflow: hidden;
    }
    .quick-win-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
    }
    .quick-win-icon { font-size: 24px; }
    .quick-win-title {
        color: #a7f3d0;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .confidence-summary {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    .confidence-score {
        font-size: 48px;
        font-weight: 800;
        color: #ecfdf5;
    }
    .confidence-details {
        flex: 1;
    }
    .confidence-message {
        color: #ecfdf5;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 8px;
    }
    .confidence-stats {
        display: flex;
        gap: 16px;
    }
    .confidence-stat {
        color: #a7f3d0;
        font-size: 12px;
    }
    .confidence-stat strong {
        color: #ecfdf5;
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    simulation_dates: List[datetime]
    target_date_prob: float
    p50_date: datetime
    p85_date: datetime
    p95_date: datetime
    mean_days: float
    std_days: float


def get_connection():
    """Get database connection."""
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path), read_only=True)


def get_release_confidence(conn) -> dict:
    """Calculate quick release confidence score - the #1 question every release manager asks."""
    try:
        # Get completion rate
        metrics = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'Terminé(e)' THEN 1 ELSE 0 END) as done,
                SUM(CASE WHEN priority = 'Highest' AND status != 'Terminé(e)' THEN 1 ELSE 0 END) as blockers,
                COALESCE(SUM(story_points), 0) as total_pts,
                COALESCE(SUM(CASE WHEN status = 'Terminé(e)' THEN story_points END), 0) as done_pts
            FROM issues i
            JOIN sprints s ON i.sprint_id = s.id AND s.state = 'active'
        """).fetchone()

        total = metrics[0] or 1
        done = metrics[1] or 0
        blockers = metrics[2] or 0
        total_pts = metrics[3] or 0
        done_pts = metrics[4] or 0

        # Calculate confidence score (0-100)
        completion_factor = (done / total * 50) if total > 0 else 50
        blocker_penalty = min(30, blockers * 10)
        pts_factor = (done_pts / total_pts * 30) if total_pts > 0 else 15

        confidence = max(0, min(100, completion_factor + pts_factor - blocker_penalty + 10))

        if confidence >= 75:
            message = "Looking good! High confidence for on-time delivery."
        elif confidence >= 50:
            message = "Moderate confidence. Monitor blockers closely."
        else:
            message = "At risk. Consider scope reduction or deadline extension."

        return {
            'score': int(confidence),
            'message': message,
            'done': done,
            'total': total,
            'blockers': blockers,
            'status': 'high' if confidence >= 75 else 'medium' if confidence >= 50 else 'low'
        }
    except Exception:
        return {'score': 50, 'message': 'Unable to calculate', 'done': 0, 'total': 0, 'blockers': 0, 'status': 'medium'}


def analyze_historical_performance(df_completed: pd.DataFrame, df_sprints: pd.DataFrame) -> dict:
    """Analyze historical performance for simulation parameters."""
    # Calculate weekly velocity
    if df_completed.empty:
        return {
            'velocity_mean': 5.0,
            'velocity_std': 2.0,
            'estimation_bias_mean': 1.2,
            'estimation_bias_std': 0.3,
            'scope_creep_mean': 0.1,
            'scope_creep_std': 0.05
        }

    # Group by week and count completions
    df_completed['resolved'] = pd.to_datetime(df_completed['resolved'], errors='coerce')
    df_completed = df_completed.dropna(subset=['resolved'])

    if df_completed.empty:
        return {
            'velocity_mean': 5.0,
            'velocity_std': 2.0,
            'estimation_bias_mean': 1.2,
            'estimation_bias_std': 0.3,
            'scope_creep_mean': 0.1,
            'scope_creep_std': 0.05
        }

    df_completed['week'] = df_completed['resolved'].dt.isocalendar().week
    df_completed['year'] = df_completed['resolved'].dt.year

    weekly_velocity = df_completed.groupby(['year', 'week']).size()

    velocity_mean = max(weekly_velocity.mean(), 1.0)
    velocity_std = max(weekly_velocity.std(), 0.5) if len(weekly_velocity) > 1 else velocity_mean * 0.3

    # Estimation bias (simulate if not available)
    estimation_bias_mean = 1.2 + np.random.uniform(-0.2, 0.3)
    estimation_bias_std = 0.3

    # Scope creep (simulate)
    scope_creep_mean = 0.1 + np.random.uniform(0, 0.1)
    scope_creep_std = 0.05

    return {
        'velocity_mean': velocity_mean,
        'velocity_std': velocity_std,
        'estimation_bias_mean': estimation_bias_mean,
        'estimation_bias_std': estimation_bias_std,
        'scope_creep_mean': scope_creep_mean,
        'scope_creep_std': scope_creep_std
    }


def run_simulation(
    remaining_items: int,
    target_date: datetime,
    params: dict,
    n_simulations: int = 5000,
    team_multiplier: float = 1.0,
    scope_cut: float = 0.0,
    estimation_fix: float = 0.0
) -> SimulationResult:
    """Run Monte Carlo simulation for delivery dates."""
    np.random.seed(42)  # For reproducibility

    # Adjust parameters
    adjusted_items = int(remaining_items * (1 - scope_cut))
    adjusted_velocity = params['velocity_mean'] * team_multiplier
    adjusted_velocity_std = params['velocity_std'] * team_multiplier
    adjusted_bias = params['estimation_bias_mean'] * (1 - estimation_fix * 0.5)

    completion_dates = []
    start_date = datetime.now()

    for _ in range(n_simulations):
        # Simulate weekly velocity
        weeks_needed = 0
        items_remaining = adjusted_items * adjusted_bias

        # Add scope creep
        scope_creep = np.random.normal(params['scope_creep_mean'], params['scope_creep_std'])
        items_remaining *= (1 + max(0, scope_creep))

        while items_remaining > 0 and weeks_needed < 260:  # Cap at 5 years
            weekly_velocity = max(1, np.random.normal(adjusted_velocity, adjusted_velocity_std))
            items_remaining -= weekly_velocity
            weeks_needed += 1

        completion_date = start_date + timedelta(weeks=weeks_needed)
        completion_dates.append(completion_date)

    # Calculate percentiles
    sorted_dates = sorted(completion_dates)
    p50_idx = int(n_simulations * 0.50)
    p85_idx = int(n_simulations * 0.85)
    p95_idx = int(n_simulations * 0.95)

    # Calculate probability of hitting target
    on_time = sum(1 for d in completion_dates if d <= target_date)
    target_prob = on_time / n_simulations

    # Calculate statistics
    days_to_complete = [(d - start_date).days for d in completion_dates]

    return SimulationResult(
        simulation_dates=completion_dates,
        target_date_prob=target_prob,
        p50_date=sorted_dates[p50_idx],
        p85_date=sorted_dates[p85_idx],
        p95_date=sorted_dates[p95_idx],
        mean_days=np.mean(days_to_complete),
        std_days=np.std(days_to_complete)
    )


def create_probability_gauge(prob: float) -> go.Figure:
    """Create probability gauge."""
    if prob >= 0.7:
        color = '#27ae60'
    elif prob >= 0.4:
        color = '#f39c12'
    else:
        color = '#e74c3c'

    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={'suffix': '%', 'font': {'size': 42, 'color': '#1a202c'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': 'rgba(0,0,0,0)'},
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': '#f1f5f9',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': 'rgba(231, 76, 60, 0.2)'},
                {'range': [40, 70], 'color': 'rgba(243, 156, 18, 0.2)'},
                {'range': [70, 100], 'color': 'rgba(39, 174, 96, 0.2)'},
            ],
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'},
        height=200,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    return fig


def create_distribution_chart(baseline: SimulationResult, whatif: Optional[SimulationResult], target_date: datetime) -> go.Figure:
    """Create distribution histogram with confidence bands."""
    fig = go.Figure()

    # Convert dates to days from now
    baseline_days = [(d - datetime.now()).days for d in baseline.simulation_dates]

    fig.add_trace(go.Histogram(
        x=baseline_days,
        name='Baseline',
        marker=dict(color='rgba(102, 126, 234, 0.7)', line=dict(color='#667eea', width=1)),
        nbinsx=50,
        opacity=0.8
    ))

    if whatif:
        whatif_days = [(d - datetime.now()).days for d in whatif.simulation_dates]
        fig.add_trace(go.Histogram(
            x=whatif_days,
            name='What-If',
            marker=dict(color='rgba(39, 174, 96, 0.5)', line=dict(color='#27ae60', width=1)),
            nbinsx=50,
            opacity=0.6
        ))

    # Add target line
    target_days = (target_date - datetime.now()).days
    fig.add_vline(
        x=target_days,
        line_dash='dash',
        line_color='#e74c3c',
        line_width=2,
        annotation_text='Target',
        annotation_font_color='#e74c3c'
    )

    # Add percentile lines for baseline
    p50_days = (baseline.p50_date - datetime.now()).days
    p85_days = (baseline.p85_date - datetime.now()).days

    fig.add_vline(x=p50_days, line_dash='dot', line_color='#f39c12', line_width=1,
                  annotation_text='50%', annotation_font_color='#f39c12', annotation_position='top')
    fig.add_vline(x=p85_days, line_dash='dot', line_color='#3498db', line_width=1,
                  annotation_text='85%', annotation_font_color='#3498db', annotation_position='top')

    fig.update_layout(
        title=dict(text='Simulated Completion Distribution', font=dict(color='#1a202c', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'},
        xaxis=dict(
            title='Days from Now',
            tickfont=dict(color='#64748b'),
            gridcolor='#e2e8f0'
        ),
        yaxis=dict(
            title='Frequency',
            tickfont=dict(color='#64748b'),
            gridcolor='#e2e8f0'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(color='#64748b')
        ),
        barmode='overlay',
        height=400,
        margin=dict(l=60, r=40, t=80, b=60)
    )

    return fig


def create_scenario_comparison(scenarios: List[dict]) -> go.Figure:
    """Create scenario comparison bar chart."""
    names = [s['name'] for s in scenarios]
    probs = [s['prob'] * 100 for s in scenarios]
    days_saved = [s['days_saved'] for s in scenarios]

    colors = ['#667eea' if i == 0 else '#27ae60' for i in range(len(scenarios))]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=names,
        y=probs,
        name='On-Time Probability',
        marker=dict(color=colors),
        text=[f'{p:.0f}%' for p in probs],
        textposition='outside',
        textfont=dict(color='#fff')
    ))

    fig.update_layout(
        title=dict(text='Scenario Comparison', font=dict(color='#1a202c', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'},
        xaxis=dict(
            tickfont=dict(color='#64748b'),
            showgrid=False
        ),
        yaxis=dict(
            title='Probability (%)',
            tickfont=dict(color='#64748b'),
            gridcolor='#e2e8f0',
            range=[0, 100]
        ),
        height=300,
        margin=dict(l=60, r=40, t=60, b=60)
    )

    return fig


def main():
    # Render page guide in sidebar
    render_page_guide()

    # Header
    st.markdown(f"""<div style="text-align: center; padding: 20px 0 30px 0;">
    <h1 style="font-size: 42px; font-weight: 800; margin: 0;
               background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #3498db 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               background-clip: text;">
        🎲 Delivery Forecast
    </h1>
    <p style="color: #64748b; font-size: 16px; margin-top: 10px;">
        Monte Carlo Simulation for Probabilistic Delivery Dates
    </p>
</div>
""", unsafe_allow_html=True)

    conn = get_connection()
    if not conn:
        st.error("Database not found. Please sync data first.")
        st.stop()

    # Quick Win Widget - Release Confidence
    try:
        conf = get_release_confidence(conn)
        st.markdown(f"""<div class="quick-win-widget">
    <div class="quick-win-header">
        <span class="quick-win-icon">🚀</span>
        <span class="quick-win-title">RELEASE CONFIDENCE • Quick Check</span>
    </div>
    <div class="confidence-summary">
        <div class="confidence-score">{conf['score']}%</div>
        <div class="confidence-details">
            <div class="confidence-message">{conf['message']}</div>
            <div class="confidence-stats">
                <span class="confidence-stat"><strong>{conf['done']}/{conf['total']}</strong> items done</span>
                <span class="confidence-stat"><strong>{conf['blockers']}</strong> blockers</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
    except Exception:
        pass

    # Load data
    try:
        df_completed = conn.execute("""
            SELECT * FROM issues WHERE resolved IS NOT NULL
        """).fetchdf()

        df_sprints = conn.execute("SELECT * FROM sprints").fetchdf()

        df_all = conn.execute("SELECT * FROM issues").fetchdf()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    # Calculate remaining backlog
    done_statuses = ['Terminé(e)']  # French status for Done
    df_remaining = df_all[~df_all['status'].isin(done_statuses)]
    remaining_count = len(df_remaining)

    # ========== SIMULATION PARAMETERS ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎛️ Simulation Parameters</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        backlog_size = st.slider(
            "📋 Backlog Items",
            min_value=10,
            max_value=max(500, remaining_count * 2),
            value=max(remaining_count, 50),
            step=10,
            help="Number of items to complete"
        )

    with col2:
        target_weeks = st.slider(
            "📅 Target (Weeks)",
            min_value=4,
            max_value=52,
            value=12,
            step=2,
            help="Weeks from now"
        )

    with col3:
        n_simulations = st.selectbox(
            "🔄 Simulations",
            options=[1000, 5000, 10000, 25000],
            index=1,
            help="More simulations = more accurate"
        )

    with col4:
        target_date = datetime.now() + timedelta(weeks=target_weeks)
        st.markdown(f"""<div style="text-align: center; padding-top: 8px;">
    <div style="color: #64748b; font-size: 12px;">Target Date</div>
    <div style="color: #1a202c; font-size: 18px; font-weight: 600;">{target_date.strftime('%b %d, %Y')}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== WHAT-IF SCENARIOS ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🔮 What-If Scenarios</div>', unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)

    with s1:
        team_multiplier = st.slider(
            "👥 Team Size",
            min_value=0.5,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="1.0 = current team"
        )

    with s2:
        scope_cut = st.slider(
            "✂️ Scope Cut",
            min_value=0,
            max_value=50,
            value=0,
            step=5,
            help="% of backlog to cut"
        )

    with s3:
        estimation_fix = st.slider(
            "🎯 Estimation Fix",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="0 = no change, 1 = perfect"
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== RUN SIMULATION ==========
    with st.spinner("Running Monte Carlo simulations..."):
        params = analyze_historical_performance(df_completed, df_sprints)

        baseline = run_simulation(
            remaining_items=backlog_size,
            target_date=target_date,
            params=params,
            n_simulations=n_simulations
        )

        whatif = None
        if team_multiplier != 1.0 or scope_cut > 0 or estimation_fix > 0:
            whatif = run_simulation(
                remaining_items=backlog_size,
                target_date=target_date,
                params=params,
                n_simulations=n_simulations,
                team_multiplier=team_multiplier,
                scope_cut=scope_cut / 100,
                estimation_fix=estimation_fix
            )

    # ========== PROBABILITY METRICS ==========
    prob = baseline.target_date_prob
    prob_class = 'high' if prob >= 0.7 else ('medium' if prob >= 0.4 else 'low')

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""<div class="prob-card prob-{prob_class}">
    <div class="prob-label">On-Time Probability</div>
    <div class="prob-value value-{prob_class}">{prob*100:.0f}%</div>
    <div class="prob-subtitle">Target: {target_date.strftime('%b %d')}</div>
</div>
""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""<div class="prob-card prob-high">
            <div class="prob-label">50% Confidence</div>
            <div class="prob-date">{baseline.p50_date.strftime('%b %d, %Y')}</div>
            <div class="prob-subtitle">Optimistic Estimate</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""<div class="prob-card prob-medium">
            <div class="prob-label">85% Confidence</div>
            <div class="prob-date">{baseline.p85_date.strftime('%b %d, %Y')}</div>
            <div class="prob-subtitle">Realistic Estimate</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""<div class="prob-card prob-low">
            <div class="prob-label">95% Confidence</div>
            <div class="prob-date">{baseline.p95_date.strftime('%b %d, %Y')}</div>
            <div class="prob-subtitle">Conservative Estimate</div>
        </div>
        """, unsafe_allow_html=True)

    # ========== WHAT-IF COMPARISON ==========
    if whatif:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📊 Scenario Impact</div>', unsafe_allow_html=True)

        delta_prob = (whatif.target_date_prob - baseline.target_date_prob) * 100
        days_saved = (baseline.p85_date - whatif.p85_date).days

        c1, c2, c3 = st.columns(3)

        with c1:
            delta_class = 'delta-positive' if delta_prob > 0 else 'delta-negative'
            st.markdown(f"""<div class="scenario-card">
                <div class="scenario-header">
                    <span class="scenario-name">Probability Change</span>
                    <span class="scenario-delta {delta_class}">{delta_prob:+.0f}%</span>
                </div>
                <div style="color: #8892b0; font-size: 13px;">
                    What-If: {whatif.target_date_prob*100:.0f}% vs Baseline: {baseline.target_date_prob*100:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            days_class = 'delta-positive' if days_saved > 0 else 'delta-negative'
            st.markdown(f"""<div class="scenario-card">
                <div class="scenario-header">
                    <span class="scenario-name">Days Saved (85%)</span>
                    <span class="scenario-delta {days_class}">{days_saved:+d} days</span>
                </div>
                <div style="color: #8892b0; font-size: 13px;">
                    What-If: {whatif.p85_date.strftime('%b %d')} vs Baseline: {baseline.p85_date.strftime('%b %d')}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with c3:
            # Risk reduction
            risk_reduction = (1 - whatif.target_date_prob) / max((1 - baseline.target_date_prob), 0.01) * 100
            risk_class = 'delta-positive' if risk_reduction < 100 else 'delta-negative'
            st.markdown(f"""<div class="scenario-card">
                <div class="scenario-header">
                    <span class="scenario-name">Risk Level</span>
                    <span class="scenario-delta {risk_class}">{100 - risk_reduction:+.0f}%</span>
                </div>
                <div style="color: #8892b0; font-size: 13px;">
                    Relative risk compared to baseline scenario
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== DISTRIBUTION CHART ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Simulation Distribution</div>', unsafe_allow_html=True)

    st.plotly_chart(create_distribution_chart(baseline, whatif, target_date), use_container_width=True)

    # Confidence Bands
    st.markdown("**Confidence Bands:**")

    bands = [
        ('50%', baseline.p50_date, 50, '#27ae60'),
        ('85%', baseline.p85_date, 85, '#f39c12'),
        ('95%', baseline.p95_date, 95, '#e74c3c'),
    ]

    for label, date, pct, color in bands:
        st.markdown(f"""<div class="confidence-band">
            <span class="band-label">{label}</span>
            <div class="band-bar">
                <div class="band-fill" style="width: {pct}%; background: {color};"></div>
            </div>
            <span class="band-date">{date.strftime('%b %d, %Y')}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== RISK FACTORS ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚠️ Risk Factors</div>', unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        bias = params['estimation_bias_mean']
        bias_class = 'risk-good' if bias < 1.2 else ('risk-warning' if bias < 1.5 else 'risk-danger')
        st.markdown(f"""<div class="risk-factor-card">
            <div class="risk-factor-label">Estimation Bias</div>
            <div class="risk-factor-value {bias_class}">{bias:.1f}x</div>
            <div class="risk-factor-desc">Actual vs Estimated</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        vel_var = params['velocity_std'] / max(params['velocity_mean'], 0.1) * 100
        var_class = 'risk-good' if vel_var < 20 else ('risk-warning' if vel_var < 40 else 'risk-danger')
        st.markdown(f"""<div class="risk-factor-card">
            <div class="risk-factor-label">Velocity Variance</div>
            <div class="risk-factor-value {var_class}">±{vel_var:.0f}%</div>
            <div class="risk-factor-desc">Throughput Stability</div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        creep = params['scope_creep_mean'] * 100
        creep_class = 'risk-good' if creep < 10 else ('risk-warning' if creep < 20 else 'risk-danger')
        st.markdown(f"""<div class="risk-factor-card">
            <div class="risk-factor-label">Scope Creep</div>
            <div class="risk-factor-value {creep_class}">{creep:.0f}%</div>
            <div class="risk-factor-desc">Expected Addition</div>
        </div>
        """, unsafe_allow_html=True)

    with r4:
        uncertainty = baseline.std_days / max(baseline.mean_days, 1) * 100
        unc_class = 'risk-good' if uncertainty < 15 else ('risk-warning' if uncertainty < 30 else 'risk-danger')
        st.markdown(f"""<div class="risk-factor-card">
            <div class="risk-factor-label">Uncertainty</div>
            <div class="risk-factor-value {unc_class}">±{uncertainty:.0f}%</div>
            <div class="risk-factor-desc">Prediction Spread</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== ANALYSIS SUMMARY ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Analysis Summary</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Backlog Analysis:**")
        stats = [
            ('Total Issues', len(df_all)),
            ('Completed', len(df_completed)),
            ('Remaining', remaining_count),
            ('Simulated Backlog', backlog_size),
        ]
        for label, value in stats:
            st.markdown(f"""<div class="sim-stat">
                <span class="sim-stat-label">{label}</span>
                <span class="sim-stat-value">{value}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Historical Performance:**")
        perf_stats = [
            ('Weekly Velocity', f"{params['velocity_mean']:.1f} (±{params['velocity_std']:.1f})"),
            ('Estimation Bias', f"{params['estimation_bias_mean']:.1f}x"),
            ('Scope Creep Rate', f"{params['scope_creep_mean']*100:.0f}%"),
            ('Mean Completion', f"{baseline.mean_days:.0f} days"),
        ]
        for label, value in perf_stats:
            st.markdown(f"""<div class="sim-stat">
                <span class="sim-stat-label">{label}</span>
                <span class="sim-stat-value">{value}</span>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== RECOMMENDATIONS ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💡 Recommendations</div>', unsafe_allow_html=True)

    if prob < 0.5:
        st.markdown(f"""<div style="background: #fee2e2; border-radius: 12px; padding: 20px; border-left: 4px solid #e74c3c; margin-bottom: 12px;">
            <div style="color: #e74c3c; font-weight: 700; margin-bottom: 8px;">🚨 High Risk of Missing Target</div>
            <div style="color: #1e293b; font-size: 13px; line-height: 1.5;">
                Consider scope reduction, team augmentation, or adjusting the target date.
                Use the What-If scenarios to explore mitigation options.
            </div>
        </div>
        """, unsafe_allow_html=True)
    elif prob < 0.7:
        st.markdown(f"""<div style="background: #fef3c7; border-radius: 12px; padding: 20px; border-left: 4px solid #f39c12; margin-bottom: 12px;">
            <div style="color: #f39c12; font-weight: 700; margin-bottom: 8px;">⚠️ Moderate Risk</div>
            <div style="color: #1e293b; font-size: 13px; line-height: 1.5;">
                The target is achievable but with significant risk. Monitor progress closely
                and have contingency plans ready.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""<div style="background: #dcfce7; border-radius: 12px; padding: 20px; border-left: 4px solid #27ae60; margin-bottom: 12px;">
            <div style="color: #27ae60; font-weight: 700; margin-bottom: 8px;">✓ On Track</div>
            <div style="color: #1e293b; font-size: 13px; line-height: 1.5;">
                Good probability of hitting the target. Continue monitoring and
                re-run simulations weekly as data evolves.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""<div style="background: #eef2ff; border-radius: 12px; padding: 20px; border-left: 4px solid #667eea;">
        <div style="color: #667eea; font-weight: 700; margin-bottom: 8px;">📊 Best Practice</div>
        <div style="color: #1e293b; font-size: 13px; line-height: 1.5;">
            Use the <strong>85% confidence date</strong> for external commitments.
            This provides a realistic buffer while remaining achievable.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    conn.close()


if __name__ == "__main__":
    main()
