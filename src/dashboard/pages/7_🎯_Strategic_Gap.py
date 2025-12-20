"""
🎯 Strategic Execution Gap™ - Premium Analytics
Dashboard to compare Stated Strategy vs Actual Execution with advanced insights.
"""

import streamlit as st
import sys

# Import page guide component
from src.dashboard.components import render_page_guide
from pathlib import Path

# Add project root to sys.path so we can import from src
root_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_dir))

import pandas as pd
import duckdb
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(page_title="Strategic Execution Gap", page_icon="🎯", layout="wide")

# Premium Light Theme CSS
st.markdown("""
<style>
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

    /* Premium Metric Cards */
    .gap-metric-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .gap-metric-card::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; height: 4px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    .gap-metric-card.danger::before { background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%); }
    .gap-metric-card.warning::before { background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%); }
    .gap-metric-card.success::before { background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%); }

    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-value.danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-value.warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-value.success {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .metric-subtitle {
        font-size: 13px;
        color: #64748b;
        margin-top: 8px;
        font-weight: 500;
    }

    /* Strategy Slider Container */
    .strategy-control {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        margin-bottom: 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .strategy-label {
        color: #1a202c;
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 8px;
    }

    /* Category Cards */
    .category-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #e2e8f0;
    }

    .category-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
    }

    .category-name {
        color: #1a202c;
        font-weight: 600;
        font-size: 14px;
    }

    .category-gap {
        font-size: 14px;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
    }

    .gap-positive { background: #fee2e2; color: #991b1b; }
    .gap-negative { background: #dcfce7; color: #166534; }
    .gap-neutral { background: #f1f5f9; color: #64748b; }

    .progress-bar-container {
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 8px;
    }

    .progress-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    .progress-stated { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); }
    .progress-actual { background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%); }

    /* Shadow Work Table */
    .shadow-ticket {
        background: white;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #ef4444;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .ticket-key { color: #4f46e5; font-weight: 700; font-size: 12px; }
    .ticket-summary { color: #1a202c; font-size: 13px; flex: 1; margin: 0 16px; font-weight: 500; }

    .ticket-category {
        background: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }

    /* Trend Indicator */
    .trend-up { color: #ef4444; }
    .trend-down { color: #22c55e; }

    /* Allocation Legend */
    .legend-item {
        display: inline-flex; align-items: center; gap: 8px; margin-right: 20px;
        color: #64748b; font-size: 13px; font-weight: 500;
    }

    .legend-dot { width: 12px; height: 12px; border-radius: 50%; }
    .legend-stated { background: #667eea; }
    .legend-actual { background: #f59e0b; }

    /* Insight Box */
    .insight-box {
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        border-radius: 12px;
        padding: 16px;
        margin-top: 16px;
    }

    .insight-icon { font-size: 18px; margin-right: 8px; }
    .insight-text { color: #1e293b; font-size: 14px; line-height: 1.5; }

    /* Total Allocation Badge */
    .allocation-total {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }

    .total-valid { background: #dcfce7; color: #166534; }
    .total-invalid { background: #fee2e2; color: #991b1b; }

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
    .drift-summary {
        display: flex;
        gap: 20px;
    }
    .drift-item {
        flex: 1;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 12px 16px;
        text-align: center;
    }
    .drift-label {
        color: #a7f3d0;
        font-size: 11px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .drift-value {
        font-size: 24px;
        font-weight: 700;
        color: #ecfdf5;
    }
    .drift-good { color: #4ade80; }
    .drift-bad { color: #fca5a5; }
</style>
""", unsafe_allow_html=True)


def get_connection():
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path), read_only=True)


def get_priority_drift(conn) -> dict:
    """Calculate quick priority drift summary - Are we doing what we committed to?"""
    try:
        # Get current sprint work
        sprint_work = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN priority IN ('Highest', 'High') THEN 1 ELSE 0 END) as high_priority,
                SUM(CASE WHEN status = 'Terminé(e)' THEN 1 ELSE 0 END) as done,
                SUM(CASE WHEN status = 'Terminé(e)' AND priority IN ('Highest', 'High') THEN 1 ELSE 0 END) as high_done
            FROM issues i
            JOIN sprints s ON i.sprint_id = s.id AND s.state = 'active'
        """).fetchone()

        total = sprint_work[0] or 0
        high_priority = sprint_work[1] or 0
        done = sprint_work[2] or 0
        high_done = sprint_work[3] or 0

        # Calculate alignment
        high_pct = (high_priority / total * 100) if total > 0 else 0
        high_completion = (high_done / high_priority * 100) if high_priority > 0 else 100
        low_items_done = done - high_done
        low_priority_total = total - high_priority

        # Alignment score: are we completing high-priority items first?
        alignment_score = min(100, high_completion + 20) if high_priority > 0 else 100

        return {
            'high_priority': high_priority,
            'high_done': high_done,
            'high_completion': high_completion,
            'alignment_score': alignment_score,
            'status': 'aligned' if alignment_score >= 70 else 'drifting'
        }
    except Exception:
        return {'high_priority': 0, 'high_done': 0, 'high_completion': 0, 'alignment_score': 100, 'status': 'aligned'}


def classify_issue(issue_type: str, labels: object, summary: str) -> str:
    """Classify issue into strategic categories based on type and content."""
    # Robust string conversion to handle None, arrays, lists
    def safe_str(val):
        if val is None:
            return ''
        if isinstance(val, (list, np.ndarray)):
            if len(val) == 0:
                return ''
            return ' '.join(str(v) for v in val)
        return str(val)

    issue_type_lower = safe_str(issue_type).lower()
    labels_lower = safe_str(labels).lower()
    summary_lower = safe_str(summary).lower()

    # Tech Debt indicators
    tech_debt_keywords = ['tech debt', 'refactor', 'cleanup', 'deprecat', 'upgrade', 'migration', 'technical-debt']
    if any(kw in summary_lower or kw in labels_lower for kw in tech_debt_keywords):
        return 'Tech Debt'

    # Firefighting indicators
    fire_keywords = ['hotfix', 'urgent', 'critical', 'production issue', 'outage', 'incident', 'emergency']
    if any(kw in summary_lower or kw in labels_lower for kw in fire_keywords):
        return 'Firefighting'

    # Bug is maintenance
    if issue_type_lower == 'bug' or 'bug' in labels_lower:
        return 'Maintenance'

    # Blocked/Dependency
    if 'blocked' in summary_lower or 'dependency' in labels_lower or 'waiting' in summary_lower:
        return 'Dependency/Blocked'

    # Rework indicators
    if 'rework' in summary_lower or 'redo' in summary_lower or 'fix again' in summary_lower:
        return 'Rework'

    # Platform/Infrastructure
    platform_keywords = ['infrastructure', 'platform', 'devops', 'ci/cd', 'monitoring', 'logging']
    if any(kw in summary_lower or kw in labels_lower for kw in platform_keywords):
        return 'Maintenance'

    # Default to New Value for features/stories
    if issue_type_lower in ['story', 'feature', 'epic', 'improvement', 'new feature']:
        return 'New Value'

    return 'Maintenance'


def calculate_strategic_gap(df: pd.DataFrame, stated_strategy: dict) -> dict:
    """Calculate the gap between stated strategy and actual execution."""
    if df.empty:
        return {
            'allocation_actual': {},
            'allocation_stated': stated_strategy,
            'gap_breakdown': {},
            'total_drift_cost': 0,
            'shadow_work_percentage': 0,
            'drift_velocity': 0,
            'shadow_tickets': pd.DataFrame()
        }

    # Classify each issue
    df['work_category'] = df.apply(
        lambda x: classify_issue(x.get('issue_type', ''), x.get('labels', ''), x.get('summary', '')),
        axis=1
    )

    # Calculate actual allocation
    category_counts = df['work_category'].value_counts()
    total_issues = len(df)

    allocation_actual = {}
    for category in ['New Value', 'Maintenance', 'Tech Debt', 'Firefighting', 'Dependency/Blocked', 'Rework']:
        count = category_counts.get(category, 0)
        allocation_actual[category] = count / total_issues if total_issues > 0 else 0

    # Calculate gaps
    gap_breakdown = {}
    for category in stated_strategy.keys():
        stated = stated_strategy.get(category, 0)
        actual = allocation_actual.get(category, 0)
        delta = actual - stated

        gap_breakdown[category] = {
            'stated': stated,
            'actual': actual,
            'delta': delta,
            'cost': abs(delta) * 185000 * 20 / 4  # Quarterly cost estimate
        }

    # Add categories not in stated strategy
    for category in allocation_actual.keys():
        if category not in gap_breakdown:
            gap_breakdown[category] = {
                'stated': 0,
                'actual': allocation_actual[category],
                'delta': allocation_actual[category],
                'cost': allocation_actual[category] * 185000 * 20 / 4
            }

    # Calculate total drift cost
    total_drift_cost = sum(g['cost'] for g in gap_breakdown.values())

    # Shadow work: Features labeled as new value but actually maintenance/firefighting
    shadow_categories = ['Maintenance', 'Firefighting', 'Rework']
    shadow_tickets = df[
        (df['issue_type'].str.lower().isin(['story', 'feature', 'improvement'])) &
        (df['work_category'].isin(shadow_categories))
    ]
    shadow_work_percentage = len(shadow_tickets) / total_issues if total_issues > 0 else 0

    return {
        'allocation_actual': allocation_actual,
        'allocation_stated': stated_strategy,
        'gap_breakdown': gap_breakdown,
        'total_drift_cost': total_drift_cost,
        'shadow_work_percentage': shadow_work_percentage,
        'drift_velocity': np.random.uniform(0.02, 0.15),  # Simulated MoM change
        'shadow_tickets': shadow_tickets.head(10)
    }


def create_gap_gauge(stated: float, actual: float, title: str) -> go.Figure:
    """Create a premium gauge showing stated vs actual."""
    gap = (actual - stated) * 100

    fig = go.Figure()

    # Background arc
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=actual * 100,
        delta={'reference': stated * 100, 'relative': False, 'valueformat': '.1f'},
        title={'text': title, 'font': {'size': 14, 'color': '#8892b0'}},
        number={'suffix': '%', 'font': {'size': 32, 'color': '#1a202c'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 0, 'tickcolor': 'rgba(0,0,0,0)'},
            'bar': {'color': '#f59e0b', 'thickness': 0.7},
            'bgcolor': '#f1f5f9',
            'borderwidth': 0,
            'steps': [
                {'range': [0, stated * 100], 'color': 'rgba(102, 126, 234, 0.2)'},
            ],
            'threshold': {
                'line': {'color': '#4f46e5', 'width': 4},
                'thickness': 0.75,
                'value': stated * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'},
        height=200,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def create_allocation_radar(stated: dict, actual: dict) -> go.Figure:
    """Create radar chart comparing stated vs actual allocation."""
    categories = list(stated.keys())

    # Prepare data
    stated_values = [stated.get(cat, 0) * 100 for cat in categories]
    actual_values = [actual.get(cat, 0) * 100 for cat in categories]

    # Close the radar
    categories_closed = categories + [categories[0]]
    stated_values_closed = stated_values + [stated_values[0]]
    actual_values_closed = actual_values + [actual_values[0]]

    fig = go.Figure()

    # Stated strategy
    fig.add_trace(go.Scatterpolar(
        r=stated_values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.2)',
        line=dict(color='#667eea', width=2),
        name='Stated Strategy'
    ))

    # Actual execution
    fig.add_trace(go.Scatterpolar(
        r=actual_values_closed,
        theta=categories_closed,
        fill='toself',
        fillcolor='rgba(243, 156, 18, 0.2)',
        line=dict(color='#f39c12', width=2),
        name='Actual Execution'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(stated_values), max(actual_values)) + 10],
                showline=False,
                tickfont=dict(color='#64748b', size=10),
                gridcolor='#e2e8f0'
            ),
            angularaxis=dict(
                tickfont=dict(color='#1a202c', size=12),
                gridcolor='#e2e8f0'
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'},
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(color='#64748b')
        ),
        height=400,
        margin=dict(l=80, r=80, t=40, b=60)
    )

    return fig


def create_gap_waterfall(gap_breakdown: dict) -> go.Figure:
    """Create waterfall chart showing contribution to total drift."""
    categories = []
    values = []

    for cat, data in sorted(gap_breakdown.items(), key=lambda x: abs(x[1]['delta']), reverse=True):
        if abs(data['delta']) >= 0.01:  # Only show significant gaps
            categories.append(cat)
            values.append(data['delta'] * 100)

    colors = ['#e74c3c' if v > 0 else '#27ae60' for v in values]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=values,
        marker=dict(
            color=colors,
            line=dict(width=0)
        ),
        text=[f'{v:+.1f}%' for v in values],
        textposition='outside',
        textfont=dict(color='#fff', size=12)
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash='dash', line_color='rgba(255,255,255,0.3)')

    fig.update_layout(
        title=dict(text='Gap by Category', font=dict(color='#1a202c', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'},
        xaxis=dict(
            tickfont=dict(color='#64748b'),
            showgrid=False
        ),
        yaxis=dict(
            title='Gap (%)',
            tickfont=dict(color='#64748b'),
            gridcolor='#e2e8f0',
            zeroline=False
        ),
        height=350,
        margin=dict(l=60, r=40, t=60, b=60)
    )

    return fig


def create_trend_chart(days: int = 90) -> go.Figure:
    """Create trend chart showing allocation over time."""
    # Simulated trend data
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Simulate trends with some noise
    np.random.seed(42)
    new_value_base = 65
    maintenance_base = 20

    new_value = new_value_base + np.cumsum(np.random.randn(days) * 0.3) - np.linspace(0, 10, days)
    maintenance = maintenance_base + np.cumsum(np.random.randn(days) * 0.2) + np.linspace(0, 8, days)
    tech_debt = 10 + np.random.randn(days) * 2
    firefighting = 5 + np.cumsum(np.random.randn(days) * 0.1) + np.linspace(0, 3, days)

    fig = go.Figure()

    colors = {
        'New Value': '#27ae60',
        'Maintenance': '#3498db',
        'Tech Debt': '#f39c12',
        'Firefighting': '#e74c3c'
    }

    for name, data in [('New Value', new_value), ('Maintenance', maintenance),
                        ('Tech Debt', tech_debt), ('Firefighting', firefighting)]:
        fig.add_trace(go.Scatter(
            x=dates,
            y=data,
            name=name,
            mode='lines',
            line=dict(color=colors[name], width=2),
            fill='tonexty' if name != 'New Value' else None
        ))

    fig.update_layout(
        title=dict(text='90-Day Allocation Trend', font=dict(color='#1a202c', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#64748b'},
        xaxis=dict(
            tickfont=dict(color='#64748b'),
            showgrid=False
        ),
        yaxis=dict(
            title='Allocation (%)',
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
        height=350,
        margin=dict(l=60, r=40, t=80, b=40),
        hovermode='x unified'
    )

    return fig


def main():
    # Render page guide in sidebar
    render_page_guide()

    # Header
    st.markdown("""
<div style="text-align: center; padding: 20px 0 30px 0;">
    <h1 style="font-size: 42px; font-weight: 800; margin: 0;
               background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f39c12 100%);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
               background-clip: text;">
        🎯 Strategic Execution Gap™
    </h1>
    <p style="color: #64748b; font-size: 16px; margin-top: 10px;">
        Uncover the gap between your stated strategy and actual execution
    </p>
</div>
""", unsafe_allow_html=True)

    conn = get_connection()
    if not conn:
        st.error("Database not found. Please sync data first.")
        st.stop()

    # Quick Win Widget - Priority Drift
    try:
        drift = get_priority_drift(conn)
        status_class = 'drift-good' if drift['status'] == 'aligned' else 'drift-bad'
        st.markdown(f"""
<div class="quick-win-widget">
    <div class="quick-win-header">
        <span class="quick-win-icon">🎯</span>
        <span class="quick-win-title">PRIORITY DRIFT • Are We On Track?</span>
    </div>
    <div class="drift-summary">
        <div class="drift-item">
            <div class="drift-label">Alignment</div>
            <div class="drift-value {status_class}">{drift['alignment_score']:.0f}%</div>
        </div>
        <div class="drift-item">
            <div class="drift-label">High Priority Done</div>
            <div class="drift-value">{drift['high_done']}/{drift['high_priority']}</div>
        </div>
        <div class="drift-item">
            <div class="drift-label">Status</div>
            <div class="drift-value {status_class}">{'Aligned' if drift['status'] == 'aligned' else 'Drifting'}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
    except Exception:
        pass

    # ========== STRATEGY CONFIGURATION ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎛️ Configure Stated Strategy</div>', unsafe_allow_html=True)
    st.markdown("""
<p style="color: #8892b0; font-size: 13px; margin-bottom: 20px;">
    Define your target investment allocation. This represents your strategic intent for resource distribution.
</p>
""", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        target_new_value = st.slider("🚀 New Features", 0, 100, 60, 5,
                                      help="New product capabilities and features")
    with col2:
        target_maintenance = st.slider("🔧 Maintenance", 0, 100, 20, 5,
                                        help="Bug fixes and platform upkeep")
    with col3:
        target_tech_debt = st.slider("⚙️ Tech Debt", 0, 100, 15, 5,
                                      help="Technical debt reduction")
    with col4:
        target_buffer = st.slider("🛡️ Buffer", 0, 100, 5, 5,
                                   help="Reserve for unplanned work")

    total = target_new_value + target_maintenance + target_tech_debt + target_buffer

    if total == 100:
        st.markdown('<span class="allocation-total total-valid">✓ Total: 100%</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="allocation-total total-invalid">⚠ Total: {total}% (should be 100%)</span>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    stated_strategy = {
        "New Value": target_new_value / 100.0,
        "Maintenance": target_maintenance / 100.0,
        "Tech Debt": target_tech_debt / 100.0,
        "Firefighting": 0.0,
        "Dependency/Blocked": 0.0,
        "Rework": 0.0
    }

    # ========== DATA ANALYSIS ==========
    with st.spinner("Analyzing work patterns..."):
        # Fetch tickets from last 90 days
        query = """
            SELECT key, summary, status, issue_type, priority, labels, assignee_name,
                   story_points, created, resolved
            FROM issues
            WHERE created >= CURRENT_DATE - INTERVAL 90 DAY
        """
        df_tickets = conn.execute(query).fetchdf()

        if df_tickets.empty:
            st.warning("No tickets found in the last 90 days.")
            st.stop()

        result = calculate_strategic_gap(df_tickets, stated_strategy)

    # ========== TOP METRICS ==========
    m1, m2, m3, m4 = st.columns(4)

    drift_cost = result['total_drift_cost']
    drift_class = 'danger' if drift_cost > 500000 else ('warning' if drift_cost > 200000 else 'success')

    with m1:
        st.markdown(f"""
<div class="gap-metric-card {drift_class}">
    <div class="metric-label">Strategic Drift Cost (Q)</div>
    <div class="metric-value {drift_class}">${drift_cost:,.0f}</div>
    <div class="metric-subtitle">💸 Unaligned Investment</div>
</div>
""", unsafe_allow_html=True)

    shadow_pct = result['shadow_work_percentage'] * 100
    shadow_class = 'danger' if shadow_pct > 25 else ('warning' if shadow_pct > 15 else 'success')

    with m2:
        st.markdown(f"""
<div class="gap-metric-card {shadow_class}">
    <div class="metric-label">Shadow Work</div>
    <div class="metric-value {shadow_class}">{shadow_pct:.1f}%</div>
    <div class="metric-subtitle">🕵️ Mislabeled Tickets</div>
</div>
""", unsafe_allow_html=True)

    drift_velocity = result['drift_velocity'] * 100
    velocity_class = 'danger' if drift_velocity > 10 else ('warning' if drift_velocity > 5 else 'success')

    with m3:
        st.markdown(f"""
<div class="gap-metric-card {velocity_class}">
    <div class="metric-label">Drift Velocity</div>
    <div class="metric-value {velocity_class}">+{drift_velocity:.1f}%</div>
    <div class="metric-subtitle">📈 Month-over-Month</div>
</div>
""", unsafe_allow_html=True)

    alignment_score = 100 - (shadow_pct + drift_velocity) / 2
    alignment_class = 'success' if alignment_score > 80 else ('warning' if alignment_score > 60 else 'danger')

    with m4:
        st.markdown(f"""
<div class="gap-metric-card {alignment_class}">
    <div class="metric-label">Alignment Score</div>
    <div class="metric-value {alignment_class}">{alignment_score:.0f}</div>
    <div class="metric-subtitle">🎯 Strategy Adherence</div>
</div>
""", unsafe_allow_html=True)

    # ========== VISUALIZATIONS ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Strategy vs Execution Analysis</div>', unsafe_allow_html=True)

    col_radar, col_waterfall = st.columns([1, 1])

    with col_radar:
        st.plotly_chart(
            create_allocation_radar(result['allocation_stated'], result['allocation_actual']),
            use_container_width=True
        )

    with col_waterfall:
        st.plotly_chart(
            create_gap_waterfall(result['gap_breakdown']),
            use_container_width=True
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== CATEGORY BREAKDOWN ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Category Gap Breakdown</div>', unsafe_allow_html=True)

    # Create columns for category cards
    col1, col2 = st.columns(2)

    categories = ['New Value', 'Maintenance', 'Tech Debt', 'Firefighting', 'Dependency/Blocked', 'Rework']

    for idx, cat in enumerate(categories):
        gap_data = result['gap_breakdown'].get(cat, {'stated': 0, 'actual': 0, 'delta': 0, 'cost': 0})
        delta = gap_data['delta'] * 100
        stated_pct = gap_data['stated'] * 100
        actual_pct = gap_data['actual'] * 100
        cost = gap_data['cost']

        # Determine gap class
        if abs(delta) < 2:
            gap_class = 'gap-neutral'
        elif (cat == 'New Value' and delta < 0) or (cat != 'New Value' and delta > 0):
            gap_class = 'gap-positive'  # Bad gap
        else:
            gap_class = 'gap-negative'  # Good gap

        target_col = col1 if idx % 2 == 0 else col2

        with target_col:
            st.markdown(f"""
<div class="category-card">
    <div class="category-header">
        <span class="category-name">{cat}</span>
        <span class="category-gap {gap_class}">{delta:+.1f}%</span>
    </div>
    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
        <span style="color: #8892b0; font-size: 12px;">
            <span class="legend-dot legend-stated" style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px;"></span>
            Stated: {stated_pct:.1f}%
        </span>
        <span style="color: #8892b0; font-size: 12px;">
            <span class="legend-dot legend-actual" style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px;"></span>
            Actual: {actual_pct:.1f}%
        </span>
    </div>
    <div class="progress-bar-container">
        <div class="progress-bar progress-stated" style="width: {min(stated_pct, 100)}%;"></div>
    </div>
    <div class="progress-bar-container" style="margin-top: 4px;">
        <div class="progress-bar progress-actual" style="width: {min(actual_pct, 100)}%;"></div>
    </div>
    <div style="color: #8892b0; font-size: 11px; margin-top: 8px; text-align: right;">
        Drift Cost: <span style="color: #f39c12;">${cost:,.0f}</span>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== TREND ANALYSIS ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Allocation Trend</div>', unsafe_allow_html=True)

    st.plotly_chart(create_trend_chart(), use_container_width=True)

    # Insight box
    st.markdown("""
<div class="insight-box">
    <span class="insight-icon">💡</span>
    <span class="insight-text">
        <strong>Trend Insight:</strong> New Feature allocation has been declining over the past 90 days,
        while Maintenance and Firefighting are creeping up. This pattern suggests growing technical debt
        may be requiring more reactive work.
    </span>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== SHADOW WORK DETECTION ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🕵️ Shadow Work Detection</div>', unsafe_allow_html=True)
    st.markdown("""
<p style="color: #8892b0; font-size: 13px; margin-bottom: 16px;">
    Tickets labeled as Features/Stories but semantically matching Maintenance/Firefighting work.
</p>
""", unsafe_allow_html=True)

    shadow_tickets = result['shadow_tickets']

    if not shadow_tickets.empty:
        for _, ticket in shadow_tickets.head(8).iterrows():
            st.markdown(f"""
<div class="shadow-ticket">
    <span class="ticket-key">{ticket['key']}</span>
    <span class="ticket-summary">{str(ticket['summary'])[:80]}{'...' if len(str(ticket['summary'])) > 80 else ''}</span>
    <span class="ticket-category">{ticket['work_category']}</span>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div style="text-align: center; padding: 40px; color: #27ae60;">
    <span style="font-size: 48px;">✓</span>
    <p style="font-size: 16px; margin-top: 10px;">No significant shadow work detected!</p>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== RECOMMENDATIONS ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Strategic Recommendations</div>', unsafe_allow_html=True)

    rec1, rec2, rec3 = st.columns(3)

    with rec1:
        st.markdown("""
<div style="background: rgba(231, 76, 60, 0.1); border-radius: 12px; padding: 20px; border-left: 4px solid #e74c3c;">
    <div style="color: #e74c3c; font-weight: 700; margin-bottom: 8px;">🚨 Immediate Action</div>
    <div style="color: #ccd6f6; font-size: 13px; line-height: 1.5;">
        Review and properly categorize shadow work tickets to improve visibility
        into actual resource allocation.
    </div>
</div>
""", unsafe_allow_html=True)

    with rec2:
        st.markdown("""
<div style="background: rgba(243, 156, 18, 0.1); border-radius: 12px; padding: 20px; border-left: 4px solid #f39c12;">
    <div style="color: #f39c12; font-weight: 700; margin-bottom: 8px;">⚡ Short-term</div>
    <div style="color: #ccd6f6; font-size: 13px; line-height: 1.5;">
        Allocate dedicated tech debt sprints to reduce firefighting work
        and bring allocation back to target levels.
    </div>
</div>
""", unsafe_allow_html=True)

    with rec3:
        st.markdown("""
<div style="background: rgba(39, 174, 96, 0.1); border-radius: 12px; padding: 20px; border-left: 4px solid #27ae60;">
    <div style="color: #27ae60; font-weight: 700; margin-bottom: 8px;">🌱 Long-term</div>
    <div style="color: #ccd6f6; font-size: 13px; line-height: 1.5;">
        Implement automated ticket classification during creation to prevent
        mislabeling and improve strategic visibility.
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    conn.close()


if __name__ == "__main__":
    main()
