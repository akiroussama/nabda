"""
⏱️ The Scope Negotiator - Strategic Timeline Simulator
"The first page that lets you bully the timeline before it bullies you."

Allows Product Managers to:
1. Visualize the "physics" of the current sprint (Scope vs Time).
2. Simuate "What if we add X?" scenarios.
3. Instantly see the impact on delivery dates.
"""

import sys

# Import page guide component
from src.dashboard.components import render_page_guide
from pathlib import Path
import streamlit as st
import pandas as pd
import duckdb
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_dir))

st.set_page_config(page_title="Scope Negotiator", page_icon="⏱️", layout="wide")

# Premium Physics/Time Theme CSS
st.markdown("""
<style>
    /* Global Theme */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Hero Section */
    .scope-hero {
        background: linear-gradient(135deg, #2c3e50 0%, #4ca1af 100%);
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 24px;
        color: white;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    
    .hero-title {
        font-size: 32px;
        font-weight: 800;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        font-size: 16px;
        opacity: 0.9;
        max-width: 600px;
    }

    /* Simulation Control Panel */
    .sim-panel {
        background: white;
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        height: 100%;
    }

    .panel-header {
        font-size: 18px;
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Date Badges */
    .date-badge {
        display: inline-flex;
        flex-direction: column;
        align-items: center;
        padding: 12px 24px;
        border-radius: 12px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        margin-right: 16px;
    }
    
    .date-badge.target {
        background: #ecfdf5;
        border-color: #d1fae5;
    }
    
    .date-badge.projected {
        background: #fff7ed;
        border-color: #ffedd5;
    }
    
    .date-badge.danger {
        background: #fef2f2;
        border-color: #fee2e2;
        animation: pulse-border 2s infinite;
    }
    
    @keyframes pulse-border {
        0% { border-color: #fee2e2; box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.2); }
        50% { border-color: #ef4444; box-shadow: 0 0 0 4px rgba(239, 68, 68, 0.1); }
        100% { border-color: #fee2e2; box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    .badge-label {
        font-size: 11px;
        text-transform: uppercase;
        color: #64748b;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .badge-value {
        font-size: 24px;
        font-weight: 800;
        color: #1e293b;
    }
    
    .badge-diff {
        font-size: 13px;
        font-weight: 600;
        margin-top: 4px;
        padding: 2px 8px;
        border-radius: 10px;
    }
    .diff-good { color: #166534; background: #dcfce7; }
    .diff-bad { color: #991b1b; background: #fee2e2; }

    /* Scope Items */
    .scope-item {
        display: flex;
        align-items: center;
        padding: 12px;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 8px;
        transition: all 0.2s;
    }
    
    .scope-item:hover {
        border-color: #cbd5e1;
        transform: translateX(2px);
    }
    
    .item-points {
        background: #e2e8f0;
        color: #475569;
        font-weight: 700;
        font-size: 12px;
        padding: 4px 8px;
        border-radius: 6px;
        margin-right: 12px;
        min-width: 32px;
        text-align: center;
    }
    
    .item-key {
        color: #4f46e5;
        font-weight: 600;
        font-size: 13px;
        margin-right: 8px;
    }
    
    .item-summary {
        font-size: 14px;
        color: #1e293b;
        flex: 1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    
    /* Recommendations */
    .recommendation-card {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin-top: 16px;
    }
    
    .rec-title {
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

</style>
""", unsafe_allow_html=True)

def get_connection():
    db_path = Path("data/jira.duckdb")
    return duckdb.connect(str(db_path), read_only=True) if db_path.exists() else None

def get_sprint_data(conn):
    """Get active sprint data and issues."""
    sprint = conn.execute("""
        SELECT id, name, start_date, end_date
        FROM sprints 
        WHERE state = 'active'
        ORDER BY start_date DESC LIMIT 1
    """).fetchone()
    
    if not sprint:
        # Fallback to last closed sprint for demo
        sprint = conn.execute("""
            SELECT id, name, start_date, end_date
            FROM sprints 
            ORDER BY start_date DESC LIMIT 1
        """).fetchone()
        
    return sprint

def get_velocity(conn):
    """Calculate recent velocity (avg of last 3 closed sprints)."""
    try:
        velocity = conn.execute("""
            SELECT AVG(total_points) as avg_velocity
            FROM (
                SELECT 
                    s.id, 
                    SUM(CASE WHEN i.status IN ('Done', 'Terminé(e)', 'Closed') THEN i.story_points ELSE 0 END) as total_points
                FROM sprints s
                JOIN issues i ON i.sprint_id = s.id
                WHERE s.state = 'closed' OR s.state = 'future' -- broadened for demo data
                GROUP BY s.id
                ORDER BY s.end_date DESC
                LIMIT 3
            )
        """).fetchone()[0]
        return velocity if velocity and velocity > 0 else 20.0 # Default fallback
    except:
        return 20.0

def get_issues(conn, sprint_id):
    """Get issues for the current sprint and backlog candidates."""
    # Active Sprint Issues
    sprint_issues = conn.execute(f"""
        SELECT key, summary, issue_type, priority, status, 
               COALESCE(story_points, 3) as story_points
        FROM issues 
        WHERE sprint_id = {sprint_id}
    """).fetchdf()
    
    # Backlog Candidates (Top 20 from backlog)
    backlog_issues = conn.execute(f"""
        SELECT key, summary, issue_type, priority, status,
               COALESCE(story_points, 3) as story_points
        FROM issues 
        WHERE sprint_id IS NULL 
        AND status NOT IN ('Done', 'Terminé(e)', 'Closed')
        ORDER BY 
            CASE priority 
                WHEN 'Highest' THEN 1 
                WHEN 'High' THEN 2 
                ELSE 3 
            END, created DESC
        LIMIT 20
    """).fetchdf()
    
    return sprint_issues, backlog_issues

def create_timeline_chart(start_date, target_date, projected_date):
    """Create a timeline visualization."""
    
    # Ensure dates are datetime objects
    if isinstance(start_date, str): start_date = pd.to_datetime(start_date)
    if isinstance(target_date, str): target_date = pd.to_datetime(target_date)
    if isinstance(projected_date, str): projected_date = pd.to_datetime(projected_date)
    
    # Create figure
    fig = go.Figure()
    
    # 1. Base Sprint Duration (Gray Bar)
    fig.add_trace(go.Bar(
        y=['Timeline'],
        x=[(target_date - start_date).total_seconds() * 1000], # Milliseconds
        base=[start_date.timestamp() * 1000],
        orientation='h',
        marker=dict(color='#e2e8f0', opacity=0.5),
        name='Planned Duration',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # 2. Projected Duration (Color Coded)
    is_delayed = projected_date > target_date
    color = '#ef4444' if is_delayed else '#22c55e'
    
    fig.add_trace(go.Bar(
        y=['Timeline'],
        x=[(projected_date - start_date).total_seconds() * 1000],
        base=[start_date.timestamp() * 1000],
        orientation='h',
        marker=dict(color=color, opacity=0.8),
        name='Projected',
        text=f"{'⚠️ DELAYED' if is_delayed else '✅ ON TRACK'}",
        textposition='auto',
        hovertemplate='Projected End: %{x|%b %d}<extra></extra>'
    ))
    
    # 3. Target Line
    fig.add_vline(
        x=target_date.timestamp() * 1000, 
        line_width=3, 
        line_dash="dash", 
        line_color="#1e293b",
        annotation_text="Target Deadline",
        annotation_position="top",
    )
    
    # Layout configuration
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(
            type='date',
            tickformat='%b %d',
            gridcolor='#f1f5f9',
            range=[
                (start_date - timedelta(days=2)).timestamp() * 1000,
                (max(target_date, projected_date) + timedelta(days=5)).timestamp() * 1000
            ]
        ),
        yaxis=dict(showticklabels=False, showgrid=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        barmode='overlay'
    )
    
    return fig

def main():
    # Render page guide in sidebar
    render_page_guide()

    # Hero Section
    st.markdown("""
<div class="scope-hero">
    <div class="hero-title">⏱️ The Scope Negotiator</div>
    <div class="hero-subtitle">
        Don't just verify the timeline—bully it. Simulate scope changes and instant delay impacts 
        to make data-driven decisions in seconds.
    </div>
</div>
""", unsafe_allow_html=True)

    conn = get_connection()
    if not conn:
        st.error("Database connection failed.")
        st.stop()
        
    sprint = get_sprint_data(conn)
    if not sprint:
        st.warning("No active sprint metrics found.")
        st.stop()
        
    sprint_id, sprint_name, start_t, end_t = sprint
    
    # Convert dates
    start_date = pd.to_datetime(start_t) if start_t else datetime.now()
    target_date = pd.to_datetime(end_t) if end_t else start_date + timedelta(days=14)
    
    # Get Data
    velocity_weekly = get_velocity(conn) # Points per sprint (assuming 2 weeks usually)
    velocity_daily = velocity_weekly / 10 # Assuming 10 working days
    
    sprint_issues_df, backlog_issues_df = get_issues(conn, sprint_id)
    
    # --- SIMULATION STATE ---
    
    # Sidebar or Top Controls
    with st.expander("⚙️ Simulation Parameters", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            sim_velocity = st.slider("Team Velocity (Points/Sprint)", 
                                     min_value=5.0, 
                                     max_value=100.0, 
                                     value=float(velocity_weekly),
                                     help="Adjust to simulate 'What if we work harder?' (Risk of burnout!)")
        with c2:
            sim_headcount = st.slider("Headcount Multiplier", 
                                      min_value=0.5, 
                                      max_value=2.0, 
                                      value=1.0, 
                                      step=0.1,
                                      help="Simulate adding/removing devs (Brooks' Law warning!)")

    # Adjusted daily velocity
    effective_daily_velocity = (sim_velocity / 10) * sim_headcount

    # --- INTERACTIVE SCOPE EDITOR ---
    
    col_sim_viz, col_scope = st.columns([1.5, 1])
    
    with col_scope:
        st.markdown('<div class="sim-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">📝 Scope Sandbox</div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Current Sprint", "Backlog (Add New)"])
        
        with tab1:
            st.markdown(f"**{len(sprint_issues_df)} items in {sprint_name}**")
            
            # Use data editor to allow enabling/disabling items
            # Add a 'keep' column defaulting to True
            sprint_issues_df['include'] = True
            
            edited_sprint = st.data_editor(
                sprint_issues_df[['include', 'key', 'summary', 'story_points', 'priority']],
                column_config={
                    "include": st.column_config.CheckboxColumn("Keep?", help="Uncheck to simulate removing this item", default=True),
                    "story_points": st.column_config.NumberColumn("Pts", width="small"),
                    "summary": st.column_config.TextColumn("Summary", width="medium"),
                },
                disabled=["key", "summary", "priority"],
                hide_index=True,
                key="editor_sprint"
            )
            
        with tab2:
            st.markdown("**Drag from Backlog** (Check to Simulate Adding)")
            
            backlog_issues_df['include'] = False
            
            edited_backlog = st.data_editor(
                backlog_issues_df[['include', 'key', 'summary', 'story_points', 'priority']],
                column_config={
                    "include": st.column_config.CheckboxColumn("Add?", help="Check to simulate adding this item", default=False),
                    "story_points": st.column_config.NumberColumn("Pts", width="small"),
                    "summary": st.column_config.TextColumn("Summary", width="medium"),
                },
                disabled=["key", "summary", "priority"],
                hide_index=True,
                key="editor_backlog"
            )

        st.markdown('</div>', unsafe_allow_html=True)

    # --- CALCULATION ENGINE ---
    
    # 1. Total Points from "Kept" Sprint Items
    kept_sprint_points = edited_sprint[edited_sprint['include']]['story_points'].sum()
    
    # 2. Total Points from "Added" Backlog Items
    added_backlog_points = edited_backlog[edited_backlog['include']]['story_points'].sum()
    
    total_simulated_points = kept_sprint_points + added_backlog_points
    
    # 3. Calculate Projected Days
    # Basic math: Points / Daily Velocity = Days Needed
    days_needed = total_simulated_points / effective_daily_velocity if effective_daily_velocity > 0 else 999
    
    # 4. Projected End Date (skipping weekends roughly by using business days logic or simple multiplier)
    # Simple logic: Add days, if weekend add 2 more. 
    # Better: user np.busday_count or similar if available, or just simple mapping.
    # We will use simple calendar days * (7/5) to account for weekends approx.
    calendar_days_needed = days_needed * (7/5)
    
    projected_end_date = start_date + timedelta(days=float(calendar_days_needed))
    
    # Delay Calculation
    delay_days = (projected_end_date - target_date).days
    
    # --- VISUALIZATION COLUMN ---
    
    with col_sim_viz:
        st.markdown('<div class="sim-panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">📊 Impact Analysis</div>', unsafe_allow_html=True)
        
        # Badges
        b1, b2, b3 = st.columns(3)
        
        with b1:
            st.markdown(f"""
<div class="date-badge target">
    <div class="badge-label">Target Deadline</div>
    <div class="badge-value">{target_date.strftime('%b %d')}</div>
    <div class="badge-diff diff-good">Locked</div>
</div>
""", unsafe_allow_html=True)
            
        with b2:
            badge_class = "danger" if delay_days > 0 else "projected"
            diff_class = "diff-bad" if delay_days > 0 else "diff-good"
            sign = "+" if delay_days > 0 else ""
            
            st.markdown(f"""
<div class="date-badge {badge_class}">
    <div class="badge-label">Projected Landing</div>
    <div class="badge-value">{projected_end_date.strftime('%b %d')}</div>
    <div class="badge-diff {diff_class}">{sign}{delay_days} Days</div>
</div>
""", unsafe_allow_html=True)
            
        with b3:
             st.markdown(f"""
<div class="date-badge">
    <div class="badge-label">Total Scope</div>
    <div class="badge-value">{total_simulated_points:.0f} pts</div>
    <div class="badge-diff diff-neutral">Cap: {sim_velocity:.0f}</div>
</div>
""", unsafe_allow_html=True)
            
        # Timeline
        st.plotly_chart(create_timeline_chart(start_date, target_date, projected_end_date), use_container_width=True)
        
        # Analysis / AI Coach
        if delay_days > 0:
            st.markdown(f"""
<div class="recommendation-card">
    <div class="rec-title">🚨 NEGOTIATOR ALERT: You are overbooked!</div>
    <p>To hit <b>{target_date.strftime('%b %d')}</b>, you must remove <b>{delay_days * effective_daily_velocity:.1f} points</b>.</p>
    <p><i>Suggested Cut:</i> Remove low priority items to save the fast lane.</p>
</div>
""", unsafe_allow_html=True)
            
            # Show suggestions
            potential_cuts = edited_sprint[edited_sprint['include']].sort_values('priority', ascending=False).head(3)
            if not potential_cuts.empty:
               st.caption("📉 Suggested items to remove (Lowest Priority):")
               for _, row in potential_cuts.iterrows():
                   st.markdown(f"- **{row['key']}**: {row['summary']} ({row['story_points']} pts)")
                   
        elif added_backlog_points > 0:
             st.markdown(f"""
<div class="recommendation-card" style="border-left-color: #22c55e;">
    <div class="rec-title">✅ SAFE TO ADD: Timeline Intact</div>
    <p>You added <b>{added_backlog_points} points</b> and are still on track.</p>
    <p>Buffer remaining: {abs(delay_days)} days.</p>
</div>
""", unsafe_allow_html=True)
        else:
             st.markdown(f"""
<div class="recommendation-card" style="border-left-color: #94a3b8;">
    <div class="rec-title">💤 STATUS QUO</div>
    <p>No changes simulated. Use the sandbox on the right to simulate "What If" scenarios.</p>
</div>
""", unsafe_allow_html=True)
            
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
