"""
🌅 Good Morning Dashboard - Premium Command Center
Your personalized morning briefing with AI-powered insights.
"""

import sys
from pathlib import Path

# Add project root to sys.path so we can import from src
root_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_dir))

import streamlit as st
import pandas as pd
import duckdb
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List

st.set_page_config(
    page_title="Good Morning Dashboard",
    page_icon="🌅",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Premium Dark Theme CSS
st.markdown("""
<style>
    /* Global Light Theme */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Main layout */
    .main > div {
        padding-top: 1rem;
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

    /* Greeting Card */
    .greeting-card {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }

    .greeting-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 70%);
    }

    .greeting-card.morning {
        background: linear-gradient(135deg, #f59e0b 0%, #ea580c 100%);
    }

    .greeting-card.afternoon {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    }

    .greeting-card.evening {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
    }

    .greeting-text {
        font-size: 36px;
        font-weight: 800;
        color: white;
        margin-bottom: 8px;
        position: relative;
        z-index: 1;
    }

    .greeting-date {
        font-size: 16px;
        color: rgba(255, 255, 255, 0.95);
        position: relative;
        z-index: 1;
        font-weight: 500;
    }

    .greeting-icon {
        position: absolute;
        right: 32px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 64px;
        opacity: 0.25;
        color: white;
    }

    /* Quick Stats Cards */
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    }

    .stat-value {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-value.positive {
        background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-value.negative {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 8px;
        font-weight: 600;
    }

    .stat-delta {
        font-size: 12px;
        margin-top: 4px;
    }

    .delta-up { color: #16a34a; }
    .delta-down { color: #dc2626; }
    .delta-neutral { color: #64748b; }

    /* Briefing Cards */
    .briefing-section {
        margin-bottom: 20px;
    }

    .briefing-header {
        font-size: 16px;
        font-weight: 700;
        color: #1a202c;
        margin-bottom: 12px;
    }

    .good-news-card {
        background: #f0fdf4;
        border-left: 4px solid #16a34a;
        border-radius: 0 12px 12px 0;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #dcfce7;
        border-left-width: 4px;
    }

    .concern-card {
        background: #fff7ed;
        border-left: 4px solid #ea580c;
        border-radius: 0 12px 12px 0;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #ffedd5;
        border-left-width: 4px;
    }

    .risk-card {
        background: #fef2f2;
        border-left: 4px solid #dc2626;
        border-radius: 0 12px 12px 0;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #fee2e2;
        border-left-width: 4px;
    }

    .card-text {
        color: #334155;
        font-size: 14px;
        line-height: 1.6;
    }

    .evidence-badge {
        display: inline-block;
        background: #e0e7ff;
        color: #4338ca;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-family: monospace;
        margin-left: 8px;
        font-weight: 600;
    }

    /* Recommendation Card */
    .recommendation-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 20px;
        margin-top: 16px;
    }

    .recommendation-title {
        color: #4f46e5;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 8px;
    }

    .recommendation-text {
        color: #1e293b;
        font-size: 15px;
        line-height: 1.5;
    }

    /* Decision Queue */
    .decision-item {
        background: white;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    .decision-item:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    .decision-item.critical { border-left: 4px solid #ef4444; }
    .decision-item.high { border-left: 4px solid #f59e0b; }
    .decision-item.medium { border-left: 4px solid #3b82f6; }
    .decision-item.low { border-left: 4px solid #22c55e; }

    .decision-priority {
        display: inline-block;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        text-align: center;
        line-height: 28px;
        font-weight: 700;
        font-size: 12px;
        margin-right: 12px;
    }

    .priority-critical { background: #fee2e2; color: #991b1b; }
    .priority-high { background: #fef3c7; color: #92400e; }
    .priority-medium { background: #dbeafe; color: #1e40af; }
    .priority-low { background: #dcfce7; color: #166534; }

    .decision-key {
        color: #4f46e5;
        font-weight: 700;
        font-size: 12px;
    }

    .decision-summary {
        color: #1e293b;
        font-size: 13px;
        margin-top: 4px;
        font-weight: 500;
    }

    .decision-reason {
        color: #64748b;
        font-size: 11px;
        margin-top: 8px;
    }

    /* AI Insights */
    .ai-insight {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .ai-icon {
        font-size: 20px;
        margin-right: 10px;
    }

    .ai-text {
        color: #334155;
        font-size: 13px;
        line-height: 1.5;
    }

    /* Team Activity */
    .team-member-row {
        display: flex;
        align-items: center;
        padding: 12px;
        background: white;
        border-radius: 8px;
        margin-bottom: 8px;
        border: 1px solid #e2e8f0;
    }

    .team-avatar {
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

    .team-name {
        color: #1e293b;
        font-weight: 600;
        font-size: 14px;
        flex: 1;
    }

    .team-stat {
        color: #64748b;
        font-size: 12px;
        background: #f1f5f9;
        padding: 2px 8px;
        border-radius: 12px;
    }

    /* Footer */
    .dashboard-footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        padding: 16px;
        margin-top: 24px;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


def get_connection():
    """Get database connection."""
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path), read_only=True)


def get_time_greeting() -> tuple:
    """Get time-appropriate greeting and CSS class."""
    hour = datetime.now().hour
    if hour < 12:
        return "Good Morning", "morning", "🌅"
    elif hour < 17:
        return "Good Afternoon", "afternoon", "☀️"
    else:
        return "Good Evening", "evening", "🌙"


def get_avatar_color(name: str) -> str:
    """Generate consistent color for avatar based on name."""
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#fa709a']
    return colors[hash(name or '') % len(colors)]


def get_morning_priority(conn, project_key: str) -> Dict[str, Any]:
    """Get the #1 priority item for today - the thing to focus on first."""
    # Find the most critical item: highest priority + oldest + in progress or blocked
    try:
        priority_item = conn.execute(f"""
            SELECT
                key,
                summary,
                priority,
                status,
                assignee_name,
                DATEDIFF('day', updated, CURRENT_TIMESTAMP) as days_stale,
                DATEDIFF('day', created, CURRENT_TIMESTAMP) as age_days
            FROM issues
            WHERE project_key = '{project_key}'
            AND status != 'Terminé(e)'
            ORDER BY
                CASE priority WHEN 'Highest' THEN 1 WHEN 'High' THEN 2 WHEN 'Medium' THEN 3 ELSE 4 END,
                CASE WHEN status = 'En cours' AND DATEDIFF('day', updated, CURRENT_TIMESTAMP) >= 3 THEN 0 ELSE 1 END,
                DATEDIFF('day', created, CURRENT_TIMESTAMP) DESC
            LIMIT 1
        """).fetchone()

        if not priority_item:
            return None

        key, summary, priority, status, assignee, days_stale, age_days = priority_item

        # Determine severity and reason
        if priority == 'Highest' or days_stale >= 5:
            severity = 'high'
            reason = f"Highest priority • {age_days} days old" if priority == 'Highest' else f"Stale for {days_stale} days"
        elif priority == 'High' or days_stale >= 3:
            severity = 'normal'
            reason = f"High priority • {age_days} days old"
        else:
            severity = 'low'
            reason = f"Next in queue • {age_days} days old"

        # Check if blocked (stale in progress)
        blocked_by = None
        if status == 'En cours' and days_stale >= 3:
            blocked_by = f"No updates for {days_stale} days - may need help"

        # Determine impact
        if priority == 'Highest':
            impact = "Critical path item"
        elif priority == 'High':
            impact = "High business value"
        else:
            impact = "Keep momentum"

        return {
            'title': f"{key}: {summary[:50]}{'...' if len(summary) > 50 else ''}",
            'reason': reason,
            'severity': severity,
            'blocked_by': blocked_by,
            'impact': impact,
            'assignee': assignee or 'Unassigned'
        }
    except Exception:
        return None


def generate_briefing_data(conn, project_key: str) -> Dict[str, Any]:
    """Generate complete briefing data from database."""
    # Get recent activity (last 24 hours)
    yesterday = datetime.now() - timedelta(days=1)

    # Completed tickets
    try:
        completed_df = conn.execute(f"""
            SELECT key, summary, assignee_name, story_points, resolved
            FROM issues
            WHERE project_key = '{project_key}'
              AND status = 'Terminé(e)'
              AND resolved >= '{yesterday.strftime('%Y-%m-%d')}'
        """).fetchdf()
    except:
        completed_df = pd.DataFrame()

    # Created tickets
    try:
        created_df = conn.execute(f"""
            SELECT key, summary, issue_type, priority, created
            FROM issues
            WHERE project_key = '{project_key}'
              AND created >= '{yesterday.strftime('%Y-%m-%d')}'
        """).fetchdf()
    except:
        created_df = pd.DataFrame()

    # Blockers
    try:
        blockers_df = conn.execute(f"""
            SELECT key, summary, assignee_name, priority
            FROM issues
            WHERE project_key = '{project_key}'
              AND (priority = 'Blocker' OR labels LIKE '%blocked%')
              AND status != 'Terminé(e)'
        """).fetchdf()
    except:
        blockers_df = pd.DataFrame()

    # High priority items
    try:
        high_priority_df = conn.execute(f"""
            SELECT key, summary, assignee_name, priority, status, created
            FROM issues
            WHERE project_key = '{project_key}'
              AND priority IN ('Highest', 'High')
              AND status != 'Terminé(e)'
            ORDER BY
                CASE priority
                    WHEN 'Highest' THEN 1
                    WHEN 'High' THEN 2
                    ELSE 3
                END,
                created ASC
            LIMIT 10
        """).fetchdf()
    except:
        high_priority_df = pd.DataFrame()

    # Team activity
    try:
        team_df = conn.execute(f"""
            SELECT
                assignee_name,
                COUNT(*) as total_issues,
                COUNT(CASE WHEN status = 'Terminé(e)' THEN 1 END) as completed,
                COUNT(CASE WHEN status = 'En cours' THEN 1 END) as in_progress
            FROM issues
            WHERE project_key = '{project_key}'
              AND assignee_name IS NOT NULL
            GROUP BY assignee_name
            ORDER BY total_issues DESC
            LIMIT 8
        """).fetchdf()
    except:
        team_df = pd.DataFrame()

    # Overall stats
    try:
        stats_df = conn.execute(f"""
            SELECT
                COUNT(*) as total_issues,
                COUNT(CASE WHEN status = 'Terminé(e)' THEN 1 END) as done,
                COUNT(CASE WHEN status = 'En cours' THEN 1 END) as in_progress,
                COUNT(CASE WHEN priority = 'Blocker' THEN 1 END) as blocked,
                SUM(COALESCE(story_points, 0)) as total_points
            FROM issues
            WHERE project_key = '{project_key}'
        """).fetchdf()
    except:
        stats_df = pd.DataFrame({'total_issues': [0], 'done': [0], 'in_progress': [0], 'blocked': [0], 'total_points': [0]})

    return {
        'completed': completed_df,
        'created': created_df,
        'blockers': blockers_df,
        'high_priority': high_priority_df,
        'team': team_df,
        'stats': stats_df.iloc[0] if not stats_df.empty else {}
    }


def generate_ai_insights(data: Dict[str, Any]) -> List[str]:
    """Generate AI-powered insights based on data patterns."""
    insights = []

    stats = data.get('stats', {})
    completed = data.get('completed', pd.DataFrame())
    blockers = data.get('blockers', pd.DataFrame())
    created = data.get('created', pd.DataFrame())

    # Velocity insight
    completed_count = len(completed)
    created_count = len(created)

    if completed_count > created_count:
        insights.append(
            f"📈 **Positive flow**: Completed {completed_count} tickets while only {created_count} were created. "
            "Your backlog is shrinking!"
        )
    elif created_count > completed_count * 1.5:
        insights.append(
            f"⚠️ **Backlog growth**: {created_count} new tickets vs {completed_count} completed. "
            "Consider reviewing incoming work."
        )

    # Blocker insight
    blocker_count = len(blockers)
    if blocker_count > 0:
        insights.append(
            f"🚧 **{blocker_count} blocker(s) active**: These should be prioritized to unblock team progress."
        )

    # Team balance insight
    team = data.get('team', pd.DataFrame())
    if not team.empty and len(team) >= 2:
        max_load = team['total_issues'].max()
        min_load = team['total_issues'].min()
        if max_load > min_load * 2:
            insights.append(
                "👥 **Workload imbalance detected**: Consider redistributing work across the team."
            )

    # Progress insight - check if stats has data (can be Series, DataFrame, or dict)
    has_stats = False
    if isinstance(stats, (pd.DataFrame, pd.Series)):
        has_stats = not stats.empty
    elif isinstance(stats, dict):
        has_stats = bool(stats)

    if has_stats:
        total = stats.get('total_issues', 0)
        done = stats.get('done', 0)
        if total > 0:
            completion_rate = (done / total) * 100
            if completion_rate > 70:
                insights.append(
                    f"🎯 **Strong progress**: {completion_rate:.0f}% of issues are complete. Keep up the momentum!"
                )
            elif completion_rate < 30:
                insights.append(
                    f"📋 **Early stages**: Only {completion_rate:.0f}% complete. Focus on moving items through the pipeline."
                )

    if not insights:
        insights.append("✅ **All systems nominal**: No significant patterns detected. Continue steady progress.")

    return insights


def get_standup_script(data: Dict[str, Any], priority_item: Dict[str, Any]) -> List[str]:
    """
    Generate a 'Standup Cheat Sheet' - exactly what to say in the meeting.
    Ultrathink: Removes the cognitive load of preparing for standup.
    """
    script = []
    
    # Yesterday
    completed = data.get('completed', pd.DataFrame())
    if not completed.empty:
        count = len(completed)
        keys = ", ".join(completed['key'].head(2).tolist())
        script.append(f"✅ **Yesterday:** Completed {count} items ({keys}).")
    else:
        script.append("✅ **Yesterday:** Focused on progress, no tickets closed yet.")
        
    # Today
    if priority_item:
        script.append(f"🎯 **Today:** Focusing on {priority_item['title'].split(':')[0]} (Critical).")
    else:
        script.append("🎯 **Today:** Picking up next highest priority item.")
        
    # Blockers
    blockers = data.get('blockers', pd.DataFrame())
    if not blockers.empty:
        top_blocker = blockers.iloc[0]['key']
        script.append(f"⛔ **Blockers:** I am blocked on {top_blocker}. Need assistance.")
    else:
        script.append("🚀 **Blockers:** No blockers. Green light.")
        
    return script



def render_greeting():
    """Render the greeting card."""
    greeting, time_class, icon = get_time_greeting()

    st.markdown(f"""
    <div class="greeting-card {time_class}">
        <div class="greeting-text">{greeting}!</div>
        <div class="greeting-date">{datetime.now().strftime('%A, %B %d, %Y')} at {datetime.now().strftime('%I:%M %p')}</div>
        <div class="greeting-icon">{icon}</div>
    </div>
    """, unsafe_allow_html=True)


def render_quick_stats(data: Dict[str, Any]):
    """Render quick stats section."""
    stats = data.get('stats', {})
    completed = data.get('completed', pd.DataFrame())
    created = data.get('created', pd.DataFrame())
    blockers = data.get('blockers', pd.DataFrame())

    completed_count = len(completed)
    created_count = len(created)
    blocker_count = len(blockers)
    total_points = sum(completed['story_points'].fillna(0)) if not completed.empty and 'story_points' in completed.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        delta_class = 'positive' if completed_count > 0 else 'neutral'
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value {delta_class}">{completed_count}</div>
            <div class="stat-label">Completed Today</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{total_points:.0f}</div>
            <div class="stat-label">Points Delivered</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        blocker_class = 'negative' if blocker_count > 2 else ('stat-value' if blocker_count > 0 else 'positive')
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value {blocker_class}">{blocker_count}</div>
            <div class="stat-label">Active Blockers</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{created_count}</div>
            <div class="stat-label">Created Today</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        in_progress = stats.get('in_progress', 0) if isinstance(stats, (dict, pd.Series)) else 0
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{in_progress}</div>
            <div class="stat-label">In Progress</div>
        </div>
        """, unsafe_allow_html=True)


def render_briefing(data: Dict[str, Any], project_key: str):
    """Render the morning briefing section."""
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Your Briefing</div>', unsafe_allow_html=True)

    completed = data.get('completed', pd.DataFrame())
    blockers = data.get('blockers', pd.DataFrame())
    created = data.get('created', pd.DataFrame())
    high_priority = data.get('high_priority', pd.DataFrame())

    # Good News
    st.markdown('<div class="briefing-section">', unsafe_allow_html=True)
    st.markdown('<div class="briefing-header">The Good News</div>', unsafe_allow_html=True)

    if not completed.empty:
        completed_count = len(completed)
        ticket_keys = ", ".join(completed['key'].head(3).tolist())
        points = completed['story_points'].sum() if 'story_points' in completed.columns else 0

        st.markdown(f"""
        <div class="good-news-card">
            <div class="card-text">
                <strong>{completed_count} ticket(s) completed</strong> today ({points:.0f} points delivered)
                <span class="evidence-badge">{ticket_keys}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="good-news-card">
            <div class="card-text">
                Team is working on making progress. Check back later for updates!
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Concerns
    if not blockers.empty or len(created) > len(completed) * 2:
        st.markdown('<div class="briefing-section">', unsafe_allow_html=True)
        st.markdown('<div class="briefing-header">The Concern</div>', unsafe_allow_html=True)

        if not blockers.empty:
            blocker_keys = ", ".join(blockers['key'].head(3).tolist())
            st.markdown(f"""
            <div class="concern-card">
                <div class="card-text">
                    <strong>{len(blockers)} active blocker(s)</strong> need attention
                    <span class="evidence-badge">{blocker_keys}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if len(created) > len(completed) * 2:
            st.markdown(f"""
            <div class="concern-card">
                <div class="card-text">
                    <strong>Backlog growing</strong>: {len(created)} created vs {len(completed)} completed
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Risks
    critical_items = high_priority[high_priority['priority'] == 'Highest'] if not high_priority.empty and 'priority' in high_priority.columns else pd.DataFrame()

    if not critical_items.empty:
        st.markdown('<div class="briefing-section">', unsafe_allow_html=True)
        st.markdown('<div class="briefing-header">The Risk</div>', unsafe_allow_html=True)

        for _, item in critical_items.head(2).iterrows():
            st.markdown(f"""
            <div class="risk-card">
                <div class="card-text">
                    <strong>{item['key']}</strong>: {str(item['summary'])[:80]}...
                    <span class="evidence-badge">Priority: Highest</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Recommendation
    if not blockers.empty:
        top_blocker = blockers.iloc[0]['key']
        recommendation = f"Focus on unblocking {top_blocker} today to keep the team moving."
    elif not critical_items.empty:
        top_critical = critical_items.iloc[0]['key']
        recommendation = f"Prioritize {top_critical} - it's marked as highest priority."
    elif len(created) > len(completed):
        recommendation = "Consider reviewing incoming work to prevent backlog growth."
    else:
        recommendation = "Keep up the great work! Monitor progress and stay proactive."

    st.markdown(f"""
    <div class="recommendation-card">
        <div class="recommendation-title">🎯 Top Recommendation</div>
        <div class="recommendation-text">{recommendation}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_decision_queue(data: Dict[str, Any]):
    """Render the decision queue section."""
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">⚡ Decision Queue</div>', unsafe_allow_html=True)

    high_priority = data.get('high_priority', pd.DataFrame())
    blockers = data.get('blockers', pd.DataFrame())

    # Combine and prioritize items
    items = []

    # Add blockers first
    if not blockers.empty:
        for _, row in blockers.head(3).iterrows():
            items.append({
                'key': row['key'],
                'summary': row['summary'],
                'severity': 'critical',
                'reason': 'Active blocker - blocking team progress'
            })

    # Add high priority items
    if not high_priority.empty:
        for _, row in high_priority.head(5 - len(items)).iterrows():
            if row['key'] not in [i['key'] for i in items]:
                priority = row.get('priority', 'Medium')
                items.append({
                    'key': row['key'],
                    'summary': row['summary'],
                    'severity': 'high' if priority == 'Highest' else 'medium',
                    'reason': f'{priority} priority - needs attention'
                })

    if not items:
        st.markdown("""
        <div style="text-align: center; padding: 40px; color: #16a34a;">
            <span style="font-size: 32px;">✓</span>
            <p style="color: #64748b; margin-top: 10px;">No items requiring immediate attention!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for idx, item in enumerate(items[:5], 1):
            severity = item['severity']
            st.markdown(f"""
            <div class="decision-item {severity}">
                <span class="decision-priority priority-{severity}">{idx}</span>
                <span class="decision-key">{item['key']}</span>
                <div class="decision-summary">{str(item['summary'])[:60]}{'...' if len(str(item['summary'])) > 60 else ''}</div>
                <div class="decision-reason">{item['reason']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_ai_insights(data: Dict[str, Any]):
    """Render AI-powered insights section."""
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 AI Insights</div>', unsafe_allow_html=True)

    insights = generate_ai_insights(data)

    for insight in insights:
        st.markdown(f"""
        <div class="ai-insight">
            <div class="ai-text">{insight}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_team_activity(data: Dict[str, Any]):
    """Render team activity section."""
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👥 Team Activity</div>', unsafe_allow_html=True)

    team = data.get('team', pd.DataFrame())

    if team.empty:
        st.markdown("""
        <p style="color: #8892b0; text-align: center; padding: 20px;">
            No team activity data available
        </p>
        """, unsafe_allow_html=True)
    else:
        for _, member in team.iterrows():
            name = member['assignee_name']
            initials = ''.join([n[0].upper() for n in str(name).split()[:2]])
            color = get_avatar_color(name)
            completed = member.get('completed', 0)
            in_progress = member.get('in_progress', 0)

            st.markdown(f"""
            <div class="team-member-row">
                <div class="team-avatar" style="background: {color};">{initials}</div>
                <div class="team-name">{name}</div>
                <div class="team-stat">{completed} done • {in_progress} active</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


def render_recent_activity(data: Dict[str, Any]):
    """Render recent activity tables."""
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Recent Activity</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Completed", "Created", "Blockers"])

    with tab1:
        completed = data.get('completed', pd.DataFrame())
        if not completed.empty:
            display_df = completed[['key', 'summary', 'assignee_name', 'story_points']].head(10)
            display_df.columns = ['Key', 'Summary', 'Assignee', 'Points']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No tickets completed in the last 24 hours")

    with tab2:
        created = data.get('created', pd.DataFrame())
        if not created.empty:
            display_df = created[['key', 'summary', 'issue_type', 'priority']].head(10)
            display_df.columns = ['Key', 'Summary', 'Type', 'Priority']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No tickets created in the last 24 hours")

    with tab3:
        blockers = data.get('blockers', pd.DataFrame())
        if not blockers.empty:
            display_df = blockers[['key', 'summary', 'assignee_name', 'priority']].head(10)
            display_df.columns = ['Key', 'Summary', 'Assignee', 'Priority']
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.success("No active blockers!")

    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Main dashboard function."""
    conn = get_connection()
    if not conn:
        st.error("Database not found. Please sync data first.")
        st.stop()

    # Get available projects
    try:
        projects = conn.execute(
            "SELECT DISTINCT project_key FROM issues WHERE project_key IS NOT NULL"
        ).fetchall()
        project_keys = [p[0] for p in projects if p[0]]
    except Exception as e:
        st.error(f"Failed to fetch projects: {e}")
        st.stop()

    if not project_keys:
        st.warning("No projects found. Please sync data first.")
        st.stop()

    # Sidebar
    with st.sidebar:
        st.markdown("### Settings")
        project_key = st.selectbox("Project", options=project_keys, index=0)

        if st.button("🔄 Refresh", type="primary"):
            st.rerun()

    # Render greeting
    render_greeting()

    # ========== QUICK WIN: YOUR #1 PRIORITY TODAY ==========
    priority_data = get_morning_priority(conn, project_key)
    if priority_data:
        severity = priority_data.get('severity', 'normal')
        severity_colors = {'low': '#3b82f6', 'normal': '#f59e0b', 'high': '#ef4444'}
        severity_color = severity_colors.get(severity, '#667eea')
        blocked_html = f'<div style="margin-top: 8px; color: #ff6b81;">⚠️ Blocked by: {priority_data["blocked_by"]}</div>' if priority_data.get('blocked_by') else ''

        st.markdown(f"""
        <div style="background: linear-gradient(135deg, {severity_color} 0%, {severity_color}dd 100%); border-radius: 20px; padding: 24px 28px; margin-bottom: 24px; color: white; box-shadow: 0 10px 40px {severity_color}66; position: relative; overflow: hidden;">
            <div style="position: absolute; top: 16px; right: 16px; background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 12px; font-size: 10px; font-weight: 600;">⏱️ 20 min saved</div>
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span style="background: rgba(255,255,255,0.2); padding: 6px 14px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 1px;">
                    <span style="width: 8px; height: 8px; border-radius: 50%; background: #fff; display: inline-block; animation: pulse 2s infinite; margin-right: 6px;"></span>
                    #1 PRIORITY TODAY
                </span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; gap: 24px;">
                <div style="flex: 1;">
                    <div style="font-size: 13px; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Focus On This First</div>
                    <div style="font-size: 28px; font-weight: 800; line-height: 1.1; margin-bottom: 8px;">{priority_data['title']}</div>
                    <div style="font-size: 14px; opacity: 0.85;">{priority_data['reason']}</div>
                    {blocked_html}
                </div>
                <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 16px 20px; min-width: 180px; text-align: center;">
                    <div style="font-size: 11px; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Impact</div>
                    <div style="font-size: 16px; font-weight: 700; line-height: 1.3;">{priority_data['impact']}</div>
                </div>
            </div>
        </div>
        <style>@keyframes pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.5; transform: scale(1.3); }} }}</style>
        """, unsafe_allow_html=True)

    # Generate data
    with st.spinner("Loading your briefing..."):
        data = generate_briefing_data(conn, project_key)

    # Quick Stats
    render_quick_stats(data)

    # ========== ULTRATHINK: STANDUP CHEAT SHEET ==========
    standup_script = get_standup_script(data, priority_data)
    
    script_html = "".join([f'<div style="margin-bottom: 8px; font-size: 14px; color: #334155;">{line}</div>' for line in standup_script])
    
    st.markdown(f"""
    <div style="background: white; border: 1px dashed #cbd5e1; border-radius: 12px; padding: 16px; margin-bottom: 24px;">
        <div style="font-size: 12px; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
            <span>🎙️ Standup Cheat Sheet</span>
            <span style="font-size: 10px; font-weight: 400; text-transform: none; background: #f1f5f9; padding: 2px 8px; border-radius: 10px;">Read this aloud</span>
        </div>
        <div>{script_html}</div>
    </div>
    """, unsafe_allow_html=True)


    # Main content layout
    col_left, col_right = st.columns([2, 1])

    with col_left:
        render_briefing(data, project_key)
        render_ai_insights(data)

    with col_right:
        render_decision_queue(data)
        render_team_activity(data)

    # Recent Activity (full width)
    render_recent_activity(data)

    # Footer
    st.markdown(f"""
    <div class="dashboard-footer">
        Generated at {datetime.now().strftime('%I:%M %p')} | Project: {project_key} |
        Made with Premium Analytics
    </div>
    """, unsafe_allow_html=True)

    conn.close()


if __name__ == "__main__":
    main()
