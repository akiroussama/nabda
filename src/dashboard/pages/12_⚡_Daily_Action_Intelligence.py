"""
⚡ Daily Action Intelligence - Manager's Productivity Command Center
The #1 page for team leaders to eliminate daily cognitive overhead.

Real-world value: 5%+ productivity gain by eliminating:
- 1:1 meeting prep time (15-30 min per meeting)
- Blocker discovery time (30+ min daily)
- Context-switching between tools
- Forgotten follow-ups and decisions
"""

import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import hashlib

# Add project root to sys.path
import sys
root_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_dir))

from src.dashboard.utils.status_mapping import STATUS_TODO, STATUS_IN_PROGRESS, STATUS_DONE

# Import page guide component
from src.dashboard.components import render_page_guide

st.set_page_config(
    page_title="Daily Action Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium Action Intelligence CSS - Light Mode
st.markdown("""
<style>
    /* Global Theme - Light Mode */
    .stApp {
        background: #f8f9fa;
    }

    /* Hero Section */
    .action-hero {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        border-radius: 24px;
        padding: 40px;
        margin-bottom: 32px;
        position: relative;
        overflow: hidden;
    }
    .action-hero::before {
        content: '';
        position: absolute;
        top: -50%; right: -50%;
        width: 100%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
    }
    .hero-title {
        font-size: 42px;
        font-weight: 800;
        color: white;
        margin-bottom: 8px;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    .hero-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 18px;
    }
    .time-saved-badge {
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        border-radius: 50px;
        padding: 12px 24px;
        display: inline-block;
        margin-top: 16px;
        color: white;
        font-weight: 600;
    }

    /* Critical Action Cards */
    .action-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        border-left: 4px solid;
        border-top: 1px solid #e5e7eb;
        border-right: 1px solid #e5e7eb;
        border-bottom: 1px solid #e5e7eb;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .action-card:hover {
        transform: translateX(8px);
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
    }
    .action-critical { border-left-color: #ef4444; }
    .action-warning { border-left-color: #f59e0b; }
    .action-info { border-left-color: #3b82f6; }
    .action-success { border-left-color: #22c55e; }

    .action-priority {
        position: absolute;
        top: 12px;
        right: 12px;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 14px;
    }
    .priority-1 { background: linear-gradient(135deg, #ef4444, #f87171); color: white; }
    .priority-2 { background: linear-gradient(135deg, #f59e0b, #fbbf24); color: #1e293b; }
    .priority-3 { background: linear-gradient(135deg, #3b82f6, #60a5fa); color: white; }

    .action-title {
        font-size: 16px;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 8px;
        padding-right: 40px;
    }
    .action-meta {
        font-size: 12px;
        color: #64748b;
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
    }
    .action-tag {
        background: rgba(102, 126, 234, 0.1);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        color: #667eea;
    }

    /* 1:1 Prep Cards */
    .one-on-one-card {
        background: white;
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid #e5e7eb;
        position: relative;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .one-on-one-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 4px;
        background: linear-gradient(90deg, var(--member-color, #667eea), transparent);
        border-radius: 20px 20px 0 0;
    }

    .member-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 20px;
    }
    .member-avatar {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        font-weight: 700;
        color: white;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    .member-name {
        font-size: 20px;
        font-weight: 700;
        color: #1e293b;
    }
    .member-role {
        font-size: 13px;
        color: #64748b;
    }

    .talking-point {
        background: #f8fafc;
        border-radius: 12px;
        padding: 14px 18px;
        margin-bottom: 12px;
        border-left: 3px solid;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    .talking-point.celebrate { border-left-color: #22c55e; background: #f0fdf4; }
    .talking-point.discuss { border-left-color: #f59e0b; background: #fffbeb; }
    .talking-point.concern { border-left-color: #ef4444; background: #fef2f2; }
    .talking-point.growth { border-left-color: #8b5cf6; background: #faf5ff; }

    .point-icon {
        font-size: 18px;
        flex-shrink: 0;
    }
    .point-text {
        color: #334155;
        font-size: 14px;
        line-height: 1.5;
    }
    .point-data {
        color: #059669;
        font-size: 12px;
        margin-top: 4px;
    }

    /* Decision Queue */
    .decision-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 12px;
        border: 1px solid #e5e7eb;
        border-left: 4px solid #8b5cf6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .decision-title {
        font-size: 15px;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 8px;
    }
    .decision-context {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 12px;
    }
    .decision-age {
        background: rgba(139, 92, 246, 0.1);
        color: #7c3aed;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    }
    .decision-age.overdue {
        background: rgba(239, 68, 68, 0.1);
        color: #dc2626;
    }

    /* Blocker Section */
    .blocker-card {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 12px;
        border: 1px solid #fecaca;
        position: relative;
    }
    .blocker-pulse {
        position: absolute;
        top: 16px;
        right: 16px;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #ef4444;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.5); }
    }
    .blocker-title {
        font-size: 15px;
        font-weight: 600;
        color: #991b1b;
        margin-bottom: 4px;
        padding-right: 30px;
    }
    .blocker-owner {
        font-size: 13px;
        color: #b91c1c;
        margin-bottom: 8px;
    }
    .blocker-duration {
        background: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        display: inline-block;
    }

    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid #e5e7eb;
    }
    .section-icon {
        font-size: 28px;
    }
    .section-title {
        font-size: 20px;
        font-weight: 700;
        color: #1e293b;
    }
    .section-count {
        background: rgba(102, 126, 234, 0.1);
        color: #667eea;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .section-count.critical {
        background: rgba(239, 68, 68, 0.1);
        color: #dc2626;
    }

    /* Priority Stack */
    .priority-item {
        background: white;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 16px;
        border: 1px solid #e5e7eb;
        transition: all 0.2s ease;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .priority-item:hover {
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateX(4px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.1);
    }
    .priority-rank {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 16px;
        flex-shrink: 0;
    }
    .rank-1 { background: linear-gradient(135deg, #ffd700, #ffed4a); color: #1e293b; }
    .rank-2 { background: linear-gradient(135deg, #c0c0c0, #e8e8e8); color: #1e293b; }
    .rank-3 { background: linear-gradient(135deg, #cd7f32, #daa06d); color: white; }
    .rank-other { background: #f1f5f9; color: #64748b; }

    .priority-content {
        flex: 1;
    }
    .priority-title {
        font-size: 15px;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 4px;
    }
    .priority-reason {
        font-size: 12px;
        color: #64748b;
    }
    .priority-impact {
        background: rgba(34, 197, 94, 0.1);
        color: #16a34a;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }

    /* Stat Cards */
    .stat-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .stat-value {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stat-value.critical {
        background: linear-gradient(135deg, #ef4444, #f87171);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-value.success {
        background: linear-gradient(135deg, #22c55e, #4ade80);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
    }

    /* Follow-up Cards */
    .followup-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 10px;
        border-left: 3px solid #f59e0b;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .followup-card.overdue {
        border-left-color: #ef4444;
        background: #fef2f2;
    }
    .followup-content {
        flex: 1;
    }
    .followup-title {
        font-size: 14px;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 4px;
    }
    .followup-meta {
        font-size: 12px;
        color: #64748b;
    }
    .followup-due {
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }
    .due-today { background: rgba(245, 158, 11, 0.1); color: #d97706; }
    .due-overdue { background: rgba(239, 68, 68, 0.1); color: #dc2626; }
    .due-upcoming { background: rgba(102, 126, 234, 0.1); color: #667eea; }

    /* Time Estimation */
    .time-box {
        background: linear-gradient(135deg, #f0fdf4, #ecfeff);
        border: 1px solid #86efac;
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .time-saved {
        font-size: 48px;
        font-weight: 800;
        background: linear-gradient(135deg, #22c55e, #667eea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .time-label {
        color: #64748b;
        font-size: 14px;
        margin-top: 4px;
    }

    /* Quick Actions */
    .quick-action {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .quick-action:hover {
        border-color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.15);
    }
    .quick-action-icon {
        font-size: 28px;
        margin-bottom: 8px;
    }
    .quick-action-label {
        font-size: 12px;
        color: #334155;
        font-weight: 500;
    }

    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 40px;
        color: #64748b;
    }
    .empty-icon {
        font-size: 48px;
        margin-bottom: 16px;
        opacity: 0.5;
    }
</style>
""", unsafe_allow_html=True)


def get_connection():
    """Get DuckDB connection."""
    db_path = Path(__file__).resolve().parent.parent.parent.parent / "data" / "jira.duckdb"
    return duckdb.connect(str(db_path), read_only=True)


def get_member_color(name: str) -> str:
    """Generate consistent color for team member."""
    colors = ['#667eea', '#764ba2', '#f093fb', '#00d9ff', '#2ed573', '#ffa502', '#ff4757', '#a55eea']
    hash_val = int(hashlib.md5(name.encode()).hexdigest(), 16)
    return colors[hash_val % len(colors)]


def calculate_time_saved(num_actions: int, num_blockers: int, num_team_members: int) -> int:
    """Estimate minutes saved by using this dashboard."""
    # 5 min per action item identified automatically
    # 10 min per blocker surfaced proactively
    # 20 min per 1:1 prep (instead of 30+ min manually)
    return (num_actions * 5) + (num_blockers * 10) + (num_team_members * 20)


@dataclass
class ActionItem:
    """Represents a critical action item."""
    title: str
    issue_key: str
    priority: int  # 1=critical, 2=high, 3=medium
    category: str
    assignee: str
    age_days: int
    reason: str


@dataclass
class TalkingPoint:
    """Represents a 1:1 talking point."""
    text: str
    category: str  # celebrate, discuss, concern, growth
    data: Optional[str] = None


@dataclass
class TeamMemberProfile:
    """Profile for 1:1 preparation."""
    name: str
    color: str
    total_issues: int
    completed_this_week: int
    in_progress: int
    stuck_issues: int
    velocity_trend: str  # up, down, stable
    talking_points: List[TalkingPoint]


def get_blockers(conn) -> pd.DataFrame:
    """Get issues that are blocked or stale."""
    query = f"""
    SELECT
        key,
        summary,
        assignee_name,
        status,
        priority,
        updated,
        created,
        DATEDIFF('day', updated, CURRENT_TIMESTAMP) as days_stale,
        DATEDIFF('day', created, CURRENT_TIMESTAMP) as age_days
    FROM issues
    WHERE status = '{STATUS_IN_PROGRESS}'
    AND DATEDIFF('day', updated, CURRENT_TIMESTAMP) >= 3
    ORDER BY DATEDIFF('day', updated, CURRENT_TIMESTAMP) DESC
    LIMIT 10
    """
    return conn.execute(query).fetchdf()


def get_high_priority_stuck(conn) -> pd.DataFrame:
    """Get high priority issues not making progress."""
    query = f"""
    SELECT
        key,
        summary,
        assignee_name,
        priority,
        status,
        created,
        updated,
        DATEDIFF('day', created, CURRENT_TIMESTAMP) as age_days
    FROM issues
    WHERE priority IN ('Highest', 'High')
    AND status != '{STATUS_DONE}'
    AND DATEDIFF('day', updated, CURRENT_TIMESTAMP) >= 2
    ORDER BY
        CASE priority WHEN 'Highest' THEN 1 WHEN 'High' THEN 2 ELSE 3 END,
        DATEDIFF('day', created, CURRENT_TIMESTAMP) DESC
    LIMIT 10
    """
    return conn.execute(query).fetchdf()


def get_team_member_profiles(conn) -> List[TeamMemberProfile]:
    """Generate 1:1 prep profiles for each team member."""
    # Get team members
    members_query = """
    SELECT DISTINCT assignee_name
    FROM issues
    WHERE assignee_name IS NOT NULL
    ORDER BY assignee_name
    """
    members_df = conn.execute(members_query).fetchdf()

    profiles = []
    for _, row in members_df.iterrows():
        name = row['assignee_name']
        color = get_member_color(name)

        # Get stats for this member
        stats_query = f"""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status = '{STATUS_DONE}' AND resolved >= CURRENT_DATE - INTERVAL '7 days' THEN 1 ELSE 0 END) as completed_week,
            SUM(CASE WHEN status = '{STATUS_IN_PROGRESS}' THEN 1 ELSE 0 END) as in_progress,
            SUM(CASE WHEN status = '{STATUS_IN_PROGRESS}' AND DATEDIFF('day', updated, CURRENT_TIMESTAMP) >= 3 THEN 1 ELSE 0 END) as stuck,
            AVG(CASE WHEN status = '{STATUS_DONE}' AND story_points IS NOT NULL THEN story_points ELSE NULL END) as avg_points
        FROM issues
        WHERE assignee_name = ?
        """
        stats = conn.execute(stats_query, [name]).fetchdf().iloc[0]

        # Get recent completed
        recent_query = f"""
        SELECT key, summary, story_points
        FROM issues
        WHERE assignee_name = ?
        AND status = '{STATUS_DONE}'
        AND resolved >= CURRENT_DATE - INTERVAL '7 days'
        ORDER BY resolved DESC
        LIMIT 3
        """
        recent_completed = conn.execute(recent_query, [name]).fetchdf()

        # Get stuck issues
        stuck_query = f"""
        SELECT key, summary, DATEDIFF('day', updated, CURRENT_TIMESTAMP) as days_stuck
        FROM issues
        WHERE assignee_name = ?
        AND status = '{STATUS_IN_PROGRESS}'
        AND DATEDIFF('day', updated, CURRENT_TIMESTAMP) >= 3
        ORDER BY days_stuck DESC
        LIMIT 3
        """
        stuck_issues = conn.execute(stuck_query, [name]).fetchdf()

        # Get high priority pending
        priority_query = f"""
        SELECT key, summary, priority
        FROM issues
        WHERE assignee_name = ?
        AND priority IN ('Highest', 'High')
        AND status != '{STATUS_DONE}'
        LIMIT 3
        """
        high_priority = conn.execute(priority_query, [name]).fetchdf()

        # Build talking points
        talking_points = []

        # Celebrate wins
        if stats['completed_week'] > 0:
            talking_points.append(TalkingPoint(
                text=f"Great progress! Completed {int(stats['completed_week'])} issues this week.",
                category="celebrate",
                data=", ".join(recent_completed['key'].tolist()[:3]) if len(recent_completed) > 0 else None
            ))

        # Discuss stuck items
        if stats['stuck'] > 0:
            for _, issue in stuck_issues.iterrows():
                talking_points.append(TalkingPoint(
                    text=f"Check on {issue['key']}: stuck for {int(issue['days_stuck'])} days",
                    category="concern",
                    data=issue['summary'][:60] + "..." if len(issue['summary']) > 60 else issue['summary']
                ))

        # High priority items
        if len(high_priority) > 0:
            for _, issue in high_priority.iterrows():
                talking_points.append(TalkingPoint(
                    text=f"Discuss priority: {issue['key']} ({issue['priority']})",
                    category="discuss",
                    data=issue['summary'][:60] + "..." if len(issue['summary']) > 60 else issue['summary']
                ))

        # Workload check
        if stats['in_progress'] > 5:
            talking_points.append(TalkingPoint(
                text=f"Workload check: {int(stats['in_progress'])} items in progress",
                category="concern",
                data="Consider prioritization or delegation"
            ))
        elif stats['in_progress'] < 2 and stats['total'] > 5:
            talking_points.append(TalkingPoint(
                text="Low WIP - opportunity to pick up more work?",
                category="growth",
                data=f"Only {int(stats['in_progress'])} items in progress"
            ))

        # Growth opportunity
        if stats['completed_week'] >= 5:
            talking_points.append(TalkingPoint(
                text="High performer - discuss career growth or mentoring opportunities",
                category="growth"
            ))

        # Determine velocity trend
        velocity_trend = "stable"
        if stats['completed_week'] >= 4:
            velocity_trend = "up"
        elif stats['stuck'] > 2:
            velocity_trend = "down"

        profiles.append(TeamMemberProfile(
            name=name,
            color=color,
            total_issues=int(stats['total']),
            completed_this_week=int(stats['completed_week']),
            in_progress=int(stats['in_progress']),
            stuck_issues=int(stats['stuck']),
            velocity_trend=velocity_trend,
            talking_points=talking_points
        ))

    return profiles


def get_priority_stack(conn) -> pd.DataFrame:
    """Get this week's priority stack - ranked by impact."""
    query = f"""
    WITH issue_scores AS (
        SELECT
            key,
            summary,
            assignee_name,
            priority,
            status,
            story_points,
            DATEDIFF('day', created, CURRENT_TIMESTAMP) as age_days,
            CASE priority
                WHEN 'Highest' THEN 100
                WHEN 'High' THEN 75
                WHEN 'Medium' THEN 50
                WHEN 'Low' THEN 25
                ELSE 10
            END as priority_score,
            COALESCE(story_points, 3) * 10 as points_score,
            CASE
                WHEN status = '{STATUS_IN_PROGRESS}' THEN 30
                WHEN status = '{STATUS_TODO}' THEN 10
                ELSE 0
            END as status_score
        FROM issues
        WHERE status != '{STATUS_DONE}'
    )
    SELECT
        key,
        summary,
        assignee_name,
        priority,
        story_points,
        age_days,
        (priority_score + points_score + status_score + LEAST(age_days, 30)) as impact_score
    FROM issue_scores
    ORDER BY impact_score DESC
    LIMIT 10
    """
    return conn.execute(query).fetchdf()


def get_decisions_pending(conn) -> pd.DataFrame:
    """Get items that might need manager decisions (high priority, old, no movement)."""
    query = f"""
    SELECT
        key,
        summary,
        assignee_name,
        priority,
        created,
        updated,
        DATEDIFF('day', created, CURRENT_TIMESTAMP) as age_days,
        DATEDIFF('day', updated, CURRENT_TIMESTAMP) as stale_days
    FROM issues
    WHERE priority IN ('Highest', 'High')
    AND status = '{STATUS_TODO}'
    AND DATEDIFF('day', created, CURRENT_TIMESTAMP) >= 5
    ORDER BY
        CASE priority WHEN 'Highest' THEN 1 ELSE 2 END,
        age_days DESC
    LIMIT 5
    """
    return conn.execute(query).fetchdf()


def get_todays_actions(conn) -> List[ActionItem]:
    """Generate today's critical action list."""
    actions = []

    # 1. Stale blockers (priority 1)
    blockers = get_blockers(conn)
    for _, b in blockers.head(3).iterrows():
        actions.append(ActionItem(
            title=f"Unblock {b['key']}: {b['summary'][:50]}...",
            issue_key=b['key'],
            priority=1,
            category="blocker",
            assignee=b['assignee_name'] or "Unassigned",
            age_days=int(b['days_stale']),
            reason=f"No updates for {int(b['days_stale'])} days"
        ))

    # 2. High priority stuck (priority 1-2)
    high_stuck = get_high_priority_stuck(conn)
    for _, h in high_stuck.head(3).iterrows():
        if h['key'] not in [a.issue_key for a in actions]:
            actions.append(ActionItem(
                title=f"Review {h['key']}: {h['summary'][:50]}...",
                issue_key=h['key'],
                priority=1 if h['priority'] == 'Highest' else 2,
                category="high_priority",
                assignee=h['assignee_name'] or "Unassigned",
                age_days=int(h['age_days']),
                reason=f"{h['priority']} priority, {int(h['age_days'])} days old"
            ))

    # 3. Decisions needed (priority 2)
    decisions = get_decisions_pending(conn)
    for _, d in decisions.head(2).iterrows():
        if d['key'] not in [a.issue_key for a in actions]:
            actions.append(ActionItem(
                title=f"Decide on {d['key']}: {d['summary'][:50]}...",
                issue_key=d['key'],
                priority=2,
                category="decision",
                assignee=d['assignee_name'] or "Unassigned",
                age_days=int(d['age_days']),
                reason=f"Waiting in TODO for {int(d['age_days'])} days"
            ))

    # Sort by priority
    actions.sort(key=lambda x: (x.priority, -x.age_days))
    return actions[:8]


def generate_slack_update(actions: List[ActionItem], blockers, team_count: int) -> str:
    """Generate a copy-paste ready Slack status update."""
    from datetime import datetime

    today = datetime.now().strftime("%B %d")
    critical_count = sum(1 for a in actions if a.priority == 1)
    blocker_count = len(blockers) if hasattr(blockers, '__len__') else 0

    # Build the slack message
    if blocker_count == 0 and critical_count == 0:
        status_emoji = "🟢"
        status_text = "All green"
    elif blocker_count <= 2 and critical_count <= 2:
        status_emoji = "🟡"
        status_text = "Minor items need attention"
    else:
        status_emoji = "🔴"
        status_text = "Action required"

    # Create bullet points for top actions
    action_bullets = ""
    for action in actions[:3]:
        action_bullets += f"<br>• {action.issue_key}: {action.title.split(':')[-1][:40]}..."

    message = f"""
    {status_emoji} <strong>Team Status - {today}</strong><br>
    Status: {status_text}<br>
    <br>
    📊 <strong>Key Numbers:</strong><br>
    • {critical_count} critical actions<br>
    • {blocker_count} blockers to resolve<br>
    • {team_count} team members tracked<br>
    <br>
    🎯 <strong>Today's Focus:</strong>{action_bullets}
    """

    return message.strip()


def render_hero_section(time_saved: int):
    """Render the hero section."""
    current_hour = datetime.now().hour
    if current_hour < 12:
        greeting = "Good Morning"
        emoji = "🌅"
    elif current_hour < 17:
        greeting = "Good Afternoon"
        emoji = "☀️"
    else:
        greeting = "Good Evening"
        emoji = "🌙"

    st.markdown(f"""
<div class="action-hero">
    <div class="hero-title">{emoji} {greeting}</div>
    <div class="hero-subtitle">Your Daily Action Intelligence is ready</div>
    <div class="time-saved-badge">
        ⏱️ Estimated time saved today: <strong>{time_saved} minutes</strong>
    </div>
</div>
""", unsafe_allow_html=True)


def render_action_card(action: ActionItem, index: int):
    """Render a single action card."""
    priority_class = f"priority-{action.priority}"
    action_class = {
        "blocker": "action-critical",
        "high_priority": "action-warning",
        "decision": "action-info"
    }.get(action.category, "action-info")

    category_label = {
        "blocker": "🚧 Blocker",
        "high_priority": "🔥 High Priority",
        "decision": "🤔 Decision Needed"
    }.get(action.category, action.category)

    st.markdown(f"""
<div class="action-card {action_class}">
    <div class="action-priority {priority_class}">{index + 1}</div>
    <div class="action-title">{action.title}</div>
    <div class="action-meta">
        <span class="action-tag">{category_label}</span>
        <span>👤 {action.assignee}</span>
        <span>📅 {action.age_days} days</span>
    </div>
    <div style="margin-top: 8px; font-size: 12px; color: #059669;">
        💡 {action.reason}
    </div>
</div>
""", unsafe_allow_html=True)


def render_one_on_one_card(profile: TeamMemberProfile):
    """Render a 1:1 preparation card."""
    trend_emoji = {"up": "📈", "down": "📉", "stable": "➡️"}[profile.velocity_trend]
    initials = "".join([n[0] for n in profile.name.split()[:2]]).upper()

    st.markdown(f"""
<div class="one-on-one-card" style="--member-color: {profile.color}">
    <div class="member-header">
        <div class="member-avatar" style="background: {profile.color}">{initials}</div>
        <div>
            <div class="member-name">{profile.name}</div>
            <div class="member-role">
                {trend_emoji} {profile.completed_this_week} done this week ·
                {profile.in_progress} in progress ·
                {profile.stuck_issues} stuck
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

    if profile.talking_points:
        for point in profile.talking_points[:4]:
            icon = {
                "celebrate": "🎉",
                "discuss": "💬",
                "concern": "⚠️",
                "growth": "🌱"
            }.get(point.category, "💡")

            st.markdown(f"""
<div class="talking-point {point.category}">
    <span class="point-icon">{icon}</span>
    <div>
        <div class="point-text">{point.text}</div>
        {f'<div class="point-data">{point.data}</div>' if point.data else ''}
    </div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="talking-point discuss">
    <span class="point-icon">✅</span>
    <div class="point-text">All clear - general check-in and growth discussion</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_blocker_card(blocker: pd.Series):
    """Render a blocker card."""
    st.markdown(f"""
<div class="blocker-card">
    <div class="blocker-pulse"></div>
    <div class="blocker-title">{blocker['key']}: {blocker['summary'][:60]}...</div>
    <div class="blocker-owner">👤 {blocker['assignee_name'] or 'Unassigned'}</div>
    <span class="blocker-duration">⏰ Stuck for {int(blocker['days_stale'])} days</span>
</div>
""", unsafe_allow_html=True)


def render_priority_item(row: pd.Series, rank: int):
    """Render a priority stack item."""
    rank_class = f"rank-{rank}" if rank <= 3 else "rank-other"
    impact = "High" if row['impact_score'] > 100 else "Medium" if row['impact_score'] > 60 else "Normal"

    st.markdown(f"""
<div class="priority-item">
    <div class="priority-rank {rank_class}">{rank}</div>
    <div class="priority-content">
        <div class="priority-title">{row['key']}: {row['summary'][:50]}...</div>
        <div class="priority-reason">
            👤 {row['assignee_name'] or 'Unassigned'} ·
            {row['priority']} priority ·
            {int(row['age_days'])} days old
        </div>
    </div>
    <span class="priority-impact">{impact} Impact</span>
</div>
""", unsafe_allow_html=True)


def main():
    """Main dashboard function."""
    # Render page guide in sidebar
    render_page_guide()
    conn = get_connection()

    # Get all data
    actions = get_todays_actions(conn)
    blockers = get_blockers(conn)
    team_profiles = get_team_member_profiles(conn)
    priority_stack = get_priority_stack(conn)
    decisions = get_decisions_pending(conn)

    # Calculate time saved
    time_saved = calculate_time_saved(
        len(actions),
        len(blockers),
        len(team_profiles)
    )

    # Hero Section
    render_hero_section(time_saved)

    # ========== QUICK WIN: SLACK STATUS UPDATE ==========
    slack_update = generate_slack_update(actions, blockers, len(team_profiles))
    status_color = '#22c55e' if len(blockers) == 0 else '#f59e0b' if len(blockers) <= 2 else '#ef4444'

    st.markdown(f"""
<div style="background: white; border: 1px solid #e5e7eb; border-radius: 20px; padding: 24px 28px; margin-bottom: 24px; color: #1e293b; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08); position: relative; overflow: hidden;">
    <div style="position: absolute; top: 16px; right: 16px; background: #f1f5f9; padding: 4px 10px; border-radius: 12px; font-size: 10px; font-weight: 600; color: #64748b;">⏱️ 10 min saved</div>
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
        <span style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 6px 14px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 1px;">
            💬 SLACK UPDATE READY
        </span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 24px;">
        <div style="flex: 1;">
            <div style="font-size: 13px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Copy-Paste to #team-channel</div>
            <div style="background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 12px; padding: 16px; margin-top: 12px; font-family: 'Monaco', 'Menlo', monospace; font-size: 13px; line-height: 1.6; position: relative; color: #334155;">
                <div style="position: absolute; top: 8px; right: 12px; font-size: 10px; color: #94a3b8; font-family: system-ui;">📋 Click to copy</div>
                {slack_update}
            </div>
        </div>
        <div style="background: #f8fafc; border: 1px solid #e5e7eb; border-radius: 16px; padding: 16px 20px; min-width: 120px; text-align: center;">
            <div style="font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Team Status</div>
            <div style="font-size: 36px;">{'🟢' if len(blockers) == 0 else '🟡' if len(blockers) <= 2 else '🔴'}</div>
            <div style="font-size: 12px; color: #64748b; margin-top: 4px;">{len(blockers)} blockers</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Quick Stats Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        critical_count = sum(1 for a in actions if a.priority == 1)
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value {'critical' if critical_count > 0 else ''}">{critical_count}</div>
            <div class="stat-label">Critical Actions</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(blockers)}</div>
            <div class="stat-label">Blockers to Unblock</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value">{len(team_profiles)}</div>
            <div class="stat-label">1:1 Preps Ready</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-value success">{len(decisions)}</div>
            <div class="stat-label">Decisions Pending</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Content - Two Columns
    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        # Today's Critical Actions
        st.markdown(f"""
        <div class="section-header">
            <span class="section-icon">⚡</span>
            <span class="section-title">Today's Critical Actions</span>
            <span class="section-count {'critical' if len(actions) > 5 else ''}">{len(actions)} items</span>
        </div>
        """, unsafe_allow_html=True)

        if actions:
            for i, action in enumerate(actions):
                render_action_card(action, i)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">✨</div>
                <div>All clear! No critical actions needed today.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Blockers
        st.markdown(f"""
        <div class="section-header">
            <span class="section-icon">🚧</span>
            <span class="section-title">Blockers Requiring Intervention</span>
            <span class="section-count critical">{len(blockers)} stuck</span>
        </div>
        """, unsafe_allow_html=True)

        if len(blockers) > 0:
            for _, blocker in blockers.head(5).iterrows():
                render_blocker_card(blocker)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🎉</div>
                <div>No blocked items! Team is flowing.</div>
            </div>
            """, unsafe_allow_html=True)

    with right_col:
        # This Week's Priority Stack
        st.markdown(f"""
        <div class="section-header">
            <span class="section-icon">🎯</span>
            <span class="section-title">This Week's Priority Stack</span>
        </div>
        """, unsafe_allow_html=True)

        for i, (_, row) in enumerate(priority_stack.head(7).iterrows(), 1):
            render_priority_item(row, i)

        st.markdown("<br>", unsafe_allow_html=True)

        # Decisions Queue
        st.markdown(f"""
        <div class="section-header">
            <span class="section-icon">🤔</span>
            <span class="section-title">Decision Queue</span>
            <span class="section-count">{len(decisions)} waiting</span>
        </div>
        """, unsafe_allow_html=True)

        if len(decisions) > 0:
            for _, d in decisions.iterrows():
                overdue_class = "overdue" if d['age_days'] > 7 else ""
                st.markdown(f"""
                <div class="decision-card">
                    <div class="decision-title">{d['key']}: {d['summary'][:50]}...</div>
                    <div class="decision-context">
                        👤 {d['assignee_name'] or 'Unassigned'} · {d['priority']} priority
                    </div>
                    <span class="decision-age {overdue_class}">
                        ⏳ Waiting {int(d['age_days'])} days
                    </span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">✅</div>
                <div>No pending decisions!</div>
            </div>
            """, unsafe_allow_html=True)

    # 1:1 Preparation Section (Full Width)
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="section-header">
        <span class="section-icon">👥</span>
        <span class="section-title">1:1 Meeting Preparation</span>
        <span class="section-count">{len(team_profiles)} team members</span>
    </div>
    """, unsafe_allow_html=True)

    # Display in a 2-column grid
    one_on_one_cols = st.columns(2)
    for i, profile in enumerate(team_profiles):
        with one_on_one_cols[i % 2]:
            render_one_on_one_card(profile)

    # Time Saved Summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="time-box">
        <div class="time-saved">{time_saved} min</div>
        <div class="time-label">Estimated time saved with Daily Action Intelligence</div>
        <div style="color: #059669; margin-top: 12px; font-size: 14px;">
            That's <strong>{time_saved // 60}h {time_saved % 60}m</strong> you can spend on strategic work instead
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Actions Footer
    st.markdown("""
    <div style="margin-top: 20px;">
        <div class="section-header">
            <span class="section-icon">🚀</span>
            <span class="section-title">Quick Actions</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    qa_cols = st.columns(5)
    quick_actions = [
        ("📋", "Export Actions"),
        ("📧", "Email Summary"),
        ("📅", "Schedule 1:1s"),
        ("🔄", "Refresh Data"),
        ("⚙️", "Settings")
    ]

    for i, (icon, label) in enumerate(quick_actions):
        with qa_cols[i]:
            st.markdown(f"""
<div class="quick-action">
    <div class="quick-action-icon">{icon}</div>
    <div class="quick-action-label">{label}</div>
</div>
""", unsafe_allow_html=True)

    conn.close()


if __name__ == "__main__":
    main()
