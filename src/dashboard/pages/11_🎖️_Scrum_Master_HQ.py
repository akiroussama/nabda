"""
🎖️ Scrum Master Command Center - The Ultimate SM Cockpit
Real-time sprint intelligence, team dynamics, impediment tracking, and AI-powered coaching.
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
from typing import Dict, List, Any, Tuple
import random

# Import page guide component
from src.dashboard.components import render_page_guide

st.set_page_config(page_title="SM Command Center", page_icon="🎖️", layout="wide")

# Premium Command Center CSS
st.markdown("""
<style>
    /* Global Light Theme */
    .stApp {
        background-color: #f8f9fa;
    }

    .command-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .command-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    }
    .command-title {
        font-size: 36px;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .command-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 16px;
        margin-top: 8px;
        font-weight: 500;
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
        gap: 10px;
    }

    .pulse-card {
        background: white;
        border-radius: 16px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        text-align: center;
        position: relative;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .pulse-card::after {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 4px;
        background: linear-gradient(90deg, var(--pulse-color, #4f46e5), transparent);
    }

    .pulse-value {
        font-size: 42px;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .pulse-label {
        color: #64748b;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
        font-weight: 600;
    }
    .pulse-trend {
        font-size: 14px;
        margin-top: 8px;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
        font-weight: 600;
    }
    .trend-up { background: #dcfce7; color: #166534; }
    .trend-down { background: #fee2e2; color: #991b1b; }
    .trend-neutral { background: #fef3c7; color: #92400e; }

    .health-score {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        position: relative;
    }
    .health-score::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 50%;
        padding: 8px;
        background: conic-gradient(var(--score-color, #4f46e5) calc(var(--score, 0) * 3.6deg), #e2e8f0 0);
        -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 12px), #fff calc(100% - 11px));
        mask: radial-gradient(farthest-side, transparent calc(100% - 12px), #fff calc(100% - 11px));
    }
    .health-value {
        font-size: 48px;
        font-weight: 800;
        color: #1a202c;
    }
    .health-label {
        color: #64748b;
        font-size: 14px;
    }

    .impediment-card {
        background: #fef2f2;
        border: 1px solid #fee2e2;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .impediment-critical {
        border-left: 4px solid #ef4444;
        animation: pulse-red 2s infinite;
    }
    @keyframes pulse-red {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    }
    .impediment-warning {
        background: #fff7ed;
        border: 1px solid #ffedd5;
        border-left: 4px solid #f59e0b;
    }

    .coaching-tip {
        background: #eff6ff;
        border: 1px solid #dbeafe;
        border-left: 4px solid #3b82f6;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .ceremony-card {
        background: white;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #e2e8f0;
        margin-bottom: 12px;
    }

    .progress-ring {
        width: 60px;
        height: 60px;
    }

    .team-member {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px;
        background: white;
        border-radius: 10px;
        margin-bottom: 8px;
        border: 1px solid #e2e8f0;
    }

    .member-avatar {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 14px;
        color: white;
    }

    .scope-bar {
        height: 24px;
        border-radius: 12px;
        background: #e2e8f0;
        overflow: hidden;
        position: relative;
    }
    .scope-original {
        height: 100%;
        background: linear-gradient(90deg, #4f46e5, #7c3aed);
        position: absolute;
        left: 0;
        top: 0;
    }
    .scope-added {
        height: 100%;
        background: #ef4444;
        position: absolute;
        top: 0;
    }
    .scope-removed {
        height: 100%;
        background: #22c55e;
        position: absolute;
        top: 0;
    }

    .digest-preview {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        font-family: 'Georgia', serif;
        color: #334155;
        max-height: 400px;
        overflow-y: auto;
    }

    .risk-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }
    .risk-critical { background: #fee2e2; color: #991b1b; }
    .risk-high { background: #ffedd5; color: #9a3412; }
    .risk-medium { background: #fef9c3; color: #854d0e; }
    .risk-low { background: #dcfce7; color: #166534; }

    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
    }

    .mini-stat {
        background: #f8fafc;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e2e8f0;
    }
    .mini-stat-value {
        font-size: 24px;
        font-weight: 700;
        color: #4f46e5;
    }
    .mini-stat-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
    }

    .action-button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
        text-align: center;
        display: inline-block;
    }
    .action-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.4);
    }
</style>
""", unsafe_allow_html=True)


def get_connection():
    db_path = Path("data/jira.duckdb")
    return duckdb.connect(str(db_path), read_only=True) if db_path.exists() else None


def get_avatar_color(name: str) -> str:
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#fa709a']
    return colors[hash(name or '') % len(colors)]


def calculate_sprint_health(conn) -> Dict[str, Any]:
    """Calculate comprehensive sprint health metrics."""
    sprint = conn.execute("""
        SELECT id, name, state, start_date, end_date
        FROM sprints
        WHERE state = 'active'
        ORDER BY start_date DESC LIMIT 1
    """).fetchone()

    if not sprint:
        sprint = conn.execute("""
            SELECT id, name, state, start_date, end_date
            FROM sprints ORDER BY start_date DESC LIMIT 1
        """).fetchone()

    if not sprint:
        return {'health_score': 0, 'sprint_name': 'No Sprint', 'risks': [], 'metrics': {}}

    sprint_id, sprint_name, state, start_date, end_date = sprint

    issues = conn.execute("""
        SELECT key, summary, status, priority, issue_type,
               COALESCE(story_points, 0) as story_points,
               assignee_name, created, updated
        FROM issues WHERE sprint_id = ?
    """, [sprint_id]).fetchdf()

    total = len(issues)
    if total == 0:
        return {'health_score': 50, 'sprint_name': sprint_name, 'risks': [], 'metrics': {}}

    # Calculate metrics
    done_statuses = ['Done', 'Terminé(e)', 'Closed', 'Resolved']
    progress_statuses = ['In Progress', 'En cours']
    blocked_statuses = ['Blocked', 'Bloqué']

    done = len(issues[issues['status'].isin(done_statuses)])
    in_progress = len(issues[issues['status'].isin(progress_statuses)])
    blocked = len(issues[issues['status'].isin(blocked_statuses)])
    todo = total - done - in_progress - blocked

    total_points = issues['story_points'].sum()
    done_points = issues[issues['status'].isin(done_statuses)]['story_points'].sum()

    # Calculate days in sprint
    if start_date and end_date:
        try:
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            now = pd.Timestamp.now()
            total_days = (end - start).days
            elapsed_days = (now - start).days
            time_progress = min(100, max(0, (elapsed_days / total_days) * 100)) if total_days > 0 else 0
        except:
            time_progress = 50
            elapsed_days = 0
            total_days = 14
    else:
        time_progress = 50
        elapsed_days = 0
        total_days = 14

    # Work progress
    work_progress = (done / total * 100) if total > 0 else 0
    points_progress = (done_points / total_points * 100) if total_points > 0 else 0

    # Health score calculation (0-100)
    health_score = 0
    risks = []

    # Factor 1: Progress vs Time (weight: 30%)
    if time_progress > 0:
        progress_ratio = work_progress / time_progress
        if progress_ratio >= 1.0:
            health_score += 30
        elif progress_ratio >= 0.8:
            health_score += 25
        elif progress_ratio >= 0.6:
            health_score += 18
            risks.append(("⚠️ Behind Schedule", f"Work progress ({work_progress:.0f}%) lags time ({time_progress:.0f}%)", "medium"))
        else:
            health_score += 10
            risks.append(("🚨 Significantly Behind", f"Only {work_progress:.0f}% done with {time_progress:.0f}% time elapsed", "critical"))
    else:
        health_score += 15

    # Factor 2: WIP Management (weight: 20%)
    wip_ratio = in_progress / total if total > 0 else 0
    if 0.1 <= wip_ratio <= 0.4:
        health_score += 20
    elif wip_ratio < 0.1:
        health_score += 12
        risks.append(("📋 Low WIP", "Consider pulling more items into progress", "low"))
    else:
        health_score += 8
        risks.append(("🔄 High WIP", f"{in_progress} items in progress - focus on finishing", "medium"))

    # Factor 3: Blockers (weight: 25%)
    if blocked == 0:
        health_score += 25
    elif blocked <= 2:
        health_score += 18
        risks.append(("🚧 Blockers Present", f"{blocked} blocked item(s) need attention", "medium"))
    else:
        health_score += 8
        risks.append(("🚨 Multiple Blockers", f"{blocked} blockers impeding progress", "critical"))

    # Factor 4: Scope Stability (weight: 15%)
    # Check for recently added items (created in last 3 days)
    recent_date = datetime.now() - timedelta(days=3)
    try:
        recently_added = len(issues[pd.to_datetime(issues['created']) > recent_date])
    except:
        recently_added = 0

    if recently_added == 0:
        health_score += 15
    elif recently_added <= 2:
        health_score += 10
    else:
        health_score += 5
        risks.append(("📥 Scope Creep", f"{recently_added} items added in last 3 days", "high"))

    # Factor 5: Team Distribution (weight: 10%)
    team_workload = issues.groupby('assignee_name').size()
    if len(team_workload) > 1:
        workload_std = team_workload.std() / team_workload.mean() if team_workload.mean() > 0 else 0
        if workload_std < 0.3:
            health_score += 10
        elif workload_std < 0.5:
            health_score += 7
        else:
            health_score += 4
            risks.append(("⚖️ Unbalanced Workload", "Work distribution is uneven across team", "medium"))
    else:
        health_score += 5

    return {
        'health_score': min(100, health_score),
        'sprint_name': sprint_name,
        'sprint_id': sprint_id,
        'risks': risks,
        'metrics': {
            'total': total,
            'done': done,
            'in_progress': in_progress,
            'blocked': blocked,
            'todo': todo,
            'total_points': total_points,
            'done_points': done_points,
            'time_progress': time_progress,
            'work_progress': work_progress,
            'days_elapsed': elapsed_days,
            'total_days': total_days,
            'recently_added': recently_added
        },
        'issues': issues
    }


def calculate_predictability_index(conn) -> Dict[str, Any]:
    """Calculate team's delivery predictability based on historical data."""
    sprints = conn.execute("""
        SELECT s.id, s.name, s.start_date,
               COUNT(i.key) as committed,
               SUM(CASE WHEN i.status IN ('Done', 'Terminé(e)', 'Closed') THEN 1 ELSE 0 END) as delivered,
               COALESCE(SUM(i.story_points), 0) as committed_points,
               COALESCE(SUM(CASE WHEN i.status IN ('Done', 'Terminé(e)', 'Closed') THEN i.story_points ELSE 0 END), 0) as delivered_points
        FROM sprints s
        LEFT JOIN issues i ON i.sprint_id = s.id
        WHERE s.start_date IS NOT NULL
        GROUP BY s.id, s.name, s.start_date
        ORDER BY s.start_date DESC
        LIMIT 8
    """).fetchdf()

    if sprints.empty or len(sprints) < 2:
        return {'predictability_index': 0, 'trend': 'neutral', 'history': []}

    # Calculate delivery ratios
    sprints['issue_ratio'] = sprints['delivered'] / sprints['committed'].replace(0, 1)
    sprints['points_ratio'] = sprints['delivered_points'] / sprints['committed_points'].replace(0, 1)

    # Predictability = consistency of delivery (low variance = high predictability)
    issue_variance = sprints['issue_ratio'].std()
    points_variance = sprints['points_ratio'].std()

    avg_delivery = sprints['issue_ratio'].mean()

    # Predictability index (0-100)
    # Low variance + high delivery = high predictability
    variance_score = max(0, 100 - (issue_variance * 200))  # Lower variance = higher score
    delivery_score = avg_delivery * 100

    predictability_index = (variance_score * 0.4 + delivery_score * 0.6)

    # Trend (comparing recent vs older sprints)
    if len(sprints) >= 4:
        recent_avg = sprints.head(2)['issue_ratio'].mean()
        older_avg = sprints.tail(2)['issue_ratio'].mean()
        if recent_avg > older_avg * 1.1:
            trend = 'improving'
        elif recent_avg < older_avg * 0.9:
            trend = 'declining'
        else:
            trend = 'stable'
    else:
        trend = 'insufficient_data'

    return {
        'predictability_index': min(100, predictability_index),
        'avg_delivery_rate': avg_delivery * 100,
        'variance': issue_variance,
        'trend': trend,
        'history': sprints.to_dict('records')
    }


def get_team_dynamics(conn) -> Dict[str, Any]:
    """Analyze team collaboration and dynamics."""
    team = conn.execute("""
        SELECT
            COALESCE(assignee_name, 'Unassigned') as name,
            COUNT(*) as total_issues,
            SUM(CASE WHEN status IN ('Done', 'Terminé(e)', 'Closed') THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status IN ('In Progress', 'En cours') THEN 1 ELSE 0 END) as in_progress,
            COALESCE(SUM(story_points), 0) as total_points,
            COALESCE(SUM(CASE WHEN status IN ('Done', 'Terminé(e)', 'Closed') THEN story_points ELSE 0 END), 0) as completed_points,
            COUNT(DISTINCT issue_type) as type_diversity
        FROM issues
        WHERE assignee_name IS NOT NULL
        GROUP BY assignee_name
        ORDER BY total_points DESC
    """).fetchdf()

    if team.empty:
        return {'members': [], 'balance_score': 0, 'insights': []}

    # Calculate team metrics
    total_work = team['total_issues'].sum()
    avg_work = team['total_issues'].mean()
    work_std = team['total_issues'].std()

    # Balance score (0-100, higher = more balanced)
    if avg_work > 0:
        cv = work_std / avg_work  # Coefficient of variation
        balance_score = max(0, 100 - (cv * 100))
    else:
        balance_score = 0

    # Generate insights
    insights = []

    # Find overloaded members
    overloaded = team[team['total_issues'] > avg_work * 1.5]
    if not overloaded.empty:
        for _, member in overloaded.iterrows():
            insights.append({
                'type': 'warning',
                'icon': '⚠️',
                'message': f"{member['name']} has {int(member['total_issues'])} items ({int(member['total_issues'] - avg_work):.0f} above average)"
            })

    # Find underutilized members
    underutilized = team[team['total_issues'] < avg_work * 0.5]
    if not underutilized.empty:
        for _, member in underutilized.iterrows():
            insights.append({
                'type': 'info',
                'icon': '💡',
                'message': f"{member['name']} has capacity ({int(member['total_issues'])} items vs {avg_work:.0f} average)"
            })

    # High performers
    if 'completed' in team.columns:
        team['completion_rate'] = team['completed'] / team['total_issues'].replace(0, 1) * 100
        high_performers = team[team['completion_rate'] > 80]
        for _, member in high_performers.iterrows():
            insights.append({
                'type': 'success',
                'icon': '⭐',
                'message': f"{member['name']} has {member['completion_rate']:.0f}% completion rate"
            })

    return {
        'members': team.to_dict('records'),
        'balance_score': balance_score,
        'insights': insights,
        'avg_workload': avg_work,
        'total_team_size': len(team)
    }


def get_impediments(conn) -> List[Dict[str, Any]]:
    """Get current impediments and blockers with escalation info."""
    blockers = conn.execute("""
        SELECT key, summary, status, priority, assignee_name, created, updated
        FROM issues
        WHERE status IN ('Blocked', 'Bloqué', 'Impediment')
           OR priority IN ('Highest', 'Blocker')
        ORDER BY
            CASE priority
                WHEN 'Highest' THEN 1
                WHEN 'Blocker' THEN 1
                WHEN 'High' THEN 2
                ELSE 3
            END,
            created ASC
    """).fetchdf()

    impediments = []
    now = datetime.now()

    for _, row in blockers.iterrows():
        try:
            created = pd.to_datetime(row['created'])
            age_days = (now - created).days
        except:
            age_days = 0

        # Determine severity
        if row['priority'] in ['Highest', 'Blocker'] or age_days > 5:
            severity = 'critical'
        elif age_days > 3 or row['priority'] == 'High':
            severity = 'high'
        else:
            severity = 'medium'

        impediments.append({
            'key': row['key'],
            'summary': row['summary'],
            'assignee': row['assignee_name'] or 'Unassigned',
            'age_days': age_days,
            'severity': severity,
            'priority': row['priority'],
            'status': row['status']
        })

    return impediments


def get_ceremony_script(conn) -> Dict[str, Any]:
    """Generate today's ceremony script based on day of week and sprint state."""
    from datetime import datetime

    day = datetime.now().weekday()  # 0=Monday, 6=Sunday
    hour = datetime.now().hour

    # Default ceremony based on common patterns
    if day == 0:  # Monday
        ceremony_type = "Sprint Planning" if hour < 12 else "Standup"
    elif day == 4:  # Friday
        ceremony_type = "Retro" if hour >= 14 else "Standup"
    else:
        ceremony_type = "Standup"

    # Get sprint metrics for context
    try:
        metrics = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'Terminé(e)' THEN 1 ELSE 0 END) as done,
                SUM(CASE WHEN status = 'En cours' THEN 1 ELSE 0 END) as in_progress,
                SUM(CASE WHEN status = 'En cours' AND DATEDIFF('day', updated, CURRENT_TIMESTAMP) >= 2 THEN 1 ELSE 0 END) as stale
            FROM issues
        """).fetchone()

        total, done, in_progress, stale = metrics
        completion = (done / max(total, 1)) * 100
    except:
        total, done, in_progress, stale, completion = 0, 0, 0, 0, 0

    ceremonies = {
        "Standup": {
            "emoji": "🧍",
            "duration": 15,
            "agenda": [
                f"Check progress: {done}/{total} items done ({completion:.0f}%)",
                f"Review {in_progress} items in progress",
                f"Surface blockers ({stale} items stale)" if stale > 0 else "Confirm no blockers",
                "Align on today's priorities"
            ],
            "key_question": "What's blocking you from finishing your current task?"
        },
        "Sprint Planning": {
            "emoji": "📋",
            "duration": 60,
            "agenda": [
                "Review velocity from last sprint",
                "Clarify sprint goal with Product Owner",
                "Break down user stories into tasks",
                "Commit to sprint scope"
            ],
            "key_question": "Do we have clarity on what 'done' looks like for each story?"
        },
        "Retro": {
            "emoji": "🔄",
            "duration": 45,
            "agenda": [
                f"Celebrate: {done} items completed",
                f"Discuss: {stale} items got stuck" if stale > 0 else "Discuss what went well",
                "Identify 1-2 improvements to try",
                "Assign action owners"
            ],
            "key_question": "What's ONE thing we can do differently next sprint?"
        }
    }

    return {"type": ceremony_type, **ceremonies.get(ceremony_type, ceremonies["Standup"])}


def generate_coaching_tips(health_data: Dict, dynamics: Dict, impediments: List) -> List[Dict[str, str]]:
    """Generate AI-powered coaching tips based on current state."""
    tips = []
    metrics = health_data.get('metrics', {})

    # Tip based on WIP
    wip = metrics.get('in_progress', 0)
    total = metrics.get('total', 1)
    if wip / total > 0.5:
        tips.append({
            'category': 'Flow',
            'icon': '🔄',
            'title': 'Reduce Work in Progress',
            'tip': f"You have {wip} items in progress ({wip/total*100:.0f}% of sprint). Consider implementing WIP limits to improve flow. Research shows limiting WIP to 2-3 items per person increases throughput.",
            'action': 'Discuss WIP limits in next standup'
        })

    # Tip based on blockers
    if len(impediments) > 0:
        critical = sum(1 for i in impediments if i['severity'] == 'critical')
        if critical > 0:
            tips.append({
                'category': 'Impediments',
                'icon': '🚨',
                'title': 'Critical Blockers Require Escalation',
                'tip': f"You have {critical} critical blocker(s) that may need management escalation. Blockers older than 3 days typically require external intervention.",
                'action': 'Schedule blocker review with stakeholders'
            })

    # Tip based on team balance
    balance = dynamics.get('balance_score', 0)
    if balance < 60:
        tips.append({
            'category': 'Team',
            'icon': '⚖️',
            'title': 'Rebalance Workload',
            'tip': f"Team workload balance score is {balance:.0f}%. Consider pair programming or task redistribution. Unbalanced teams have 40% higher burnout risk.",
            'action': 'Review assignments in planning'
        })

    # Tip based on progress
    work_progress = metrics.get('work_progress', 0)
    time_progress = metrics.get('time_progress', 0)
    if time_progress > 50 and work_progress < time_progress * 0.7:
        tips.append({
            'category': 'Delivery',
            'icon': '📉',
            'title': 'Sprint at Risk',
            'tip': f"With {time_progress:.0f}% time elapsed and only {work_progress:.0f}% complete, consider scope reduction. It's better to deliver fewer stories well than to rush all stories.",
            'action': 'Discuss scope with Product Owner'
        })

    # Tip based on scope
    recently_added = metrics.get('recently_added', 0)
    if recently_added > 2:
        tips.append({
            'category': 'Scope',
            'icon': '📥',
            'title': 'Protect Sprint Scope',
            'tip': f"{recently_added} items were added recently. Mid-sprint changes reduce predictability by 35% on average. Consider a 'change freeze' for remaining days.",
            'action': 'Reinforce sprint commitment with PO'
        })

    # General improvement tip
    if health_data.get('health_score', 0) > 75:
        tips.append({
            'category': 'Excellence',
            'icon': '🏆',
            'title': 'Maintain Momentum',
            'tip': "Sprint health is strong! Consider documenting what's working well for the retrospective. High-performing sprints are 60% more likely when teams celebrate wins.",
            'action': 'Note successes for retro'
        })

    return tips


def generate_stakeholder_digest(health_data: Dict, dynamics: Dict, predictability: Dict) -> str:
    """Generate a professional stakeholder status digest."""
    metrics = health_data.get('metrics', {})
    sprint_name = health_data.get('sprint_name', 'Current Sprint')
    health_score = health_data.get('health_score', 0)

    # Determine status emoji and text
    if health_score >= 80:
        status_emoji = "🟢"
        status_text = "On Track"
    elif health_score >= 60:
        status_emoji = "🟡"
        status_text = "Minor Concerns"
    elif health_score >= 40:
        status_emoji = "🟠"
        status_text = "Needs Attention"
    else:
        status_emoji = "🔴"
        status_text = "At Risk"

    digest = f"""
## {sprint_name} - Status Report
**Generated:** {datetime.now().strftime('%B %d, %Y at %H:%M')}

---

### Executive Summary
{status_emoji} **Overall Status:** {status_text} (Health Score: {health_score}/100)

### Progress Metrics
| Metric | Current | Target |
|--------|---------|--------|
| Issues Completed | {metrics.get('done', 0)}/{metrics.get('total', 0)} | 100% |
| Story Points | {metrics.get('done_points', 0):.0f}/{metrics.get('total_points', 0):.0f} | 100% |
| Time Elapsed | Day {metrics.get('days_elapsed', 0)} of {metrics.get('total_days', 14)} | - |
| Work Progress | {metrics.get('work_progress', 0):.0f}% | {metrics.get('time_progress', 0):.0f}% |

### Team Health
- **Team Size:** {dynamics.get('total_team_size', 0)} members
- **Workload Balance:** {dynamics.get('balance_score', 0):.0f}%
- **Predictability Index:** {predictability.get('predictability_index', 0):.0f}%

### Key Risks & Mitigations
"""

    risks = health_data.get('risks', [])
    if risks:
        for risk_name, risk_desc, severity in risks:
            digest += f"- {risk_name}: {risk_desc}\n"
    else:
        digest += "- No significant risks identified\n"

    digest += f"""
### Blockers Requiring Attention
- **Active Blockers:** {metrics.get('blocked', 0)}
- **Items Added Mid-Sprint:** {metrics.get('recently_added', 0)}

---
*This report was auto-generated by SM Command Center*
"""

    return digest


def create_health_gauge(score: float) -> go.Figure:
    """Create a premium health score gauge."""
    if score >= 80:
        color = '#27ae60'
    elif score >= 60:
        color = '#f39c12'
    elif score >= 40:
        color = '#e67e22'
    else:
        color = '#e74c3c'

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        number={'font': {'size': 48, 'color': '#1a202c'}, 'suffix': ''},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b", 'tickfont': {'color': '#64748b'}},
            'bar': {'color': color, 'thickness': 0.7},
            'bgcolor': '#e2e8f0',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 40], 'color': '#fee2e2'},
                {'range': [40, 60], 'color': '#ffedd5'},
                {'range': [60, 80], 'color': '#ffedd5'},
                {'range': [80, 100], 'color': '#dcfce7'}
            ],
            'threshold': {
                'line': {'color': "#1a202c", 'width': 3},
                'thickness': 0.8,
                'value': score
            }
        }
    ))

    fig.update_layout(
        height=220,
        margin=dict(l=30, r=30, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#334155'}
    )

    return fig


def create_predictability_chart(history: List[Dict]) -> go.Figure:
    """Create predictability trend chart."""
    if not history:
        return go.Figure()

    df = pd.DataFrame(history)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['name'],
        y=df['issue_ratio'] * 100,
        mode='lines+markers',
        name='Delivery Rate',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)'
    ))

    # Add target line
    fig.add_hline(y=100, line_dash="dash", line_color="#27ae60",
                  annotation_text="100% Target", annotation_position="right")

    # Add average line
    avg = df['issue_ratio'].mean() * 100
    fig.add_hline(y=avg, line_dash="dot", line_color="#f39c12",
                  annotation_text=f"Avg: {avg:.0f}%", annotation_position="left")

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, tickfont=dict(color='#64748b'), tickangle=-45),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)',
                   tickfont=dict(color='#64748b'), title='Delivery %'),
        legend=dict(font=dict(color='#64748b'))
    )

    return fig


def create_team_radar(dynamics: Dict) -> go.Figure:
    """Create team dynamics radar chart."""
    members = dynamics.get('members', [])
    if not members or len(members) < 2:
        return go.Figure()

    df = pd.DataFrame(members)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=df['total_issues'],
        theta=df['name'],
        fill='toself',
        name='Assigned',
        line_color='#667eea',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=df['completed'],
        theta=df['name'],
        fill='toself',
        name='Completed',
        line_color='#27ae60',
        fillcolor='rgba(39, 174, 96, 0.3)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, showticklabels=True, tickfont=dict(color='#64748b'),
                           gridcolor='rgba(0,0,0,0.1)'),
            angularaxis=dict(tickfont=dict(color='#334155'), gridcolor='rgba(0,0,0,0.1)')
        ),
        showlegend=True,
        legend=dict(font=dict(color='#64748b'), orientation='h', y=-0.1),
        height=300,
        margin=dict(l=60, r=60, t=40, b=60),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def main():

    # Render page guide in sidebar
    render_page_guide()
    # Header
    st.markdown("""
    <div class="command-header">
        <h1 class="command-title">🎖️ Scrum Master Command Center</h1>
        <p class="command-subtitle">Real-time sprint intelligence • Team dynamics • AI-powered coaching</p>
    </div>
    """, unsafe_allow_html=True)

    conn = get_connection()
    if not conn:
        st.error("Database not found. Please sync data first.")
        st.stop()

    # ========== QUICK WIN: TODAY'S CEREMONY SCRIPT ==========
    ceremony_data = get_ceremony_script(conn)
    if ceremony_data:
        agenda_html = "".join([f'<li style="padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.1); display: flex; align-items: center; gap: 10px; font-size: 13px;"><span>→</span> {item}</li>' for item in ceremony_data['agenda'][:4]])

        st.markdown(f"""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 20px; padding: 24px 28px; margin-bottom: 24px; color: white; box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4); position: relative; overflow: hidden;">
    <div style="position: absolute; top: 16px; right: 16px; background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 12px; font-size: 10px; font-weight: 600;">⏱️ 15 min saved</div>
    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
        <span style="background: rgba(255,255,255,0.2); padding: 6px 14px; border-radius: 20px; font-size: 11px; font-weight: 700; letter-spacing: 1px;">
            {ceremony_data['emoji']} {ceremony_data['type'].upper()} GUIDE
        </span>
    </div>
    <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 24px;">
        <div style="flex: 1;">
            <div style="font-size: 13px; opacity: 0.9; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px;">Today's Ceremony</div>
            <div style="font-size: 24px; font-weight: 800; line-height: 1.1; margin-bottom: 8px;">{ceremony_data['duration']} min • {len(ceremony_data['agenda'])} items</div>
            <ul style="list-style: none; padding: 0; margin: 12px 0 0 0;">{agenda_html}</ul>
        </div>
        <div style="background: rgba(255,255,255,0.15); backdrop-filter: blur(10px); border-radius: 16px; padding: 16px 20px; min-width: 200px; text-align: center;">
            <div style="font-size: 11px; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Key Question to Ask</div>
            <div style="font-size: 15px; font-weight: 700; line-height: 1.3;">"{ceremony_data['key_question']}"</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Calculate all metrics
    health_data = calculate_sprint_health(conn)
    predictability = calculate_predictability_index(conn)
    dynamics = get_team_dynamics(conn)
    impediments = get_impediments(conn)
    coaching_tips = generate_coaching_tips(health_data, dynamics, impediments)

    metrics = health_data.get('metrics', {})

    # ========== ROW 1: Sprint Pulse ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📊 Sprint Pulse</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        # Health Score Gauge
        st.markdown(f"**{health_data.get('sprint_name', 'Sprint')}**")
        fig = create_health_gauge(health_data.get('health_score', 0))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Status badge
        score = health_data.get('health_score', 0)
        if score >= 80:
            status = ("🟢 Excellent", "trend-up")
        elif score >= 60:
            status = ("🟡 Good", "trend-neutral")
        elif score >= 40:
            status = ("🟠 At Risk", "trend-down")
        else:
            status = ("🔴 Critical", "trend-down")

        st.markdown(f'<div class="pulse-trend {status[1]}" style="text-align:center;">{status[0]}</div>',
                    unsafe_allow_html=True)

    with col2:
        # Key metrics grid
        st.markdown("#### Quick Stats")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
<div class="pulse-card" style="--pulse-color: #667eea;">
    <div class="pulse-value">{metrics.get('done', 0)}/{metrics.get('total', 0)}</div>
    <div class="pulse-label">Issues Done</div>
</div>
""", unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
<div class="pulse-card" style="--pulse-color: #764ba2;">
    <div class="pulse-value">{metrics.get('done_points', 0):.0f}</div>
    <div class="pulse-label">Points Delivered</div>
</div>
""", unsafe_allow_html=True)

        with m3:
            wip = metrics.get('in_progress', 0)
            wip_color = '#27ae60' if wip <= 5 else '#f39c12' if wip <= 10 else '#e74c3c'
            st.markdown(f"""
<div class="pulse-card" style="--pulse-color: {wip_color};">
    <div class="pulse-value">{wip}</div>
    <div class="pulse-label">Work in Progress</div>
</div>
""", unsafe_allow_html=True)

        with m4:
            blocked = metrics.get('blocked', 0)
            blocked_color = '#27ae60' if blocked == 0 else '#e74c3c'
            st.markdown(f"""
<div class="pulse-card" style="--pulse-color: {blocked_color};">
    <div class="pulse-value">{blocked}</div>
    <div class="pulse-label">Blockers</div>
</div>
""", unsafe_allow_html=True)

        # Progress bars
        st.markdown("#### Progress Tracking")
        work_pct = metrics.get('work_progress', 0)
        time_pct = metrics.get('time_progress', 0)

        st.markdown(f"""
<div style="margin-bottom: 16px;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
        <span style="color: #334155;">Work Progress</span>
        <span style="color: #667eea; font-weight: 600;">{work_pct:.0f}%</span>
    </div>
    <div style="background: #e2e8f0; border-radius: 10px; height: 12px; overflow: hidden;">
        <div style="background: linear-gradient(90deg, #667eea, #764ba2); width: {work_pct}%; height: 100%;"></div>
    </div>
</div>
<div style="margin-bottom: 16px;">
    <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
        <span style="color: #334155;">Time Elapsed</span>
        <span style="color: #f39c12; font-weight: 600;">{time_pct:.0f}%</span>
    </div>
    <div style="background: #e2e8f0; border-radius: 10px; height: 12px; overflow: hidden;">
        <div style="background: linear-gradient(90deg, #f39c12, #e67e22); width: {time_pct}%; height: 100%;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

    with col3:
        # Predictability Index
        st.markdown("#### Predictability")
        pred_score = predictability.get('predictability_index', 0)
        pred_trend = predictability.get('trend', 'stable')

        trend_icon = "📈" if pred_trend == 'improving' else "📉" if pred_trend == 'declining' else "➡️"
        trend_color = '#27ae60' if pred_trend == 'improving' else '#e74c3c' if pred_trend == 'declining' else '#f39c12'

        st.markdown(f"""
<div class="pulse-card">
    <div class="pulse-value">{pred_score:.0f}%</div>
    <div class="pulse-label">Predictability Index</div>
    <div class="pulse-trend" style="background: {trend_color}22; color: {trend_color};">
        {trend_icon} {pred_trend.title()}
    </div>
</div>
""", unsafe_allow_html=True)

        avg_delivery = predictability.get('avg_delivery_rate', 0)
        st.markdown(f"""
<div style="margin-top: 16px; padding: 12px; background: rgba(0,0,0,0.03); border-radius: 8px;">
    <div style="color: #64748b; font-size: 12px;">Avg Delivery Rate</div>
    <div style="color: #334155; font-size: 20px; font-weight: 600;">{avg_delivery:.0f}%</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== ROW 2: Risks & Impediments ==========
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🚨 Risk Radar</div>', unsafe_allow_html=True)

        risks = health_data.get('risks', [])
        if risks:
            for risk_name, risk_desc, severity in risks:
                severity_class = 'risk-critical' if severity == 'critical' else \
                                'risk-high' if severity == 'high' else \
                                'risk-medium' if severity == 'medium' else 'risk-low'
                card_class = 'impediment-critical' if severity == 'critical' else 'impediment-warning'

                st.markdown(f"""
<div class="{card_class}">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <span style="color: #1a202c; font-weight: 600;">{risk_name}</span>
        <span class="risk-badge {severity_class}">{severity.upper()}</span>
    </div>
    <div style="color: #64748b; font-size: 13px;">{risk_desc}</div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 40px; color: #27ae60;">
                <div style="font-size: 48px;">✅</div>
                <div style="font-size: 16px; margin-top: 8px;">No significant risks detected</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🚧 Impediment War Room</div>', unsafe_allow_html=True)

        if impediments:
            for imp in impediments[:5]:  # Show top 5
                severity_class = 'impediment-critical' if imp['severity'] == 'critical' else 'impediment-warning'
                badge_class = 'risk-critical' if imp['severity'] == 'critical' else \
                             'risk-high' if imp['severity'] == 'high' else 'risk-medium'

                st.markdown(f"""
<div class="{severity_class}">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <span style="color: #4f46e5; font-size: 12px; font-weight: 600;">{imp['key']}</span>
        <span class="risk-badge {badge_class}">{imp['age_days']}d old</span>
    </div>
    <div style="color: #1a202c; font-size: 14px; margin-bottom: 6px;">{imp['summary'][:60]}...</div>
    <div style="color: #64748b; font-size: 12px;">👤 {imp['assignee']} • {imp['status']}</div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 40px; color: #27ae60;">
                <div style="font-size: 48px;">🎉</div>
                <div style="font-size: 16px; margin-top: 8px;">No blockers! Team is flowing</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== ROW 3: Team Dynamics & Coaching ==========
    col_team, col_coaching = st.columns([1, 1])

    with col_team:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👥 Team Dynamics Radar</div>', unsafe_allow_html=True)

        # Balance score
        balance = dynamics.get('balance_score', 0)
        balance_color = '#27ae60' if balance >= 70 else '#f39c12' if balance >= 50 else '#e74c3c'

        st.markdown(f"""
<div style="display: flex; gap: 16px; margin-bottom: 16px;">
    <div class="mini-stat" style="flex: 1;">
        <div class="mini-stat-value" style="color: {balance_color};">{balance:.0f}%</div>
        <div class="mini-stat-label">Balance Score</div>
    </div>
    <div class="mini-stat" style="flex: 1;">
        <div class="mini-stat-value">{dynamics.get('total_team_size', 0)}</div>
        <div class="mini-stat-label">Team Size</div>
    </div>
    <div class="mini-stat" style="flex: 1;">
        <div class="mini-stat-value">{dynamics.get('avg_workload', 0):.1f}</div>
        <div class="mini-stat-label">Avg Items</div>
    </div>
</div>
""", unsafe_allow_html=True)

        # Radar chart
        fig = create_team_radar(dynamics)
        if fig.data:
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Team insights
        insights = dynamics.get('insights', [])
        if insights:
            st.markdown("**Team Insights:**")
            for insight in insights[:3]:
                color = '#27ae60' if insight['type'] == 'success' else \
                       '#f39c12' if insight['type'] == 'warning' else '#3498db'
                st.markdown(f"""
<div style="padding: 8px 12px; background: {color}22; border-left: 3px solid {color};
            border-radius: 4px; margin-bottom: 8px; font-size: 13px; color: #334155;">
    {insight['icon']} {insight['message']}
</div>
""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    with col_coaching:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">🧠 AI Coaching Tips</div>', unsafe_allow_html=True)

        if coaching_tips:
            for tip in coaching_tips[:4]:
                st.markdown(f"""
<div class="coaching-tip">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <span style="color: #3b82f6; font-size: 12px; font-weight: 600;">{tip['category'].upper()}</span>
        <span style="font-size: 20px;">{tip['icon']}</span>
    </div>
    <div style="color: #1a202c; font-weight: 600; margin-bottom: 6px;">{tip['title']}</div>
    <div style="color: #64748b; font-size: 13px; margin-bottom: 8px;">{tip['tip']}</div>
    <div style="color: #4f46e5; font-size: 12px; font-weight: 500;">
        💡 Action: {tip['action']}
    </div>
</div>
""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 40px; color: #3498db;">
                <div style="font-size: 48px;">💡</div>
                <div style="font-size: 16px; margin-top: 8px;">Everything looks great! Keep up the good work.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== ROW 4: Predictability Trend & Stakeholder Digest ==========
    col_pred, col_digest = st.columns([1, 1])

    with col_pred:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📈 Predictability Trend</div>', unsafe_allow_html=True)

        history = predictability.get('history', [])
        if history:
            fig = create_predictability_chart(history)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

            st.markdown("""
            <div style="padding: 12px; background: rgba(0,0,0,0.03); border-radius: 8px; margin-top: 12px;">
                <div style="color: #64748b; font-size: 12px; margin-bottom: 4px;">📊 What is Predictability?</div>
                <div style="color: #334155; font-size: 13px;">
                    Measures how consistently your team delivers on commitments.
                    High predictability = stakeholder trust + better planning.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Complete more sprints to see predictability trends.")

        st.markdown('</div>', unsafe_allow_html=True)

    with col_digest:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📋 Stakeholder Digest Generator</div>', unsafe_allow_html=True)

        if st.button("🚀 Generate Status Report", type="primary"):
            digest = generate_stakeholder_digest(health_data, dynamics, predictability)
            st.session_state['digest'] = digest

        if 'digest' in st.session_state:
            st.markdown('<div class="digest-preview">', unsafe_allow_html=True)
            st.markdown(st.session_state['digest'])
            st.markdown('</div>', unsafe_allow_html=True)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "📥 Download Markdown",
                    st.session_state['digest'],
                    file_name=f"sprint_status_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
            with col_dl2:
                st.button("📧 Copy to Clipboard", help="Copy report to clipboard")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 40px; color: #94a3b8;">
                <div style="font-size: 48px;">📋</div>
                <div style="font-size: 14px; margin-top: 8px;">
                    Generate a professional status report<br/>for stakeholders with one click
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== ROW 5: Ceremony Tracker ==========
    st.markdown('<div class="section-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📅 Ceremony Health Check</div>', unsafe_allow_html=True)

    ceremonies = [
        {"name": "Daily Standup", "icon": "☀️", "frequency": "Daily", "health": 85,
         "tip": "Keep it under 15 minutes. Focus on blockers, not status."},
        {"name": "Sprint Planning", "icon": "📝", "frequency": "Sprint Start", "health": 78,
         "tip": "Ensure stories have acceptance criteria before committing."},
        {"name": "Sprint Review", "icon": "🎭", "frequency": "Sprint End", "health": 72,
         "tip": "Invite stakeholders. Demo working software, not slides."},
        {"name": "Retrospective", "icon": "🔄", "frequency": "Sprint End", "health": 65,
         "tip": "Track action items. Follow up on previous retro actions."},
        {"name": "Backlog Refinement", "icon": "✂️", "frequency": "Weekly", "health": 70,
         "tip": "Keep it to 10% of sprint capacity. Focus on upcoming sprints."}
    ]

    c1, c2, c3, c4, c5 = st.columns(5)
    cols = [c1, c2, c3, c4, c5]

    for i, ceremony in enumerate(ceremonies):
        with cols[i]:
            health = ceremony['health']
            health_color = '#27ae60' if health >= 80 else '#f39c12' if health >= 60 else '#e74c3c'

            st.markdown(f"""
<div class="ceremony-card">
    <div style="text-align: center; font-size: 32px; margin-bottom: 8px;">{ceremony['icon']}</div>
    <div style="text-align: center; color: #1a202c; font-weight: 600; font-size: 14px; margin-bottom: 4px;">
        {ceremony['name']}
    </div>
    <div style="text-align: center; color: #64748b; font-size: 11px; margin-bottom: 12px;">
        {ceremony['frequency']}
    </div>
    <div style="background: #e2e8f0; border-radius: 8px; height: 8px; overflow: hidden;">
        <div style="background: {health_color}; width: {health}%; height: 100%;"></div>
    </div>
    <div style="text-align: center; color: {health_color}; font-size: 12px; margin-top: 8px;">
        {health}% Health
    </div>
</div>
""", unsafe_allow_html=True)

            with st.expander("💡 Tip"):
                st.write(ceremony['tip'])

    st.markdown('</div>', unsafe_allow_html=True)

    # ========== Footer ==========
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #94a3b8; font-size: 12px;">
        🎖️ Scrum Master Command Center | Sprint Intelligence Platform |
        Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
    """, unsafe_allow_html=True)

    conn.close()


if __name__ == "__main__":
    main()
