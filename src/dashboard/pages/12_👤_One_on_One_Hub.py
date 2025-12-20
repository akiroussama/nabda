"""
👤 1-on-1 Intelligence Hub - The Manager's Secret Weapon
Auto-generated agendas, performance insights, action tracking, and review data.
Saves 4-8 hours/week on 1-on-1 preparation and performance management.
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
from typing import Dict, List, Any, Tuple, Optional
import hashlib
import json

st.set_page_config(page_title="1-on-1 Hub", page_icon="👤", layout="wide")

# Premium 1-on-1 Hub CSS
st.markdown("""
<style>
    .hub-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #3d7ab5 100%);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hub-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='40' height='40' viewBox='0 0 40 40' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M20 20c0-5.5-4.5-10-10-10S0 14.5 0 20s4.5 10 10 10 10-4.5 10-10zm10 0c0 5.5 4.5 10 10 10s10-4.5 10-10-4.5-10-10-10-10 4.5-10 10z'/%3E%3C/g%3E%3C/svg%3E");
    }
    .hub-title {
        font-size: 36px;
        font-weight: 800;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    .hub-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 16px;
        margin-top: 8px;
    }
    .hub-stat {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        padding: 8px 16px;
        border-radius: 20px;
        margin: 4px;
        font-size: 13px;
        color: white;
    }

    .section-container {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.05);
    }

    .section-title {
        font-size: 18px;
        font-weight: 600;
        color: #fff;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .member-selector {
        background: linear-gradient(145deg, #252541 0%, #1e1e32 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        cursor: pointer;
        transition: all 0.2s;
        margin-bottom: 12px;
    }
    .member-selector:hover {
        transform: translateX(4px);
        border-color: #667eea44;
    }
    .member-selector.active {
        border-color: #667eea;
        background: linear-gradient(145deg, #2a2a50 0%, #232340 100%);
    }

    .member-avatar {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 16px;
        color: white;
    }

    .member-name {
        font-size: 16px;
        font-weight: 600;
        color: #fff;
    }

    .member-role {
        font-size: 12px;
        color: #8892b0;
    }

    .insight-card {
        background: linear-gradient(145deg, #252541 0%, #1e1e32 100%);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 12px;
    }

    .talking-point {
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 10px;
        border-left: 4px solid;
    }
    .talking-point.recognition {
        background: #27ae6015;
        border-color: #27ae60;
    }
    .talking-point.discussion {
        background: #f39c1215;
        border-color: #f39c12;
    }
    .talking-point.concern {
        background: #e74c3c15;
        border-color: #e74c3c;
    }
    .talking-point.growth {
        background: #3498db15;
        border-color: #3498db;
    }

    .metric-card {
        background: rgba(255,255,255,0.03);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #667eea;
    }
    .metric-label {
        font-size: 11px;
        color: #8892b0;
        text-transform: uppercase;
    }
    .metric-trend {
        font-size: 12px;
        margin-top: 4px;
    }
    .trend-up { color: #27ae60; }
    .trend-down { color: #e74c3c; }
    .trend-neutral { color: #8892b0; }

    .action-item {
        background: rgba(255,255,255,0.02);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .action-checkbox {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        border: 2px solid #667eea;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .action-checkbox.done {
        background: #27ae60;
        border-color: #27ae60;
    }

    .agenda-section {
        background: linear-gradient(145deg, #1a2d3d 0%, #152025 100%);
        border: 1px solid #3498db44;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .timeline-item {
        position: relative;
        padding-left: 24px;
        padding-bottom: 16px;
        border-left: 2px solid #667eea33;
    }
    .timeline-item::before {
        content: '';
        position: absolute;
        left: -5px;
        top: 4px;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #667eea;
    }
    .timeline-item:last-child {
        border-left: none;
    }

    .performance-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
    }
    .badge-exceptional { background: #27ae60; color: white; }
    .badge-strong { background: #3498db; color: white; }
    .badge-meets { background: #f39c12; color: white; }
    .badge-developing { background: #e67e22; color: white; }
    .badge-needs-attention { background: #e74c3c; color: white; }

    .review-section {
        background: #0f0f23;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 20px;
        font-family: 'Georgia', serif;
        color: #ccd6f6;
    }

    .copy-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 12px;
        cursor: pointer;
    }

    .week-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        background: rgba(102, 126, 234, 0.2);
        color: #667eea;
        margin-right: 8px;
    }
</style>
""", unsafe_allow_html=True)


def get_connection():
    db_path = Path("data/jira.duckdb")
    return duckdb.connect(str(db_path), read_only=True) if db_path.exists() else None


def get_avatar_color(name: str) -> str:
    colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#43e97b', '#fa709a']
    return colors[hash(name or '') % len(colors)]


def get_team_members(conn) -> List[Dict[str, Any]]:
    """Get all team members with their basic metrics."""
    members = conn.execute("""
        SELECT
            COALESCE(assignee_name, 'Unassigned') as name,
            COUNT(*) as total_issues,
            SUM(CASE WHEN status IN ('Done', 'Terminé(e)', 'Closed', 'Resolved') THEN 1 ELSE 0 END) as completed,
            SUM(CASE WHEN status IN ('In Progress', 'En cours') THEN 1 ELSE 0 END) as in_progress,
            COALESCE(SUM(story_points), 0) as total_points,
            COALESCE(SUM(CASE WHEN status IN ('Done', 'Terminé(e)', 'Closed', 'Resolved')
                         THEN story_points ELSE 0 END), 0) as completed_points,
            COUNT(DISTINCT issue_type) as type_diversity,
            MAX(updated) as last_activity
        FROM issues
        WHERE assignee_name IS NOT NULL
        GROUP BY assignee_name
        ORDER BY total_points DESC
    """).fetchdf()

    result = []
    for _, row in members.iterrows():
        completion_rate = (row['completed'] / row['total_issues'] * 100) if row['total_issues'] > 0 else 0
        result.append({
            'name': row['name'],
            'total_issues': int(row['total_issues']),
            'completed': int(row['completed']),
            'in_progress': int(row['in_progress']),
            'total_points': float(row['total_points']),
            'completed_points': float(row['completed_points']),
            'completion_rate': completion_rate,
            'type_diversity': int(row['type_diversity']),
            'last_activity': row['last_activity']
        })

    return result


def get_member_details(conn, member_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific team member."""
    # Recent work (last 30 days)
    recent_work = conn.execute("""
        SELECT key, summary, status, priority, issue_type,
               COALESCE(story_points, 0) as story_points,
               created, updated, resolved
        FROM issues
        WHERE assignee_name = ?
        ORDER BY updated DESC
        LIMIT 20
    """, [member_name]).fetchdf()

    # Completed recently
    completed_recent = conn.execute("""
        SELECT key, summary, issue_type, COALESCE(story_points, 0) as story_points, resolved
        FROM issues
        WHERE assignee_name = ?
          AND status IN ('Done', 'Terminé(e)', 'Closed', 'Resolved')
        ORDER BY resolved DESC
        LIMIT 10
    """, [member_name]).fetchdf()

    # Currently working on
    current_work = conn.execute("""
        SELECT key, summary, priority, issue_type, COALESCE(story_points, 0) as story_points, updated
        FROM issues
        WHERE assignee_name = ?
          AND status IN ('In Progress', 'En cours')
        ORDER BY priority DESC, updated DESC
    """, [member_name]).fetchdf()

    # Blocked items
    blocked = conn.execute("""
        SELECT key, summary, priority, created
        FROM issues
        WHERE assignee_name = ?
          AND (status IN ('Blocked', 'Bloqué') OR priority IN ('Highest', 'Blocker'))
        ORDER BY created ASC
    """, [member_name]).fetchdf()

    # Weekly velocity (last 8 weeks)
    velocity = conn.execute("""
        SELECT
            DATE_TRUNC('week', resolved) as week,
            COUNT(*) as issues_completed,
            COALESCE(SUM(story_points), 0) as points_completed
        FROM issues
        WHERE assignee_name = ?
          AND status IN ('Done', 'Terminé(e)', 'Closed', 'Resolved')
          AND resolved IS NOT NULL
          AND resolved >= CURRENT_DATE - INTERVAL '56 days'
        GROUP BY DATE_TRUNC('week', resolved)
        ORDER BY week DESC
    """, [member_name]).fetchdf()

    # Issue type distribution
    type_dist = conn.execute("""
        SELECT issue_type, COUNT(*) as count
        FROM issues
        WHERE assignee_name = ?
        GROUP BY issue_type
        ORDER BY count DESC
    """, [member_name]).fetchdf()

    # Priority distribution
    priority_dist = conn.execute("""
        SELECT priority, COUNT(*) as count
        FROM issues
        WHERE assignee_name = ?
          AND priority IS NOT NULL
        GROUP BY priority
    """, [member_name]).fetchdf()

    # Complexity analysis (based on story points)
    complexity = conn.execute("""
        SELECT
            CASE
                WHEN story_points <= 2 THEN 'Low (1-2)'
                WHEN story_points <= 5 THEN 'Medium (3-5)'
                WHEN story_points <= 8 THEN 'High (6-8)'
                ELSE 'Complex (8+)'
            END as complexity,
            COUNT(*) as count
        FROM issues
        WHERE assignee_name = ?
          AND story_points IS NOT NULL
        GROUP BY complexity
        ORDER BY MIN(story_points)
    """, [member_name]).fetchdf()

    return {
        'recent_work': recent_work,
        'completed_recent': completed_recent,
        'current_work': current_work,
        'blocked': blocked,
        'velocity': velocity,
        'type_distribution': type_dist,
        'priority_distribution': priority_dist,
        'complexity': complexity
    }


def generate_talking_points(member: Dict, details: Dict) -> List[Dict[str, Any]]:
    """Generate smart talking points for the 1-on-1."""
    talking_points = []

    # 1. Recognition - Recent completions
    completed = details.get('completed_recent', pd.DataFrame())
    if not completed.empty:
        high_value = completed[completed['story_points'] >= 5]
        if not high_value.empty:
            for _, item in high_value.head(2).iterrows():
                talking_points.append({
                    'category': 'recognition',
                    'icon': '🏆',
                    'title': 'High-Impact Delivery',
                    'content': f"Completed {item['key']}: {item['summary'][:50]}... ({int(item['story_points'])} points)",
                    'action': 'Acknowledge the effort and impact'
                })

        if len(completed) >= 3:
            talking_points.append({
                'category': 'recognition',
                'icon': '⭐',
                'title': 'Consistent Output',
                'content': f"Completed {len(completed)} items recently. Strong delivery momentum.",
                'action': 'Recognize consistency and ask what\'s working well'
            })

    # 2. Current Focus
    current = details.get('current_work', pd.DataFrame())
    if not current.empty:
        talking_points.append({
            'category': 'discussion',
            'icon': '🔄',
            'title': 'Current Focus',
            'content': f"Working on {len(current)} items: {', '.join(current['key'].head(3).tolist())}",
            'action': 'Ask about progress and any blockers'
        })

    # 3. Blockers
    blocked = details.get('blocked', pd.DataFrame())
    if not blocked.empty:
        talking_points.append({
            'category': 'concern',
            'icon': '🚧',
            'title': 'Active Blockers',
            'content': f"{len(blocked)} blocked item(s): {blocked['key'].iloc[0]} - {blocked['summary'].iloc[0][:40]}...",
            'action': 'Discuss what support is needed to unblock'
        })

    # 4. Velocity Trend
    velocity = details.get('velocity', pd.DataFrame())
    if not velocity.empty and len(velocity) >= 2:
        recent_avg = velocity.head(2)['points_completed'].mean()
        older_avg = velocity.tail(2)['points_completed'].mean() if len(velocity) >= 4 else recent_avg

        if recent_avg > older_avg * 1.2:
            talking_points.append({
                'category': 'recognition',
                'icon': '📈',
                'title': 'Velocity Increasing',
                'content': f"Output has increased by {((recent_avg/older_avg)-1)*100:.0f}% recently",
                'action': 'Acknowledge growth and ensure sustainable pace'
            })
        elif recent_avg < older_avg * 0.7:
            talking_points.append({
                'category': 'concern',
                'icon': '📉',
                'title': 'Velocity Decline',
                'content': f"Output has decreased by {((1-recent_avg/older_avg))*100:.0f}% recently",
                'action': 'Explore if there are impediments or personal factors'
            })

    # 5. Complexity Growth
    complexity = details.get('complexity', pd.DataFrame())
    if not complexity.empty:
        complex_items = complexity[complexity['complexity'].isin(['High (6-8)', 'Complex (8+)'])]
        if not complex_items.empty and complex_items['count'].sum() > 2:
            talking_points.append({
                'category': 'growth',
                'icon': '🎯',
                'title': 'Taking on Complexity',
                'content': f"Handling {int(complex_items['count'].sum())} high-complexity items",
                'action': 'Discuss if ready for more challenging work'
            })

    # 6. Type Diversity
    type_dist = details.get('type_distribution', pd.DataFrame())
    if not type_dist.empty and len(type_dist) >= 3:
        talking_points.append({
            'category': 'growth',
            'icon': '🌟',
            'title': 'Versatility',
            'content': f"Working across {len(type_dist)} different issue types",
            'action': 'Discuss preferences and career development'
        })

    # 7. Workload check
    if member.get('in_progress', 0) > 5:
        talking_points.append({
            'category': 'concern',
            'icon': '⚠️',
            'title': 'High WIP',
            'content': f"{member['in_progress']} items in progress simultaneously",
            'action': 'Discuss prioritization and focus'
        })

    # 8. Career/Growth placeholder
    talking_points.append({
        'category': 'growth',
        'icon': '🚀',
        'title': 'Career Development',
        'content': 'Check in on career goals and growth opportunities',
        'action': 'Ask about skills they want to develop'
    })

    return talking_points


def calculate_performance_metrics(member: Dict, details: Dict) -> Dict[str, Any]:
    """Calculate performance metrics for review."""
    velocity = details.get('velocity', pd.DataFrame())
    completed = details.get('completed_recent', pd.DataFrame())
    complexity = details.get('complexity', pd.DataFrame())

    # Average weekly velocity
    avg_velocity = velocity['points_completed'].mean() if not velocity.empty else 0

    # Velocity trend
    if not velocity.empty and len(velocity) >= 4:
        recent = velocity.head(2)['points_completed'].mean()
        older = velocity.tail(2)['points_completed'].mean()
        velocity_trend = ((recent / older) - 1) * 100 if older > 0 else 0
    else:
        velocity_trend = 0

    # Completion rate
    completion_rate = member.get('completion_rate', 0)

    # Complexity score (weighted by points)
    if not complexity.empty:
        complexity_weights = {'Low (1-2)': 1, 'Medium (3-5)': 2, 'High (6-8)': 3, 'Complex (8+)': 4}
        total_items = complexity['count'].sum()
        weighted_sum = sum(
            complexity_weights.get(row['complexity'], 1) * row['count']
            for _, row in complexity.iterrows()
        )
        complexity_score = (weighted_sum / total_items) if total_items > 0 else 1
    else:
        complexity_score = 1

    # Overall performance rating
    score = 0
    if avg_velocity >= 15:
        score += 30
    elif avg_velocity >= 10:
        score += 25
    elif avg_velocity >= 5:
        score += 15
    else:
        score += 5

    if completion_rate >= 80:
        score += 25
    elif completion_rate >= 60:
        score += 20
    elif completion_rate >= 40:
        score += 10

    if velocity_trend > 10:
        score += 20
    elif velocity_trend >= 0:
        score += 15
    elif velocity_trend >= -10:
        score += 10

    if complexity_score >= 2.5:
        score += 25
    elif complexity_score >= 2:
        score += 20
    elif complexity_score >= 1.5:
        score += 15
    else:
        score += 10

    # Rating
    if score >= 85:
        rating = ('Exceptional', 'badge-exceptional')
    elif score >= 70:
        rating = ('Strong Performer', 'badge-strong')
    elif score >= 55:
        rating = ('Meets Expectations', 'badge-meets')
    elif score >= 40:
        rating = ('Developing', 'badge-developing')
    else:
        rating = ('Needs Attention', 'badge-needs-attention')

    return {
        'avg_velocity': avg_velocity,
        'velocity_trend': velocity_trend,
        'completion_rate': completion_rate,
        'complexity_score': complexity_score,
        'overall_score': score,
        'rating': rating,
        'total_completed': int(member.get('completed', 0)),
        'total_points': float(member.get('completed_points', 0))
    }


def generate_1on1_agenda(member: Dict, talking_points: List, metrics: Dict) -> str:
    """Generate a printable 1-on-1 agenda."""
    now = datetime.now()

    agenda = f"""# 1-on-1 Meeting: {member['name']}
**Date:** {now.strftime('%B %d, %Y')}
**Prepared by:** 1-on-1 Intelligence Hub

---

## 📊 Quick Stats
| Metric | Value |
|--------|-------|
| Completion Rate | {metrics['completion_rate']:.0f}% |
| Avg Weekly Velocity | {metrics['avg_velocity']:.1f} pts |
| Velocity Trend | {'+' if metrics['velocity_trend'] > 0 else ''}{metrics['velocity_trend']:.0f}% |
| Performance Rating | {metrics['rating'][0]} |

---

## 💬 Talking Points

"""

    for point in talking_points:
        emoji_map = {'recognition': '✅', 'discussion': '💬', 'concern': '⚠️', 'growth': '🌱'}
        agenda += f"### {point['icon']} {point['title']}\n"
        agenda += f"- **Context:** {point['content']}\n"
        agenda += f"- **Suggested Action:** {point['action']}\n\n"

    agenda += """---

## 📝 Action Items from This Meeting
- [ ] _Action 1_
- [ ] _Action 2_
- [ ] _Action 3_

## 🎯 Follow-up for Next 1-on-1
- _Topic 1_
- _Topic 2_

---
*Generated by 1-on-1 Intelligence Hub*
"""
    return agenda


def generate_performance_review(member: Dict, metrics: Dict, details: Dict) -> str:
    """Generate performance review data."""
    now = datetime.now()
    quarter = (now.month - 1) // 3 + 1

    review = f"""# Performance Review: {member['name']}
**Period:** Q{quarter} {now.year}
**Generated:** {now.strftime('%B %d, %Y')}

---

## Executive Summary

**Overall Rating:** {metrics['rating'][0]}
**Performance Score:** {metrics['overall_score']}/100

---

## Key Metrics

### Productivity
- **Total Items Completed:** {metrics['total_completed']}
- **Total Story Points Delivered:** {metrics['total_points']:.0f}
- **Average Weekly Velocity:** {metrics['avg_velocity']:.1f} points
- **Completion Rate:** {metrics['completion_rate']:.0f}%

### Growth Indicators
- **Velocity Trend:** {'+' if metrics['velocity_trend'] > 0 else ''}{metrics['velocity_trend']:.0f}% (vs previous period)
- **Complexity Score:** {metrics['complexity_score']:.1f}/4.0

---

## Strengths Observed

"""

    # Add strengths based on metrics
    if metrics['completion_rate'] >= 70:
        review += "- **High Completion Rate:** Consistently delivers on commitments\n"
    if metrics['velocity_trend'] > 10:
        review += "- **Improving Velocity:** Demonstrating growth in output capacity\n"
    if metrics['complexity_score'] >= 2.5:
        review += "- **Handles Complexity:** Successfully tackles challenging work\n"
    if metrics['avg_velocity'] >= 12:
        review += "- **Strong Output:** Above-average productivity\n"

    review += """
---

## Areas for Development

"""

    if metrics['completion_rate'] < 60:
        review += "- **Completion Rate:** Focus on finishing started work before taking new items\n"
    if metrics['velocity_trend'] < -10:
        review += "- **Velocity Decline:** Investigate blockers or capacity issues\n"
    if metrics['complexity_score'] < 1.5:
        review += "- **Complexity Exposure:** Consider taking on more challenging work\n"

    review += """
---

## Recommendations

1. _Manager to add specific recommendations_
2. _Based on discussion in 1-on-1_
3. _Career development goals_

---

## Goals for Next Quarter

1. _Goal 1_
2. _Goal 2_
3. _Goal 3_

---
*Data sourced from project management system. Review with employee for context.*
"""
    return review


def create_velocity_chart(velocity_df: pd.DataFrame) -> go.Figure:
    """Create velocity trend chart."""
    if velocity_df.empty:
        return go.Figure()

    df = velocity_df.sort_values('week')

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['week'],
        y=df['points_completed'],
        mode='lines+markers+text',
        name='Points',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)',
        text=[f"{int(v)}" for v in df['points_completed']],
        textposition='top center',
        textfont=dict(color='#667eea')
    ))

    # Add trend line
    if len(df) >= 3:
        z = np.polyfit(range(len(df)), df['points_completed'].values, 1)
        p = np.poly1d(z)
        trend_values = p(range(len(df)))

        fig.add_trace(go.Scatter(
            x=df['week'],
            y=trend_values,
            mode='lines',
            name='Trend',
            line=dict(color='#f39c12', width=2, dash='dash')
        ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, tickfont=dict(color='#8892b0')),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                   tickfont=dict(color='#8892b0'), title='Points'),
        legend=dict(font=dict(color='#8892b0'), orientation='h', y=-0.15),
        hovermode='x unified'
    )

    return fig


def create_type_distribution_chart(type_df: pd.DataFrame) -> go.Figure:
    """Create issue type distribution chart."""
    if type_df.empty:
        return go.Figure()

    colors = ['#667eea', '#764ba2', '#f093fb', '#27ae60', '#3498db', '#f39c12']

    fig = go.Figure(data=[go.Pie(
        labels=type_df['issue_type'],
        values=type_df['count'],
        hole=0.6,
        marker=dict(colors=colors[:len(type_df)]),
        textinfo='percent',
        textfont=dict(color='white')
    )])

    fig.update_layout(
        height=200,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True,
        legend=dict(font=dict(color='#8892b0', size=10), orientation='h', y=-0.2)
    )

    return fig


def main():
    # Header
    st.markdown("""
<div class="hub-header">
    <h1 class="hub-title">👤 1-on-1 Intelligence Hub</h1>
    <p class="hub-subtitle">Auto-generated agendas • Performance insights • Action tracking</p>
    <div style="margin-top: 16px;">
        <span class="hub-stat">⏱️ Saves 4-8 hrs/week</span>
        <span class="hub-stat">📊 Data-driven conversations</span>
        <span class="hub-stat">📋 Review-ready data</span>
    </div>
</div>
""", unsafe_allow_html=True)

    conn = get_connection()
    if not conn:
        st.error("Database not found. Please sync data first.")
        st.stop()

    # Get team members
    team_members = get_team_members(conn)

    if not team_members:
        st.warning("No team members found in the data.")
        st.stop()

    # Layout: Member selector | Main content
    col_selector, col_content = st.columns([1, 3])

    with col_selector:
        st.markdown('<div class="section-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">👥 Team</div>', unsafe_allow_html=True)

        selected_member = None
        for i, member in enumerate(team_members):
            avatar_color = get_avatar_color(member['name'])
            initials = ''.join([n[0].upper() for n in member['name'].split()[:2]])

            if st.button(
                f"{member['name']}",
                key=f"member_{i}",
                use_container_width=True
            ):
                st.session_state['selected_member'] = member['name']

        # Get selected member
        if 'selected_member' in st.session_state:
            selected_name = st.session_state['selected_member']
            selected_member = next((m for m in team_members if m['name'] == selected_name), team_members[0])
        else:
            selected_member = team_members[0]
            st.session_state['selected_member'] = selected_member['name']

        st.markdown('</div>', unsafe_allow_html=True)

    with col_content:
        if selected_member:
            # Get detailed info
            details = get_member_details(conn, selected_member['name'])
            talking_points = generate_talking_points(selected_member, details)
            metrics = calculate_performance_metrics(selected_member, details)

            # Member header
            avatar_color = get_avatar_color(selected_member['name'])
            initials = ''.join([n[0].upper() for n in selected_member['name'].split()[:2]])

            st.markdown(f"""
<div style="display: flex; align-items: center; gap: 20px; margin-bottom: 24px;">
    <div class="member-avatar" style="background: {avatar_color}; width: 72px; height: 72px; font-size: 24px;">
        {initials}
    </div>
    <div>
        <div style="font-size: 28px; font-weight: 700; color: #fff;">{selected_member['name']}</div>
        <div style="color: #8892b0;">
            {selected_member['total_issues']} total issues •
            {selected_member['completed_points']:.0f} points delivered
        </div>
        <span class="performance-badge {metrics['rating'][1]}">{metrics['rating'][0]}</span>
    </div>
</div>
""", unsafe_allow_html=True)

            # Quick metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                trend_class = 'trend-up' if metrics['velocity_trend'] > 0 else 'trend-down' if metrics['velocity_trend'] < 0 else 'trend-neutral'
                st.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{metrics['avg_velocity']:.1f}</div>
    <div class="metric-label">Avg Weekly Pts</div>
    <div class="metric-trend {trend_class}">
        {'↑' if metrics['velocity_trend'] > 0 else '↓' if metrics['velocity_trend'] < 0 else '→'}
        {abs(metrics['velocity_trend']):.0f}%
    </div>
</div>
""", unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{metrics['completion_rate']:.0f}%</div>
    <div class="metric-label">Completion Rate</div>
</div>
""", unsafe_allow_html=True)

            with m3:
                st.markdown(f"""
<div class="metric-card">
    <div class="metric-value">{selected_member['in_progress']}</div>
    <div class="metric-label">In Progress</div>
</div>
""", unsafe_allow_html=True)

            with m4:
                blocked_count = len(details.get('blocked', pd.DataFrame()))
                blocked_color = '#e74c3c' if blocked_count > 0 else '#27ae60'
                st.markdown(f"""
<div class="metric-card">
    <div class="metric-value" style="color: {blocked_color};">{blocked_count}</div>
    <div class="metric-label">Blocked</div>
</div>
""", unsafe_allow_html=True)

            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📋 1-on-1 Agenda", "📈 Performance", "📊 Activity", "📄 Review Data"])

            with tab1:
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">💬 Auto-Generated Talking Points</div>', unsafe_allow_html=True)

                if talking_points:
                    for point in talking_points:
                        css_class = point['category']
                        st.markdown(f"""
<div class="talking-point {css_class}">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
        <span style="font-weight: 600; color: #fff;">{point['icon']} {point['title']}</span>
        <span style="font-size: 11px; color: #8892b0; text-transform: uppercase;">{point['category']}</span>
    </div>
    <div style="color: #ccd6f6; font-size: 14px; margin-bottom: 6px;">{point['content']}</div>
    <div style="color: #667eea; font-size: 12px;">💡 {point['action']}</div>
</div>
""", unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

                # Generate agenda
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">📄 Printable Agenda</div>', unsafe_allow_html=True)

                if st.button("🚀 Generate 1-on-1 Agenda", type="primary"):
                    agenda = generate_1on1_agenda(selected_member, talking_points, metrics)
                    st.session_state['agenda'] = agenda

                if 'agenda' in st.session_state:
                    st.markdown('<div class="review-section" style="max-height: 400px; overflow-y: auto;">', unsafe_allow_html=True)
                    st.markdown(st.session_state['agenda'])
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.download_button(
                        "📥 Download Agenda",
                        st.session_state['agenda'],
                        file_name=f"1on1_{selected_member['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown"
                    )

                st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                col_chart1, col_chart2 = st.columns([2, 1])

                with col_chart1:
                    st.markdown('<div class="section-container">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">📈 Velocity Trend (8 weeks)</div>', unsafe_allow_html=True)

                    velocity_df = details.get('velocity', pd.DataFrame())
                    if not velocity_df.empty:
                        fig = create_velocity_chart(velocity_df)
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    else:
                        st.info("Not enough data for velocity trend.")

                    st.markdown('</div>', unsafe_allow_html=True)

                with col_chart2:
                    st.markdown('<div class="section-container">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">📊 Work Types</div>', unsafe_allow_html=True)

                    type_df = details.get('type_distribution', pd.DataFrame())
                    if not type_df.empty:
                        fig = create_type_distribution_chart(type_df)
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                    st.markdown('</div>', unsafe_allow_html=True)

                # Complexity analysis
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">🎯 Complexity Handling</div>', unsafe_allow_html=True)

                complexity_df = details.get('complexity', pd.DataFrame())
                if not complexity_df.empty:
                    cols = st.columns(len(complexity_df))
                    colors = {'Low (1-2)': '#27ae60', 'Medium (3-5)': '#3498db',
                             'High (6-8)': '#f39c12', 'Complex (8+)': '#e74c3c'}
                    for i, (_, row) in enumerate(complexity_df.iterrows()):
                        with cols[i]:
                            color = colors.get(row['complexity'], '#667eea')
                            st.markdown(f"""
<div class="metric-card">
    <div class="metric-value" style="color: {color};">{int(row['count'])}</div>
    <div class="metric-label">{row['complexity']}</div>
</div>
""", unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            with tab3:
                col_current, col_completed = st.columns(2)

                with col_current:
                    st.markdown('<div class="section-container">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">🔄 Currently Working On</div>', unsafe_allow_html=True)

                    current = details.get('current_work', pd.DataFrame())
                    if not current.empty:
                        for _, item in current.iterrows():
                            priority_colors = {'Highest': '#e74c3c', 'High': '#e67e22',
                                             'Medium': '#f39c12', 'Low': '#3498db'}
                            p_color = priority_colors.get(item['priority'], '#8892b0')
                            st.markdown(f"""
<div class="action-item">
    <div style="flex: 1;">
        <div style="color: #667eea; font-size: 12px; font-weight: 600;">{item['key']}</div>
        <div style="color: #ccd6f6; font-size: 14px;">{item['summary'][:50]}...</div>
        <div style="display: flex; gap: 8px; margin-top: 4px;">
            <span style="color: {p_color}; font-size: 11px;">{item['priority']}</span>
            <span style="color: #8892b0; font-size: 11px;">• {int(item['story_points'])} pts</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
                    else:
                        st.info("No items currently in progress.")

                    st.markdown('</div>', unsafe_allow_html=True)

                with col_completed:
                    st.markdown('<div class="section-container">', unsafe_allow_html=True)
                    st.markdown('<div class="section-title">✅ Recently Completed</div>', unsafe_allow_html=True)

                    completed = details.get('completed_recent', pd.DataFrame())
                    if not completed.empty:
                        for _, item in completed.head(5).iterrows():
                            st.markdown(f"""
<div class="action-item">
    <div class="action-checkbox done">✓</div>
    <div style="flex: 1;">
        <div style="color: #667eea; font-size: 12px; font-weight: 600;">{item['key']}</div>
        <div style="color: #ccd6f6; font-size: 14px;">{item['summary'][:50]}...</div>
        <div style="color: #27ae60; font-size: 11px;">{int(item['story_points'])} pts delivered</div>
    </div>
</div>
""", unsafe_allow_html=True)
                    else:
                        st.info("No recent completions.")

                    st.markdown('</div>', unsafe_allow_html=True)

            with tab4:
                st.markdown('<div class="section-container">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">📄 Performance Review Data Generator</div>', unsafe_allow_html=True)

                st.markdown("""
<div style="padding: 16px; background: rgba(102, 126, 234, 0.1); border-radius: 12px; margin-bottom: 16px;">
    <div style="color: #667eea; font-weight: 600; margin-bottom: 8px;">⏱️ Save hours on performance reviews</div>
    <div style="color: #8892b0; font-size: 14px;">
        Generate a data-rich performance review document with one click.
        All metrics are pulled directly from work data.
    </div>
</div>
""", unsafe_allow_html=True)

                if st.button("📊 Generate Performance Review Data", type="primary"):
                    review = generate_performance_review(selected_member, metrics, details)
                    st.session_state['review'] = review

                if 'review' in st.session_state:
                    st.markdown('<div class="review-section" style="max-height: 500px; overflow-y: auto;">', unsafe_allow_html=True)
                    st.markdown(st.session_state['review'])
                    st.markdown('</div>', unsafe_allow_html=True)

                    col_dl1, col_dl2 = st.columns(2)
                    with col_dl1:
                        st.download_button(
                            "📥 Download Review (Markdown)",
                            st.session_state['review'],
                            file_name=f"performance_review_{selected_member['name'].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
                            mime="text/markdown"
                        )

                st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown(f"""
<div style="text-align: center; color: #8892b0; font-size: 12px;">
    👤 1-on-1 Intelligence Hub | Saving managers 4-8 hours weekly |
    Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
""", unsafe_allow_html=True)

    conn.close()


if __name__ == "__main__":
    main()
