"""
📬 Waiting-On Inbox - Follow-Through OS
The PM's command center for everything they're waiting on.
Eliminates dropped balls through active queue management with timers, nudges, and escalation.

Target: 2 hours/week saved + 1-2 fewer dropped balls
"""

import streamlit as st
import pandas as pd
import duckdb
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Optional, List, Dict, Any
import hashlib

# Import page guide component
from src.dashboard.components import render_page_guide

st.set_page_config(page_title="Waiting-On Inbox", page_icon="📬", layout="wide")

# ============================================================================
# PREMIUM CSS - FOLLOW-THROUGH OS THEME
# ============================================================================
st.markdown("""
<style>
    /* Main Header */
    .inbox-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f4c75 50%, #3282b8 100%);
        border-radius: 20px;
        padding: 28px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .inbox-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 60%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    .inbox-title {
        font-size: 32px;
        font-weight: 800;
        color: white;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .inbox-subtitle {
        color: rgba(255,255,255,0.85);
        font-size: 15px;
        margin-top: 6px;
    }
    .inbox-stats {
        display: flex;
        gap: 16px;
        margin-top: 16px;
    }
    .inbox-stat {
        background: rgba(255,255,255,0.12);
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 13px;
        color: white;
        font-weight: 500;
    }
    .inbox-stat.alert {
        background: rgba(239, 68, 68, 0.3);
        border: 1px solid rgba(239, 68, 68, 0.5);
    }

    /* Quick Win Widget - Overdue Alert */
    .overdue-widget {
        background: linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 20px;
        border: 1px solid rgba(252, 165, 165, 0.3);
        box-shadow: 0 8px 32px rgba(127, 29, 29, 0.4);
        position: relative;
        overflow: hidden;
    }
    .overdue-widget::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -30%;
        width: 80%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        pointer-events: none;
    }
    .overdue-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 14px;
    }
    .overdue-icon {
        font-size: 28px;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.1); }
    }
    .overdue-title {
        color: #fecaca;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    .overdue-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 16px;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        margin-bottom: 8px;
        border-left: 4px solid #fca5a5;
        transition: all 0.2s;
    }
    .overdue-item:hover {
        background: rgba(255,255,255,0.15);
        transform: translateX(4px);
    }
    .overdue-person {
        color: #fecaca;
        font-weight: 600;
        font-size: 13px;
    }
    .overdue-desc {
        color: white;
        font-size: 14px;
        flex: 1;
        margin: 0 16px;
    }
    .overdue-days {
        background: rgba(0,0,0,0.3);
        color: #fee2e2;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        white-space: nowrap;
    }

    /* Inbox Groups */
    .inbox-group {
        background: white;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .group-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid #f1f5f9;
    }
    .group-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 16px;
        font-weight: 700;
        color: #1e293b;
    }
    .group-badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }
    .badge-overdue { background: #fee2e2; color: #991b1b; }
    .badge-due-soon { background: #fef3c7; color: #92400e; }
    .badge-on-track { background: #dcfce7; color: #166534; }

    /* Waiting Item Card */
    .waiting-item {
        background: #f8fafc;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        border: 1px solid #e2e8f0;
        transition: all 0.2s;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .waiting-item:hover {
        border-color: #cbd5e1;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        transform: translateY(-1px);
    }
    .waiting-item.overdue {
        border-left: 4px solid #ef4444;
        background: #fef2f2;
    }
    .waiting-item.due-soon {
        border-left: 4px solid #f59e0b;
        background: #fffbeb;
    }
    .waiting-item.on-track {
        border-left: 4px solid #22c55e;
    }

    .item-avatar {
        width: 44px;
        height: 44px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 14px;
        color: white;
        flex-shrink: 0;
    }

    .item-content {
        flex: 1;
        min-width: 0;
    }
    .item-description {
        font-size: 14px;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .item-meta {
        font-size: 12px;
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .item-linked {
        background: #e0e7ff;
        color: #4338ca;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        font-family: monospace;
    }

    .item-timing {
        text-align: right;
        flex-shrink: 0;
    }
    .timing-days {
        font-size: 14px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .timing-days.overdue { color: #dc2626; }
    .timing-days.due-soon { color: #d97706; }
    .timing-days.on-track { color: #16a34a; }
    .timing-date {
        font-size: 11px;
        color: #94a3b8;
    }

    .item-actions {
        display: flex;
        gap: 8px;
        flex-shrink: 0;
    }
    .action-btn {
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        border: none;
        cursor: pointer;
        transition: all 0.2s;
    }
    .action-btn.nudge {
        background: #fef3c7;
        color: #92400e;
    }
    .action-btn.nudge:hover {
        background: #fde68a;
    }
    .action-btn.complete {
        background: #dcfce7;
        color: #166534;
    }
    .action-btn.complete:hover {
        background: #bbf7d0;
    }
    .action-btn.cancel {
        background: #f1f5f9;
        color: #64748b;
    }
    .action-btn.cancel:hover {
        background: #e2e8f0;
    }

    /* Urgency Tags */
    .urgency-tag {
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        text-transform: uppercase;
    }
    .urgency-blocker { background: #fee2e2; color: #991b1b; }
    .urgency-high { background: #ffedd5; color: #9a3412; }
    .urgency-medium { background: #fef3c7; color: #92400e; }
    .urgency-low { background: #e0f2fe; color: #075985; }

    /* Create Form */
    .create-form {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid #bae6fd;
    }
    .form-title {
        font-size: 16px;
        font-weight: 700;
        color: #0c4a6e;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Stats Footer */
    .stats-footer {
        background: white;
        border-radius: 16px;
        padding: 20px 24px;
        border: 1px solid #e2e8f0;
        display: flex;
        justify-content: space-around;
        text-align: center;
    }
    .stat-item {
        padding: 0 20px;
    }
    .stat-value {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
        margin-top: 4px;
    }
    .stat-value.danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        -webkit-background-clip: text;
    }
    .stat-value.success {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        -webkit-background-clip: text;
    }

    /* Empty State */
    .empty-state {
        text-align: center;
        padding: 60px 40px;
        color: #64748b;
    }
    .empty-icon {
        font-size: 64px;
        margin-bottom: 16px;
        opacity: 0.5;
    }
    .empty-title {
        font-size: 20px;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 8px;
    }
    .empty-text {
        font-size: 14px;
        max-width: 400px;
        margin: 0 auto;
    }

    /* Nudge History */
    .nudge-history {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 4px;
    }
    .nudge-count {
        background: #fef3c7;
        color: #92400e;
        padding: 1px 6px;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def get_connection():
    """Get database connection."""
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path))


def ensure_tables_exist(conn) -> bool:
    """Ensure waiting_on tables exist, create if not."""
    try:
        conn.execute("SELECT 1 FROM waiting_on LIMIT 1")
        return True
    except:
        # Create tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS waiting_on (
                id INTEGER PRIMARY KEY,
                created_by VARCHAR NOT NULL,
                created_by_name VARCHAR,
                waiting_for VARCHAR NOT NULL,
                waiting_for_name VARCHAR,
                description VARCHAR(280) NOT NULL,
                context TEXT,
                evidence_required VARCHAR(280),
                expected_by DATE NOT NULL,
                urgency VARCHAR DEFAULT 'medium',
                linked_issue_key VARCHAR,
                linked_issue_summary VARCHAR,
                source VARCHAR DEFAULT 'manual',
                status VARCHAR DEFAULT 'active',
                acknowledged_at TIMESTAMP,
                completed_at TIMESTAMP,
                completion_evidence TEXT,
                canceled_reason TEXT,
                nudge_count INTEGER DEFAULT 0,
                last_nudged_at TIMESTAMP,
                escalated_at TIMESTAMP,
                escalated_to VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS waiting_on_history (
                id INTEGER PRIMARY KEY,
                waiting_on_id INTEGER NOT NULL,
                action VARCHAR NOT NULL,
                actor VARCHAR,
                note TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        return True


def get_team_members(conn) -> List[Dict]:
    """Get list of team members for dropdown."""
    try:
        df = conn.execute("""
            SELECT DISTINCT
                COALESCE(assignee_id, assignee_name) as id,
                COALESCE(assignee_name, 'Unknown') as name
            FROM issues
            WHERE assignee_name IS NOT NULL
            ORDER BY name
        """).fetchdf()
        return df.to_dict('records')
    except:
        return []


def get_linked_issues(conn) -> List[Dict]:
    """Get open issues for linking."""
    try:
        df = conn.execute("""
            SELECT key, summary, status, assignee_name
            FROM issues
            WHERE status NOT IN ('Done', 'Terminé(e)', 'Closed', 'Resolved')
            ORDER BY updated DESC
            LIMIT 100
        """).fetchdf()
        return df.to_dict('records')
    except:
        return []


def get_waiting_items(conn, user_filter: Optional[str] = None) -> pd.DataFrame:
    """Get all active waiting-on items."""
    query = """
        SELECT
            id, created_by, created_by_name, waiting_for, waiting_for_name,
            description, context, evidence_required, expected_by, urgency,
            linked_issue_key, linked_issue_summary, source, status,
            acknowledged_at, nudge_count, last_nudged_at, created_at
        FROM waiting_on
        WHERE status IN ('active', 'acknowledged')
    """
    if user_filter:
        query += f" AND created_by_name = '{user_filter}'"
    query += " ORDER BY expected_by ASC"

    try:
        return conn.execute(query).fetchdf()
    except:
        return pd.DataFrame()


def create_waiting_item(conn, data: Dict) -> bool:
    """Create a new waiting-on item."""
    try:
        # Get next ID
        result = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM waiting_on").fetchone()
        next_id = result[0]

        conn.execute("""
            INSERT INTO waiting_on (
                id, created_by, created_by_name, waiting_for, waiting_for_name,
                description, context, evidence_required, expected_by, urgency,
                linked_issue_key, linked_issue_summary, source, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'manual', 'active')
        """, [
            next_id,
            data.get('created_by', 'current_user'),
            data.get('created_by_name', 'You'),
            data.get('waiting_for', ''),
            data.get('waiting_for_name', ''),
            data.get('description', ''),
            data.get('context', ''),
            data.get('evidence_required', ''),
            data.get('expected_by'),
            data.get('urgency', 'medium'),
            data.get('linked_issue_key'),
            data.get('linked_issue_summary'),
        ])

        # Log history
        history_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM waiting_on_history").fetchone()[0]
        conn.execute("""
            INSERT INTO waiting_on_history (id, waiting_on_id, action, actor, note)
            VALUES (?, ?, 'created', ?, ?)
        """, [history_id, next_id, data.get('created_by_name', 'You'), f"Waiting for {data.get('waiting_for_name', '')}"])

        return True
    except Exception as e:
        st.error(f"Failed to create: {e}")
        return False


def nudge_item(conn, item_id: int) -> bool:
    """Send a nudge for an item."""
    try:
        conn.execute("""
            UPDATE waiting_on
            SET nudge_count = nudge_count + 1,
                last_nudged_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, [item_id])

        history_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM waiting_on_history").fetchone()[0]
        conn.execute("""
            INSERT INTO waiting_on_history (id, waiting_on_id, action, actor, note)
            VALUES (?, ?, 'nudged', 'You', 'Sent reminder')
        """, [history_id, item_id])

        return True
    except:
        return False


def complete_item(conn, item_id: int, evidence: str = "") -> bool:
    """Mark an item as completed."""
    try:
        conn.execute("""
            UPDATE waiting_on
            SET status = 'completed',
                completed_at = CURRENT_TIMESTAMP,
                completion_evidence = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, [evidence, item_id])

        history_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM waiting_on_history").fetchone()[0]
        conn.execute("""
            INSERT INTO waiting_on_history (id, waiting_on_id, action, actor, note)
            VALUES (?, ?, 'completed', 'You', ?)
        """, [history_id, item_id, evidence or 'Marked as received'])

        return True
    except:
        return False


def cancel_item(conn, item_id: int, reason: str = "") -> bool:
    """Cancel an item."""
    try:
        conn.execute("""
            UPDATE waiting_on
            SET status = 'canceled',
                canceled_reason = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, [reason, item_id])

        history_id = conn.execute("SELECT COALESCE(MAX(id), 0) + 1 FROM waiting_on_history").fetchone()[0]
        conn.execute("""
            INSERT INTO waiting_on_history (id, waiting_on_id, action, actor, note)
            VALUES (?, ?, 'canceled', 'You', ?)
        """, [history_id, item_id, reason or 'No longer needed'])

        return True
    except:
        return False


def detect_auto_items(conn) -> List[Dict]:
    """Detect potential waiting-on items from Jira data."""
    detected = []

    # 1. Blocked issues without waiting-on entry
    try:
        blocked = conn.execute("""
            SELECT i.key, i.summary, i.assignee_name, i.updated
            FROM issues i
            WHERE i.status IN ('Blocked', 'Waiting', 'On Hold')
            AND NOT EXISTS (
                SELECT 1 FROM waiting_on w
                WHERE w.linked_issue_key = i.key
                AND w.status = 'active'
            )
            ORDER BY i.updated DESC
            LIMIT 5
        """).fetchdf()

        for _, row in blocked.iterrows():
            detected.append({
                'type': 'blocked',
                'key': row['key'],
                'summary': row['summary'],
                'person': row['assignee_name'] or 'Unassigned',
                'reason': f"Issue is blocked/waiting"
            })
    except:
        pass

    # 2. Recent handoffs (assignee changed in last 2 days)
    try:
        handoffs = conn.execute("""
            SELECT
                c.issue_key,
                c.to_value as new_assignee,
                c.from_value as old_assignee,
                c.changed_at,
                i.summary
            FROM issue_changelog c
            JOIN issues i ON c.issue_key = i.key
            WHERE c.field = 'assignee'
            AND c.changed_at > CURRENT_TIMESTAMP - INTERVAL '2 days'
            AND c.from_value IS NOT NULL
            AND c.to_value IS NOT NULL
            AND NOT EXISTS (
                SELECT 1 FROM waiting_on w
                WHERE w.linked_issue_key = c.issue_key
                AND w.status = 'active'
                AND w.source = 'auto_handoff'
            )
            ORDER BY c.changed_at DESC
            LIMIT 5
        """).fetchdf()

        for _, row in handoffs.iterrows():
            detected.append({
                'type': 'handoff',
                'key': row['issue_key'],
                'summary': row['summary'],
                'person': row['new_assignee'],
                'old_person': row['old_assignee'],
                'reason': f"Handed off from {row['old_assignee']}"
            })
    except:
        pass

    return detected


def get_waiting_stats(conn) -> Dict:
    """Get statistics for the inbox."""
    stats = {
        'total_active': 0,
        'overdue': 0,
        'due_soon': 0,
        'on_track': 0,
        'completed_this_week': 0,
        'avg_days_to_complete': 0,
        'on_time_rate': 0
    }

    try:
        # Active counts
        result = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN expected_by < CURRENT_DATE THEN 1 ELSE 0 END) as overdue,
                SUM(CASE WHEN expected_by >= CURRENT_DATE AND expected_by <= CURRENT_DATE + INTERVAL '2 days' THEN 1 ELSE 0 END) as due_soon,
                SUM(CASE WHEN expected_by > CURRENT_DATE + INTERVAL '2 days' THEN 1 ELSE 0 END) as on_track
            FROM waiting_on
            WHERE status IN ('active', 'acknowledged')
        """).fetchone()

        if result:
            stats['total_active'] = result[0] or 0
            stats['overdue'] = result[1] or 0
            stats['due_soon'] = result[2] or 0
            stats['on_track'] = result[3] or 0

        # Completed this week
        completed = conn.execute("""
            SELECT COUNT(*) FROM waiting_on
            WHERE status = 'completed'
            AND completed_at >= CURRENT_DATE - INTERVAL '7 days'
        """).fetchone()
        stats['completed_this_week'] = completed[0] if completed else 0

        # On-time rate (last 30 days)
        on_time = conn.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN completed_at::DATE <= expected_by THEN 1 ELSE 0 END) as on_time
            FROM waiting_on
            WHERE status = 'completed'
            AND completed_at >= CURRENT_DATE - INTERVAL '30 days'
        """).fetchone()

        if on_time and on_time[0] > 0:
            stats['on_time_rate'] = round((on_time[1] / on_time[0]) * 100)

    except:
        pass

    return stats


def get_avatar_color(name: str) -> str:
    """Generate consistent color for avatar."""
    colors = ['#6366f1', '#8b5cf6', '#d946ef', '#ec4899', '#f43f5e', '#f97316', '#eab308', '#22c55e', '#14b8a6', '#3b82f6']
    return colors[hash(name or 'Unknown') % len(colors)]


def get_initials(name: str) -> str:
    """Get initials from name."""
    if not name or name == 'Unknown':
        return '?'
    parts = name.split()
    return ''.join([p[0].upper() for p in parts[:2]])


def calculate_days_remaining(expected_by) -> tuple:
    """Calculate days remaining and status."""
    if expected_by is None:
        return 0, 'on_track', 'No date'

    if isinstance(expected_by, str):
        expected_by = datetime.strptime(expected_by[:10], '%Y-%m-%d').date()
    elif isinstance(expected_by, datetime):
        expected_by = expected_by.date()

    today = date.today()
    delta = (expected_by - today).days

    if delta < 0:
        return delta, 'overdue', f"{abs(delta)}d overdue"
    elif delta == 0:
        return 0, 'due-soon', "Due today"
    elif delta <= 2:
        return delta, 'due-soon', f"Due in {delta}d"
    else:
        return delta, 'on-track', f"Due in {delta}d"


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():

    # Render page guide in sidebar
    render_page_guide()
    conn = get_connection()
    if not conn:
        st.error("Database not found. Please sync data first.")
        st.stop()

    ensure_tables_exist(conn)

    # Get data
    stats = get_waiting_stats(conn)
    items_df = get_waiting_items(conn)
    team_members = get_team_members(conn)

    # ========== HEADER ==========
    st.markdown(f"""
    <div class="inbox-header">
        <div class="inbox-title">📬 Waiting-On Inbox</div>
        <div class="inbox-subtitle">Everything you're waiting on from others — tracked, timed, and followed through</div>
        <div class="inbox-stats">
            <span class="inbox-stat{'  alert' if stats['overdue'] > 0 else ''}">
                {'🔴 ' if stats['overdue'] > 0 else ''}{stats['overdue']} Overdue
            </span>
            <span class="inbox-stat">{stats['due_soon']} Due Soon</span>
            <span class="inbox-stat">{stats['on_track']} On Track</span>
            <span class="inbox-stat">✅ {stats['completed_this_week']} This Week</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ========== QUICK WIN: OVERDUE ALERT ==========
    if stats['overdue'] > 0 and not items_df.empty:
        overdue_items = []
        for _, row in items_df.iterrows():
            days, status, _ = calculate_days_remaining(row['expected_by'])
            if status == 'overdue':
                overdue_items.append({
                    'id': row['id'],
                    'person': row['waiting_for_name'] or 'Unknown',
                    'desc': row['description'][:50] + ('...' if len(row['description']) > 50 else ''),
                    'days': abs(days)
                })

        if overdue_items:
            items_html = ""
            for item in overdue_items[:3]:
                items_html += f"""
                <div class="overdue-item">
                    <span class="overdue-person">@{item['person']}</span>
                    <span class="overdue-desc">{item['desc']}</span>
                    <span class="overdue-days">{item['days']}d overdue</span>
                </div>
                """

            st.markdown(f"""
            <div class="overdue-widget">
                <div class="overdue-header">
                    <span class="overdue-icon">🚨</span>
                    <span class="overdue-title">Action Required — {len(overdue_items)} Items Overdue</span>
                </div>
                {items_html}
            </div>
            """, unsafe_allow_html=True)

    # ========== CREATE NEW / AUTO-DETECT TABS ==========
    tab1, tab2, tab3 = st.tabs(["📬 Inbox", "➕ Create New", "🔍 Auto-Detect"])

    with tab2:
        st.markdown('<div class="create-form">', unsafe_allow_html=True)
        st.markdown('<div class="form-title">➕ Create New Waiting-On Item</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            # Who are you waiting for?
            member_options = ["Select person..."] + [m['name'] for m in team_members]
            waiting_for_name = st.selectbox("Who are you waiting for?", member_options)

            # What do you need?
            description = st.text_input("What do you need from them?", placeholder="API spec document")

            # Why does it matter?
            context = st.text_area("Why does it matter? (optional)", placeholder="Needed for sprint review demo", height=80)

        with col2:
            # When do you need it?
            expected_by = st.date_input("When do you need it?", value=date.today() + timedelta(days=3))

            # Urgency
            urgency = st.selectbox("Urgency", ["low", "medium", "high", "blocker"], index=1)

            # Link to issue
            issues = get_linked_issues(conn)
            issue_options = ["No linked issue"] + [f"{i['key']}: {i['summary'][:40]}" for i in issues]
            linked_issue = st.selectbox("Link to Jira issue (optional)", issue_options)

        if st.button("➕ Create Waiting-On Item", type="primary", use_container_width=True):
            if waiting_for_name == "Select person...":
                st.error("Please select who you're waiting for")
            elif not description:
                st.error("Please describe what you need")
            else:
                # Find the member ID
                waiting_for_id = waiting_for_name
                for m in team_members:
                    if m['name'] == waiting_for_name:
                        waiting_for_id = m['id']
                        break

                # Parse linked issue
                linked_key = None
                linked_summary = None
                if linked_issue != "No linked issue":
                    linked_key = linked_issue.split(":")[0]
                    linked_summary = linked_issue.split(": ", 1)[1] if ": " in linked_issue else ""

                data = {
                    'created_by': 'current_user',
                    'created_by_name': 'You',
                    'waiting_for': waiting_for_id,
                    'waiting_for_name': waiting_for_name,
                    'description': description,
                    'context': context,
                    'expected_by': expected_by,
                    'urgency': urgency,
                    'linked_issue_key': linked_key,
                    'linked_issue_summary': linked_summary
                }

                if create_waiting_item(conn, data):
                    st.success(f"✅ Created! {waiting_for_name} will be notified.")
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown("### 🔍 Auto-Detected Items")
        st.markdown("*Items that might need tracking based on Jira activity*")

        detected = detect_auto_items(conn)

        if detected:
            for item in detected:
                icon = "🚫" if item['type'] == 'blocked' else "🔄"
                with st.container():
                    col1, col2, col3 = st.columns([1, 4, 1])
                    with col1:
                        st.markdown(f"### {icon}")
                    with col2:
                        st.markdown(f"**{item['key']}**: {item['summary'][:60]}")
                        st.caption(f"{item['reason']} • Waiting for: {item['person']}")
                    with col3:
                        if st.button("Track", key=f"track_{item['key']}"):
                            data = {
                                'created_by': 'current_user',
                                'created_by_name': 'You',
                                'waiting_for': item['person'],
                                'waiting_for_name': item['person'],
                                'description': f"Resolution for {item['key']}",
                                'context': item['summary'],
                                'expected_by': date.today() + timedelta(days=3),
                                'urgency': 'medium',
                                'linked_issue_key': item['key'],
                                'linked_issue_summary': item['summary']
                            }
                            if create_waiting_item(conn, data):
                                st.success("Added to inbox!")
                                st.rerun()
                    st.divider()
        else:
            st.info("No auto-detected items. Your Jira data looks clean!")

    with tab1:
        if items_df.empty:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">📭</div>
                <div class="empty-title">Your inbox is empty</div>
                <div class="empty-text">
                    Nothing to wait on! Create a new item when you need something from someone,
                    or check the Auto-Detect tab for suggestions.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Group items by status
            overdue_items = []
            due_soon_items = []
            on_track_items = []

            for _, row in items_df.iterrows():
                days, status, timing_text = calculate_days_remaining(row['expected_by'])
                item_data = {
                    'id': row['id'],
                    'person': row['waiting_for_name'] or 'Unknown',
                    'description': row['description'],
                    'context': row['context'],
                    'expected_by': row['expected_by'],
                    'urgency': row['urgency'],
                    'linked_key': row['linked_issue_key'],
                    'linked_summary': row['linked_issue_summary'],
                    'nudge_count': row['nudge_count'] or 0,
                    'last_nudged': row['last_nudged_at'],
                    'acknowledged': row['acknowledged_at'] is not None,
                    'days': days,
                    'timing_text': timing_text,
                    'status_class': status
                }

                if status == 'overdue':
                    overdue_items.append(item_data)
                elif status == 'due-soon':
                    due_soon_items.append(item_data)
                else:
                    on_track_items.append(item_data)

            # Render groups
            def render_group(title: str, items: List[Dict], badge_class: str, icon: str):
                if not items:
                    return

                st.markdown(f"""
                <div class="inbox-group">
                    <div class="group-header">
                        <div class="group-title">
                            <span>{icon}</span>
                            <span>{title}</span>
                        </div>
                        <span class="group-badge {badge_class}">{len(items)} items</span>
                    </div>
                """, unsafe_allow_html=True)

                for item in items:
                    avatar_color = get_avatar_color(item['person'])
                    initials = get_initials(item['person'])

                    urgency_html = ""
                    if item['urgency'] in ['high', 'blocker']:
                        urgency_html = f'<span class="urgency-tag urgency-{item["urgency"]}">{item["urgency"]}</span>'

                    linked_html = ""
                    if item['linked_key']:
                        linked_html = f'<span class="item-linked">{item["linked_key"]}</span>'

                    nudge_html = ""
                    if item['nudge_count'] > 0:
                        nudge_html = f'<div class="nudge-history">Nudged <span class="nudge-count">{item["nudge_count"]}x</span></div>'

                    ack_icon = "✓ " if item['acknowledged'] else ""

                    st.markdown(f"""
                    <div class="waiting-item {item['status_class']}">
                        <div class="item-avatar" style="background: {avatar_color};">{initials}</div>
                        <div class="item-content">
                            <div class="item-description">
                                {ack_icon}{item['description'][:60]}{'...' if len(item['description']) > 60 else ''}
                                {urgency_html}
                            </div>
                            <div class="item-meta">
                                <span>From @{item['person']}</span>
                                {linked_html}
                            </div>
                            {nudge_html}
                        </div>
                        <div class="item-timing">
                            <div class="timing-days {item['status_class']}">{item['timing_text']}</div>
                            <div class="timing-date">{item['expected_by']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Action buttons (using Streamlit for interactivity)
                    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
                    with col1:
                        if st.button("📤 Nudge", key=f"nudge_{item['id']}", help="Send reminder"):
                            if nudge_item(conn, item['id']):
                                st.toast(f"Nudge sent to {item['person']}!")
                                st.rerun()
                    with col2:
                        if st.button("✅ Done", key=f"done_{item['id']}", help="Mark as received"):
                            if complete_item(conn, item['id']):
                                st.toast("Marked as complete!")
                                st.rerun()
                    with col3:
                        if st.button("❌ Cancel", key=f"cancel_{item['id']}", help="No longer needed"):
                            if cancel_item(conn, item['id']):
                                st.toast("Canceled")
                                st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

            # Render all groups
            render_group("🔴 Overdue", overdue_items, "badge-overdue", "🔴")
            render_group("🟡 Due Soon", due_soon_items, "badge-due-soon", "🟡")
            render_group("🟢 On Track", on_track_items, "badge-on-track", "🟢")

    # ========== STATS FOOTER ==========
    st.markdown("---")
    st.markdown(f"""
    <div class="stats-footer">
        <div class="stat-item">
            <div class="stat-value">{stats['total_active']}</div>
            <div class="stat-label">Active Items</div>
        </div>
        <div class="stat-item">
            <div class="stat-value danger">{stats['overdue']}</div>
            <div class="stat-label">Overdue</div>
        </div>
        <div class="stat-item">
            <div class="stat-value success">{stats['completed_this_week']}</div>
            <div class="stat-label">Completed This Week</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">{stats['on_time_rate']}%</div>
            <div class="stat-label">On-Time Rate (30d)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    conn.close()


if __name__ == "__main__":
    main()
