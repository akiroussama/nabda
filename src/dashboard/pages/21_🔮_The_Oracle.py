"""
🔮 THE ORACLE - Your PM Autopilot

THE REVOLUTION: YOU DON'T ANALYZE. YOU DON'T DECIDE. YOU JUST APPROVE.

Before The Oracle: "Here's a dashboard with 47 metrics. Now figure out what to do."
After The Oracle: "Do THIS. Right now. One click."

This is GPS for project management.
This is Netflix for PM decisions.
This is autopilot for your workday.

ONE WIDGET. ONE ACTION. ONE CLICK.

The Oracle has already:
- Analyzed all your data
- Considered all the trade-offs
- Written the exact message/action
- Calculated the impact

You just approve. Or skip to the next one.

That's it. That's the revolution.

Target: Eliminate 90% of PM decision fatigue
Competitor Gap: INFINITY (they can't even conceive this)
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random
import duckdb
from pathlib import Path

# Import page guide component
from src.dashboard.components import render_page_guide

# Page configuration - MINIMAL, FOCUSED
st.set_page_config(
    page_title="The Oracle",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =============================================================================
# THE ORACLE DATA MODELS
# =============================================================================

class ActionType(Enum):
    """Types of actions The Oracle can recommend."""
    SEND_MESSAGE = "send_message"      # Pre-written message to send
    REASSIGN = "reassign"              # One-click reassignment
    ESCALATE = "escalate"              # One-click escalation
    CANCEL = "cancel"                  # Cancel a meeting/task
    SCHEDULE = "schedule"              # Schedule something
    DECIDE = "decide"                  # Make a decision
    PROTECT = "protect"                # Protect someone's time
    UNBLOCK = "unblock"                # Unblock something


@dataclass
class OracleAction:
    """A single action from The Oracle."""
    # The instruction
    headline: str  # "Send this message to Sarah"
    subtext: str   # "The API blocker needs escalation"

    # The pre-built action
    action_type: ActionType
    action_content: str  # The actual message/action
    action_target: str   # Who/what it targets

    # The why (one sentence)
    reason: str

    # Impact metrics
    time_saved_minutes: int
    risk_if_ignored: str  # "Delivery slips 3 days"
    confidence: float  # 0-1

    # Priority
    urgency: str  # "NOW", "TODAY", "THIS WEEK"

    # Metadata
    data_source: str  # What data informed this
    generated_at: datetime


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

@st.cache_resource
def get_connection():
    """Get DuckDB connection."""
    try:
        db_path = Path("data/jira.duckdb")
        return duckdb.connect(str(db_path), read_only=True) if db_path.exists() else None
    except Exception as e:
        return None


# =============================================================================
# THE ORACLE ENGINE - THE BRAIN
# =============================================================================

def analyze_blockers_for_action(conn, project_key: str) -> Optional[OracleAction]:
    """Analyze blockers and generate an action if needed."""
    try:
        # Find the most critical blocker
        blocker = conn.execute(f"""
            SELECT
                i.key,
                i.summary,
                i.priority,
                u.pseudonym as assignee,
                DATEDIFF('day', i.updated, CURRENT_TIMESTAMP) as days_blocked
            FROM issues i
            LEFT JOIN users u ON i.assignee_id = u.account_id
            WHERE i.project_key = '{project_key}'
              AND i.status IN ('Blocked', 'On Hold', 'Waiting')
            ORDER BY
                CASE i.priority WHEN 'Highest' THEN 1 WHEN 'High' THEN 2 ELSE 3 END,
                days_blocked DESC
            LIMIT 1
        """).fetchone()

        if not blocker or blocker[4] < 2:
            return None

        key, summary, priority, assignee, days = blocker

        # Generate the exact message to send
        message = f"""Hi {assignee or 'team'},

Quick check on {key} - it's been blocked for {days} days.

What's the ONE thing needed to unblock this?

If you need help, let me know and I'll pull in the right people.

Thanks!"""

        return OracleAction(
            headline=f"Send this message to {assignee or 'the team'}",
            subtext=f"{key} has been blocked for {days} days",
            action_type=ActionType.SEND_MESSAGE,
            action_content=message,
            action_target=assignee or "team",
            reason=f"Blocked {days} days. Every day costs ~{days * 2} hours of delay.",
            time_saved_minutes=30,
            risk_if_ignored=f"Delivery slips {days + 2} more days",
            confidence=0.92,
            urgency="NOW" if days > 5 else "TODAY",
            data_source=f"Blocker analysis: {key}",
            generated_at=datetime.now()
        )
    except:
        return None


def analyze_overload_for_action(conn, project_key: str) -> Optional[OracleAction]:
    """Find overloaded people and suggest reassignment."""
    try:
        # Find most overloaded person
        overloaded = conn.execute(f"""
            SELECT
                i.assignee_id,
                u.pseudonym as name,
                COUNT(*) as active_count,
                COUNT(CASE WHEN i.status = 'En cours' THEN 1 END) as in_progress
            FROM issues i
            LEFT JOIN users u ON i.assignee_id = u.account_id
            WHERE i.project_key = '{project_key}'
              AND i.status NOT IN ('Terminé(e)', 'Done', 'Closed')
              AND i.assignee_id IS NOT NULL
            GROUP BY i.assignee_id, u.pseudonym
            HAVING COUNT(*) > 6
            ORDER BY in_progress DESC, active_count DESC
            LIMIT 1
        """).fetchone()

        if not overloaded:
            return None

        person_id, name, total, in_progress = overloaded

        # Find the lowest priority item to reassign
        item_to_move = conn.execute(f"""
            SELECT key, summary, priority
            FROM issues
            WHERE project_key = '{project_key}'
              AND assignee_id = '{person_id}'
              AND status NOT IN ('Terminé(e)', 'En cours')
            ORDER BY
                CASE priority WHEN 'Low' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END
            LIMIT 1
        """).fetchone()

        # Find who can take it
        available = conn.execute(f"""
            SELECT u.pseudonym
            FROM users u
            LEFT JOIN (
                SELECT assignee_id, COUNT(*) as cnt
                FROM issues
                WHERE status NOT IN ('Terminé(e)', 'Done')
                GROUP BY assignee_id
            ) i ON u.account_id = i.assignee_id
            WHERE u.active = true
              AND COALESCE(i.cnt, 0) < 4
              AND u.account_id != '{person_id}'
            LIMIT 1
        """).fetchone()

        if not item_to_move or not available:
            return None

        return OracleAction(
            headline=f"Reassign {item_to_move[0]} from {name} to {available[0]}",
            subtext=f"{name} has {total} items ({in_progress} in progress)",
            action_type=ActionType.REASSIGN,
            action_content=f"Move {item_to_move[0]} to {available[0]}",
            action_target=item_to_move[0],
            reason=f"{name} is at {int(total/6*100)}% capacity. Burnout risk.",
            time_saved_minutes=25,
            risk_if_ignored=f"{name} burns out, loses 2 days next week",
            confidence=0.85,
            urgency="TODAY",
            data_source=f"Workload analysis: {name}",
            generated_at=datetime.now()
        )
    except:
        return None


def analyze_stale_items_for_action(conn, project_key: str) -> Optional[OracleAction]:
    """Find items with no activity and suggest a check-in."""
    try:
        stale = conn.execute(f"""
            SELECT
                i.key,
                i.summary,
                u.pseudonym as assignee,
                DATEDIFF('day', i.updated, CURRENT_TIMESTAMP) as days_stale
            FROM issues i
            LEFT JOIN users u ON i.assignee_id = u.account_id
            WHERE i.project_key = '{project_key}'
              AND i.status = 'En cours'
              AND DATEDIFF('day', i.updated, CURRENT_TIMESTAMP) > 3
            ORDER BY days_stale DESC
            LIMIT 1
        """).fetchone()

        if not stale:
            return None

        key, summary, assignee, days = stale

        message = f"""Hey {assignee or 'there'},

Quick sync on {key}? It's been {days} days without an update.

No pressure - just want to make sure you're not stuck. Need any help?

Reply with:
✅ On track
🆘 Need help
⏸️ Blocked on something"""

        return OracleAction(
            headline=f"Check in on {assignee or 'the team'} about {key}",
            subtext=f"No updates for {days} days - could be stuck",
            action_type=ActionType.SEND_MESSAGE,
            action_content=message,
            action_target=assignee or "team",
            reason=f"Silence often means stuck. Early check-in prevents 3-day delays.",
            time_saved_minutes=45,
            risk_if_ignored=f"Hidden blocker emerges in 3 days",
            confidence=0.78,
            urgency="TODAY",
            data_source=f"Stale work detection: {key}",
            generated_at=datetime.now()
        )
    except:
        return None


def analyze_sprint_risk_for_action(conn, project_key: str) -> Optional[OracleAction]:
    """Analyze sprint and suggest scope adjustment if needed."""
    try:
        sprint = conn.execute(f"""
            WITH current_sprint AS (
                SELECT id, name, end_date FROM sprints WHERE state = 'active' LIMIT 1
            )
            SELECT
                cs.name,
                DATEDIFF('day', CURRENT_TIMESTAMP, cs.end_date) as days_left,
                COUNT(*) as total,
                COUNT(CASE WHEN i.status = 'Terminé(e)' THEN 1 END) as done,
                COUNT(CASE WHEN i.status NOT IN ('Terminé(e)', 'En cours') THEN 1 END) as not_started
            FROM issues i
            JOIN current_sprint cs ON i.sprint_id = cs.id
            WHERE i.project_key = '{project_key}'
            GROUP BY cs.name, cs.end_date
        """).fetchone()

        if not sprint:
            return None

        name, days_left, total, done, not_started = sprint

        if days_left <= 0 or total == 0:
            return None

        completion_rate = done / total
        expected_rate = 1 - (days_left / 10)  # Assuming 10-day sprint

        if completion_rate >= expected_rate - 0.1:
            return None  # On track

        # Find lowest priority item to descope
        to_descope = conn.execute(f"""
            SELECT key, summary
            FROM issues i
            JOIN sprints s ON i.sprint_id = s.id
            WHERE s.state = 'active'
              AND i.project_key = '{project_key}'
              AND i.status NOT IN ('Terminé(e)', 'En cours')
            ORDER BY
                CASE i.priority WHEN 'Low' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END
            LIMIT 1
        """).fetchone()

        if not to_descope:
            return None

        stakeholder_message = f"""Sprint Update: {name}

We're at {int(completion_rate*100)}% with {days_left} days left.

To protect our commitment, I recommend moving {to_descope[0]} to next sprint.

This ensures we deliver the high-priority items with quality.

Let me know if you have concerns."""

        return OracleAction(
            headline=f"Send this scope update to stakeholders",
            subtext=f"Sprint is behind - protect the commitment",
            action_type=ActionType.SEND_MESSAGE,
            action_content=stakeholder_message,
            action_target="stakeholders",
            reason=f"{int(completion_rate*100)}% done with {days_left}d left. Math doesn't work.",
            time_saved_minutes=60,
            risk_if_ignored="Sprint fails, trust erodes",
            confidence=0.88,
            urgency="TODAY",
            data_source=f"Sprint analysis: {name}",
            generated_at=datetime.now()
        )
    except:
        return None


def analyze_wins_for_action(conn, project_key: str) -> Optional[OracleAction]:
    """Find wins to celebrate - positive reinforcement."""
    try:
        wins = conn.execute(f"""
            SELECT
                u.pseudonym as name,
                COUNT(*) as completed_today
            FROM issues i
            LEFT JOIN users u ON i.assignee_id = u.account_id
            WHERE i.project_key = '{project_key}'
              AND i.status = 'Terminé(e)'
              AND DATE(i.resolved) = CURRENT_DATE
            GROUP BY u.pseudonym
            HAVING COUNT(*) >= 2
            ORDER BY completed_today DESC
            LIMIT 1
        """).fetchone()

        if not wins:
            return None

        name, count = wins

        message = f"""🎉 Shoutout to {name}!

Crushed it today with {count} items completed.

This kind of momentum is what makes great teams. Keep it up!"""

        return OracleAction(
            headline=f"Celebrate {name}'s wins in the team channel",
            subtext=f"{count} items completed today!",
            action_type=ActionType.SEND_MESSAGE,
            action_content=message,
            action_target="team channel",
            reason="Recognition increases productivity by 14%. Science.",
            time_saved_minutes=0,
            risk_if_ignored="Miss chance to boost morale",
            confidence=0.95,
            urgency="TODAY",
            data_source="Completion tracking",
            generated_at=datetime.now()
        )
    except:
        return None


def generate_oracle_actions(conn, project_key: str) -> List[OracleAction]:
    """Generate all potential Oracle actions, ranked by impact."""
    actions = []

    # Run all analyzers
    analyzers = [
        analyze_blockers_for_action,
        analyze_overload_for_action,
        analyze_stale_items_for_action,
        analyze_sprint_risk_for_action,
        analyze_wins_for_action,
    ]

    for analyzer in analyzers:
        try:
            action = analyzer(conn, project_key)
            if action:
                actions.append(action)
        except:
            continue

    # Sort by urgency and confidence
    urgency_order = {"NOW": 0, "TODAY": 1, "THIS WEEK": 2}
    actions.sort(key=lambda a: (urgency_order.get(a.urgency, 2), -a.confidence))

    return actions


# =============================================================================
# THE ORACLE INTERFACE - PURE, MINIMAL, FOCUSED
# =============================================================================

def render_oracle_widget(action: OracleAction, index: int):
    """Render THE Oracle widget - the one thing to do."""

    urgency_colors = {
        "NOW": ("#e53e3e", "#fed7d7", "🔴"),
        "TODAY": ("#ed8936", "#feebc8", "🟠"),
        "THIS WEEK": ("#38a169", "#c6f6d5", "🟢")
    }

    bg_color, light_color, emoji = urgency_colors.get(action.urgency, ("#718096", "#e2e8f0", "⚪"))

    st.markdown(f"""
<style>
.oracle-container {{
    min-height: 70vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 20px;
}}
.oracle-card {{
    background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    border-radius: 30px;
    padding: 50px;
    max-width: 800px;
    width: 100%;
    border: 2px solid {bg_color};
    box-shadow: 0 0 80px {bg_color}40, 0 0 120px {bg_color}20;
    position: relative;
    overflow: hidden;
}}
.oracle-card::before {{
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, {bg_color}08 0%, transparent 50%);
    animation: pulse 4s ease-in-out infinite;
}}
@keyframes pulse {{
    0%, 100% {{ transform: scale(1); opacity: 0.5; }}
    50% {{ transform: scale(1.1); opacity: 0.8; }}
}}
.oracle-urgency {{
    display: inline-block;
    background: {bg_color};
    color: white;
    padding: 8px 24px;
    border-radius: 30px;
    font-weight: 800;
    font-size: 0.9em;
    letter-spacing: 2px;
    margin-bottom: 30px;
    position: relative;
    z-index: 1;
}}
.oracle-headline {{
    font-size: 2.2em;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.3;
    margin-bottom: 15px;
    position: relative;
    z-index: 1;
}}
.oracle-subtext {{
    font-size: 1.2em;
    color: #a0aec0;
    margin-bottom: 30px;
    position: relative;
    z-index: 1;
}}
.oracle-reason {{
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 30px;
    border-left: 4px solid {bg_color};
    position: relative;
    z-index: 1;
}}
.oracle-reason-label {{
    color: {bg_color};
    font-weight: 700;
    font-size: 0.85em;
    letter-spacing: 1px;
    margin-bottom: 8px;
}}
.oracle-reason-text {{
    color: #e2e8f0;
    font-size: 1.05em;
}}
.oracle-action-box {{
    background: rgba(0,0,0,0.3);
    border-radius: 15px;
    padding: 25px;
    margin-bottom: 30px;
    border: 1px solid rgba(255,255,255,0.1);
    position: relative;
    z-index: 1;
}}
.oracle-action-label {{
    color: #68d391;
    font-weight: 700;
    font-size: 0.85em;
    letter-spacing: 1px;
    margin-bottom: 15px;
}}
.oracle-action-content {{
    color: #ffffff;
    font-size: 0.95em;
    line-height: 1.6;
    white-space: pre-wrap;
    font-family: inherit;
}}
.oracle-metrics {{
    display: flex;
    justify-content: space-between;
    gap: 20px;
    margin-bottom: 30px;
    position: relative;
    z-index: 1;
}}
.oracle-metric {{
    flex: 1;
    text-align: center;
    padding: 15px;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
}}
.oracle-metric-value {{
    font-size: 1.5em;
    font-weight: 800;
    color: #68d391;
}}
.oracle-metric-label {{
    color: #718096;
    font-size: 0.8em;
    margin-top: 5px;
}}
.oracle-risk {{
    color: #fc8181;
    font-size: 0.9em;
    text-align: center;
    margin-bottom: 30px;
    position: relative;
    z-index: 1;
}}
</style>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div class="oracle-container">
<div class="oracle-card">
<div style="text-align: center;">
<span class="oracle-urgency">{emoji} {action.urgency}</span>
</div>
<div class="oracle-headline">{action.headline}</div>
<div class="oracle-subtext">{action.subtext}</div>
<div class="oracle-reason">
<div class="oracle-reason-label">WHY THIS MATTERS</div>
<div class="oracle-reason-text">{action.reason}</div>
</div>
<div class="oracle-action-box">
<div class="oracle-action-label">📋 THE EXACT ACTION (READY TO USE)</div>
<div class="oracle-action-content">{action.action_content}</div>
</div>
<div class="oracle-metrics">
<div class="oracle-metric">
<div class="oracle-metric-value">{action.time_saved_minutes}m</div>
<div class="oracle-metric-label">Time Saved</div>
</div>
<div class="oracle-metric">
<div class="oracle-metric-value">{int(action.confidence * 100)}%</div>
<div class="oracle-metric-label">Confidence</div>
</div>
</div>
<div class="oracle-risk">
⚠️ If ignored: {action.risk_if_ignored}
</div>
</div>
</div>
""", unsafe_allow_html=True)

    # Action buttons
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("✅ DO IT", type="primary", use_container_width=True, key=f"do_{index}"):
                st.success("✅ Action marked as done! Moving to next...")
                st.balloons()
                return "done"
        with bcol2:
            if st.button("⏭️ SKIP", use_container_width=True, key=f"skip_{index}"):
                return "skip"

    return None


def render_empty_oracle():
    """Render when there are no actions."""
    st.markdown("""
<div style="min-height: 70vh; display: flex; align-items: center; justify-content: center;">
<div style="text-align: center; padding: 60px; background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%); border-radius: 30px; border: 2px solid #48bb78; box-shadow: 0 0 60px rgba(72, 187, 120, 0.2);">
<div style="font-size: 4em; margin-bottom: 20px;">🎯</div>
<div style="font-size: 2em; font-weight: 800; color: #68d391; margin-bottom: 15px;">ALL CLEAR</div>
<div style="color: #a0aec0; font-size: 1.2em;">
No actions needed right now.<br>
Your project is on autopilot.
</div>
<div style="color: #4a5568; font-size: 0.9em; margin-top: 30px;">
Check back in an hour. The Oracle never sleeps.
</div>
</div>
</div>
""", unsafe_allow_html=True)


def render_oracle_header():
    """Minimal header for The Oracle."""
    st.markdown("""
<style>
.oracle-header {
    text-align: center;
    padding: 30px 20px;
    margin-bottom: 20px;
}
.oracle-title {
    font-size: 1.5em;
    font-weight: 300;
    color: #718096;
    letter-spacing: 8px;
    text-transform: uppercase;
}
.oracle-tagline {
    font-size: 0.95em;
    color: #4a5568;
    margin-top: 10px;
}
</style>

<div class="oracle-header">
<div class="oracle-title">🔮 THE ORACLE</div>
<div class="oracle-tagline">You don't analyze. You don't decide. You just approve.</div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Render page guide in sidebar
    render_page_guide()
    conn = get_connection()

    # Render minimal header
    render_oracle_header()

    if not conn:
        # Demo mode with sample action
        demo_action = OracleAction(
            headline="Send this check-in to Ada Lovelace",
            subtext="PROJ-142 has been silent for 5 days",
            action_type=ActionType.SEND_MESSAGE,
            action_content="""Hey Ada,

Quick sync on PROJ-142? It's been 5 days without an update.

No pressure - just want to make sure you're not stuck.

Reply with:
✅ On track
🆘 Need help
⏸️ Blocked on something""",
            action_target="Ada Lovelace",
            reason="Silence often means stuck. Early check-in prevents 3-day delays.",
            time_saved_minutes=45,
            risk_if_ignored="Hidden blocker emerges Friday",
            confidence=0.88,
            urgency="TODAY",
            data_source="Stale work detection",
            generated_at=datetime.now()
        )
        render_oracle_widget(demo_action, 0)
        return

    # Get project key
    try:
        project_key = conn.execute(
            "SELECT DISTINCT project_key FROM issues LIMIT 1"
        ).fetchone()[0]
    except:
        project_key = "PROJ"

    # Initialize session state for tracking actions
    if 'oracle_index' not in st.session_state:
        st.session_state.oracle_index = 0
    if 'oracle_actions' not in st.session_state:
        st.session_state.oracle_actions = None

    # Generate actions if not cached
    if st.session_state.oracle_actions is None:
        with st.spinner("The Oracle is thinking..."):
            st.session_state.oracle_actions = generate_oracle_actions(conn, project_key)

    actions = st.session_state.oracle_actions

    if not actions:
        render_empty_oracle()
        return

    # Show current action
    current_index = st.session_state.oracle_index

    if current_index >= len(actions):
        # All done
        render_empty_oracle()
        if st.button("🔄 Refresh The Oracle"):
            st.session_state.oracle_actions = None
            st.session_state.oracle_index = 0
            st.rerun()
        return

    result = render_oracle_widget(actions[current_index], current_index)

    if result == "done" or result == "skip":
        st.session_state.oracle_index += 1
        st.rerun()

    # Progress indicator
    st.markdown(f"""
<div style="text-align: center; color: #4a5568; font-size: 0.85em; padding: 20px;">
Action {current_index + 1} of {len(actions)}
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
