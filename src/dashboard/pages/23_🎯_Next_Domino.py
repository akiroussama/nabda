"""
🎯 NEXT DOMINO - The Single Highest-Impact Intervention

THE ONE WIDGET THAT MATTERS:
"What is the one thing I should fix RIGHT NOW to prevent the biggest schedule slip?"

Most dashboards show status. This shows LEVERAGE.
It continuously computes the project's single highest-impact intervention
so the PM stops scanning boards and starts making one decisive move.

Scoring: Expected Delay Prevented = (Slip if unaddressed) × (Probability) × (Ease-of-fix bonus)
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import duckdb
import hashlib

# Page configuration
st.set_page_config(
    page_title="Next Domino",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# LLM CLIENT
# =============================================================================

def get_llm_client():
    """Get or create LLM client."""
    try:
        import sys
        sys.path.insert(0, 'src')
        from intelligence.llm_client import GeminiClient

        api_key = os.environ.get('GOOGLE_API_KEY', '')
        if api_key:
            return GeminiClient(
                api_key=api_key,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 2048,
                }
            )
        return None
    except Exception:
        return None


# =============================================================================
# DATA MODELS
# =============================================================================

class DominoType(Enum):
    """Types of dominoes that can cause schedule slips."""
    BLOCKER = "blocker"
    DEPENDENCY_CHAIN = "dependency_chain"
    RESOURCE_BOTTLENECK = "resource_bottleneck"
    STALE_WORK = "stale_work"
    MISSING_REVIEWER = "missing_reviewer"
    OVERLOADED_OWNER = "overloaded_owner"
    UNASSIGNED_CRITICAL = "unassigned_critical"
    DEADLINE_RISK = "deadline_risk"


class RescueType(Enum):
    """Types of rescue actions."""
    REASSIGN = "reassign"
    SPLIT_TASK = "split_task"
    DESCOPE = "descope"
    PING_OWNER = "ping_owner"
    ADD_REVIEWER = "add_reviewer"
    ESCALATE = "escalate"
    UNBLOCK = "unblock"


@dataclass
class RescuePath:
    """A possible rescue action for the domino."""
    type: RescueType
    title: str
    description: str
    time_to_fix: str  # e.g., "2 mins", "8 mins"
    time_to_fix_mins: int
    days_saved: float
    action_content: str  # Pre-filled message or action
    target: str  # Who to target
    confidence: float = 0.8


@dataclass
class DominoCandidate:
    """A potential domino item in the project."""
    id: str
    key: str
    summary: str
    type: DominoType
    assignee: Optional[str]
    assignee_id: Optional[str]

    # Impact metrics
    slip_days: float  # Days of slip if unaddressed
    probability: float  # Probability of slip (0-1)
    ease_of_fix: float  # How easy to fix (0-1, higher = easier)

    # Scoring
    impact_score: float = 0.0  # Computed: slip × prob × ease bonus

    # Context
    why_domino: str = ""  # One-line proof
    blocks_count: int = 0  # How many tasks it blocks
    blocked_teams: List[str] = field(default_factory=list)
    is_critical_path: bool = False
    days_stale: int = 0

    # Rescue paths
    rescue_paths: List[RescuePath] = field(default_factory=list)

    def compute_score(self):
        """Compute the impact score."""
        # Ease-of-fix bonus: easier fixes get priority (1.0 to 2.0 multiplier)
        ease_bonus = 1.0 + self.ease_of_fix
        self.impact_score = self.slip_days * self.probability * ease_bonus
        return self.impact_score


@dataclass
class DominoAction:
    """A one-click action for the domino."""
    id: str
    type: str  # "approve_fix", "ask_owner", "escalate"
    label: str
    description: str
    action_content: str
    target: str
    executed: bool = False
    executed_at: Optional[datetime] = None


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

@st.cache_resource
def get_connection():
    """Get DuckDB connection."""
    try:
        return duckdb.connect("jira_data.duckdb", read_only=True)
    except Exception:
        return None


# =============================================================================
# DOMINO DETECTION ENGINE
# =============================================================================

def detect_blocked_dominoes(conn, project_key: str) -> List[DominoCandidate]:
    """Detect blocked items that could cause slips."""
    dominoes = []

    try:
        blocked = conn.execute(f"""
            SELECT
                i.id, i.key, i.summary,
                u.pseudonym as assignee, i.assignee_id,
                i.story_points,
                DATEDIFF('day', i.updated, CURRENT_TIMESTAMP) as days_blocked,
                i.priority
            FROM issues i
            LEFT JOIN users u ON i.assignee_id = u.account_id
            WHERE i.project_key = '{project_key}'
              AND i.status IN ('Blocked', 'On Hold', 'Waiting')
            ORDER BY days_blocked DESC
            LIMIT 10
        """).fetchall()

        for row in blocked:
            issue_id, key, summary, assignee, assignee_id, points, days_blocked, priority = row

            # Calculate slip impact
            slip_days = min(days_blocked * 1.5, 10)  # Compounding effect
            probability = min(0.6 + (days_blocked * 0.05), 0.95)
            ease_of_fix = 0.7 if days_blocked < 3 else 0.4

            domino = DominoCandidate(
                id=str(issue_id),
                key=key,
                summary=summary or "Unknown task",
                type=DominoType.BLOCKER,
                assignee=assignee,
                assignee_id=assignee_id,
                slip_days=slip_days,
                probability=probability,
                ease_of_fix=ease_of_fix,
                why_domino=f"Blocked for {days_blocked} days; each day adds 1.5x delay compound",
                days_stale=days_blocked,
                is_critical_path=priority in ('Highest', 'High')
            )
            domino.compute_score()
            dominoes.append(domino)

    except Exception:
        pass

    return dominoes


def detect_dependency_dominoes(conn, project_key: str) -> List[DominoCandidate]:
    """Detect items blocking multiple other tasks."""
    dominoes = []

    try:
        # Find items with many dependents
        blockers = conn.execute(f"""
            WITH blocking_items AS (
                SELECT
                    i.id, i.key, i.summary,
                    u.pseudonym as assignee, i.assignee_id,
                    i.status, i.priority,
                    COUNT(DISTINCT l.target_issue_id) as blocks_count
                FROM issues i
                LEFT JOIN users u ON i.assignee_id = u.account_id
                LEFT JOIN issue_links l ON i.id = l.source_issue_id
                    AND l.link_type IN ('blocks', 'is blocked by')
                WHERE i.project_key = '{project_key}'
                  AND i.status NOT IN ('Terminé(e)', 'Done', 'Closed')
                GROUP BY i.id, i.key, i.summary, u.pseudonym, i.assignee_id, i.status, i.priority
                HAVING COUNT(DISTINCT l.target_issue_id) >= 2
            )
            SELECT * FROM blocking_items
            ORDER BY blocks_count DESC
            LIMIT 5
        """).fetchall()

        for row in blockers:
            issue_id, key, summary, assignee, assignee_id, status, priority, blocks_count = row

            slip_days = blocks_count * 1.5  # Each blocked task adds delay
            probability = 0.7 if status == 'En cours' else 0.85
            ease_of_fix = 0.5

            domino = DominoCandidate(
                id=str(issue_id),
                key=key,
                summary=summary or "Unknown task",
                type=DominoType.DEPENDENCY_CHAIN,
                assignee=assignee,
                assignee_id=assignee_id,
                slip_days=slip_days,
                probability=probability,
                ease_of_fix=ease_of_fix,
                why_domino=f"Blocks {blocks_count} downstream tasks; critical dependency chain",
                blocks_count=blocks_count,
                is_critical_path=True
            )
            domino.compute_score()
            dominoes.append(domino)

    except Exception:
        pass

    return dominoes


def detect_stale_dominoes(conn, project_key: str) -> List[DominoCandidate]:
    """Detect stale in-progress items that may be secretly stuck."""
    dominoes = []

    try:
        stale = conn.execute(f"""
            SELECT
                i.id, i.key, i.summary,
                u.pseudonym as assignee, i.assignee_id,
                i.story_points, i.priority,
                DATEDIFF('day', i.updated, CURRENT_TIMESTAMP) as days_stale
            FROM issues i
            LEFT JOIN users u ON i.assignee_id = u.account_id
            WHERE i.project_key = '{project_key}'
              AND i.status = 'En cours'
              AND DATEDIFF('day', i.updated, CURRENT_TIMESTAMP) >= 4
            ORDER BY days_stale DESC
            LIMIT 8
        """).fetchall()

        for row in stale:
            issue_id, key, summary, assignee, assignee_id, points, priority, days_stale = row

            slip_days = days_stale * 0.8
            probability = min(0.5 + (days_stale * 0.08), 0.9)
            ease_of_fix = 0.9  # Usually just needs a ping

            domino = DominoCandidate(
                id=str(issue_id),
                key=key,
                summary=summary or "Unknown task",
                type=DominoType.STALE_WORK,
                assignee=assignee,
                assignee_id=assignee_id,
                slip_days=slip_days,
                probability=probability,
                ease_of_fix=ease_of_fix,
                why_domino=f"No updates for {days_stale} days; likely secretly blocked",
                days_stale=days_stale
            )
            domino.compute_score()
            dominoes.append(domino)

    except Exception:
        pass

    return dominoes


def detect_overload_dominoes(conn, project_key: str) -> List[DominoCandidate]:
    """Detect overloaded team members creating bottlenecks."""
    dominoes = []

    try:
        overloaded = conn.execute(f"""
            SELECT
                u.pseudonym as assignee, i.assignee_id,
                COUNT(*) as task_count,
                COUNT(CASE WHEN i.status = 'En cours' THEN 1 END) as in_progress,
                MAX(i.priority) as highest_priority,
                MIN(i.id) as sample_id,
                MIN(i.key) as sample_key,
                MIN(i.summary) as sample_summary
            FROM issues i
            JOIN users u ON i.assignee_id = u.account_id
            WHERE i.project_key = '{project_key}'
              AND i.status NOT IN ('Terminé(e)', 'Done', 'Closed')
            GROUP BY u.pseudonym, i.assignee_id
            HAVING COUNT(*) > 7 OR COUNT(CASE WHEN i.status = 'En cours' THEN 1 END) > 4
            ORDER BY task_count DESC
            LIMIT 5
        """).fetchall()

        for row in overloaded:
            assignee, assignee_id, task_count, in_progress, priority, sample_id, sample_key, sample_summary = row

            slip_days = (task_count - 5) * 0.5 + (in_progress - 2) * 1.0
            probability = 0.75
            ease_of_fix = 0.6

            domino = DominoCandidate(
                id=str(sample_id),
                key=sample_key,
                summary=f"{assignee}: {task_count} tasks ({in_progress} in progress)",
                type=DominoType.OVERLOADED_OWNER,
                assignee=assignee,
                assignee_id=assignee_id,
                slip_days=slip_days,
                probability=probability,
                ease_of_fix=ease_of_fix,
                why_domino=f"{assignee} has {task_count} tasks, {in_progress} in parallel; 30% slower delivery rate"
            )
            domino.compute_score()
            dominoes.append(domino)

    except Exception:
        pass

    return dominoes


def detect_unassigned_dominoes(conn, project_key: str) -> List[DominoCandidate]:
    """Detect critical unassigned items."""
    dominoes = []

    try:
        unassigned = conn.execute(f"""
            SELECT
                i.id, i.key, i.summary, i.priority, i.story_points,
                s.end_date as sprint_end,
                DATEDIFF('day', CURRENT_TIMESTAMP, s.end_date) as days_to_sprint_end
            FROM issues i
            LEFT JOIN sprints s ON i.sprint_id = s.id
            WHERE i.project_key = '{project_key}'
              AND i.assignee_id IS NULL
              AND i.status NOT IN ('Terminé(e)', 'Done', 'Closed', 'Backlog')
              AND i.priority IN ('Highest', 'High', 'Medium')
            ORDER BY
                CASE i.priority
                    WHEN 'Highest' THEN 1
                    WHEN 'High' THEN 2
                    ELSE 3
                END,
                days_to_sprint_end
            LIMIT 5
        """).fetchall()

        for row in unassigned:
            issue_id, key, summary, priority, points, sprint_end, days_left = row

            urgency = 3 if priority == 'Highest' else 2 if priority == 'High' else 1
            slip_days = urgency * 2
            probability = 0.8
            ease_of_fix = 0.95  # Just needs assignment

            domino = DominoCandidate(
                id=str(issue_id),
                key=key,
                summary=summary or "Unknown task",
                type=DominoType.UNASSIGNED_CRITICAL,
                assignee=None,
                assignee_id=None,
                slip_days=slip_days,
                probability=probability,
                ease_of_fix=ease_of_fix,
                why_domino=f"{priority} priority task with no owner; guaranteed slip without action",
                is_critical_path=priority == 'Highest'
            )
            domino.compute_score()
            dominoes.append(domino)

    except Exception:
        pass

    return dominoes


def find_the_domino(conn, project_key: str) -> Optional[DominoCandidate]:
    """Find THE single highest-impact domino."""

    all_dominoes = []

    # Run all detectors
    all_dominoes.extend(detect_blocked_dominoes(conn, project_key))
    all_dominoes.extend(detect_dependency_dominoes(conn, project_key))
    all_dominoes.extend(detect_stale_dominoes(conn, project_key))
    all_dominoes.extend(detect_overload_dominoes(conn, project_key))
    all_dominoes.extend(detect_unassigned_dominoes(conn, project_key))

    if not all_dominoes:
        return None

    # Sort by impact score (highest first)
    all_dominoes.sort(key=lambda d: d.impact_score, reverse=True)

    return all_dominoes[0]


# =============================================================================
# RESCUE PATH GENERATOR
# =============================================================================

def generate_rescue_paths(domino: DominoCandidate, conn, project_key: str, llm_client) -> List[RescuePath]:
    """Generate ranked rescue paths for the domino."""

    paths = []

    # Path 1: Based on domino type, generate primary rescue
    if domino.type == DominoType.BLOCKER:
        paths.append(RescuePath(
            type=RescueType.PING_OWNER,
            title="Ping for blocker status",
            description="Send status check to get unblock timeline",
            time_to_fix="2 mins",
            time_to_fix_mins=2,
            days_saved=domino.slip_days * 0.3,
            action_content=f"""Hi {domino.assignee or 'team'},

Quick check on {domino.key}: {domino.summary}

It's been blocked for {domino.days_stale} days. Can you share:
1. What's currently blocking this?
2. Who can help unblock?
3. Expected resolution time?

This is on the critical path - every day of delay compounds.

Thanks!""",
            target=domino.assignee or "owner"
        ))

        paths.append(RescuePath(
            type=RescueType.ESCALATE,
            title="Escalate to remove blocker",
            description="Schedule 12-min micro-sync with decision makers",
            time_to_fix="15 mins",
            time_to_fix_mins=15,
            days_saved=domino.slip_days * 0.7,
            action_content=f"""🚨 Blocker Escalation: {domino.key}

SITUATION: {domino.summary}
BLOCKED FOR: {domino.days_stale} days
IMPACT: +{domino.slip_days:.1f} days slip to schedule

DECISION NEEDED: How do we unblock this today?

Required attendees: {domino.assignee or 'Owner'}, Tech Lead, PM
Duration: 12 minutes (hard stop)
Agenda:
1. What's blocking (2 min)
2. Options to unblock (5 min)
3. Decision & owner (5 min)""",
            target="leadership"
        ))

    elif domino.type == DominoType.DEPENDENCY_CHAIN:
        paths.append(RescuePath(
            type=RescueType.SPLIT_TASK,
            title="Split into unblocked chunks",
            description="Break task so downstream work can start",
            time_to_fix="8 mins",
            time_to_fix_mins=8,
            days_saved=domino.slip_days * 0.5,
            action_content=f"""Task Split Proposal: {domino.key}

CURRENT: {domino.summary}
BLOCKS: {domino.blocks_count} downstream tasks

SPLIT PROPOSAL:
1. {domino.key}-A: [Core functionality] - Can ship independently
2. {domino.key}-B: [Integration work] - Depends on external input

This unblocks {domino.blocks_count} tasks immediately.
Approve to create the split.""",
            target=domino.assignee or "tech lead"
        ))

        paths.append(RescuePath(
            type=RescueType.REASSIGN,
            title="Add parallel resource",
            description="Assign additional person to accelerate",
            time_to_fix="3 mins",
            time_to_fix_mins=3,
            days_saved=domino.slip_days * 0.4,
            action_content=f"""Resource Request: {domino.key}

This task blocks {domino.blocks_count} others.
Recommend adding a second person to parallelize work.

Current: {domino.assignee or 'Unassigned'}
Suggested pairing: [Select from available team]

Expected acceleration: 40% faster completion.""",
            target="resource manager"
        ))

    elif domino.type == DominoType.STALE_WORK:
        paths.append(RescuePath(
            type=RescueType.PING_OWNER,
            title="Status check",
            description="Quick ping to surface hidden blockers",
            time_to_fix="2 mins",
            time_to_fix_mins=2,
            days_saved=domino.slip_days * 0.4,
            action_content=f"""Hi {domino.assignee or 'team'},

Checking in on {domino.key}: {domino.summary}

No updates in {domino.days_stale} days. Quick questions:
- Is this still actively being worked on?
- Any blockers we should know about?
- Need any help?

If you're stuck, let's chat - happy to help unblock.

Thanks!""",
            target=domino.assignee or "owner"
        ))

        paths.append(RescuePath(
            type=RescueType.REASSIGN,
            title="Reassign to available person",
            description="Move to someone with bandwidth",
            time_to_fix="3 mins",
            time_to_fix_mins=3,
            days_saved=domino.slip_days * 0.6,
            action_content=f"""Reassignment Proposal: {domino.key}

CURRENT OWNER: {domino.assignee or 'Unassigned'}
STALE FOR: {domino.days_stale} days

This task appears stuck. Recommend reassigning to:
[Available team member with capacity]

Approve to reassign.""",
            target="team lead"
        ))

    elif domino.type == DominoType.OVERLOADED_OWNER:
        paths.append(RescuePath(
            type=RescueType.REASSIGN,
            title="Redistribute workload",
            description="Move tasks to team members with capacity",
            time_to_fix="5 mins",
            time_to_fix_mins=5,
            days_saved=domino.slip_days * 0.5,
            action_content=f"""Workload Rebalance: {domino.assignee}

SITUATION: {domino.assignee} is overloaded
CURRENT LOAD: {domino.summary}

RECOMMENDATION:
Move 2-3 lower priority tasks to team members with capacity.
This will reduce context-switching and improve throughput by ~30%.

Tasks to redistribute:
1. [Lower priority task 1]
2. [Lower priority task 2]

Approve to rebalance.""",
            target="team lead"
        ))

        paths.append(RescuePath(
            type=RescueType.DESCOPE,
            title="Descope non-essential work",
            description="Move nice-to-haves to next sprint",
            time_to_fix="5 mins",
            time_to_fix_mins=5,
            days_saved=domino.slip_days * 0.4,
            action_content=f"""Scope Adjustment Proposal

{domino.assignee} is overloaded. Recommend moving these to next sprint:
- [Nice-to-have feature 1]
- [Nice-to-have feature 2]

This preserves bandwidth for critical path items.
Impact: {domino.slip_days * 0.4:.1f} days saved on schedule.

Approve to move to backlog.""",
            target="product owner"
        ))

    elif domino.type == DominoType.UNASSIGNED_CRITICAL:
        paths.append(RescuePath(
            type=RescueType.REASSIGN,
            title="Assign to available person",
            description="Assign to team member with bandwidth",
            time_to_fix="2 mins",
            time_to_fix_mins=2,
            days_saved=domino.slip_days * 0.8,
            action_content=f"""Assignment Needed: {domino.key}

TASK: {domino.summary}
PRIORITY: High/Critical
STATUS: Unassigned

This critical task needs an owner immediately.
Recommend: [Team member with relevant skills and capacity]

Approve to assign.""",
            target="team lead"
        ))

    # Add escalation as a universal fallback
    if not any(p.type == RescueType.ESCALATE for p in paths):
        paths.append(RescuePath(
            type=RescueType.ESCALATE,
            title="Escalate for decision",
            description="Schedule 12-min sync with stakeholders",
            time_to_fix="15 mins",
            time_to_fix_mins=15,
            days_saved=domino.slip_days * 0.6,
            action_content=f"""🚨 Escalation: {domino.key}

ISSUE: {domino.summary}
WHY: {domino.why_domino}
IMPACT: +{domino.slip_days:.1f} days slip risk

Need decision on best path forward.
12-min sync with required stakeholders.""",
            target="leadership"
        ))

    # Sort by days_saved / time_to_fix ratio (bang for buck)
    paths.sort(key=lambda p: p.days_saved / max(p.time_to_fix_mins, 1), reverse=True)

    return paths[:3]  # Return top 3


def generate_ask_owner_message(domino: DominoCandidate) -> str:
    """Generate pre-filled message for Ask Owner action."""
    return f"""Hi {domino.assignee or 'there'},

I'm tracking {domino.key} and noticed it might need attention.

**Current status:** {domino.type.value.replace('_', ' ').title()}
**Why it matters:** {domino.why_domino}
**Impact if delayed:** +{domino.slip_days:.1f} days to schedule

I have two quick questions:
1. What's the current blocker or challenge?
2. What would help you move this forward today?

Options to discuss:
- A) I can help remove a blocker
- B) We can reassign or pair on this
- C) We can adjust scope if needed

Let me know - happy to jump on a quick call.

Thanks!"""


def generate_escalation_invite(domino: DominoCandidate) -> str:
    """Generate escalation meeting invite."""
    return f"""📅 MICRO-SYNC: Unblock {domino.key}

**Duration:** 12 minutes (hard stop)
**Required:** {domino.assignee or 'Task Owner'}, Tech Lead, PM

**Agenda:**
1. Current state (2 min)
2. Blocker analysis (3 min)
3. Options (4 min)
4. Decision & owner (3 min)

**Context:**
- Task: {domino.summary}
- Impact: +{domino.slip_days:.1f} days slip
- Why urgent: {domino.why_domino}

**Goal:** Leave with a clear decision and owner for next action."""


# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def get_domino_state() -> Dict:
    """Get or initialize domino widget state."""
    if 'domino_state' not in st.session_state:
        st.session_state.domino_state = {
            'current_domino': None,
            'rescue_paths': [],
            'last_computed': None,
            'actions_taken': 0,
            'days_saved': 0.0,
            'audit_trail': [],
            'auto_refresh': False
        }
    return st.session_state.domino_state


def save_domino_state(state: Dict):
    """Save domino state."""
    st.session_state.domino_state = state


def add_audit_entry(state: Dict, action: str, details: str, impact: str = None):
    """Add to audit trail."""
    entry = {
        'timestamp': datetime.now().isoformat(),
        'action': action,
        'details': details,
        'impact': impact
    }
    state['audit_trail'].insert(0, entry)
    state['audit_trail'] = state['audit_trail'][:20]


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_domino_widget(domino: DominoCandidate, rescue_paths: List[RescuePath], state: Dict):
    """Render the main domino widget."""

    # Severity color based on slip days
    if domino.slip_days >= 5:
        color = "#e53e3e"
        glow = "0 0 80px #e53e3e40"
    elif domino.slip_days >= 3:
        color = "#ed8936"
        glow = "0 0 60px #ed893640"
    else:
        color = "#ecc94b"
        glow = "0 0 40px #ecc94b40"

    confidence = int(domino.probability * 100)

    st.markdown(f"""
    <style>
    @keyframes pulse {{
        0%, 100% {{ box-shadow: {glow}; }}
        50% {{ box-shadow: 0 0 100px {color}60; }}
    }}
    @keyframes slideIn {{
        from {{ transform: translateY(-20px); opacity: 0; }}
        to {{ transform: translateY(0); opacity: 1; }}
    }}
    .domino-main {{
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        border-radius: 24px;
        padding: 40px;
        border: 3px solid {color};
        animation: pulse 3s ease-in-out infinite, slideIn 0.5s ease-out;
        position: relative;
        overflow: hidden;
    }}
    .domino-main::before {{
        content: '🎯';
        position: absolute;
        top: 20px;
        right: 30px;
        font-size: 4em;
        opacity: 0.15;
    }}
    .domino-label {{
        display: inline-block;
        background: {color}22;
        color: {color};
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85em;
        letter-spacing: 2px;
        margin-bottom: 15px;
    }}
    .domino-headline {{
        font-size: 2.4em;
        font-weight: 800;
        color: #fff;
        line-height: 1.2;
        margin-bottom: 20px;
    }}
    .domino-key {{
        color: {color};
        font-weight: 700;
    }}
    .impact-meter {{
        display: flex;
        align-items: center;
        gap: 20px;
        background: rgba(0,0,0,0.3);
        border-radius: 16px;
        padding: 20px;
        margin: 20px 0;
    }}
    .impact-number {{
        font-size: 3em;
        font-weight: 900;
        color: {color};
        line-height: 1;
    }}
    .impact-label {{
        color: #a0aec0;
        font-size: 0.9em;
    }}
    .impact-confidence {{
        background: rgba(255,255,255,0.1);
        padding: 5px 12px;
        border-radius: 20px;
        color: #718096;
        font-size: 0.85em;
    }}
    .why-proof {{
        background: rgba(255,255,255,0.05);
        border-left: 3px solid {color};
        padding: 15px 20px;
        margin: 20px 0;
        color: #e2e8f0;
        font-size: 1.05em;
        border-radius: 0 12px 12px 0;
    }}
    </style>

    <div class="domino-main">
        <div class="domino-label">NEXT DOMINO</div>

        <div class="domino-headline">
            <span class="domino-key">{domino.key}:</span> {domino.summary[:80]}{'...' if len(domino.summary) > 80 else ''}
        </div>

        <div class="impact-meter">
            <div>
                <div class="impact-number">+{domino.slip_days:.1f}d</div>
                <div class="impact-label">slip if nothing changes</div>
            </div>
            <div class="impact-confidence">
                {confidence}% confidence
            </div>
        </div>

        <div class="why-proof">
            💡 <strong>Why it's the domino:</strong> {domino.why_domino}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_rescue_paths(rescue_paths: List[RescuePath], domino: DominoCandidate, state: Dict):
    """Render the ranked rescue paths."""

    st.markdown("""
    <div style="margin-top: 30px; margin-bottom: 15px;">
        <span style="color: #68d391; font-weight: 700; font-size: 1.1em; letter-spacing: 1px;">
            ⚡ FASTEST RESCUE PATHS
        </span>
        <span style="color: #718096; font-size: 0.9em; margin-left: 10px;">
            (ranked by impact/effort)
        </span>
    </div>
    """, unsafe_allow_html=True)

    for i, path in enumerate(rescue_paths):
        rank = i + 1
        rank_colors = ["#48bb78", "#4299e1", "#9f7aea"]
        color = rank_colors[i] if i < 3 else "#718096"

        with st.container():
            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(26,26,46,0.95), rgba(22,33,62,0.95));
                border-radius: 16px;
                padding: 20px;
                margin-bottom: 12px;
                border: 1px solid {color}40;
                display: flex;
                gap: 20px;
                align-items: start;
            ">
                <div style="
                    background: {color};
                    color: #000;
                    width: 32px;
                    height: 32px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: 800;
                    flex-shrink: 0;
                ">{rank}</div>

                <div style="flex: 1;">
                    <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap; gap: 10px;">
                        <div>
                            <div style="color: #fff; font-weight: 700; font-size: 1.1em;">
                                {path.title}
                            </div>
                            <div style="color: #a0aec0; font-size: 0.9em; margin-top: 4px;">
                                {path.description}
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="color: {color}; font-weight: 700;">
                                ⏱️ {path.time_to_fix}
                            </div>
                            <div style="color: #68d391; font-size: 0.85em;">
                                saves {path.days_saved:.1f}d
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Expandable action content
            with st.expander(f"View action content", expanded=False):
                st.code(path.action_content, language=None)

                if st.button(f"✅ Execute This Fix", key=f"fix_{i}", type="primary"):
                    state['actions_taken'] = state.get('actions_taken', 0) + 1
                    state['days_saved'] = state.get('days_saved', 0) + path.days_saved
                    add_audit_entry(
                        state,
                        f"Executed: {path.title}",
                        f"For {domino.key}",
                        f"Saved {path.days_saved:.1f} days"
                    )
                    save_domino_state(state)
                    st.success(f"✅ Fix executed! Saved {path.days_saved:.1f} days.")
                    st.balloons()


def render_one_click_actions(domino: DominoCandidate, rescue_paths: List[RescuePath], state: Dict):
    """Render the one-click action buttons."""

    st.markdown("""
    <div style="margin-top: 30px; margin-bottom: 20px;">
        <span style="color: #f6ad55; font-weight: 700; font-size: 1.1em; letter-spacing: 1px;">
            🚀 ONE-CLICK ACTIONS
        </span>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(72, 187, 120, 0.2), rgba(72, 187, 120, 0.1));
            border: 2px solid #48bb78;
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            height: 160px;
        ">
            <div style="font-size: 2em; margin-bottom: 10px;">✅</div>
            <div style="color: #48bb78; font-weight: 700; font-size: 1.1em;">Approve Fix</div>
            <div style="color: #a0aec0; font-size: 0.85em; margin-top: 8px;">
                Apply the #1 rescue path
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Approve Top Fix", key="approve_fix", type="primary", use_container_width=True):
            if rescue_paths:
                path = rescue_paths[0]
                state['actions_taken'] = state.get('actions_taken', 0) + 1
                state['days_saved'] = state.get('days_saved', 0) + path.days_saved
                add_audit_entry(
                    state,
                    f"Approved: {path.title}",
                    f"For {domino.key}",
                    f"Saved {path.days_saved:.1f} days"
                )
                save_domino_state(state)
                st.success(f"✅ Approved! {path.title}")
                st.balloons()

    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(66, 153, 225, 0.2), rgba(66, 153, 225, 0.1));
            border: 2px solid #4299e1;
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            height: 160px;
        ">
            <div style="font-size: 2em; margin-bottom: 10px;">💬</div>
            <div style="color: #4299e1; font-weight: 700; font-size: 1.1em;">Ask Owner</div>
            <div style="color: #a0aec0; font-size: 0.85em; margin-top: 8px;">
                Pre-filled DM with context + options
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Open Message", key="ask_owner", use_container_width=True):
            message = generate_ask_owner_message(domino)
            state['actions_taken'] = state.get('actions_taken', 0) + 1
            add_audit_entry(state, "Asked Owner", f"Contacted {domino.assignee or 'owner'} about {domino.key}")
            save_domino_state(state)

            st.markdown("**📋 Copy this message:**")
            st.code(message, language=None)
            st.success("Message ready to send!")

    with col3:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, rgba(237, 137, 54, 0.2), rgba(237, 137, 54, 0.1));
            border: 2px solid #ed8936;
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            height: 160px;
        ">
            <div style="font-size: 2em; margin-bottom: 10px;">🚨</div>
            <div style="color: #ed8936; font-weight: 700; font-size: 1.1em;">Escalate</div>
            <div style="color: #a0aec0; font-size: 0.85em; margin-top: 8px;">
                12-min micro-sync with key people
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Schedule Sync", key="escalate", use_container_width=True):
            invite = generate_escalation_invite(domino)
            state['actions_taken'] = state.get('actions_taken', 0) + 1
            add_audit_entry(state, "Escalated", f"Scheduled sync for {domino.key}")
            save_domino_state(state)

            st.markdown("**📅 Meeting Invite:**")
            st.code(invite, language=None)
            st.success("Escalation scheduled!")


def render_no_domino():
    """Render when no domino is found."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.1), rgba(72, 187, 120, 0.05));
        border: 2px solid rgba(72, 187, 120, 0.3);
        border-radius: 24px;
        padding: 60px;
        text-align: center;
    ">
        <div style="font-size: 4em; margin-bottom: 20px;">✨</div>
        <div style="color: #68d391; font-size: 1.8em; font-weight: 800;">
            NO DOMINOES DETECTED
        </div>
        <div style="color: #a0aec0; font-size: 1.1em; margin-top: 15px;">
            Your project is running smoothly. No critical interventions needed right now.
        </div>
        <div style="color: #718096; font-size: 0.9em; margin-top: 20px;">
            Keep it up! The next scan will run automatically when data changes.
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_header(state: Dict):
    """Render the page header with stats."""

    st.markdown(f"""
    <style>
    .header-stats {{
        display: flex;
        gap: 30px;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }}
    .stat-card {{
        background: linear-gradient(135deg, rgba(26,26,46,0.8), rgba(22,33,62,0.8));
        border-radius: 16px;
        padding: 20px 30px;
        text-align: center;
        border: 1px solid rgba(255,255,255,0.1);
        min-width: 140px;
    }}
    .stat-number {{
        font-size: 2.2em;
        font-weight: 800;
        color: #68d391;
    }}
    .stat-label {{
        color: #718096;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 5px;
    }}
    </style>

    <div class="header-stats">
        <div class="stat-card">
            <div class="stat-number">{state.get('actions_taken', 0)}</div>
            <div class="stat-label">Actions Taken</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{state.get('days_saved', 0):.1f}d</div>
            <div class="stat-label">Days Saved</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">{len(state.get('audit_trail', []))}</div>
            <div class="stat-label">Interventions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_audit_trail(state: Dict):
    """Render the audit trail."""
    trail = state.get('audit_trail', [])

    if not trail:
        st.info("No actions taken yet. Your intervention history will appear here.")
        return

    for entry in trail[:10]:
        timestamp = entry.get('timestamp', '')
        try:
            timestamp = datetime.fromisoformat(timestamp).strftime('%H:%M')
        except:
            timestamp = timestamp[:5]

        impact = entry.get('impact', '')

        st.markdown(f"""
        <div style="
            display: flex;
            gap: 15px;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        ">
            <div style="color: #718096; font-size: 0.85em; min-width: 50px;">
                {timestamp}
            </div>
            <div style="flex: 1;">
                <div style="color: #e2e8f0; font-weight: 600;">{entry.get('action', '')}</div>
                <div style="color: #a0aec0; font-size: 0.9em;">{entry.get('details', '')}</div>
                {f'<div style="color: #68d391; font-size: 0.85em; margin-top: 3px;">✅ {impact}</div>' if impact else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_why_number_one(domino: DominoCandidate):
    """Render the audit trail of why this is #1."""

    st.markdown("""
    <div style="margin-top: 30px; margin-bottom: 15px;">
        <span style="color: #9f7aea; font-weight: 700; font-size: 1.1em; letter-spacing: 1px;">
            📊 WHY THIS IS #1
        </span>
        <span style="color: #718096; font-size: 0.9em; margin-left: 10px;">
            (audit trail)
        </span>
    </div>
    """, unsafe_allow_html=True)

    ease_bonus = 1.0 + domino.ease_of_fix

    st.markdown(f"""
    <div style="
        background: rgba(159, 122, 234, 0.1);
        border: 1px solid rgba(159, 122, 234, 0.3);
        border-radius: 16px;
        padding: 20px;
    ">
        <div style="color: #e2e8f0; font-size: 0.95em; margin-bottom: 15px;">
            <strong>Scoring Formula:</strong> Expected Delay Prevented = Slip × Probability × Ease Bonus
        </div>

        <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 15px;">
            <div style="text-align: center; padding: 10px 20px; background: rgba(0,0,0,0.2); border-radius: 10px;">
                <div style="color: #fc8181; font-size: 1.5em; font-weight: 700;">{domino.slip_days:.1f}d</div>
                <div style="color: #718096; font-size: 0.8em;">Slip if unaddressed</div>
            </div>
            <div style="color: #718096; font-size: 1.5em; align-self: center;">×</div>
            <div style="text-align: center; padding: 10px 20px; background: rgba(0,0,0,0.2); border-radius: 10px;">
                <div style="color: #f6ad55; font-size: 1.5em; font-weight: 700;">{int(domino.probability * 100)}%</div>
                <div style="color: #718096; font-size: 0.8em;">Probability</div>
            </div>
            <div style="color: #718096; font-size: 1.5em; align-self: center;">×</div>
            <div style="text-align: center; padding: 10px 20px; background: rgba(0,0,0,0.2); border-radius: 10px;">
                <div style="color: #68d391; font-size: 1.5em; font-weight: 700;">{ease_bonus:.1f}x</div>
                <div style="color: #718096; font-size: 0.8em;">Ease bonus</div>
            </div>
            <div style="color: #718096; font-size: 1.5em; align-self: center;">=</div>
            <div style="text-align: center; padding: 10px 20px; background: rgba(159, 122, 234, 0.2); border-radius: 10px; border: 1px solid #9f7aea;">
                <div style="color: #9f7aea; font-size: 1.5em; font-weight: 700;">{domino.impact_score:.2f}</div>
                <div style="color: #718096; font-size: 0.8em;">Impact Score</div>
            </div>
        </div>

        <div style="color: #a0aec0; font-size: 0.9em;">
            <strong>Type:</strong> {domino.type.value.replace('_', ' ').title()} |
            <strong>Critical Path:</strong> {'Yes' if domino.is_critical_path else 'No'} |
            <strong>Days Stale:</strong> {domino.days_stale}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""

    # Initialize
    conn = get_connection()
    llm_client = get_llm_client()
    state = get_domino_state()

    # Get project key
    project_key = "PROJ"
    if conn:
        try:
            result = conn.execute("SELECT DISTINCT project_key FROM issues LIMIT 1").fetchone()
            if result:
                project_key = result[0]
        except:
            pass

    # Title
    st.markdown("""
    <div style="text-align: center; margin-bottom: 20px;">
        <div style="font-size: 1em; color: #718096; letter-spacing: 3px; margin-bottom: 10px;">
            THE SINGLE HIGHEST-IMPACT INTERVENTION
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Header stats
    render_header(state)

    # Refresh controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=state.get('auto_refresh', False))
        state['auto_refresh'] = auto_refresh

    with col3:
        if st.button("🔄 Recompute Now", use_container_width=True):
            state['current_domino'] = None
            save_domino_state(state)
            st.rerun()

    with col1:
        if state.get('last_computed'):
            try:
                last = datetime.fromisoformat(state['last_computed'])
                st.markdown(f"*Last computed: {last.strftime('%H:%M:%S')}*")
            except:
                pass

    st.markdown("---")

    # Find or use cached domino
    if not state.get('current_domino') or auto_refresh:
        if conn:
            with st.spinner("Computing next domino..."):
                domino = find_the_domino(conn, project_key)

                if domino:
                    rescue_paths = generate_rescue_paths(domino, conn, project_key, llm_client)
                    state['current_domino'] = asdict(domino)
                    state['current_domino']['type'] = domino.type.value
                    state['rescue_paths'] = [asdict(p) for p in rescue_paths]
                    for p in state['rescue_paths']:
                        p['type'] = p['type'].value if hasattr(p['type'], 'value') else p['type']
                else:
                    state['current_domino'] = None
                    state['rescue_paths'] = []

                state['last_computed'] = datetime.now().isoformat()
                save_domino_state(state)

    # Render domino
    if state.get('current_domino'):
        # Reconstruct domino object
        domino_dict = state['current_domino']
        domino = DominoCandidate(
            id=domino_dict['id'],
            key=domino_dict['key'],
            summary=domino_dict['summary'],
            type=DominoType(domino_dict['type']),
            assignee=domino_dict.get('assignee'),
            assignee_id=domino_dict.get('assignee_id'),
            slip_days=domino_dict['slip_days'],
            probability=domino_dict['probability'],
            ease_of_fix=domino_dict['ease_of_fix'],
            impact_score=domino_dict.get('impact_score', 0),
            why_domino=domino_dict.get('why_domino', ''),
            blocks_count=domino_dict.get('blocks_count', 0),
            is_critical_path=domino_dict.get('is_critical_path', False),
            days_stale=domino_dict.get('days_stale', 0)
        )

        # Reconstruct rescue paths
        rescue_paths = []
        for p in state.get('rescue_paths', []):
            rescue_paths.append(RescuePath(
                type=RescueType(p['type']),
                title=p['title'],
                description=p['description'],
                time_to_fix=p['time_to_fix'],
                time_to_fix_mins=p['time_to_fix_mins'],
                days_saved=p['days_saved'],
                action_content=p['action_content'],
                target=p['target'],
                confidence=p.get('confidence', 0.8)
            ))

        # Render main widget
        render_domino_widget(domino, rescue_paths, state)

        # Render rescue paths
        render_rescue_paths(rescue_paths, domino, state)

        # Render one-click actions
        render_one_click_actions(domino, rescue_paths, state)

        # Why #1 audit trail
        render_why_number_one(domino)

    else:
        render_no_domino()

    # Tabs for additional info
    st.markdown("---")

    tab1, tab2 = st.tabs(["📜 Action History", "📊 Success Metrics"])

    with tab1:
        render_audit_trail(state)

    with tab2:
        st.markdown("""
        ### Success Metrics for This Widget

        Track these to measure impact:

        | Metric | Description | Target |
        |--------|-------------|--------|
        | **MTTU** | Mean Time to Unblock | ↓ 50% |
        | **Late Surprises** | Issues discovered in final week | ↓ 70% |
        | **Manual Follow-ups** | PM-initiated status checks | ↓ 60% |
        | **Forecast Accuracy** | Plan vs actual variance | ↑ 25% |
        """)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Actions This Session", state.get('actions_taken', 0))
        with col2:
            st.metric("Days Saved", f"{state.get('days_saved', 0):.1f}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #718096; font-size: 0.85em; padding: 20px;">
        🎯 Next Domino • The Single Highest-Impact Intervention<br>
        <span style="font-size: 0.8em;">Most dashboards show status. This shows leverage.</span>
    </div>
    """, unsafe_allow_html=True)

    # Auto-refresh logic
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
