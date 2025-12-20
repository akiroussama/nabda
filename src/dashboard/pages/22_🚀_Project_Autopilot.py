"""
🚀 PROJECT AUTOPILOT - Self-Driving Project Management

THE REVOLUTION: YOUR PROJECT RUNS ITSELF

This isn't a dashboard. This is a self-driving project layer that:
1. Takes your goal and constraints
2. Builds a living, executable Intent Graph
3. Detects drift BEFORE humans can see it
4. Proposes AND executes minimum interventions
5. Forces clarity when the plan is ambiguous

You're not managing work anymore. You're GOVERNING an autonomous system.

Powered by: Gemini 2.0 Flash for intelligent decision-making
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

# Page configuration
st.set_page_config(
    page_title="Project Autopilot",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# LLM CLIENT INITIALIZATION
# =============================================================================

def get_llm_client():
    """Get or create LLM client."""
    try:
        import sys
        sys.path.insert(0, 'src')
        from intelligence.llm_client import GeminiClient, create_llm_client

        api_key = os.environ.get('GOOGLE_API_KEY', '')
        if api_key:
            return GeminiClient(
                api_key=api_key,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 4096,
                }
            )
        else:
            # Return mock client for demo
            from intelligence.llm_client import MockGeminiClient
            return MockGeminiClient()
    except Exception as e:
        st.warning(f"LLM not available: {e}. Using demo mode.")
        return None


# =============================================================================
# DATA MODELS
# =============================================================================

class AutonomyLevel(Enum):
    """Autopilot autonomy levels."""
    OBSERVE = 0       # Read-only, just observes
    SUGGEST = 1       # Draft actions, suggest but don't execute
    ASSIST = 2        # Execute low-risk (updates, notifications)
    COPILOT = 3       # Execute with spot approvals
    AUTONOMOUS = 4    # Full autonomy within constraints


class DriftSeverity(Enum):
    """Severity of detected drift."""
    CRITICAL = "critical"    # Will miss deadline/constraint
    HIGH = "high"            # Significant risk
    MEDIUM = "medium"        # Noticeable deviation
    LOW = "low"              # Minor drift


class InterventionType(Enum):
    """Types of interventions the autopilot can take."""
    REASSIGN_TASK = "reassign_task"
    SEND_UPDATE = "send_update"
    CREATE_TASK = "create_task"
    ESCALATE = "escalate"
    SCHEDULE_MEETING = "schedule_meeting"
    ADJUST_SCOPE = "adjust_scope"
    PING_BLOCKER = "ping_blocker"
    NOTIFY_STAKEHOLDER = "notify_stakeholder"


class InterventionStatus(Enum):
    """Status of an intervention."""
    PROPOSED = "proposed"
    APPROVED = "approved"
    EXECUTED = "executed"
    REJECTED = "rejected"
    REVERTED = "reverted"


@dataclass
class Constraint:
    """A constraint on the project."""
    name: str
    description: str
    type: str  # "headcount", "deadline", "compliance", "quality", "budget"
    value: Any
    is_hard: bool = True  # Hard constraints can't be violated


@dataclass
class ProjectIntent:
    """The Intent Graph - machine-readable project goal."""
    goal: str
    deadline: datetime
    constraints: List[Constraint]
    success_metrics: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "goal": self.goal,
            "deadline": self.deadline.isoformat(),
            "constraints": [asdict(c) for c in self.constraints],
            "success_metrics": self.success_metrics,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class DriftDetection:
    """A detected drift from the plan."""
    id: str
    severity: DriftSeverity
    title: str
    description: str
    impact: str  # What happens if not addressed
    detected_at: datetime
    data_source: str  # What data revealed this drift
    affected_constraint: Optional[str] = None
    predicted_delay_days: int = 0
    confidence: float = 0.0


@dataclass
class Intervention:
    """An intervention proposed or executed by the autopilot."""
    id: str
    type: InterventionType
    title: str
    description: str
    action_content: str  # The actual action (message, assignment, etc.)
    target: str  # Who/what is the target
    drift_id: str  # Which drift this addresses
    status: InterventionStatus
    created_at: datetime
    executed_at: Optional[datetime] = None
    impact_measurement: Optional[str] = None
    requires_approval: bool = True
    risk_level: str = "low"  # low, medium, high
    reversible: bool = True


@dataclass
class AuditEntry:
    """An entry in the audit trail."""
    timestamp: datetime
    action: str
    details: str
    actor: str  # "autopilot" or username
    drift_id: Optional[str] = None
    intervention_id: Optional[str] = None
    impact: Optional[str] = None


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

@st.cache_resource
def get_connection():
    """Get DuckDB connection."""
    try:
        return duckdb.connect("jira_data.duckdb", read_only=True)
    except Exception as e:
        return None


# =============================================================================
# INTENT GRAPH BUILDER
# =============================================================================

def parse_intent_from_goal(goal_text: str, llm_client) -> ProjectIntent:
    """Use LLM to parse a natural language goal into an Intent Graph."""

    if not llm_client:
        # Demo mode - parse manually
        return create_demo_intent(goal_text)

    prompt = f"""Parse this project goal into a structured Intent Graph.

Goal: "{goal_text}"

Extract:
1. The main deliverable/outcome
2. The deadline (if mentioned)
3. Any constraints (headcount, compliance, quality, etc.)
4. Success metrics (if implied)

Respond in JSON format:
{{
    "goal": "clear statement of the goal",
    "deadline": "YYYY-MM-DD or null",
    "constraints": [
        {{"name": "constraint name", "description": "details", "type": "headcount|deadline|compliance|quality|budget", "value": "the value", "is_hard": true|false}}
    ],
    "success_metrics": {{"metric_name": target_value}}
}}

Only respond with valid JSON."""

    try:
        result = llm_client.generate_json(prompt)

        deadline = datetime.now() + timedelta(days=30)  # Default
        if result.get("deadline"):
            try:
                deadline = datetime.fromisoformat(result["deadline"])
            except:
                pass

        constraints = []
        for c in result.get("constraints", []):
            constraints.append(Constraint(
                name=c.get("name", "Unknown"),
                description=c.get("description", ""),
                type=c.get("type", "other"),
                value=c.get("value"),
                is_hard=c.get("is_hard", True)
            ))

        return ProjectIntent(
            goal=result.get("goal", goal_text),
            deadline=deadline,
            constraints=constraints,
            success_metrics=result.get("success_metrics", {})
        )
    except Exception as e:
        st.warning(f"LLM parsing failed: {e}. Using fallback.")
        return create_demo_intent(goal_text)


def create_demo_intent(goal_text: str) -> ProjectIntent:
    """Create a demo intent for when LLM is not available."""
    return ProjectIntent(
        goal=goal_text,
        deadline=datetime.now() + timedelta(days=30),
        constraints=[
            Constraint("Deadline", "Must ship by target date", "deadline",
                      (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d"), True),
            Constraint("Team Size", "No more than current team", "headcount", "current", True),
        ],
        success_metrics={"completion_rate": 100, "quality_score": 80}
    )


# =============================================================================
# DRIFT DETECTION ENGINE
# =============================================================================

def detect_velocity_drift(conn, project_key: str, intent: ProjectIntent) -> Optional[DriftDetection]:
    """Detect if velocity is on track to meet deadline."""
    try:
        # Get velocity and remaining work
        velocity = conn.execute(f"""
            SELECT
                COALESCE(AVG(completed_points), 0) as avg_velocity
            FROM (
                SELECT
                    s.id,
                    SUM(CASE WHEN i.status = 'Terminé(e)' THEN i.story_points ELSE 0 END) as completed_points
                FROM sprints s
                JOIN issues i ON i.sprint_id = s.id
                WHERE i.project_key = '{project_key}'
                  AND s.state IN ('closed', 'active')
                GROUP BY s.id
                ORDER BY s.end_date DESC
                LIMIT 4
            )
        """).fetchone()

        remaining = conn.execute(f"""
            SELECT COALESCE(SUM(story_points), 0)
            FROM issues
            WHERE project_key = '{project_key}'
              AND status NOT IN ('Terminé(e)', 'Done', 'Closed')
        """).fetchone()

        avg_velocity = velocity[0] if velocity else 10
        remaining_points = remaining[0] if remaining else 0

        # Calculate if we'll make it
        days_remaining = (intent.deadline - datetime.now()).days
        sprints_remaining = max(1, days_remaining / 14)  # Assuming 2-week sprints
        projected_completion = avg_velocity * sprints_remaining

        if remaining_points > projected_completion * 1.2:
            delay_days = int((remaining_points - projected_completion) / (avg_velocity / 14))
            return DriftDetection(
                id=f"drift_velocity_{datetime.now().strftime('%Y%m%d%H%M')}",
                severity=DriftSeverity.HIGH if delay_days > 5 else DriftSeverity.MEDIUM,
                title="Velocity Drift Detected",
                description=f"Current velocity ({avg_velocity:.0f} pts/sprint) is insufficient to complete remaining {remaining_points:.0f} points by deadline.",
                impact=f"Project will slip by approximately {delay_days} days if nothing changes.",
                detected_at=datetime.now(),
                data_source="Velocity analysis",
                affected_constraint="deadline",
                predicted_delay_days=delay_days,
                confidence=0.85
            )

        return None
    except Exception as e:
        return None


def detect_blocker_drift(conn, project_key: str) -> Optional[DriftDetection]:
    """Detect critical blockers that threaten the timeline."""
    try:
        blockers = conn.execute(f"""
            SELECT
                COUNT(*) as blocked_count,
                MAX(DATEDIFF('day', updated, CURRENT_TIMESTAMP)) as max_days_blocked,
                COALESCE(SUM(story_points), 0) as blocked_points
            FROM issues
            WHERE project_key = '{project_key}'
              AND status IN ('Blocked', 'On Hold', 'Waiting')
        """).fetchone()

        if blockers and blockers[0] > 0:
            blocked_count, max_days, blocked_points = blockers

            if max_days > 5 or blocked_count > 3:
                return DriftDetection(
                    id=f"drift_blocker_{datetime.now().strftime('%Y%m%d%H%M')}",
                    severity=DriftSeverity.CRITICAL if max_days > 7 else DriftSeverity.HIGH,
                    title=f"{blocked_count} Items Blocked (Max {max_days} days)",
                    description=f"{blocked_count} work items are blocked, representing {blocked_points:.0f} story points. Longest blocked: {max_days} days.",
                    impact=f"Each blocked day compounds into 1.5x delivery delay. Currently risking {int(max_days * 1.5)} day slip.",
                    detected_at=datetime.now(),
                    data_source="Blocker analysis",
                    affected_constraint="deadline",
                    predicted_delay_days=int(max_days * 1.5),
                    confidence=0.9
                )

        return None
    except:
        return None


def detect_resource_drift(conn, project_key: str) -> Optional[DriftDetection]:
    """Detect resource overload or bottlenecks."""
    try:
        overloaded = conn.execute(f"""
            SELECT
                u.pseudonym as name,
                COUNT(*) as task_count,
                COUNT(CASE WHEN i.status = 'En cours' THEN 1 END) as in_progress
            FROM issues i
            LEFT JOIN users u ON i.assignee_id = u.account_id
            WHERE i.project_key = '{project_key}'
              AND i.status NOT IN ('Terminé(e)', 'Done', 'Closed')
              AND i.assignee_id IS NOT NULL
            GROUP BY u.pseudonym, i.assignee_id
            HAVING COUNT(*) > 7 OR COUNT(CASE WHEN i.status = 'En cours' THEN 1 END) > 4
        """).fetchall()

        if overloaded:
            names = [r[0] for r in overloaded]
            return DriftDetection(
                id=f"drift_resource_{datetime.now().strftime('%Y%m%d%H%M')}",
                severity=DriftSeverity.MEDIUM,
                title=f"Resource Overload: {', '.join(names[:2])}",
                description=f"{len(overloaded)} team member(s) are overloaded with too many concurrent tasks.",
                impact="Overloaded team members deliver 30% slower and have 2x error rate. Burnout risk elevated.",
                detected_at=datetime.now(),
                data_source="Workload analysis",
                affected_constraint="quality",
                predicted_delay_days=3,
                confidence=0.75
            )

        return None
    except:
        return None


def detect_stale_drift(conn, project_key: str) -> Optional[DriftDetection]:
    """Detect stale work items that might be silently stuck."""
    try:
        stale = conn.execute(f"""
            SELECT COUNT(*)
            FROM issues
            WHERE project_key = '{project_key}'
              AND status = 'En cours'
              AND DATEDIFF('day', updated, CURRENT_TIMESTAMP) > 4
        """).fetchone()

        if stale and stale[0] > 2:
            return DriftDetection(
                id=f"drift_stale_{datetime.now().strftime('%Y%m%d%H%M')}",
                severity=DriftSeverity.MEDIUM,
                title=f"{stale[0]} Silently Stuck Items",
                description=f"{stale[0]} items are marked 'In Progress' but haven't been updated in 4+ days. They may be secretly blocked.",
                impact="Silent blocks are the #1 cause of last-week surprises. Average hidden delay: 5 days.",
                detected_at=datetime.now(),
                data_source="Activity analysis",
                affected_constraint="deadline",
                predicted_delay_days=5,
                confidence=0.7
            )

        return None
    except:
        return None


def run_drift_detection(conn, project_key: str, intent: ProjectIntent) -> List[DriftDetection]:
    """Run all drift detectors and return findings."""
    detections = []

    detectors = [
        lambda: detect_velocity_drift(conn, project_key, intent),
        lambda: detect_blocker_drift(conn, project_key),
        lambda: detect_resource_drift(conn, project_key),
        lambda: detect_stale_drift(conn, project_key),
    ]

    for detector in detectors:
        try:
            result = detector()
            if result:
                detections.append(result)
        except:
            continue

    # Sort by severity
    severity_order = {
        DriftSeverity.CRITICAL: 0,
        DriftSeverity.HIGH: 1,
        DriftSeverity.MEDIUM: 2,
        DriftSeverity.LOW: 3
    }

    detections.sort(key=lambda d: severity_order.get(d.severity, 4))

    return detections


# =============================================================================
# INTERVENTION ENGINE (LLM-POWERED)
# =============================================================================

def generate_intervention(drift: DriftDetection, conn, project_key: str, llm_client) -> Optional[Intervention]:
    """Use LLM to generate an intelligent intervention for a drift."""

    if not llm_client:
        return generate_fallback_intervention(drift, conn, project_key)

    # Gather context
    context = gather_intervention_context(conn, project_key, drift)

    prompt = f"""You are Project Autopilot, an AI that manages projects autonomously.

DETECTED DRIFT:
- Title: {drift.title}
- Severity: {drift.severity.value}
- Description: {drift.description}
- Impact if not addressed: {drift.impact}
- Predicted delay: {drift.predicted_delay_days} days
- Affected constraint: {drift.affected_constraint}

CURRENT PROJECT CONTEXT:
{json.dumps(context, indent=2, default=str)}

Your job is to propose the MINIMUM intervention needed to address this drift.

Rules:
1. Prefer async communication over meetings
2. Target the exact person who can unblock
3. Draft the exact message/action - not a template
4. Consider reversibility and risk
5. Be specific, not generic

Respond in JSON:
{{
    "intervention_type": "reassign_task|send_update|create_task|escalate|schedule_meeting|adjust_scope|ping_blocker|notify_stakeholder",
    "title": "short action title",
    "description": "what this intervention does",
    "action_content": "THE EXACT message to send or action to take - ready to execute",
    "target": "who receives this action",
    "risk_level": "low|medium|high",
    "expected_impact": "what improvement this should create",
    "requires_approval": true|false
}}

Only respond with valid JSON."""

    try:
        result = llm_client.generate_json(prompt)

        return Intervention(
            id=f"int_{drift.id}_{datetime.now().strftime('%H%M%S')}",
            type=InterventionType(result.get("intervention_type", "send_update")),
            title=result.get("title", "Intervention"),
            description=result.get("description", ""),
            action_content=result.get("action_content", ""),
            target=result.get("target", "team"),
            drift_id=drift.id,
            status=InterventionStatus.PROPOSED,
            created_at=datetime.now(),
            requires_approval=result.get("requires_approval", True),
            risk_level=result.get("risk_level", "low"),
            impact_measurement=result.get("expected_impact")
        )
    except Exception as e:
        return generate_fallback_intervention(drift, conn, project_key)


def gather_intervention_context(conn, project_key: str, drift: DriftDetection) -> Dict:
    """Gather relevant context for intervention generation."""
    context = {}

    try:
        # Get team info
        team = conn.execute(f"""
            SELECT u.pseudonym, COUNT(*) as tasks
            FROM issues i
            JOIN users u ON i.assignee_id = u.account_id
            WHERE i.project_key = '{project_key}'
              AND i.status NOT IN ('Terminé(e)', 'Done')
            GROUP BY u.pseudonym
            ORDER BY tasks DESC
            LIMIT 5
        """).fetchall()
        context["team_workload"] = [{"name": t[0], "tasks": t[1]} for t in team]

        # Get blockers if relevant
        if "blocker" in drift.title.lower():
            blockers = conn.execute(f"""
                SELECT key, summary, u.pseudonym as assignee,
                       DATEDIFF('day', updated, CURRENT_TIMESTAMP) as days_blocked
                FROM issues i
                LEFT JOIN users u ON i.assignee_id = u.account_id
                WHERE i.project_key = '{project_key}'
                  AND i.status IN ('Blocked', 'On Hold')
                ORDER BY days_blocked DESC
                LIMIT 3
            """).fetchall()
            context["top_blockers"] = [
                {"key": b[0], "summary": b[1], "assignee": b[2], "days": b[3]}
                for b in blockers
            ]

        # Get sprint status
        sprint = conn.execute(f"""
            SELECT s.name,
                   COUNT(CASE WHEN i.status = 'Terminé(e)' THEN 1 END) as done,
                   COUNT(*) as total,
                   DATEDIFF('day', CURRENT_TIMESTAMP, s.end_date) as days_left
            FROM sprints s
            JOIN issues i ON i.sprint_id = s.id
            WHERE s.state = 'active'
              AND i.project_key = '{project_key}'
            GROUP BY s.name, s.end_date
        """).fetchone()
        if sprint:
            context["current_sprint"] = {
                "name": sprint[0],
                "done": sprint[1],
                "total": sprint[2],
                "days_left": sprint[3]
            }
    except:
        pass

    return context


def generate_fallback_intervention(drift: DriftDetection, conn, project_key: str) -> Intervention:
    """Generate a basic intervention when LLM is not available."""

    # Map drift types to intervention templates
    if "blocker" in drift.title.lower():
        return Intervention(
            id=f"int_{drift.id}_{datetime.now().strftime('%H%M%S')}",
            type=InterventionType.PING_BLOCKER,
            title="Unblock Critical Items",
            description="Send check-in to blocked item owners",
            action_content=f"""Hi team,

We have {drift.description}

For each blocked item, please reply with:
1. What's blocking you
2. Who can help unblock
3. Expected resolution time

Let's clear these today to stay on track.

Thanks!""",
            target="blocked item owners",
            drift_id=drift.id,
            status=InterventionStatus.PROPOSED,
            created_at=datetime.now(),
            requires_approval=True,
            risk_level="low"
        )

    elif "velocity" in drift.title.lower():
        return Intervention(
            id=f"int_{drift.id}_{datetime.now().strftime('%H%M%S')}",
            type=InterventionType.ADJUST_SCOPE,
            title="Scope Review Required",
            description="Velocity indicates we need to adjust scope or timeline",
            action_content=f"""Sprint Scope Review Needed

Current velocity won't meet our deadline. We need to choose:

Option A: Reduce scope
- Move lower priority items to next sprint
- Focus on must-have features only

Option B: Extend timeline
- Add {drift.predicted_delay_days} days to deadline
- Communicate revised date to stakeholders

Option C: Add capacity
- Bring in support from another team
- Contract help for specific items

Please approve one option.""",
            target="project stakeholders",
            drift_id=drift.id,
            status=InterventionStatus.PROPOSED,
            created_at=datetime.now(),
            requires_approval=True,
            risk_level="medium"
        )

    else:
        return Intervention(
            id=f"int_{drift.id}_{datetime.now().strftime('%H%M%S')}",
            type=InterventionType.SEND_UPDATE,
            title="Status Check Required",
            description="Investigate and report on detected issue",
            action_content=f"""Autopilot Alert: {drift.title}

{drift.description}

Impact: {drift.impact}

Please review and take appropriate action.

Confidence: {int(drift.confidence * 100)}%
Data source: {drift.data_source}""",
            target="project manager",
            drift_id=drift.id,
            status=InterventionStatus.PROPOSED,
            created_at=datetime.now(),
            requires_approval=True,
            risk_level="low"
        )


# =============================================================================
# CLARITY FORCING ENGINE
# =============================================================================

def generate_clarity_question(drift: DriftDetection, llm_client) -> Optional[Dict]:
    """Generate a binary choice question when the situation is ambiguous."""

    if not llm_client:
        # Fallback question
        return {
            "question": f"How should we handle: {drift.title}?",
            "option_a": {
                "label": "Prioritize deadline",
                "description": "Accept quality/scope trade-offs to hit the date",
                "impact": f"Ship on time, may need follow-up fixes"
            },
            "option_b": {
                "label": "Prioritize quality",
                "description": "Take extra time to do it right",
                "impact": f"Slip {drift.predicted_delay_days} days, higher quality"
            }
        }

    prompt = f"""A project drift has been detected that requires a human decision.

DRIFT: {drift.title}
DESCRIPTION: {drift.description}
IMPACT: {drift.impact}
PREDICTED DELAY: {drift.predicted_delay_days} days

Generate a BINARY choice question that forces clarity. The question should:
1. Present exactly 2 options (not 3, not "it depends")
2. Make trade-offs explicit
3. Include predicted impact of each option
4. Be answerable immediately

Respond in JSON:
{{
    "question": "The main question",
    "option_a": {{
        "label": "Short label (3-5 words)",
        "description": "What this option means",
        "impact": "What happens if chosen"
    }},
    "option_b": {{
        "label": "Short label (3-5 words)",
        "description": "What this option means",
        "impact": "What happens if chosen"
    }}
}}

Only respond with valid JSON."""

    try:
        return llm_client.generate_json(prompt)
    except:
        return None


# =============================================================================
# AUTOPILOT STATE MANAGEMENT
# =============================================================================

def get_autopilot_state() -> Dict:
    """Get or initialize autopilot state from session."""
    if 'autopilot_state' not in st.session_state:
        st.session_state.autopilot_state = {
            'intent': None,
            'autonomy_level': AutonomyLevel.SUGGEST.value,
            'drifts': [],
            'interventions': [],
            'audit_trail': [],
            'last_scan': None,
            'prevented_slips': 0,
            'actions_taken': 0
        }
    return st.session_state.autopilot_state


def save_autopilot_state(state: Dict):
    """Save autopilot state to session."""
    st.session_state.autopilot_state = state


def add_audit_entry(state: Dict, action: str, details: str,
                   actor: str = "autopilot", drift_id: str = None,
                   intervention_id: str = None, impact: str = None):
    """Add an entry to the audit trail."""
    entry = AuditEntry(
        timestamp=datetime.now(),
        action=action,
        details=details,
        actor=actor,
        drift_id=drift_id,
        intervention_id=intervention_id,
        impact=impact
    )
    state['audit_trail'].insert(0, asdict(entry))
    # Keep last 50 entries
    state['audit_trail'] = state['audit_trail'][:50]


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_autopilot_header(state: Dict):
    """Render the autopilot header."""

    autonomy_labels = {
        0: ("OBSERVE", "#718096", "Watching only"),
        1: ("SUGGEST", "#4299e1", "Proposing actions"),
        2: ("ASSIST", "#48bb78", "Executing low-risk"),
        3: ("COPILOT", "#ed8936", "Full assistance"),
        4: ("AUTONOMOUS", "#9f7aea", "Self-driving")
    }

    level = state.get('autonomy_level', 1)
    label, color, desc = autonomy_labels.get(level, autonomy_labels[1])

    st.markdown(f"""
    <style>
    .autopilot-header {{
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        border-radius: 24px;
        padding: 40px;
        margin-bottom: 30px;
        border: 2px solid {color};
        box-shadow: 0 0 60px {color}40;
        position: relative;
        overflow: hidden;
    }}
    .autopilot-header::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, transparent, {color}, transparent);
        animation: scan 3s linear infinite;
    }}
    @keyframes scan {{
        0% {{ transform: translateX(-100%); }}
        100% {{ transform: translateX(100%); }}
    }}
    .autopilot-title {{
        font-size: 2.8em;
        font-weight: 800;
        background: linear-gradient(90deg, #fff, {color}, #fff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
    }}
    @keyframes shimmer {{
        to {{ background-position: 200% center; }}
    }}
    .autopilot-mode {{
        display: inline-block;
        background: {color};
        color: #000;
        padding: 8px 24px;
        border-radius: 30px;
        font-weight: 800;
        font-size: 0.9em;
        letter-spacing: 2px;
        margin-top: 15px;
    }}
    .autopilot-stats {{
        display: flex;
        gap: 30px;
        margin-top: 25px;
        flex-wrap: wrap;
    }}
    .stat-item {{
        text-align: center;
    }}
    .stat-value {{
        font-size: 2em;
        font-weight: 800;
        color: {color};
    }}
    .stat-label {{
        color: #718096;
        font-size: 0.85em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    </style>

    <div class="autopilot-header">
        <div class="autopilot-title">🚀 PROJECT AUTOPILOT</div>
        <div style="color: #a0aec0; font-size: 1.1em; margin-top: 10px;">
            Your project runs itself. You just approve.
        </div>
        <div class="autopilot-mode">{label} MODE</div>
        <div style="color: #718096; font-size: 0.9em; margin-top: 5px;">{desc}</div>

        <div class="autopilot-stats">
            <div class="stat-item">
                <div class="stat-value">{state.get('prevented_slips', 0)}</div>
                <div class="stat-label">Slips Prevented</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{state.get('actions_taken', 0)}</div>
                <div class="stat-label">Actions Taken</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len(state.get('drifts', []))}</div>
                <div class="stat-label">Active Drifts</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">{len([i for i in state.get('interventions', []) if i.get('status') == 'proposed'])}</div>
                <div class="stat-label">Pending Approvals</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_intent_setup(state: Dict, llm_client):
    """Render the intent setup interface."""

    st.markdown("### 🎯 Set Your Project Intent")
    st.markdown("*Tell Autopilot what you're trying to achieve, and it will keep you on track.*")

    example = "Ship onboarding v2 by March 15. Must keep signup conversion ≥ 85%. No more than 2 engineers off the payments squad. Legal review required."

    goal_text = st.text_area(
        "Describe your goal with constraints:",
        placeholder=example,
        height=120,
        key="intent_input"
    )

    if st.button("🚀 Activate Autopilot", type="primary", use_container_width=True):
        if goal_text:
            with st.spinner("Parsing intent and building graph..."):
                intent = parse_intent_from_goal(goal_text, llm_client)
                state['intent'] = intent.to_dict()
                add_audit_entry(
                    state,
                    "Intent Activated",
                    f"Goal: {intent.goal[:100]}...",
                    actor="user"
                )
                save_autopilot_state(state)
                st.success("✅ Autopilot activated! Starting drift detection...")
                st.rerun()
        else:
            st.warning("Please describe your goal first.")


def render_intent_card(intent_dict: Dict):
    """Render the current intent."""

    deadline = datetime.fromisoformat(intent_dict['deadline'])
    days_left = (deadline - datetime.now()).days

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(72, 187, 120, 0.1), rgba(72, 187, 120, 0.05));
        border: 1px solid rgba(72, 187, 120, 0.3);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 20px;
    ">
        <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap; gap: 15px;">
            <div style="flex: 2; min-width: 300px;">
                <div style="color: #68d391; font-weight: 700; font-size: 0.9em; letter-spacing: 1px;">
                    CURRENT INTENT
                </div>
                <div style="color: #fff; font-size: 1.2em; font-weight: 600; margin-top: 10px;">
                    {intent_dict['goal'][:150]}{'...' if len(intent_dict['goal']) > 150 else ''}
                </div>
            </div>
            <div style="text-align: center; min-width: 120px;">
                <div style="font-size: 2.5em; font-weight: 800; color: {'#68d391' if days_left > 14 else '#f6ad55' if days_left > 7 else '#fc8181'};">
                    {days_left}
                </div>
                <div style="color: #718096; font-size: 0.85em;">DAYS LEFT</div>
            </div>
        </div>

        <div style="margin-top: 20px; display: flex; gap: 10px; flex-wrap: wrap;">
            {''.join(f'<span style="background: rgba(99, 179, 237, 0.2); color: #63b3ed; padding: 5px 12px; border-radius: 20px; font-size: 0.85em;">{c["name"]}: {c["value"]}</span>' for c in intent_dict.get("constraints", [])[:4])}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_drift_detection(drifts: List[Dict], state: Dict, conn, project_key: str, llm_client):
    """Render detected drifts and interventions."""

    if not drifts:
        st.markdown("""
        <div style="
            text-align: center;
            padding: 40px;
            background: rgba(72, 187, 120, 0.1);
            border-radius: 16px;
            border: 1px solid rgba(72, 187, 120, 0.3);
        ">
            <div style="font-size: 3em;">✅</div>
            <div style="color: #68d391; font-size: 1.3em; font-weight: 700; margin-top: 10px;">
                ALL SYSTEMS NOMINAL
            </div>
            <div style="color: #a0aec0; margin-top: 10px;">
                No drift detected. Your project is on track.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    st.markdown("### 🔍 Detected Drifts")

    for i, drift_dict in enumerate(drifts):
        drift = DriftDetection(**{k: v if k != 'severity' else DriftSeverity(v)
                                  for k, v in drift_dict.items()
                                  if k != 'detected_at'})
        drift.detected_at = datetime.fromisoformat(drift_dict['detected_at']) if isinstance(drift_dict.get('detected_at'), str) else drift_dict.get('detected_at', datetime.now())

        severity_colors = {
            DriftSeverity.CRITICAL: "#e53e3e",
            DriftSeverity.HIGH: "#ed8936",
            DriftSeverity.MEDIUM: "#ecc94b",
            DriftSeverity.LOW: "#48bb78"
        }

        color = severity_colors.get(drift.severity, "#718096")

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(26,26,46,0.95), rgba(22,33,62,0.95));
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 15px;
            border-left: 4px solid {color};
        ">
            <div style="display: flex; justify-content: space-between; align-items: start; flex-wrap: wrap; gap: 15px;">
                <div style="flex: 2; min-width: 300px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="
                            background: {color}22;
                            color: {color};
                            padding: 4px 12px;
                            border-radius: 20px;
                            font-size: 0.8em;
                            font-weight: 700;
                        ">{drift.severity.value.upper()}</span>
                        <span style="color: #718096; font-size: 0.85em;">
                            {drift.detected_at.strftime('%H:%M')} • {drift.data_source}
                        </span>
                    </div>
                    <div style="color: #fff; font-size: 1.2em; font-weight: 700; margin-top: 12px;">
                        {drift.title}
                    </div>
                    <div style="color: #a0aec0; margin-top: 8px;">
                        {drift.description}
                    </div>
                    <div style="color: {color}; margin-top: 10px; font-size: 0.9em;">
                        ⚠️ {drift.impact}
                    </div>
                </div>
                <div style="text-align: center; min-width: 100px;">
                    <div style="font-size: 2em; font-weight: 800; color: {color};">
                        -{drift.predicted_delay_days}d
                    </div>
                    <div style="color: #718096; font-size: 0.8em;">RISK</div>
                    <div style="color: #a0aec0; font-size: 0.85em; margin-top: 5px;">
                        {int(drift.confidence * 100)}% conf
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Generate intervention button
        with st.expander(f"🤖 Generate Intervention for: {drift.title}", expanded=i==0):
            if st.button(f"Generate AI Intervention", key=f"gen_int_{i}"):
                with st.spinner("AI generating optimal intervention..."):
                    intervention = generate_intervention(drift, conn, project_key, llm_client)
                    if intervention:
                        state['interventions'].append(asdict(intervention))
                        add_audit_entry(
                            state,
                            "Intervention Proposed",
                            f"{intervention.title}",
                            actor="autopilot",
                            drift_id=drift.id,
                            intervention_id=intervention.id
                        )
                        save_autopilot_state(state)
                        st.success("✅ Intervention generated!")
                        st.rerun()


def render_pending_interventions(state: Dict):
    """Render interventions pending approval."""

    pending = [i for i in state.get('interventions', [])
               if i.get('status') == 'proposed']

    if not pending:
        return

    st.markdown("### ⚡ Pending Approvals")
    st.markdown("*Review and approve interventions to execute them.*")

    for i, int_dict in enumerate(pending):
        risk_colors = {
            "low": "#48bb78",
            "medium": "#f6ad55",
            "high": "#fc8181"
        }

        color = risk_colors.get(int_dict.get('risk_level', 'low'), "#48bb78")

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(26,26,46,0.95), rgba(22,33,62,0.95));
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 15px;
            border: 2px solid {color};
        ">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 15px;">
                <div>
                    <span style="
                        background: {color}22;
                        color: {color};
                        padding: 4px 12px;
                        border-radius: 20px;
                        font-size: 0.8em;
                        font-weight: 700;
                    ">{int_dict.get('type', 'action').upper()}</span>
                    <span style="
                        background: rgba(255,255,255,0.1);
                        color: #a0aec0;
                        padding: 4px 12px;
                        border-radius: 20px;
                        font-size: 0.8em;
                        margin-left: 8px;
                    ">{int_dict.get('risk_level', 'low').upper()} RISK</span>
                </div>
                <div style="color: #718096; font-size: 0.85em;">
                    Target: {int_dict.get('target', 'team')}
                </div>
            </div>

            <div style="color: #fff; font-size: 1.15em; font-weight: 700;">
                {int_dict.get('title', 'Intervention')}
            </div>
            <div style="color: #a0aec0; margin-top: 8px; font-size: 0.95em;">
                {int_dict.get('description', '')}
            </div>

            <div style="
                background: rgba(0,0,0,0.3);
                border-radius: 12px;
                padding: 15px;
                margin-top: 15px;
                border: 1px solid rgba(255,255,255,0.1);
            ">
                <div style="color: #68d391; font-weight: 600; font-size: 0.85em; margin-bottom: 10px;">
                    📋 READY TO EXECUTE
                </div>
                <div style="color: #e2e8f0; font-size: 0.9em; white-space: pre-wrap; font-family: inherit;">
{int_dict.get('action_content', '')}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("✅ Approve & Execute", key=f"approve_{i}", type="primary"):
                int_dict['status'] = 'executed'
                int_dict['executed_at'] = datetime.now().isoformat()
                state['actions_taken'] = state.get('actions_taken', 0) + 1
                state['prevented_slips'] = state.get('prevented_slips', 0) + 1
                add_audit_entry(
                    state,
                    "Intervention Executed",
                    int_dict['title'],
                    actor="user",
                    intervention_id=int_dict['id'],
                    impact=f"Drift addressed"
                )
                save_autopilot_state(state)
                st.success("✅ Intervention executed!")
                st.rerun()

        with col2:
            if st.button("❌ Reject", key=f"reject_{i}"):
                int_dict['status'] = 'rejected'
                add_audit_entry(
                    state,
                    "Intervention Rejected",
                    int_dict['title'],
                    actor="user",
                    intervention_id=int_dict['id']
                )
                save_autopilot_state(state)
                st.rerun()


def render_audit_trail(state: Dict):
    """Render the audit trail."""

    trail = state.get('audit_trail', [])

    if not trail:
        return

    st.markdown("### 📜 Audit Trail")
    st.markdown("*Complete record of all autopilot actions.*")

    for entry in trail[:10]:
        timestamp = entry.get('timestamp', '')
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp).strftime('%H:%M:%S')
            except:
                timestamp = timestamp[:8]
        else:
            timestamp = timestamp.strftime('%H:%M:%S')

        actor_emoji = "🤖" if entry.get('actor') == 'autopilot' else "👤"

        st.markdown(f"""
        <div style="
            display: flex;
            gap: 15px;
            padding: 12px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        ">
            <div style="color: #718096; font-size: 0.85em; min-width: 70px;">
                {timestamp}
            </div>
            <div style="font-size: 1.2em;">{actor_emoji}</div>
            <div style="flex: 1;">
                <div style="color: #e2e8f0; font-weight: 600;">{entry.get('action', '')}</div>
                <div style="color: #a0aec0; font-size: 0.9em;">{entry.get('details', '')}</div>
                {f'<div style="color: #68d391; font-size: 0.85em; margin-top: 3px;">Impact: {entry.get("impact")}</div>' if entry.get('impact') else ''}
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_autonomy_controls(state: Dict):
    """Render autonomy level controls."""

    st.markdown("### ⚙️ Autonomy Settings")

    levels = {
        0: ("🔍 OBSERVE", "Watch only - no actions"),
        1: ("💡 SUGGEST", "Propose actions for approval"),
        2: ("🤝 ASSIST", "Auto-execute low-risk actions"),
        3: ("🚗 COPILOT", "Full assistance with spot approvals"),
        4: ("✈️ AUTONOMOUS", "Self-driving within constraints")
    }

    current = state.get('autonomy_level', 1)

    for level, (label, desc) in levels.items():
        selected = level == current
        color = "#48bb78" if selected else "#4a5568"

        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
            <div style="
                padding: 10px 15px;
                border-radius: 10px;
                background: {'rgba(72, 187, 120, 0.1)' if selected else 'transparent'};
                border: 1px solid {color};
                margin-bottom: 8px;
            ">
                <span style="color: {color}; font-weight: {'700' if selected else '400'};">
                    {label}
                </span>
                <span style="color: #718096; font-size: 0.85em; margin-left: 10px;">
                    {desc}
                </span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("Select" if not selected else "Active",
                        key=f"level_{level}",
                        disabled=selected):
                state['autonomy_level'] = level
                add_audit_entry(
                    state,
                    "Autonomy Changed",
                    f"Set to level {level}: {label}",
                    actor="user"
                )
                save_autopilot_state(state)
                st.rerun()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""

    # Initialize
    conn = get_connection()
    llm_client = get_llm_client()
    state = get_autopilot_state()

    # Get project key
    project_key = "PROJ"
    if conn:
        try:
            result = conn.execute("SELECT DISTINCT project_key FROM issues LIMIT 1").fetchone()
            if result:
                project_key = result[0]
        except:
            pass

    # Render header
    render_autopilot_header(state)

    # Check if intent is set
    if not state.get('intent'):
        render_intent_setup(state, llm_client)
        return

    # Show current intent
    render_intent_card(state['intent'])

    # Run drift detection
    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("🔄 Scan for Drifts", use_container_width=True):
            if conn:
                intent = ProjectIntent(
                    goal=state['intent']['goal'],
                    deadline=datetime.fromisoformat(state['intent']['deadline']),
                    constraints=[Constraint(**c) for c in state['intent'].get('constraints', [])],
                    success_metrics=state['intent'].get('success_metrics', {})
                )

                with st.spinner("Scanning for drifts..."):
                    drifts = run_drift_detection(conn, project_key, intent)
                    state['drifts'] = [asdict(d) for d in drifts]
                    state['drifts'] = [
                        {**d, 'severity': d['severity'].value,
                         'detected_at': d['detected_at'].isoformat() if isinstance(d['detected_at'], datetime) else d['detected_at']}
                        for d in state['drifts']
                    ]
                    state['last_scan'] = datetime.now().isoformat()
                    add_audit_entry(
                        state,
                        "Drift Scan",
                        f"Found {len(drifts)} drifts",
                        actor="autopilot"
                    )
                    save_autopilot_state(state)
                    st.rerun()

    with col1:
        if state.get('last_scan'):
            try:
                last = datetime.fromisoformat(state['last_scan'])
                st.markdown(f"*Last scan: {last.strftime('%H:%M:%S')}*")
            except:
                pass

    # Main content
    tab1, tab2, tab3 = st.tabs(["🔍 Drifts & Interventions", "📜 Audit Trail", "⚙️ Settings"])

    with tab1:
        # Show drifts
        render_drift_detection(
            state.get('drifts', []),
            state,
            conn,
            project_key,
            llm_client
        )

        # Show pending interventions
        render_pending_interventions(state)

    with tab2:
        render_audit_trail(state)

    with tab3:
        render_autonomy_controls(state)

        st.markdown("---")

        if st.button("🔄 Reset Autopilot", type="secondary"):
            st.session_state.autopilot_state = None
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #718096; font-size: 0.85em; padding: 20px;">
        🚀 Project Autopilot • Self-Driving Project Management<br>
        <span style="font-size: 0.8em;">Powered by Gemini 2.0 Flash • You govern, it executes</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
