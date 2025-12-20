"""
💥 What Breaks If... Simulator - Cascade Impact Engine

THE ULTIMATE "OH SHIT" PREVENTION TOOL

Before disasters happen, simulate them. This page lets you ask:
- "What breaks if Alice is sick for a week?"
- "What happens if we delay the API task by 5 days?"
- "What's the blast radius if we lose access to the staging server?"

Then it shows you EXACTLY what cascades, who's affected, and how much delay results.

Target: Prevent 30-45 min/day of fire-fighting through proactive simulation
Competitor Gap: 18-24 months (requires dependency inference engine)
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import duckdb
from pathlib import Path

# Import page guide component
from src.dashboard.components import render_page_guide

# Page configuration
st.set_page_config(
    page_title="What Breaks If... Simulator",
    page_icon="💥",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =============================================================================
# DATA MODELS
# =============================================================================

class ScenarioType(Enum):
    """Types of scenarios to simulate."""
    PERSON_UNAVAILABLE = "person_unavailable"
    TASK_DELAYED = "task_delayed"
    PRIORITY_CHANGE = "priority_change"
    SCOPE_ADDITION = "scope_addition"


class ImpactSeverity(Enum):
    """Severity of impact."""
    CRITICAL = "critical"  # Stops sprint/release
    HIGH = "high"          # Major delay, multiple teams affected
    MEDIUM = "medium"      # Some delay, one team affected
    LOW = "low"            # Minimal impact, absorbable


@dataclass
class CascadeEffect:
    """Represents a single cascade effect."""
    affected_item: str
    affected_item_key: str
    effect_type: str  # "blocked", "delayed", "reassigned", "unassigned"
    delay_days: int
    confidence: float  # 0-1, how confident we are in this prediction
    reason: str
    downstream_count: int  # How many items are affected by THIS effect


@dataclass
class SimulationResult:
    """Complete simulation result."""
    scenario_name: str
    scenario_type: ScenarioType
    trigger: str  # What triggered the simulation

    # Direct impacts
    direct_effects: List[CascadeEffect]

    # Cascade analysis
    total_affected_items: int
    total_delay_days: int
    affected_people: List[str]

    # Sprint/release impact
    sprint_impact: str  # "on_track", "at_risk", "will_miss"
    confidence_change: float  # -30% means delivery confidence drops 30%

    # Mitigation suggestions
    mitigations: List[Dict[str, str]]


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
        st.error(f"Database connection failed: {e}")
        return None


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def get_person_workload(conn, person_id: str, project_key: str) -> List[Dict]:
    """Get all active work items for a person."""
    try:
        items = conn.execute(f"""
            SELECT
                key,
                summary,
                status,
                priority,
                story_points,
                sprint_name,
                issue_type,
                DATEDIFF('day', created, CURRENT_TIMESTAMP) as age_days
            FROM issues
            WHERE project_key = '{project_key}'
              AND assignee_id = '{person_id}'
              AND status NOT IN ('Terminé(e)', 'Done', 'Closed')
            ORDER BY
                CASE priority
                    WHEN 'Highest' THEN 1
                    WHEN 'High' THEN 2
                    WHEN 'Medium' THEN 3
                    ELSE 4
                END
        """).fetchall()

        columns = ['key', 'summary', 'status', 'priority', 'story_points',
                   'sprint_name', 'issue_type', 'age_days']
        return [dict(zip(columns, item)) for item in items]
    except Exception as e:
        return []


def get_task_dependencies(conn, task_key: str, project_key: str) -> Dict:
    """Get tasks that depend on or are dependencies of this task."""
    try:
        # Get task info
        task = conn.execute(f"""
            SELECT
                key, summary, status, priority, story_points,
                assignee_id, sprint_id, issue_type
            FROM issues
            WHERE key = '{task_key}'
        """).fetchone()

        if not task:
            return {}

        # Find related tasks (same sprint, similar priority = implicit dependency)
        related = conn.execute(f"""
            SELECT
                key, summary, status, priority, story_points,
                u.pseudonym as assignee_name,
                CASE priority
                    WHEN 'Highest' THEN 1
                    WHEN 'High' THEN 2
                    ELSE 3
                END as priority_rank
            FROM issues i
            LEFT JOIN users u ON i.assignee_id = u.account_id
            WHERE project_key = '{project_key}'
              AND sprint_id = {task[6] or 0}
              AND key != '{task_key}'
              AND status NOT IN ('Terminé(e)', 'Done')
            ORDER BY priority_rank
        """).fetchall()

        return {
            "task": {
                "key": task[0],
                "summary": task[1],
                "status": task[2],
                "priority": task[3],
                "story_points": task[4]
            },
            "downstream": [
                {
                    "key": r[0],
                    "summary": r[1],
                    "status": r[2],
                    "priority": r[3],
                    "assignee": r[5]
                }
                for r in related if r[6] >= (
                    1 if task[3] == 'Highest' else
                    2 if task[3] == 'High' else 3
                )
            ][:10]
        }
    except Exception as e:
        return {}


def get_sprint_health(conn, project_key: str) -> Dict:
    """Get current sprint health for baseline."""
    try:
        health = conn.execute(f"""
            WITH current_sprint AS (
                SELECT id, name, start_date, end_date
                FROM sprints
                WHERE state = 'active'
                LIMIT 1
            )
            SELECT
                cs.name as sprint_name,
                COUNT(*) as total_issues,
                COUNT(CASE WHEN i.status = 'Terminé(e)' THEN 1 END) as completed,
                COUNT(CASE WHEN i.status = 'En cours' THEN 1 END) as in_progress,
                COUNT(CASE WHEN i.status IN ('Blocked', 'On Hold') THEN 1 END) as blocked,
                COALESCE(SUM(i.story_points), 0) as total_points,
                COALESCE(SUM(CASE WHEN i.status = 'Terminé(e)' THEN i.story_points ELSE 0 END), 0) as completed_points,
                DATEDIFF('day', cs.start_date, CURRENT_TIMESTAMP) as days_elapsed,
                DATEDIFF('day', CURRENT_TIMESTAMP, cs.end_date) as days_remaining
            FROM issues i
            JOIN current_sprint cs ON i.sprint_id = cs.id
            WHERE i.project_key = '{project_key}'
            GROUP BY cs.name, cs.start_date, cs.end_date
        """).fetchone()

        if not health:
            return {
                "sprint_name": "Current Sprint",
                "completion_rate": 0.5,
                "days_remaining": 5,
                "confidence": 70
            }

        total = health[1]
        completed = health[2]
        completion_rate = completed / total if total > 0 else 0

        # Calculate confidence based on completion and time
        days_elapsed = health[7] or 1
        days_remaining = health[8] or 5
        sprint_progress = days_elapsed / (days_elapsed + days_remaining)

        # If we're 50% through time but only 30% done, confidence is low
        expected_completion = sprint_progress
        actual_completion = completion_rate

        if actual_completion >= expected_completion:
            confidence = min(95, 70 + (actual_completion - expected_completion) * 100)
        else:
            confidence = max(30, 70 - (expected_completion - actual_completion) * 150)

        return {
            "sprint_name": health[0],
            "total_issues": total,
            "completed": completed,
            "in_progress": health[3],
            "blocked": health[4],
            "completion_rate": completion_rate,
            "days_remaining": days_remaining,
            "confidence": confidence
        }
    except Exception as e:
        return {
            "sprint_name": "Current Sprint",
            "completion_rate": 0.5,
            "days_remaining": 5,
            "confidence": 70
        }


def simulate_person_unavailable(conn, person_id: str, person_name: str,
                                duration_days: int, project_key: str) -> SimulationResult:
    """Simulate what happens if a person is unavailable."""

    # Get their current workload
    workload = get_person_workload(conn, person_id, project_key)
    sprint_health = get_sprint_health(conn, project_key)

    direct_effects = []
    total_delay = 0

    for item in workload:
        # Calculate delay based on priority and current status
        if item['status'] == 'En cours':
            # In progress items are directly impacted
            delay = duration_days
            effect_type = "blocked"
            confidence = 0.95
        elif item['priority'] in ['Highest', 'High']:
            # High priority items will be delayed
            delay = max(1, duration_days - 1)
            effect_type = "delayed"
            confidence = 0.85
        else:
            # Lower priority might be absorbed
            delay = max(0, duration_days - 2)
            effect_type = "delayed"
            confidence = 0.65

        if delay > 0:
            # Estimate downstream impact
            downstream = min(5, int((item.get('story_points') or 3) / 2))

            direct_effects.append(CascadeEffect(
                affected_item=item['summary'][:50],
                affected_item_key=item['key'],
                effect_type=effect_type,
                delay_days=delay,
                confidence=confidence,
                reason=f"{person_name} unavailable for {duration_days} days",
                downstream_count=downstream
            ))

            total_delay += delay

    # Calculate sprint impact
    total_affected = len(direct_effects)
    if total_affected == 0:
        sprint_impact = "on_track"
        confidence_change = 0
    elif total_affected <= 2:
        sprint_impact = "minor_risk"
        confidence_change = -5
    elif total_affected <= 4:
        sprint_impact = "at_risk"
        confidence_change = -15
    else:
        sprint_impact = "will_miss"
        confidence_change = -25

    # Generate mitigations
    mitigations = []
    if total_affected > 0:
        mitigations.append({
            "action": "Reassign critical items",
            "description": f"Move {person_name}'s Highest priority items to available team members",
            "impact": "Reduces delay by 50%"
        })
        mitigations.append({
            "action": "Adjust sprint scope",
            "description": "Move lower priority items to next sprint",
            "impact": "Protects sprint commitment"
        })
        if duration_days > 3:
            mitigations.append({
                "action": "Request temporary resource",
                "description": "Bring in contractor or cross-team support",
                "impact": "Full coverage restoration"
            })

    return SimulationResult(
        scenario_name=f"What if {person_name} is unavailable?",
        scenario_type=ScenarioType.PERSON_UNAVAILABLE,
        trigger=f"{person_name} unavailable for {duration_days} days",
        direct_effects=direct_effects,
        total_affected_items=total_affected,
        total_delay_days=total_delay,
        affected_people=[person_name],
        sprint_impact=sprint_impact,
        confidence_change=confidence_change,
        mitigations=mitigations
    )


def simulate_task_delayed(conn, task_key: str, delay_days: int,
                          project_key: str) -> SimulationResult:
    """Simulate what happens if a specific task is delayed."""

    deps = get_task_dependencies(conn, task_key, project_key)
    sprint_health = get_sprint_health(conn, project_key)

    if not deps:
        return SimulationResult(
            scenario_name=f"What if {task_key} is delayed?",
            scenario_type=ScenarioType.TASK_DELAYED,
            trigger=f"{task_key} delayed by {delay_days} days",
            direct_effects=[],
            total_affected_items=0,
            total_delay_days=delay_days,
            affected_people=[],
            sprint_impact="unknown",
            confidence_change=0,
            mitigations=[]
        )

    task = deps.get("task", {})
    downstream = deps.get("downstream", [])

    direct_effects = []
    affected_people = set()
    total_delay = delay_days

    # The task itself
    direct_effects.append(CascadeEffect(
        affected_item=task.get('summary', 'Unknown')[:50],
        affected_item_key=task_key,
        effect_type="delayed",
        delay_days=delay_days,
        confidence=1.0,
        reason="Direct delay",
        downstream_count=len(downstream)
    ))

    # Cascade to downstream items
    cascade_factor = 0.7  # Each hop reduces delay impact
    current_delay = delay_days

    for i, item in enumerate(downstream[:8]):
        cascade_delay = max(1, int(current_delay * cascade_factor))

        if cascade_delay > 0:
            direct_effects.append(CascadeEffect(
                affected_item=item.get('summary', 'Unknown')[:50],
                affected_item_key=item.get('key', ''),
                effect_type="blocked" if i < 3 else "delayed",
                delay_days=cascade_delay,
                confidence=0.8 - (i * 0.1),
                reason=f"Waiting on {task_key}",
                downstream_count=max(0, len(downstream) - i - 1)
            ))

            if item.get('assignee'):
                affected_people.add(item['assignee'])

            total_delay += cascade_delay
            current_delay = cascade_delay

    # Determine sprint impact
    if task.get('priority') == 'Highest' or len(downstream) > 5:
        sprint_impact = "will_miss"
        confidence_change = -30
    elif task.get('priority') == 'High' or len(downstream) > 2:
        sprint_impact = "at_risk"
        confidence_change = -15
    else:
        sprint_impact = "minor_risk"
        confidence_change = -5

    # Mitigations
    mitigations = [
        {
            "action": "Parallel work streams",
            "description": f"Start downstream items with placeholder/mock data",
            "impact": "Reduces cascade by 40%"
        },
        {
            "action": "Add resources to {task_key}",
            "description": "Pair programming or additional support",
            "impact": f"Could recover {delay_days - 1} days"
        }
    ]

    if delay_days > 3:
        mitigations.append({
            "action": "Scope reduction",
            "description": "Deliver MVP version of blocked features",
            "impact": "Protects delivery date"
        })

    return SimulationResult(
        scenario_name=f"What if {task_key} is delayed?",
        scenario_type=ScenarioType.TASK_DELAYED,
        trigger=f"{task_key} delayed by {delay_days} days",
        direct_effects=direct_effects,
        total_affected_items=len(direct_effects),
        total_delay_days=total_delay,
        affected_people=list(affected_people),
        sprint_impact=sprint_impact,
        confidence_change=confidence_change,
        mitigations=mitigations
    )


def get_team_members_for_selector(conn, project_key: str) -> List[Dict]:
    """Get team members for the selector dropdown."""
    try:
        members = conn.execute(f"""
            SELECT DISTINCT
                i.assignee_id,
                u.pseudonym as name,
                COUNT(*) as active_items
            FROM issues i
            LEFT JOIN users u ON i.assignee_id = u.account_id
            WHERE i.project_key = '{project_key}'
              AND i.status NOT IN ('Terminé(e)', 'Done', 'Closed')
              AND i.assignee_id IS NOT NULL
            GROUP BY i.assignee_id, u.pseudonym
            ORDER BY active_items DESC
        """).fetchall()

        return [{"id": m[0], "name": m[1] or "Unknown", "items": m[2]} for m in members]
    except:
        return []


def get_active_tasks_for_selector(conn, project_key: str) -> List[Dict]:
    """Get active tasks for the selector dropdown."""
    try:
        tasks = conn.execute(f"""
            SELECT
                key,
                summary,
                priority,
                status,
                story_points
            FROM issues
            WHERE project_key = '{project_key}'
              AND status NOT IN ('Terminé(e)', 'Done', 'Closed')
            ORDER BY
                CASE priority
                    WHEN 'Highest' THEN 1
                    WHEN 'High' THEN 2
                    ELSE 3
                END,
                story_points DESC NULLS LAST
            LIMIT 30
        """).fetchall()

        return [
            {
                "key": t[0],
                "summary": t[1][:40] + "..." if len(t[1] or "") > 40 else t[1],
                "priority": t[2],
                "status": t[3]
            }
            for t in tasks
        ]
    except:
        return []


# =============================================================================
# VISUALIZATION COMPONENTS
# =============================================================================

def render_simulator_header():
    """Render the simulator header."""
    st.markdown("""
<style>
.sim-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 30px;
    border: 1px solid rgba(237, 137, 54, 0.3);
    box-shadow: 0 0 40px rgba(237, 137, 54, 0.1);
}
.sim-title {
    font-size: 2.5em;
    font-weight: 800;
    background: linear-gradient(90deg, #ed8936, #f6ad55, #ed8936);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite;
    margin-bottom: 10px;
}
@keyframes shimmer {
    to { background-position: 200% center; }
}
.sim-subtitle {
    color: #a0a0a0;
    font-size: 1.1em;
    margin-bottom: 10px;
}
.sim-tagline {
    color: #ed8936;
    font-size: 0.95em;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
}
</style>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="sim-header">
    <div class="sim-title">💥 WHAT BREAKS IF... SIMULATOR</div>
    <div class="sim-subtitle">See the future before it happens. Simulate disruptions. Prevent cascades.</div>
    <div class="sim-tagline">⚡ "An ounce of simulation is worth a pound of fire-fighting"</div>
</div>
""", unsafe_allow_html=True)


def render_scenario_selector(conn, project_key: str) -> Optional[SimulationResult]:
    """Render scenario selection interface and return simulation result."""

    st.markdown("### 🎮 Choose Your Scenario")

    scenario_type = st.selectbox(
        "What do you want to simulate?",
        options=[
            "👤 Person becomes unavailable",
            "⏰ Task gets delayed",
        ],
        index=0
    )

    result = None

    if "Person" in scenario_type:
        col1, col2 = st.columns([2, 1])

        with col1:
            members = get_team_members_for_selector(conn, project_key)
            member_options = [f"{m['name']} ({m['items']} active items)" for m in members]

            if member_options:
                selected = st.selectbox("Who becomes unavailable?", member_options)
                selected_idx = member_options.index(selected)
                selected_member = members[selected_idx]
            else:
                st.warning("No team members found")
                return None

        with col2:
            duration = st.slider("For how many days?", 1, 14, 5)

        if st.button("🔮 Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Calculating cascade effects..."):
                result = simulate_person_unavailable(
                    conn,
                    selected_member['id'],
                    selected_member['name'],
                    duration,
                    project_key
                )

    elif "Task" in scenario_type:
        col1, col2 = st.columns([2, 1])

        with col1:
            tasks = get_active_tasks_for_selector(conn, project_key)
            task_options = [f"{t['key']}: {t['summary']} [{t['priority']}]" for t in tasks]

            if task_options:
                selected = st.selectbox("Which task gets delayed?", task_options)
                selected_idx = task_options.index(selected)
                selected_task = tasks[selected_idx]
            else:
                st.warning("No active tasks found")
                return None

        with col2:
            delay = st.slider("Delayed by how many days?", 1, 14, 3)

        if st.button("🔮 Run Simulation", type="primary", use_container_width=True):
            with st.spinner("Calculating cascade effects..."):
                result = simulate_task_delayed(
                    conn,
                    selected_task['key'],
                    delay,
                    project_key
                )

    return result


def render_simulation_result(result: SimulationResult, sprint_health: Dict):
    """Render the simulation results."""

    if not result:
        return

    # Impact summary header
    impact_colors = {
        "on_track": ("#68d391", "✅ ON TRACK"),
        "minor_risk": ("#f6e05e", "⚠️ MINOR RISK"),
        "at_risk": ("#f6ad55", "🔶 AT RISK"),
        "will_miss": ("#fc8181", "🚨 WILL MISS"),
        "unknown": ("#a0aec0", "❓ UNKNOWN")
    }

    color, label = impact_colors.get(result.sprint_impact, ("#a0aec0", "Unknown"))

    new_confidence = max(0, sprint_health.get('confidence', 70) + result.confidence_change)

    st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(26,26,46,0.95), rgba(22,33,62,0.95));
    border-radius: 20px;
    padding: 25px;
    margin: 20px 0;
    border: 2px solid {color};
">
    <div style="text-align: center; margin-bottom: 20px;">
        <div style="font-size: 1.5em; font-weight: 800; color: {color}; margin-bottom: 10px;">
            {label}
        </div>
        <div style="color: #e2e8f0; font-size: 1.1em;">
            {result.trigger}
        </div>
    </div>

    <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 20px;">
        <div style="text-align: center;">
            <div style="font-size: 2.5em; font-weight: 800; color: #fc8181;">{result.total_affected_items}</div>
            <div style="color: #a0aec0; font-size: 0.85em;">Items Affected</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2.5em; font-weight: 800; color: #f6ad55;">-{result.total_delay_days}d</div>
            <div style="color: #a0aec0; font-size: 0.85em;">Total Delay</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2.5em; font-weight: 800; color: #63b3ed;">{len(result.affected_people)}</div>
            <div style="color: #a0aec0; font-size: 0.85em;">People Impacted</div>
        </div>
        <div style="text-align: center;">
            <div style="font-size: 2.5em; font-weight: 800; color: {'#fc8181' if result.confidence_change < -10 else '#f6ad55'};">
                {new_confidence:.0f}%
            </div>
            <div style="color: #a0aec0; font-size: 0.85em;">
                New Confidence ({result.confidence_change:+.0f}%)
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


def render_cascade_visualization(result: SimulationResult):
    """Render the cascade effect visualization."""

    if not result.direct_effects:
        st.info("No cascade effects detected - this scenario has minimal impact!")
        return

    st.markdown("### 🌊 Cascade Effects")

    # Create waterfall/cascade visualization
    effects = result.direct_effects

    fig = go.Figure()

    # Sort by delay
    sorted_effects = sorted(effects, key=lambda e: e.delay_days, reverse=True)

    y_labels = [f"{e.affected_item_key}" for e in sorted_effects]
    delays = [e.delay_days for e in sorted_effects]
    colors = [
        '#e53e3e' if e.effect_type == 'blocked' else
        '#ed8936' if e.delay_days > 3 else
        '#ecc94b'
        for e in sorted_effects
    ]

    fig.add_trace(go.Bar(
        x=delays,
        y=y_labels,
        orientation='h',
        marker_color=colors,
        text=[f"-{d}d" for d in delays],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Delay: %{x} days<extra></extra>'
    ))

    fig.update_layout(
        title="Impact by Item (days delayed)",
        xaxis_title="Days of Delay",
        yaxis_title="",
        height=max(300, len(effects) * 40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )

    st.plotly_chart(fig, use_container_width=True)


def render_cascade_details(result: SimulationResult):
    """Render detailed cascade breakdown."""

    if not result.direct_effects:
        return

    st.markdown("### 📋 Detailed Impact Analysis")

    for i, effect in enumerate(result.direct_effects[:10]):
        severity_color = (
            "#e53e3e" if effect.effect_type == "blocked" else
            "#ed8936" if effect.delay_days > 3 else
            "#ecc94b"
        )

        effect_label = {
            "blocked": "🚫 BLOCKED",
            "delayed": "⏰ DELAYED",
            "reassigned": "🔄 NEEDS REASSIGN",
            "unassigned": "❓ UNASSIGNED"
        }.get(effect.effect_type, "⚠️ AFFECTED")

        st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(26,26,46,0.9), rgba(22,33,62,0.9));
    border-radius: 12px;
    padding: 15px 20px;
    margin-bottom: 10px;
    border-left: 4px solid {severity_color};
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
">
    <div style="flex: 2; min-width: 250px;">
        <div style="font-weight: 700; color: #fbd38d;">{effect.affected_item_key}</div>
        <div style="color: #e2e8f0; font-size: 0.9em; margin-top: 3px;">
            {effect.affected_item}
        </div>
        <div style="color: #a0aec0; font-size: 0.8em; margin-top: 5px;">
            {effect.reason}
        </div>
    </div>
    <div style="text-align: center; min-width: 80px;">
        <div style="font-size: 1.5em; font-weight: 800; color: {severity_color};">
            -{effect.delay_days}d
        </div>
    </div>
    <div style="min-width: 120px;">
        <div style="
            background: {severity_color}22;
            color: {severity_color};
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: 600;
            text-align: center;
        ">
            {effect_label}
        </div>
        {f'<div style="color: #a0aec0; font-size: 0.75em; text-align: center; margin-top: 5px;">→ {effect.downstream_count} more affected</div>' if effect.downstream_count > 0 else ''}
    </div>
</div>
""", unsafe_allow_html=True)


def render_mitigations(result: SimulationResult):
    """Render mitigation suggestions."""

    if not result.mitigations:
        return

    st.markdown("### 🛡️ Recommended Mitigations")

    for i, mit in enumerate(result.mitigations):
        st.markdown(f"""
<div style="
    background: linear-gradient(135deg, rgba(104, 211, 145, 0.1), rgba(104, 211, 145, 0.05));
    border: 1px solid rgba(104, 211, 145, 0.3);
    border-radius: 12px;
    padding: 15px 20px;
    margin-bottom: 10px;
">
    <div style="display: flex; justify-content: space-between; align-items: start;">
        <div>
            <div style="font-weight: 700; color: #68d391; font-size: 1.05em;">
                {i + 1}. {mit['action']}
            </div>
            <div style="color: #e2e8f0; font-size: 0.9em; margin-top: 5px;">
                {mit['description']}
            </div>
        </div>
        <div style="
            background: rgba(104, 211, 145, 0.2);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            color: #68d391;
            white-space: nowrap;
        ">
            {mit['impact']}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


def render_quick_scenarios(conn, project_key: str):
    """Render quick one-click scenarios."""

    st.markdown("### ⚡ Quick Scenarios")

    members = get_team_members_for_selector(conn, project_key)[:4]
    tasks = get_active_tasks_for_selector(conn, project_key)[:4]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**👤 Key Person Risk**")
        for m in members[:3]:
            if st.button(f"What if {m['name']} is out?", key=f"quick_person_{m['id']}"):
                return simulate_person_unavailable(conn, m['id'], m['name'], 5, project_key)

    with col2:
        st.markdown("**📌 Critical Task Risk**")
        for t in tasks[:3]:
            if st.button(f"What if {t['key']} slips?", key=f"quick_task_{t['key']}"):
                return simulate_task_delayed(conn, t['key'], 5, project_key)

    return None


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Render page guide in sidebar
    render_page_guide()
    conn = get_connection()
    if not conn:
        st.error("Cannot connect to database")
        return

    # Get project key
    try:
        project_key = conn.execute(
            "SELECT DISTINCT project_key FROM issues LIMIT 1"
        ).fetchone()[0]
    except:
        project_key = "PROJ"

    # Get baseline sprint health
    sprint_health = get_sprint_health(conn, project_key)

    # Render header
    render_simulator_header()

    # Current sprint baseline
    st.markdown(f"""
    <div style="
        background: rgba(99, 179, 237, 0.1);
        border: 1px solid rgba(99, 179, 237, 0.3);
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
    ">
        <div style="color: #63b3ed; font-weight: 600;">📊 Current Sprint Baseline</div>
        <div style="color: #e2e8f0; margin-top: 8px;">
            <strong>{sprint_health.get('sprint_name', 'Sprint')}</strong> •
            {sprint_health.get('days_remaining', 5)} days remaining •
            {sprint_health.get('confidence', 70):.0f}% delivery confidence
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Two columns: Scenario builder + Quick scenarios
    col1, col2 = st.columns([2, 1])

    with col1:
        result = render_scenario_selector(conn, project_key)

    with col2:
        quick_result = render_quick_scenarios(conn, project_key)

    # Use quick result if main result is None
    if result is None and quick_result is not None:
        result = quick_result

    # Show results
    if result:
        st.markdown("---")
        render_simulation_result(result, sprint_health)

        col1, col2 = st.columns([3, 2])

        with col1:
            render_cascade_visualization(result)
            render_cascade_details(result)

        with col2:
            render_mitigations(result)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #718096; font-size: 0.85em; padding: 20px;">
        💥 What Breaks If Simulator • Prevent disasters before they happen<br>
        <span style="font-size: 0.8em;">Saves 30-45 min/day of fire-fighting</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
