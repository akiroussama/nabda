"""
Autonomous Agents - The Self-Driving Organization

This revolutionary module implements autonomous work agents that manage themselves.
Work figures out the how. Humans set the intent and intervene only for true judgment calls.

The Paradigm Shift:
- Before: Humans manage work. Tools help them do it better.
- After: Work manages itself. Humans set intent and make judgment calls.

Core Components:
1. Intent Console - Set high-level goals, agents figure out execution
2. Intervention Inbox - Only see moments requiring human judgment
3. Agent Observatory - Watch agents work in real-time
4. Policy Layer - Set rules, not tasks
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Tuple
import sys
import json
import hashlib

# Path setup
root_dir = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(root_dir))

# Import work agents engine
from src.features.work_agents import (
    WorkAgentEngine, WorkAgent, Intent, Policy, Intervention,
    AgentState, AgentType, HealthStatus, AutonomyLevel,
    EscalationReason, PolicyType, Evidence,
    generate_demo_work_agents
)

# Import LLM client
try:
    from src.intelligence.llm_client import create_llm_client, GeminiClient
    from config.settings import get_settings
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Import page guide component
from src.dashboard.components import render_page_guide

# Page configuration
st.set_page_config(
    page_title="Autonomous Agents",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Premium CSS - Self-Driving Organization Theme
st.markdown("""
<style>
    /* Base theme - Light with futuristic accents */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    }

    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Revolution Header */
    .revolution-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }

    .revolution-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        pointer-events: none;
    }

    .revolution-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
        z-index: 1;
    }

    .revolution-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        position: relative;
        z-index: 1;
    }

    /* Status Banner */
    .status-banner {
        display: flex;
        gap: 2rem;
        margin-top: 1.5rem;
        position: relative;
        z-index: 1;
    }

    .status-item {
        text-align: center;
    }

    .status-value {
        font-size: 2rem;
        font-weight: 700;
        color: #22c55e;
    }

    .status-value.warning { color: #f59e0b; }
    .status-value.critical { color: #ef4444; }

    .status-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #64748b;
    }

    /* Section Cards */
    .section-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }

    .section-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }

    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
    }

    .section-subtitle {
        font-size: 0.875rem;
        color: #64748b;
    }

    /* Intent Console */
    .intent-console {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border-radius: 16px;
        padding: 2rem;
        color: white;
    }

    .intent-input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        font-size: 1rem;
        width: 100%;
        margin-bottom: 1rem;
    }

    .intent-input::placeholder {
        color: rgba(255, 255, 255, 0.5);
    }

    /* Intervention Cards */
    .intervention-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        border-left: 4px solid #f59e0b;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }

    .intervention-card:hover {
        transform: translateX(4px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    .intervention-card.critical {
        border-left-color: #ef4444;
        animation: pulse-critical 2s infinite;
    }

    .intervention-card.high {
        border-left-color: #f59e0b;
    }

    .intervention-card.medium {
        border-left-color: #3b82f6;
    }

    .intervention-card.low {
        border-left-color: #22c55e;
    }

    @keyframes pulse-critical {
        0%, 100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
        50% { box-shadow: 0 0 0 8px rgba(239, 68, 68, 0); }
    }

    .intervention-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.75rem;
    }

    .intervention-urgency {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        background: #fef3c7;
        color: #92400e;
    }

    .intervention-urgency.critical { background: #fee2e2; color: #991b1b; }
    .intervention-urgency.high { background: #fef3c7; color: #92400e; }
    .intervention-urgency.medium { background: #dbeafe; color: #1e40af; }
    .intervention-urgency.low { background: #dcfce7; color: #166534; }

    .intervention-title {
        font-weight: 600;
        color: #1e293b;
        font-size: 1rem;
        flex: 1;
        margin-right: 1rem;
    }

    .intervention-context {
        font-size: 0.875rem;
        color: #64748b;
        margin-bottom: 0.75rem;
    }

    .intervention-actions {
        display: flex;
        gap: 0.5rem;
    }

    /* Agent Cards */
    .agent-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
    }

    .agent-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.1);
    }

    .agent-card.executing {
        border-left: 3px solid #22c55e;
    }

    .agent-card.planning {
        border-left: 3px solid #3b82f6;
    }

    .agent-card.negotiating {
        border-left: 3px solid #8b5cf6;
    }

    .agent-card.blocked {
        border-left: 3px solid #ef4444;
    }

    .agent-card.awaiting_human {
        border-left: 3px solid #f59e0b;
        animation: pulse-warning 2s infinite;
    }

    @keyframes pulse-warning {
        0%, 100% { box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.3); }
        50% { box-shadow: 0 0 0 4px rgba(245, 158, 11, 0); }
    }

    .agent-name {
        font-weight: 600;
        color: #1e293b;
        font-size: 0.95rem;
    }

    .agent-intent {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.25rem;
    }

    .agent-progress {
        margin-top: 0.75rem;
    }

    .progress-bar {
        height: 6px;
        background: #e2e8f0;
        border-radius: 3px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #3b82f6, #22c55e);
        border-radius: 3px;
        transition: width 0.5s ease;
    }

    .agent-meta {
        display: flex;
        justify-content: space-between;
        margin-top: 0.5rem;
        font-size: 0.75rem;
        color: #94a3b8;
    }

    /* Policy Cards */
    .policy-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        border: 1px solid #e2e8f0;
    }

    .policy-card.enabled {
        background: white;
        border-color: #22c55e;
    }

    .policy-name {
        font-weight: 600;
        color: #1e293b;
        font-size: 0.9rem;
    }

    .policy-rule {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.25rem;
        font-style: italic;
    }

    .policy-type {
        display: inline-block;
        font-size: 0.65rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        background: #e0e7ff;
        color: #3730a3;
        margin-top: 0.5rem;
    }

    /* Metrics Grid */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid #e2e8f0;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }

    .metric-value.green { color: #22c55e; }
    .metric-value.amber { color: #f59e0b; }
    .metric-value.red { color: #ef4444; }
    .metric-value.blue { color: #3b82f6; }

    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Empty State */
    .empty-inbox {
        text-align: center;
        padding: 3rem;
        color: #64748b;
    }

    .empty-inbox-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }

    .empty-inbox-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #22c55e;
    }

    .empty-inbox-subtitle {
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: #f1f5f9;
        padding: 0.25rem;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background: white;
        border-radius: 8px;
    }

    /* Network Graph Container */
    .network-container {
        background: #0f172a;
        border-radius: 12px;
        padding: 1rem;
        min-height: 400px;
    }

    /* Action Buttons */
    .action-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        border: none;
    }

    .action-btn.primary {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
    }

    .action-btn.secondary {
        background: #f1f5f9;
        color: #475569;
    }

    .action-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* Management Tax Savings */
    .savings-card {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }

    .savings-value {
        font-size: 3rem;
        font-weight: 800;
    }

    .savings-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* Autonomy Level Badge */
    .autonomy-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        font-size: 0.7rem;
        font-weight: 600;
    }

    .autonomy-badge.observer {
        background: #dbeafe;
        color: #1e40af;
    }

    .autonomy-badge.guardrailed {
        background: #fef3c7;
        color: #92400e;
    }

    .autonomy-badge.autonomous {
        background: #dcfce7;
        color: #166534;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE & INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state with demo data."""
    if 'work_agent_engine' not in st.session_state:
        # Initialize LLM client if available
        llm_client = None
        if LLM_AVAILABLE:
            try:
                settings = get_settings()
                api_key = settings.llm.google_api_key if hasattr(settings, 'llm') else None
                if api_key:
                    llm_client = create_llm_client(api_key=api_key)
            except Exception as e:
                st.warning(f"LLM client not available: {e}")

        engine = WorkAgentEngine(llm_client=llm_client)
        generate_demo_work_agents(engine)
        st.session_state.work_agent_engine = engine

    if 'selected_view' not in st.session_state:
        st.session_state.selected_view = "observatory"

    if 'show_intent_form' not in st.session_state:
        st.session_state.show_intent_form = False

    if 'autonomy_level' not in st.session_state:
        st.session_state.autonomy_level = AutonomyLevel.GUARDRAILED


def get_engine() -> WorkAgentEngine:
    """Get the work agent engine from session state."""
    return st.session_state.work_agent_engine


def simulate_agent_tick():
    """Simulate one tick of agent activity - makes the system feel alive."""
    import random

    engine = get_engine()

    for agent in engine.agents.values():
        if agent.state == AgentState.EXECUTING:
            # Simulate progress
            progress_delta = random.uniform(0.01, 0.05)
            new_progress = min(1.0, agent.progress + progress_delta)
            agent.progress = new_progress

            # Small chance of completion
            if new_progress >= 1.0:
                agent.state = AgentState.COMPLETED
                agent.actual_completion = datetime.now()

            # Small chance of needing intervention
            if random.random() < 0.02:
                reasons = [
                    EscalationReason.SCOPE_AMBIGUITY,
                    EscalationReason.RESOURCE_CONFLICT,
                    EscalationReason.QUALITY_GATE_FAILURE
                ]
                agent.request_intervention(
                    reason=random.choice(reasons),
                    summary="Agent detected an issue requiring human judgment",
                    context="Automated detection during execution",
                    evidence=[Evidence(
                        type="observation",
                        source="agent_self_monitor",
                        value="Anomaly detected",
                        confidence=0.75
                    )],
                    options=[
                        {"id": "proceed", "label": "Proceed as planned"},
                        {"id": "adjust", "label": "Adjust approach"},
                        {"id": "pause", "label": "Pause for review"}
                    ],
                    urgency=random.choice(["low", "medium", "high"])
                )

        elif agent.state == AgentState.PLANNING:
            # Chance to start executing
            if random.random() < 0.1:
                agent.state = AgentState.EXECUTING
                agent.actual_start = datetime.now()

        elif agent.state == AgentState.NEGOTIATING:
            # Chance to resolve negotiation
            if random.random() < 0.15:
                agent.state = AgentState.EXECUTING
                agent.metrics.negotiations_successful += 1


# =============================================================================
# HEADER COMPONENT
# =============================================================================

def render_revolution_header():
    """Render the revolutionary header with system status."""
    engine = get_engine()
    health = engine.get_system_health()

    # Determine status colors
    status_class = ""
    if health["status"] == "warning":
        status_class = "warning"
    elif health["status"] == "critical":
        status_class = "critical"

    st.markdown(f"""
    <div class="revolution-header">
        <div class="revolution-title">The Self-Driving Organization</div>
        <div class="revolution-subtitle">
            Work that manages itself, so humans can do what only humans can do.
        </div>
        <div class="status-banner">
            <div class="status-item">
                <div class="status-value {'green' if health['pending_interventions'] == 0 else 'warning' if health['pending_interventions'] < 3 else 'critical'}">
                    {health['pending_interventions']}
                </div>
                <div class="status-label">Interventions Needed</div>
            </div>
            <div class="status-item">
                <div class="status-value">{health['total_agents']}</div>
                <div class="status-label">Active Agents</div>
            </div>
            <div class="status-item">
                <div class="status-value" style="color: #22c55e;">{health['autonomy_rate']:.0%}</div>
                <div class="status-label">Autonomy Rate</div>
            </div>
            <div class="status-item">
                <div class="status-value" style="color: #3b82f6;">{health['active_intents']}</div>
                <div class="status-label">Active Intents</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# INTENT CONSOLE
# =============================================================================

def render_intent_console():
    """Render the Intent Console - where humans set high-level goals."""
    engine = get_engine()

    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <div class="section-icon" style="background: linear-gradient(135deg, #4f46e5, #7c3aed);">
                🎯
            </div>
            <div>
                <div class="section-title">Intent Console</div>
                <div class="section-subtitle">Set outcomes, not tasks. Agents figure out the how.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # New Intent Form
    with st.expander("➕ Create New Intent", expanded=st.session_state.show_intent_form):
        col1, col2 = st.columns([2, 1])

        with col1:
            intent_title = st.text_input(
                "What do you want to achieve?",
                placeholder="e.g., 'Launch new onboarding flow by Q1'",
                key="intent_title"
            )

            intent_description = st.text_area(
                "Describe the desired outcome",
                placeholder="Reduce time-to-value by 40% through a streamlined onboarding experience that guides users to their first success moment within 5 minutes.",
                height=100,
                key="intent_description"
            )

        with col2:
            priority = st.slider("Priority", 1, 100, 50, key="intent_priority")
            deadline = st.date_input(
                "Target Deadline",
                value=date.today() + timedelta(days=30),
                key="intent_deadline"
            )

            st.markdown("**Autonomy Level**")
            autonomy = st.radio(
                "How much should agents decide alone?",
                ["Observer", "Guardrailed", "Autonomous"],
                index=1,
                key="autonomy_selection",
                help="Observer: Agents suggest, you decide\nGuardrailed: Agents act within strict rules\nAutonomous: Minimal oversight"
            )

        if st.button("🚀 Activate Intent", type="primary", use_container_width=True):
            if intent_title:
                intent = engine.create_intent(
                    title=intent_title,
                    description=intent_description,
                    owner_id="current_user",
                    outcome=intent_description,
                    deadline=deadline,
                    priority=priority
                )
                engine.decompose_intent(intent.id)
                engine.activate_intent(intent.id)
                st.success(f"✨ Intent activated! {len(intent.agent_ids)} agents created and working.")
                st.rerun()
            else:
                st.warning("Please provide a title for your intent.")

    # Active Intents
    st.markdown("### Active Intents")

    active_intents = [i for i in engine.intents.values() if i.status == "active"]

    if not active_intents:
        st.info("No active intents. Create one above to get started!")
    else:
        for intent in active_intents:
            # Calculate overall progress
            agent_progress = [
                engine.agents[aid].progress
                for aid in intent.agent_ids
                if aid in engine.agents
            ]
            overall_progress = sum(agent_progress) / len(agent_progress) if agent_progress else 0

            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    st.markdown(f"**{intent.title}**")
                    st.caption(intent.outcome[:100] + "..." if len(intent.outcome) > 100 else intent.outcome)

                with col2:
                    st.metric("Progress", f"{overall_progress:.0%}")

                with col3:
                    st.metric("Agents", len(intent.agent_ids))

                # Progress bar
                st.progress(overall_progress)
                st.divider()


# =============================================================================
# INTERVENTION INBOX
# =============================================================================

def render_intervention_inbox():
    """Render the Intervention Inbox - only moments requiring human judgment."""
    engine = get_engine()
    interventions = engine.get_pending_interventions()

    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <div class="section-icon" style="background: linear-gradient(135deg, #f59e0b, #d97706);">
                📥
            </div>
            <div>
                <div class="section-title">Intervention Inbox</div>
                <div class="section-subtitle">Only moments requiring your judgment. Goal: Keep this empty.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not interventions:
        st.markdown("""
        <div class="empty-inbox">
            <div class="empty-inbox-icon">✨</div>
            <div class="empty-inbox-title">Inbox Zero!</div>
            <div class="empty-inbox-subtitle">
                All agents are healthy. No decisions needed.<br>
                You can spend your time on strategy instead of status.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Show time saved
        tax = engine.get_management_tax_calculation()
        st.markdown(f"""
        <div class="savings-card" style="margin-top: 1rem;">
            <div class="savings-value">{tax['recoverable_hours']:.0f}h</div>
            <div class="savings-label">Management time saved this week</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"**{len(interventions)} decisions need your attention**")

        for intervention in interventions:
            agent = engine.agents.get(intervention.agent_id)
            agent_name = agent.name if agent else "Unknown Agent"

            urgency_class = intervention.urgency
            reason_emoji = {
                EscalationReason.POLICY_VIOLATION: "⚠️",
                EscalationReason.RESOURCE_CONFLICT: "👥",
                EscalationReason.DEADLINE_RISK: "⏰",
                EscalationReason.SCOPE_AMBIGUITY: "❓",
                EscalationReason.STAKEHOLDER_DECISION: "🎯",
                EscalationReason.ETHICAL_JUDGMENT: "⚖️",
                EscalationReason.STRATEGIC_PIVOT: "🔄",
                EscalationReason.CONFLICT_RESOLUTION: "🤝",
                EscalationReason.QUALITY_GATE_FAILURE: "🔍",
                EscalationReason.EXTERNAL_DEPENDENCY: "🔗",
            }.get(intervention.reason, "📋")

            st.markdown(f"""
            <div class="intervention-card {urgency_class}">
                <div class="intervention-header">
                    <div class="intervention-title">{reason_emoji} {intervention.summary}</div>
                    <span class="intervention-urgency {urgency_class}">{intervention.urgency}</span>
                </div>
                <div class="intervention-context">
                    <strong>Agent:</strong> {agent_name}<br>
                    {intervention.context}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Action buttons
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                if st.button("✅ Approve", key=f"approve_{intervention.id}", type="primary"):
                    if agent:
                        agent.resolve_intervention(
                            intervention.id,
                            "approved",
                            "current_user"
                        )
                        st.success("Decision recorded. Agent continuing.")
                        st.rerun()

            with col2:
                if st.button("🔄 Override", key=f"override_{intervention.id}"):
                    if agent:
                        agent.resolve_intervention(
                            intervention.id,
                            "overridden",
                            "current_user"
                        )
                        st.info("Override recorded. Agent will adapt.")
                        st.rerun()

            with col3:
                if st.button("⏸️ Defer", key=f"defer_{intervention.id}"):
                    st.warning("Decision deferred. Agent will wait.")

            st.divider()


# =============================================================================
# AGENT OBSERVATORY
# =============================================================================

def render_agent_observatory():
    """Render the Agent Observatory - watch agents work in real-time."""
    engine = get_engine()

    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <div class="section-icon" style="background: linear-gradient(135deg, #06b6d4, #0891b2);">
                🔭
            </div>
            <div>
                <div class="section-title">Agent Observatory</div>
                <div class="section-subtitle">Watch your agents work. Intervene if you want, but the system discourages it.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Agent state summary
    state_counts = engine._count_agents_by_state()

    cols = st.columns(6)
    state_info = [
        ("Executing", "executing", "🟢"),
        ("Planning", "planning", "🔵"),
        ("Negotiating", "negotiating", "🟣"),
        ("Blocked", "blocked", "🔴"),
        ("Awaiting Human", "awaiting_human", "🟡"),
        ("Completed", "completed", "✅"),
    ]

    for col, (label, state, emoji) in zip(cols, state_info):
        count = state_counts.get(state, 0)
        col.metric(f"{emoji} {label}", count)

    st.divider()

    # Agent Network Visualization
    st.markdown("### Agent Network")

    # Create network graph
    fig = create_agent_network_graph(engine)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Agent Cards
    st.markdown("### All Agents")

    # Filter by state
    filter_state = st.selectbox(
        "Filter by state",
        ["All"] + [s.value for s in AgentState],
        key="agent_filter"
    )

    agents = list(engine.agents.values())
    if filter_state != "All":
        agents = [a for a in agents if a.state.value == filter_state]

    # Sort by progress
    agents.sort(key=lambda a: (-a.progress if a.state != AgentState.COMPLETED else 1))

    for agent in agents:
        state_class = agent.state.value.replace("_", "-")
        autonomy_class = agent.autonomy_level.name.lower()

        st.markdown(f"""
        <div class="agent-card {agent.state.value}">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div class="agent-name">{agent.name}</div>
                    <div class="agent-intent">{agent.intent[:80]}{'...' if len(agent.intent) > 80 else ''}</div>
                </div>
                <span class="autonomy-badge {autonomy_class}">
                    {'👁️ Observer' if agent.autonomy_level == AutonomyLevel.OBSERVER else '🛡️ Guardrailed' if agent.autonomy_level == AutonomyLevel.GUARDRAILED else '🚀 Autonomous'}
                </span>
            </div>
            <div class="agent-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {agent.progress * 100}%;"></div>
                </div>
            </div>
            <div class="agent-meta">
                <span>{agent.agent_type.value.replace('_', ' ').title()}</span>
                <span>{agent.state.value.replace('_', ' ').title()}</span>
                <span>{agent.progress:.0%} complete</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Expandable details
        with st.expander(f"View {agent.name} Details"):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Success Criteria:**")
                for criterion in agent.success_criteria:
                    st.markdown(f"- {criterion}")

                st.markdown("**Metrics:**")
                st.markdown(f"- Decisions made: {agent.metrics.decisions_made}")
                st.markdown(f"- Autonomy rate: {agent.metrics.autonomy_rate:.0%}")
                st.markdown(f"- Negotiations: {agent.metrics.negotiations_initiated} ({agent.metrics.negotiation_success_rate:.0%} success)")

            with col2:
                st.markdown("**Recent Actions:**")
                for action in agent.actions[-5:]:
                    st.markdown(f"- {action.timestamp.strftime('%m/%d %H:%M')}: {action.description[:50]}")

                if agent.pending_interventions:
                    st.warning(f"⚠️ {len(agent.pending_interventions)} pending intervention(s)")


def create_agent_network_graph(engine: WorkAgentEngine) -> go.Figure:
    """Create a network visualization of agents."""
    agents = list(engine.agents.values())

    if not agents:
        fig = go.Figure()
        fig.add_annotation(
            text="No agents active",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="#64748b")
        )
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#0f172a',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    # Position nodes in a circle
    n = len(agents)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    radius = 2

    x_nodes = radius * np.cos(angles)
    y_nodes = radius * np.sin(angles)

    # Node colors by state
    state_colors = {
        AgentState.DORMANT: "#64748b",
        AgentState.PLANNING: "#3b82f6",
        AgentState.NEGOTIATING: "#8b5cf6",
        AgentState.EXECUTING: "#22c55e",
        AgentState.BLOCKED: "#ef4444",
        AgentState.AWAITING_HUMAN: "#f59e0b",
        AgentState.ADAPTING: "#06b6d4",
        AgentState.COMPLETED: "#10b981",
        AgentState.FAILED: "#991b1b",
    }

    node_colors = [state_colors.get(a.state, "#64748b") for a in agents]
    node_sizes = [30 + a.progress * 20 for a in agents]

    # Create edges for dependencies
    edge_x = []
    edge_y = []
    agent_id_to_idx = {a.id: i for i, a in enumerate(agents)}

    for i, agent in enumerate(agents):
        for dep_id in agent.dependency_agent_ids:
            if dep_id in agent_id_to_idx:
                j = agent_id_to_idx[dep_id]
                edge_x.extend([x_nodes[j], x_nodes[i], None])
                edge_y.extend([y_nodes[j], y_nodes[i], None])

    # Create figure
    fig = go.Figure()

    # Add edges
    if edge_x:
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=1, color='rgba(148, 163, 184, 0.3)'),
            hoverinfo='none'
        ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=x_nodes, y=y_nodes,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white'),
            opacity=0.9
        ),
        text=[a.name[:15] + "..." if len(a.name) > 15 else a.name for a in agents],
        textposition="top center",
        textfont=dict(color='white', size=10),
        hovertemplate="<b>%{customdata[0]}</b><br>" +
                      "State: %{customdata[1]}<br>" +
                      "Progress: %{customdata[2]:.0%}<br>" +
                      "<extra></extra>",
        customdata=[(a.name, a.state.value, a.progress) for a in agents]
    ))

    fig.update_layout(
        height=400,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='#0f172a',
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False, range=[-3, 3]),
        yaxis=dict(visible=False, range=[-3, 3]),
    )

    return fig


# =============================================================================
# POLICY LAYER
# =============================================================================

def render_policy_layer():
    """Render the Policy Layer - set rules, not tasks."""
    engine = get_engine()

    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <div class="section-icon" style="background: linear-gradient(135deg, #ec4899, #be185d);">
                📜
            </div>
            <div>
                <div class="section-title">Policy Layer</div>
                <div class="section-subtitle">Set rules, not tasks. Agents follow policies autonomously.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Add new policy
    with st.expander("➕ Add Custom Policy"):
        col1, col2 = st.columns([2, 1])

        with col1:
            policy_name = st.text_input("Policy Name", placeholder="e.g., 'Friday Release Freeze'")
            policy_rule = st.text_area(
                "Rule (natural language)",
                placeholder="No deployments allowed on Fridays after 2 PM",
                height=80
            )

        with col2:
            policy_type = st.selectbox(
                "Type",
                [t.value for t in PolicyType],
                key="new_policy_type"
            )
            policy_priority = st.slider("Priority", 1, 100, 50)

        if st.button("Add Policy", type="primary"):
            if policy_name and policy_rule:
                policy_id = f"policy_{hashlib.sha256(policy_name.encode()).hexdigest()[:8]}"
                new_policy = Policy(
                    id=policy_id,
                    name=policy_name,
                    type=PolicyType(policy_type),
                    description=policy_rule,
                    rule=policy_rule,
                    parameters={},
                    priority=policy_priority
                )
                engine.policies[policy_id] = new_policy
                st.success(f"Policy '{policy_name}' added!")
                st.rerun()

    # Active policies
    st.markdown("### Active Policies")

    # Group by type
    policies_by_type: Dict[PolicyType, List[Policy]] = {}
    for policy in engine.policies.values():
        if policy.type not in policies_by_type:
            policies_by_type[policy.type] = []
        policies_by_type[policy.type].append(policy)

    for policy_type, policies in policies_by_type.items():
        st.markdown(f"#### {policy_type.value.title()} Policies")

        for policy in sorted(policies, key=lambda p: -p.priority):
            enabled_class = "enabled" if policy.enabled else ""

            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"""
                <div class="policy-card {enabled_class}">
                    <div class="policy-name">{policy.name}</div>
                    <div class="policy-rule">"{policy.rule}"</div>
                    <span class="policy-type">Priority: {policy.priority}</span>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                toggle = st.checkbox(
                    "Enabled",
                    value=policy.enabled,
                    key=f"policy_toggle_{policy.id}"
                )
                if toggle != policy.enabled:
                    policy.enabled = toggle
                    st.rerun()


# =============================================================================
# MANAGEMENT TAX CALCULATOR
# =============================================================================

def render_management_tax():
    """Render the Management Tax calculation."""
    engine = get_engine()
    tax = engine.get_management_tax_calculation()

    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <div class="section-icon" style="background: linear-gradient(135deg, #22c55e, #16a34a);">
                💰
            </div>
            <div>
                <div class="section-title">Management Tax Calculator</div>
                <div class="section-subtitle">See how much time autonomous agents are saving you.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics
    cols = st.columns(4)

    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value red">{tax['total_management_hours']:.1f}h</div>
            <div class="metric-label">Weekly Management Tax</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value green">{tax['recoverable_hours']:.1f}h</div>
            <div class="metric-label">Hours Recovered</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value blue">{tax['recovery_percentage']:.0%}</div>
            <div class="metric-label">Time Reclaimed</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[3]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #8b5cf6;">{tax['capacity_multiplier']:.1f}x</div>
            <div class="metric-label">Capacity Multiplier</div>
        </div>
        """, unsafe_allow_html=True)

    # Activity breakdown
    st.markdown("### Activity Breakdown")

    df = pd.DataFrame(tax['activities'])
    df['recoverable'] = df['hours_per_week'] * df['automatable']
    df['remaining'] = df['hours_per_week'] - df['recoverable']

    # Stacked bar chart
    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=df['activity'],
        x=df['recoverable'],
        name='Automated',
        orientation='h',
        marker_color='#22c55e',
        text=[f"{x:.1f}h" for x in df['recoverable']],
        textposition='inside'
    ))

    fig.add_trace(go.Bar(
        y=df['activity'],
        x=df['remaining'],
        name='Still Manual',
        orientation='h',
        marker_color='#f87171',
        text=[f"{x:.1f}h" for x in df['remaining']],
        textposition='inside'
    ))

    fig.update_layout(
        barmode='stack',
        height=350,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation='h', y=1.1),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Hours per Week",
        yaxis=dict(autorange="reversed")
    )

    st.plotly_chart(fig, use_container_width=True)

    # The terrifying moment
    st.info(
        "💡 **The Terrifying Moment**: When you go on vacation for 2 weeks, return, "
        "and find that work continued without you. Nothing broke. You realize you were "
        "the bottleneck all along."
    )


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()

    # Render page guide in sidebar
    render_page_guide()

    # Simulate agent activity on each render (makes it feel alive)
    simulate_agent_tick()

    # Header
    render_revolution_header()

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔭 Agent Observatory",
        "📥 Intervention Inbox",
        "🎯 Intent Console",
        "📜 Policy Layer",
        "💰 Management Tax"
    ])

    with tab1:
        render_agent_observatory()

    with tab2:
        render_intervention_inbox()

    with tab3:
        render_intent_console()

    with tab4:
        render_policy_layer()

    with tab5:
        render_management_tax()

    # Sidebar with quick actions
    with st.sidebar:
        st.markdown("## Quick Actions")

        # Auto-refresh toggle
        auto_refresh = st.checkbox("⚡ Auto-refresh (5s)", value=False)
        if auto_refresh:
            import time
            time.sleep(0.1)  # Small delay to prevent immediate rerun
            st.rerun()

        if st.button("🔄 Refresh Agents", use_container_width=True):
            st.rerun()

        if st.button("🧹 Clear All Interventions", use_container_width=True):
            engine = get_engine()
            for agent in engine.agents.values():
                for intervention in agent.interventions:
                    if not intervention.resolved_at:
                        intervention.resolved_at = datetime.now()
                        intervention.resolution = "bulk_cleared"
            st.success("All interventions cleared!")
            st.rerun()

        st.divider()

        st.markdown("## Autonomy Level")
        level = st.radio(
            "System-wide autonomy",
            ["Observer", "Guardrailed", "Autonomous"],
            index=1,
            help="Observer: Agents suggest, you decide\nGuardrailed: Agents act within strict rules\nAutonomous: Minimal oversight"
        )

        st.divider()

        st.markdown("## System Info")
        engine = get_engine()
        health = engine.get_system_health()

        st.metric("Total Agents", health['total_agents'])
        st.metric("Healthy", health['healthy_agents'])
        st.metric("Pending Interventions", health['pending_interventions'])

        st.divider()

        st.markdown("### The Revolution")
        st.caption(
            "You don't manage work. You set intent. "
            "Work figures itself out. You're a passenger who "
            "intervenes only for judgment calls."
        )


if __name__ == "__main__":
    main()
