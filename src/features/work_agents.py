"""
Work Agents Engine - Autonomous Work Management System

This module implements the revolutionary "Self-Driving Organization" concept where
work manages itself through autonomous agents that negotiate, adapt, and escalate
only when hitting true human-judgment boundaries.

Core Concepts:
- Work Agents: Autonomous entities that understand their purpose, constraints, and can make decisions
- Intent Layer: Human sets high-level goals, agents figure out the how
- Negotiation Protocol: Agents negotiate resources and dependencies with each other
- Policy Guardrails: Rules that govern agent autonomy and escalation triggers
- Intervention System: Minimal human involvement only for true judgment calls
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Tuple, Set
import hashlib
import json
import random
from abc import ABC, abstractmethod


# =============================================================================
# ENUMS - Agent States, Types, and Priorities
# =============================================================================

class AgentState(Enum):
    """State of a work agent in the autonomous system."""
    DORMANT = "dormant"              # Created but not activated
    PLANNING = "planning"            # Agent is decomposing work
    NEGOTIATING = "negotiating"      # Negotiating resources/dependencies
    EXECUTING = "executing"          # Active work in progress
    BLOCKED = "blocked"              # Waiting on dependency or resource
    AWAITING_HUMAN = "awaiting_human"  # Needs human judgment
    ADAPTING = "adapting"            # Replanning due to change
    COMPLETED = "completed"          # Successfully finished
    FAILED = "failed"                # Could not complete


class AgentType(Enum):
    """Types of work agents based on work category."""
    FEATURE = "feature"              # New value delivery
    BUG_FIX = "bug_fix"             # Defect resolution
    TECH_DEBT = "tech_debt"         # Technical debt reduction
    MAINTENANCE = "maintenance"      # Operational maintenance
    RESEARCH = "research"           # Investigation/spike
    INTEGRATION = "integration"     # System integration
    MIGRATION = "migration"         # Data/system migration
    COORDINATION = "coordination"   # Meta-agent for orchestration


class EscalationReason(Enum):
    """Reasons an agent escalates to human intervention."""
    POLICY_VIOLATION = "policy_violation"
    RESOURCE_CONFLICT = "resource_conflict"
    DEADLINE_RISK = "deadline_risk"
    SCOPE_AMBIGUITY = "scope_ambiguity"
    STAKEHOLDER_DECISION = "stakeholder_decision"
    ETHICAL_JUDGMENT = "ethical_judgment"
    STRATEGIC_PIVOT = "strategic_pivot"
    CONFLICT_RESOLUTION = "conflict_resolution"
    QUALITY_GATE_FAILURE = "quality_gate_failure"
    EXTERNAL_DEPENDENCY = "external_dependency"


class NegotiationOutcome(Enum):
    """Outcome of agent-to-agent negotiation."""
    AGREED = "agreed"
    COUNTER_PROPOSED = "counter_proposed"
    REJECTED = "rejected"
    ESCALATED = "escalated"
    TIMED_OUT = "timed_out"


class PolicyType(Enum):
    """Types of organizational policies."""
    CAPACITY = "capacity"            # Resource utilization limits
    PRIORITY = "priority"            # Work ordering rules
    QUALITY = "quality"              # Quality gate requirements
    TIMELINE = "timeline"            # Deadline and timing rules
    ESCALATION = "escalation"        # When to involve humans
    COMMUNICATION = "communication"  # Stakeholder update rules


class HealthStatus(Enum):
    """Health status of an agent."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AutonomyLevel(Enum):
    """Level of autonomy granted to agent."""
    OBSERVER = 1          # Agent suggests, human decides
    GUARDRAILED = 2       # Agent acts within strict policies
    AUTONOMOUS = 3        # Agent acts with minimal escalation


# =============================================================================
# CORE DATA STRUCTURES
# =============================================================================

@dataclass
class Evidence:
    """Evidence supporting agent decisions or escalations."""
    type: str                        # data_point, observation, calculation, inference
    source: str                      # Where this evidence comes from
    value: Any                       # The evidence value
    confidence: float                # 0.0 to 1.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "type": self.type,
            "source": self.source,
            "value": str(self.value),
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Policy:
    """An organizational policy that governs agent behavior."""
    id: str
    name: str
    type: PolicyType
    description: str
    rule: str                        # Natural language rule
    parameters: Dict[str, Any]       # Numeric thresholds etc.
    enabled: bool = True
    priority: int = 50               # Higher = more important
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "rule": self.rule,
            "parameters": self.parameters,
            "enabled": self.enabled,
            "priority": self.priority
        }


@dataclass
class ResourceRequest:
    """A request for resources from an agent."""
    resource_type: str               # developer, qa, designer, etc.
    resource_id: Optional[str]       # Specific resource or None for any
    hours_needed: float
    priority: int                    # 1-10, higher = more urgent
    flexibility: float               # 0.0-1.0, how flexible on timing
    start_date: date
    end_date: date
    justification: str
    requesting_agent_id: str


@dataclass
class NegotiationProposal:
    """A proposal in agent-to-agent negotiation."""
    id: str
    from_agent_id: str
    to_agent_id: str
    proposal_type: str               # resource_request, timeline_shift, scope_trade
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    response: Optional[NegotiationOutcome] = None
    response_details: Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "from": self.from_agent_id,
            "to": self.to_agent_id,
            "type": self.proposal_type,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "response": self.response.value if self.response else None,
            "response_details": self.response_details
        }


@dataclass
class Intervention:
    """A request for human intervention."""
    id: str
    agent_id: str
    reason: EscalationReason
    summary: str
    context: str
    evidence: List[Evidence]
    options: List[Dict[str, str]]    # Possible actions human can take
    urgency: str                     # low, medium, high, critical
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    resolution: Optional[str] = None
    resolved_by: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "reason": self.reason.value,
            "summary": self.summary,
            "context": self.context,
            "evidence": [e.to_dict() for e in self.evidence],
            "options": self.options,
            "urgency": self.urgency,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution": self.resolution,
            "resolved_by": self.resolved_by
        }


@dataclass
class AgentAction:
    """An action taken by an agent."""
    id: str
    agent_id: str
    action_type: str                 # reserve_resource, negotiate, adapt, escalate, complete
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    result: Optional[str] = None
    evidence: List[Evidence] = field(default_factory=list)
    reversible: bool = True

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "action_type": self.action_type,
            "description": self.description,
            "timestamp": self.timestamp.isoformat(),
            "result": self.result,
            "evidence": [e.to_dict() for e in self.evidence],
            "reversible": self.reversible
        }


@dataclass
class AgentMetrics:
    """Performance metrics for an agent."""
    decisions_made: int = 0
    decisions_without_escalation: int = 0
    negotiations_initiated: int = 0
    negotiations_successful: int = 0
    adaptations: int = 0
    escalations: int = 0
    time_in_state: Dict[str, timedelta] = field(default_factory=dict)
    completion_accuracy: float = 0.0

    @property
    def autonomy_rate(self) -> float:
        """Percentage of decisions made without escalation."""
        if self.decisions_made == 0:
            return 1.0
        return self.decisions_without_escalation / self.decisions_made

    @property
    def negotiation_success_rate(self) -> float:
        """Percentage of successful negotiations."""
        if self.negotiations_initiated == 0:
            return 1.0
        return self.negotiations_successful / self.negotiations_initiated


# =============================================================================
# WORK AGENT - The Core Autonomous Entity
# =============================================================================

@dataclass
class WorkAgent:
    """
    An autonomous work agent that understands its purpose, constraints,
    and can make decisions within policy guardrails.
    """
    id: str
    name: str
    agent_type: AgentType

    # Purpose & Context
    intent: str                      # The "why" - high-level goal
    success_criteria: List[str]      # How we know it's done
    constraints: Dict[str, Any]      # Time, resources, dependencies

    # Linked work items
    issue_keys: List[str] = field(default_factory=list)
    epic_key: Optional[str] = None
    sprint_id: Optional[int] = None

    # State management
    state: AgentState = AgentState.DORMANT
    health: HealthStatus = HealthStatus.UNKNOWN
    autonomy_level: AutonomyLevel = AutonomyLevel.OBSERVER

    # Relationships
    parent_agent_id: Optional[str] = None
    child_agent_ids: List[str] = field(default_factory=list)
    dependency_agent_ids: List[str] = field(default_factory=list)

    # Execution tracking
    progress: float = 0.0            # 0.0 to 1.0
    estimated_completion: Optional[datetime] = None
    actual_start: Optional[datetime] = None
    actual_completion: Optional[datetime] = None

    # History
    actions: List[AgentAction] = field(default_factory=list)
    negotiations: List[NegotiationProposal] = field(default_factory=list)
    interventions: List[Intervention] = field(default_factory=list)

    # Metrics
    metrics: AgentMetrics = field(default_factory=AgentMetrics)

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate unique agent ID."""
        data = f"{self.name}-{self.intent}-{datetime.now().isoformat()}"
        return f"agent_{hashlib.sha256(data.encode()).hexdigest()[:12]}"

    def to_dict(self) -> Dict:
        """Convert agent to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "agent_type": self.agent_type.value,
            "intent": self.intent,
            "success_criteria": self.success_criteria,
            "constraints": self.constraints,
            "issue_keys": self.issue_keys,
            "epic_key": self.epic_key,
            "sprint_id": self.sprint_id,
            "state": self.state.value,
            "health": self.health.value,
            "autonomy_level": self.autonomy_level.value,
            "parent_agent_id": self.parent_agent_id,
            "child_agent_ids": self.child_agent_ids,
            "dependency_agent_ids": self.dependency_agent_ids,
            "progress": self.progress,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "actual_start": self.actual_start.isoformat() if self.actual_start else None,
            "actual_completion": self.actual_completion.isoformat() if self.actual_completion else None,
            "metrics": {
                "decisions_made": self.metrics.decisions_made,
                "autonomy_rate": self.metrics.autonomy_rate,
                "negotiation_success_rate": self.metrics.negotiation_success_rate,
                "escalations": self.metrics.escalations
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def log_action(self, action_type: str, description: str,
                   result: Optional[str] = None,
                   evidence: Optional[List[Evidence]] = None) -> AgentAction:
        """Log an action taken by this agent."""
        action = AgentAction(
            id=f"action_{hashlib.sha256(f'{self.id}-{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}",
            agent_id=self.id,
            action_type=action_type,
            description=description,
            result=result,
            evidence=evidence or []
        )
        self.actions.append(action)
        self.updated_at = datetime.now()
        return action

    def request_intervention(self, reason: EscalationReason,
                            summary: str, context: str,
                            evidence: List[Evidence],
                            options: List[Dict[str, str]],
                            urgency: str = "medium") -> Intervention:
        """Request human intervention."""
        intervention = Intervention(
            id=f"int_{hashlib.sha256(f'{self.id}-{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}",
            agent_id=self.id,
            reason=reason,
            summary=summary,
            context=context,
            evidence=evidence,
            options=options,
            urgency=urgency
        )
        self.interventions.append(intervention)
        self.state = AgentState.AWAITING_HUMAN
        self.metrics.escalations += 1
        self.updated_at = datetime.now()
        return intervention

    def resolve_intervention(self, intervention_id: str,
                            resolution: str, resolved_by: str):
        """Resolve a pending intervention."""
        for intervention in self.interventions:
            if intervention.id == intervention_id:
                intervention.resolved_at = datetime.now()
                intervention.resolution = resolution
                intervention.resolved_by = resolved_by
                break

        # Resume if no more pending interventions
        pending = [i for i in self.interventions if not i.resolved_at]
        if not pending:
            self.state = AgentState.EXECUTING
        self.updated_at = datetime.now()

    def update_progress(self, progress: float,
                       estimated_completion: Optional[datetime] = None):
        """Update agent progress."""
        self.progress = max(0.0, min(1.0, progress))
        if estimated_completion:
            self.estimated_completion = estimated_completion
        if progress >= 1.0:
            self.state = AgentState.COMPLETED
            self.actual_completion = datetime.now()
        self.updated_at = datetime.now()

    @property
    def pending_interventions(self) -> List[Intervention]:
        """Get unresolved interventions."""
        return [i for i in self.interventions if not i.resolved_at]

    @property
    def is_blocked(self) -> bool:
        """Check if agent is in blocked state."""
        return self.state in [AgentState.BLOCKED, AgentState.AWAITING_HUMAN]

    @property
    def days_in_current_state(self) -> float:
        """Days in current state."""
        return (datetime.now() - self.updated_at).total_seconds() / 86400


# =============================================================================
# INTENT - High-Level Goal Definition
# =============================================================================

@dataclass
class Intent:
    """
    A high-level intent that humans set, which agents decompose and execute.
    This replaces traditional project/epic creation.
    """
    id: str
    title: str
    description: str                 # Natural language description
    owner_id: str                    # Human owner

    # Goal definition
    outcome: str                     # What success looks like
    success_metrics: List[Dict[str, Any]]  # Measurable criteria

    # Constraints
    deadline: Optional[date] = None
    budget_hours: Optional[float] = None
    priority: int = 50               # 1-100, higher = more important

    # Policies
    policies: List[str] = field(default_factory=list)  # Policy IDs to apply

    # Generated agents
    root_agent_id: Optional[str] = None
    agent_ids: List[str] = field(default_factory=list)

    # Status
    status: str = "draft"            # draft, active, completed, cancelled
    overall_progress: float = 0.0
    estimated_completion: Optional[date] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    activated_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "owner_id": self.owner_id,
            "outcome": self.outcome,
            "success_metrics": self.success_metrics,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "budget_hours": self.budget_hours,
            "priority": self.priority,
            "policies": self.policies,
            "root_agent_id": self.root_agent_id,
            "agent_ids": self.agent_ids,
            "status": self.status,
            "overall_progress": self.overall_progress,
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "created_at": self.created_at.isoformat(),
            "activated_at": self.activated_at.isoformat() if self.activated_at else None
        }


# =============================================================================
# WORK AGENT ENGINE - Orchestration & Intelligence
# =============================================================================

class WorkAgentEngine:
    """
    The orchestration engine that manages work agents, handles negotiations,
    enforces policies, and coordinates autonomous execution.
    """

    def __init__(self, llm_client=None, db_connection=None):
        self.llm_client = llm_client
        self.db = db_connection

        # Runtime state
        self.agents: Dict[str, WorkAgent] = {}
        self.intents: Dict[str, Intent] = {}
        self.policies: Dict[str, Policy] = {}

        # Default policies
        self._initialize_default_policies()

    def _initialize_default_policies(self):
        """Set up default organizational policies."""
        default_policies = [
            Policy(
                id="policy_capacity_80",
                name="Engineering Capacity Limit",
                type=PolicyType.CAPACITY,
                description="Engineering capacity cannot exceed 80%",
                rule="Total assigned story points must not exceed 80% of team velocity",
                parameters={"max_utilization": 0.80, "buffer_percentage": 0.20}
            ),
            Policy(
                id="policy_customer_priority",
                name="Customer-Facing Priority",
                type=PolicyType.PRIORITY,
                description="Customer-facing work takes priority over internal",
                rule="Customer-facing tickets are prioritized above internal work of same priority level",
                parameters={"priority_boost": 10}
            ),
            Policy(
                id="policy_delay_escalation",
                name="Delay Escalation Threshold",
                type=PolicyType.ESCALATION,
                description="Escalate any delay greater than 3 days",
                rule="If estimated completion slips by more than 3 days, escalate to human",
                parameters={"delay_threshold_days": 3}
            ),
            Policy(
                id="policy_qa_required",
                name="QA Sign-off Required",
                type=PolicyType.QUALITY,
                description="Never ship without QA sign-off",
                rule="All work must have QA verification before marked complete",
                parameters={"qa_required": True}
            ),
            Policy(
                id="policy_wip_limit",
                name="WIP Limit Per Developer",
                type=PolicyType.CAPACITY,
                description="Limit work in progress per developer",
                rule="No developer should have more than 3 active tickets",
                parameters={"max_wip": 3}
            ),
            Policy(
                id="policy_blocker_response",
                name="Blocker Response Time",
                type=PolicyType.TIMELINE,
                description="Blockers must be addressed within 4 hours",
                rule="Any blocked ticket must have action within 4 hours during work hours",
                parameters={"response_hours": 4}
            ),
        ]

        for policy in default_policies:
            self.policies[policy.id] = policy

    def create_intent(self, title: str, description: str,
                      owner_id: str, outcome: str,
                      deadline: Optional[date] = None,
                      priority: int = 50) -> Intent:
        """Create a new intent from natural language."""
        intent_id = f"intent_{hashlib.sha256(f'{title}-{datetime.now().isoformat()}'.encode()).hexdigest()[:10]}"

        intent = Intent(
            id=intent_id,
            title=title,
            description=description,
            owner_id=owner_id,
            outcome=outcome,
            success_metrics=[],  # Will be generated by LLM
            deadline=deadline,
            priority=priority,
            policies=list(self.policies.keys())  # Apply all policies by default
        )

        self.intents[intent_id] = intent
        return intent

    def decompose_intent(self, intent_id: str) -> List[WorkAgent]:
        """
        Use LLM to decompose intent into autonomous work agents.
        This is the magic - turning human intent into executable agent network.
        """
        intent = self.intents.get(intent_id)
        if not intent:
            raise ValueError(f"Intent {intent_id} not found")

        # Generate agent network using LLM
        if self.llm_client:
            agents = self._llm_decompose_intent(intent)
        else:
            # Fallback: create single agent
            agents = [self._create_placeholder_agent(intent)]

        # Register agents
        for agent in agents:
            self.agents[agent.id] = agent
            intent.agent_ids.append(agent.id)

        if agents:
            intent.root_agent_id = agents[0].id

        return agents

    def _llm_decompose_intent(self, intent: Intent) -> List[WorkAgent]:
        """Use Gemini to decompose intent into agents."""
        prompt = f"""You are an autonomous work agent architect. Decompose this high-level intent into a network of autonomous work agents.

INTENT:
Title: {intent.title}
Description: {intent.description}
Desired Outcome: {intent.outcome}
Deadline: {intent.deadline or 'Not specified'}
Priority: {intent.priority}/100

Create a structured decomposition following these principles:
1. Each agent should have a clear, single purpose
2. Agents should be able to work autonomously within their scope
3. Define dependencies between agents
4. Specify success criteria for each agent
5. Identify resource requirements

Respond in JSON format:
{{
    "agents": [
        {{
            "name": "Agent name",
            "type": "feature|bug_fix|tech_debt|maintenance|research|integration|migration|coordination",
            "intent": "What this agent aims to achieve",
            "success_criteria": ["Criterion 1", "Criterion 2"],
            "dependencies": ["Name of dependent agent if any"],
            "estimated_hours": 10,
            "skills_needed": ["frontend", "backend", "qa"],
            "priority": 1-10
        }}
    ],
    "execution_order": ["Agent 1 name", "Agent 2 name"],
    "parallel_groups": [["Agent 1", "Agent 2"], ["Agent 3"]]
}}"""

        try:
            response = self.llm_client.generate_json(
                prompt=prompt,
                system_prompt="You are an expert work decomposition engine. Return valid JSON only.",
                default={"agents": []}
            )

            agents = []
            agent_map = {}  # name -> id

            for idx, agent_spec in enumerate(response.get("agents", [])):
                agent = WorkAgent(
                    id="",  # Will be auto-generated
                    name=agent_spec.get("name", f"Agent {idx+1}"),
                    agent_type=AgentType(agent_spec.get("type", "feature")),
                    intent=agent_spec.get("intent", intent.outcome),
                    success_criteria=agent_spec.get("success_criteria", []),
                    constraints={
                        "estimated_hours": agent_spec.get("estimated_hours", 8),
                        "skills_needed": agent_spec.get("skills_needed", []),
                        "priority": agent_spec.get("priority", 5)
                    }
                )
                agents.append(agent)
                agent_map[agent.name] = agent.id

            # Wire up dependencies
            for idx, agent_spec in enumerate(response.get("agents", [])):
                for dep_name in agent_spec.get("dependencies", []):
                    if dep_name in agent_map:
                        agents[idx].dependency_agent_ids.append(agent_map[dep_name])

            return agents

        except Exception as e:
            # Fallback to single agent
            return [self._create_placeholder_agent(intent)]

    def _create_placeholder_agent(self, intent: Intent) -> WorkAgent:
        """Create a placeholder agent when LLM is unavailable."""
        return WorkAgent(
            id="",
            name=f"Execute: {intent.title[:30]}",
            agent_type=AgentType.FEATURE,
            intent=intent.outcome,
            success_criteria=["Complete all related tickets", "Pass quality gates"],
            constraints={"priority": intent.priority}
        )

    def activate_intent(self, intent_id: str) -> Intent:
        """Activate an intent and start agent execution."""
        intent = self.intents.get(intent_id)
        if not intent:
            raise ValueError(f"Intent {intent_id} not found")

        intent.status = "active"
        intent.activated_at = datetime.now()

        # Activate root agent(s)
        for agent_id in intent.agent_ids:
            agent = self.agents.get(agent_id)
            if agent and not agent.dependency_agent_ids:
                # No dependencies - can start immediately
                agent.state = AgentState.PLANNING
                agent.actual_start = datetime.now()

        return intent

    def get_pending_interventions(self) -> List[Intervention]:
        """Get all pending interventions across all agents."""
        interventions = []
        for agent in self.agents.values():
            interventions.extend(agent.pending_interventions)

        # Sort by urgency and creation time
        urgency_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        interventions.sort(
            key=lambda i: (urgency_order.get(i.urgency, 4), i.created_at)
        )

        return interventions

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        total_agents = len(self.agents)
        if total_agents == 0:
            return {
                "status": "idle",
                "total_agents": 0,
                "healthy_agents": 0,
                "warning_agents": 0,
                "critical_agents": 0,
                "pending_interventions": 0,
                "autonomy_rate": 1.0
            }

        healthy = sum(1 for a in self.agents.values() if a.health == HealthStatus.HEALTHY)
        warning = sum(1 for a in self.agents.values() if a.health == HealthStatus.WARNING)
        critical = sum(1 for a in self.agents.values() if a.health == HealthStatus.CRITICAL)
        pending = len(self.get_pending_interventions())

        total_decisions = sum(a.metrics.decisions_made for a in self.agents.values())
        autonomous_decisions = sum(a.metrics.decisions_without_escalation for a in self.agents.values())
        autonomy_rate = autonomous_decisions / total_decisions if total_decisions > 0 else 1.0

        if critical > 0 or pending > 5:
            status = "critical"
        elif warning > total_agents * 0.2 or pending > 2:
            status = "warning"
        else:
            status = "healthy"

        return {
            "status": status,
            "total_agents": total_agents,
            "healthy_agents": healthy,
            "warning_agents": warning,
            "critical_agents": critical,
            "pending_interventions": pending,
            "autonomy_rate": autonomy_rate,
            "agents_by_state": self._count_agents_by_state(),
            "active_intents": len([i for i in self.intents.values() if i.status == "active"])
        }

    def _count_agents_by_state(self) -> Dict[str, int]:
        """Count agents by state."""
        counts = {}
        for agent in self.agents.values():
            state = agent.state.value
            counts[state] = counts.get(state, 0) + 1
        return counts

    def check_policy_violations(self, agent: WorkAgent) -> List[Dict[str, Any]]:
        """Check if agent is violating any policies."""
        violations = []

        for policy in self.policies.values():
            if not policy.enabled:
                continue

            violation = self._check_single_policy(agent, policy)
            if violation:
                violations.append(violation)

        return violations

    def _check_single_policy(self, agent: WorkAgent, policy: Policy) -> Optional[Dict]:
        """Check a single policy against agent state."""
        if policy.type == PolicyType.ESCALATION:
            threshold = policy.parameters.get("delay_threshold_days", 3)
            if agent.estimated_completion and agent.constraints.get("deadline"):
                delay = (agent.estimated_completion.date() - agent.constraints["deadline"]).days
                if delay > threshold:
                    return {
                        "policy_id": policy.id,
                        "policy_name": policy.name,
                        "violation": f"Estimated delay of {delay} days exceeds threshold of {threshold}",
                        "severity": "high"
                    }

        elif policy.type == PolicyType.CAPACITY:
            max_wip = policy.parameters.get("max_wip")
            if max_wip and len(agent.issue_keys) > max_wip:
                return {
                    "policy_id": policy.id,
                    "policy_name": policy.name,
                    "violation": f"WIP of {len(agent.issue_keys)} exceeds limit of {max_wip}",
                    "severity": "medium"
                }

        return None

    def simulate_agent_decisions(self, agent_id: str,
                                 scenario: str = "normal") -> List[Dict[str, Any]]:
        """
        Simulate what decisions an agent would make.
        Used for Observer mode and validation.
        """
        agent = self.agents.get(agent_id)
        if not agent:
            return []

        decisions = []

        # Check for blockers
        if agent.state == AgentState.BLOCKED:
            decisions.append({
                "type": "escalation",
                "action": "Request intervention for blocked state",
                "reason": "Agent has been blocked for too long",
                "confidence": 0.85
            })

        # Check policy violations
        violations = self.check_policy_violations(agent)
        for v in violations:
            decisions.append({
                "type": "policy_response",
                "action": f"Address {v['policy_name']} violation",
                "reason": v["violation"],
                "confidence": 0.90
            })

        # Progress update
        if agent.state == AgentState.EXECUTING:
            decisions.append({
                "type": "progress_update",
                "action": "Update progress based on ticket completion",
                "reason": "Regular progress tracking",
                "confidence": 0.95
            })

        return decisions

    def get_agent_network_graph(self, intent_id: str) -> Dict[str, Any]:
        """Get agent network for visualization."""
        intent = self.intents.get(intent_id)
        if not intent:
            return {"nodes": [], "edges": []}

        nodes = []
        edges = []

        for agent_id in intent.agent_ids:
            agent = self.agents.get(agent_id)
            if not agent:
                continue

            nodes.append({
                "id": agent.id,
                "label": agent.name,
                "type": agent.agent_type.value,
                "state": agent.state.value,
                "health": agent.health.value,
                "progress": agent.progress
            })

            for dep_id in agent.dependency_agent_ids:
                edges.append({
                    "from": dep_id,
                    "to": agent.id,
                    "type": "dependency"
                })

            if agent.parent_agent_id:
                edges.append({
                    "from": agent.parent_agent_id,
                    "to": agent.id,
                    "type": "parent"
                })

        return {"nodes": nodes, "edges": edges}

    def get_management_tax_calculation(self) -> Dict[str, Any]:
        """Calculate the management tax being saved."""
        # Simulated metrics based on typical PM activities
        activities = [
            {"activity": "Assigning work", "hours_per_week": 3.0, "automatable": 0.95},
            {"activity": "Checking progress", "hours_per_week": 5.0, "automatable": 1.00},
            {"activity": "Updating stakeholders", "hours_per_week": 2.5, "automatable": 0.90},
            {"activity": "Resolving simple blockers", "hours_per_week": 2.0, "automatable": 0.80},
            {"activity": "Replanning", "hours_per_week": 3.0, "automatable": 0.70},
            {"activity": "Status meetings", "hours_per_week": 4.0, "automatable": 0.85},
            {"activity": "Resource allocation", "hours_per_week": 2.0, "automatable": 0.75},
        ]

        total_hours = sum(a["hours_per_week"] for a in activities)
        recoverable_hours = sum(
            a["hours_per_week"] * a["automatable"]
            for a in activities
        )

        return {
            "activities": activities,
            "total_management_hours": total_hours,
            "recoverable_hours": recoverable_hours,
            "recovery_percentage": recoverable_hours / total_hours if total_hours > 0 else 0,
            "capacity_multiplier": 1 / (1 - (recoverable_hours / 40)) if recoverable_hours < 40 else 4.0
        }


# =============================================================================
# AGENT NEGOTIATION PROTOCOL
# =============================================================================

class NegotiationProtocol:
    """
    Protocol for agent-to-agent negotiations.
    Agents can negotiate resources, timelines, and scope.
    """

    def __init__(self, engine: WorkAgentEngine):
        self.engine = engine
        self.pending_negotiations: List[NegotiationProposal] = []

    def propose_resource_request(self, from_agent: WorkAgent,
                                  resource_type: str,
                                  hours_needed: float,
                                  justification: str) -> NegotiationProposal:
        """Create a resource request proposal."""
        proposal = NegotiationProposal(
            id=f"neg_{hashlib.sha256(f'{from_agent.id}-{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}",
            from_agent_id=from_agent.id,
            to_agent_id="system",  # Resource requests go to system
            proposal_type="resource_request",
            details={
                "resource_type": resource_type,
                "hours_needed": hours_needed,
                "justification": justification,
                "priority": from_agent.constraints.get("priority", 5)
            }
        )

        self.pending_negotiations.append(proposal)
        from_agent.negotiations.append(proposal)
        from_agent.metrics.negotiations_initiated += 1

        return proposal

    def propose_timeline_shift(self, from_agent: WorkAgent,
                               to_agent: WorkAgent,
                               days_requested: int,
                               reason: str) -> NegotiationProposal:
        """Propose a timeline shift to dependent agent."""
        proposal = NegotiationProposal(
            id=f"neg_{hashlib.sha256(f'{from_agent.id}-{to_agent.id}-{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}",
            from_agent_id=from_agent.id,
            to_agent_id=to_agent.id,
            proposal_type="timeline_shift",
            details={
                "days_requested": days_requested,
                "reason": reason,
                "current_deadline": from_agent.constraints.get("deadline"),
                "impact_assessment": f"Requesting {days_requested} day extension"
            }
        )

        self.pending_negotiations.append(proposal)
        from_agent.negotiations.append(proposal)
        from_agent.metrics.negotiations_initiated += 1

        return proposal

    def resolve_negotiation(self, proposal_id: str,
                           outcome: NegotiationOutcome,
                           details: Optional[Dict] = None):
        """Resolve a pending negotiation."""
        for prop in self.pending_negotiations:
            if prop.id == proposal_id:
                prop.response = outcome
                prop.response_details = details or {}

                # Update agent metrics
                from_agent = self.engine.agents.get(prop.from_agent_id)
                if from_agent and outcome == NegotiationOutcome.AGREED:
                    from_agent.metrics.negotiations_successful += 1

                self.pending_negotiations.remove(prop)
                break


# =============================================================================
# DEMO DATA GENERATOR
# =============================================================================

def generate_demo_work_agents(engine: WorkAgentEngine) -> None:
    """Generate demo data for the autonomous work agents system."""

    # Create sample intents
    intents = [
        {
            "title": "Launch New Onboarding Flow",
            "description": "Reduce time-to-value by 40% through a streamlined onboarding experience",
            "outcome": "Users complete onboarding in under 5 minutes with 90% success rate",
            "priority": 85
        },
        {
            "title": "Performance Optimization Sprint",
            "description": "Improve application performance to meet SLA requirements",
            "outcome": "All pages load under 2 seconds, API responses under 200ms",
            "priority": 70
        },
        {
            "title": "Security Compliance Audit",
            "description": "Ensure SOC2 compliance for enterprise customers",
            "outcome": "Pass SOC2 audit with zero critical findings",
            "priority": 95
        }
    ]

    # Create intents and agents
    for intent_data in intents:
        intent = engine.create_intent(
            title=intent_data["title"],
            description=intent_data["description"],
            owner_id="demo_user",
            outcome=intent_data["outcome"],
            deadline=date.today() + timedelta(days=random.randint(14, 60)),
            priority=intent_data["priority"]
        )

        # Create sample agents manually (without LLM)
        agent_types = [
            (AgentType.RESEARCH, "Research & Analysis"),
            (AgentType.FEATURE, "Implementation"),
            (AgentType.INTEGRATION, "Integration & Testing"),
            (AgentType.COORDINATION, "Coordination & Rollout")
        ]

        parent_id = None
        for agent_type, suffix in agent_types:
            agent = WorkAgent(
                id="",
                name=f"{intent.title[:20]} - {suffix}",
                agent_type=agent_type,
                intent=f"{suffix} for {intent.title}",
                success_criteria=["Complete on time", "Meet quality standards"],
                constraints={"priority": intent.priority, "deadline": intent.deadline},
                parent_agent_id=parent_id,
                state=random.choice([AgentState.EXECUTING, AgentState.PLANNING, AgentState.NEGOTIATING]),
                health=random.choice([HealthStatus.HEALTHY, HealthStatus.HEALTHY, HealthStatus.WARNING]),
                progress=random.uniform(0.1, 0.9),
                estimated_completion=datetime.now() + timedelta(days=random.randint(5, 30))
            )

            # Simulate some metrics
            agent.metrics.decisions_made = random.randint(5, 50)
            agent.metrics.decisions_without_escalation = int(agent.metrics.decisions_made * random.uniform(0.7, 0.95))
            agent.metrics.negotiations_initiated = random.randint(1, 10)
            agent.metrics.negotiations_successful = int(agent.metrics.negotiations_initiated * random.uniform(0.6, 0.9))

            engine.agents[agent.id] = agent
            intent.agent_ids.append(agent.id)
            parent_id = agent.id

        intent.root_agent_id = intent.agent_ids[0] if intent.agent_ids else None
        intent.status = "active"
        intent.activated_at = datetime.now() - timedelta(days=random.randint(1, 14))
        intent.overall_progress = sum(
            engine.agents[aid].progress for aid in intent.agent_ids
        ) / len(intent.agent_ids) if intent.agent_ids else 0

    # Create some pending interventions
    sample_interventions = [
        {
            "reason": EscalationReason.RESOURCE_CONFLICT,
            "summary": "Designer capacity conflict between two high-priority agents",
            "urgency": "high"
        },
        {
            "reason": EscalationReason.SCOPE_AMBIGUITY,
            "summary": "Unclear requirements for authentication flow",
            "urgency": "medium"
        },
        {
            "reason": EscalationReason.STAKEHOLDER_DECISION,
            "summary": "Need product decision on feature priority",
            "urgency": "low"
        }
    ]

    # Add interventions to random agents
    agents_list = list(engine.agents.values())
    if agents_list:
        for int_data in sample_interventions:
            agent = random.choice(agents_list)
            agent.request_intervention(
                reason=int_data["reason"],
                summary=int_data["summary"],
                context="Auto-generated demo intervention",
                evidence=[Evidence(
                    type="observation",
                    source="system",
                    value="Demo data",
                    confidence=0.8
                )],
                options=[
                    {"id": "approve", "label": "Approve agent recommendation"},
                    {"id": "override", "label": "Override with different approach"},
                    {"id": "defer", "label": "Defer decision"}
                ],
                urgency=int_data["urgency"]
            )
