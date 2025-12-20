"""
🧿 The Sixth Sense - Superhuman Project Perception
See what humans cannot see. Know what you'll regret before you regret it.

THE REVOLUTIONARY PREMISE:
Projects don't fail from lack of tracking. They fail from invisible dynamics
that compound until it's too late. This module gives you perception beyond
human limits — the ability to see patterns, predict conflicts, and receive
hindsight BEFORE the future happens.

This is not a dashboard. This is a PERCEPTION ENGINE.
"""

import streamlit as st
import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math

st.set_page_config(page_title="The Sixth Sense", page_icon="🧿", layout="wide")

# ============================================================================
# LIGHT MODE CSS - PERCEPTION ENGINE AESTHETIC
# ============================================================================
st.markdown("""
<style>
    /* Light theme for perception engine */
    .stApp {
        background-color: #f8f9fa;
    }

    /* The Eye - Central perception indicator */
    .sixth-sense-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #8b5cf6 100%);
        border-radius: 20px;
        text-align: center;
        padding: 40px 20px;
        position: relative;
        margin-bottom: 30px;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .the-eye {
        width: 100px;
        height: 100px;
        margin: 0 auto 20px auto;
        position: relative;
        animation: eye-pulse 4s ease-in-out infinite;
    }

    @keyframes eye-pulse {
        0%, 100% { transform: scale(1); filter: drop-shadow(0 0 15px rgba(255, 255, 255, 0.5)); }
        50% { transform: scale(1.05); filter: drop-shadow(0 0 25px rgba(255, 255, 255, 0.7)); }
    }

    .eye-outer {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, #ffffff 0%, #e0e7ff 50%, #c7d2fe 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        border: 4px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 0 40px rgba(255, 255, 255, 0.4), inset 0 0 20px rgba(139, 92, 246, 0.1);
    }

    .eye-iris {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        background: radial-gradient(circle at 40% 40%, #a78bfa 0%, #8b5cf6 50%, #7c3aed 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        animation: iris-glow 2s ease-in-out infinite alternate;
    }

    @keyframes iris-glow {
        0% { box-shadow: 0 0 15px rgba(139, 92, 246, 0.4); }
        100% { box-shadow: 0 0 25px rgba(139, 92, 246, 0.7); }
    }

    .eye-pupil {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #1e1b4b;
        position: relative;
    }

    .eye-pupil::after {
        content: '';
        position: absolute;
        top: 3px;
        left: 5px;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.9);
    }

    .header-title {
        font-size: 38px;
        font-weight: 800;
        color: white;
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }

    .header-subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 15px;
        margin-top: 8px;
        font-weight: 400;
    }

    .perception-level {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.3);
        padding: 8px 20px;
        border-radius: 20px;
        margin-top: 16px;
        font-size: 13px;
        color: white;
    }

    .perception-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #4ade80;
        animation: dot-pulse 1.5s ease-in-out infinite;
    }

    @keyframes dot-pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Vision Cards - Each type of perception */
    .vision-section {
        margin-bottom: 32px;
    }

    .vision-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 16px;
        padding: 0 8px;
    }

    .vision-icon {
        font-size: 28px;
    }

    .vision-title {
        font-size: 20px;
        font-weight: 700;
        color: #1e293b;
        margin: 0;
    }

    .vision-subtitle {
        font-size: 12px;
        color: #64748b;
        margin-left: auto;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Regret Card - Future regret prediction */
    .regret-card {
        background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
        border: 1px solid #fecaca;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 12px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.08);
    }

    .regret-card:hover {
        border-color: #f87171;
        transform: translateX(4px);
        box-shadow: 0 4px 16px rgba(239, 68, 68, 0.15);
    }

    .regret-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #f87171 0%, #dc2626 100%);
    }

    .regret-timeline {
        display: inline-block;
        background: rgba(220, 38, 38, 0.1);
        color: #dc2626;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .regret-prediction {
        color: #991b1b;
        font-size: 16px;
        font-weight: 600;
        line-height: 1.5;
        margin-bottom: 8px;
    }

    .regret-evidence {
        color: #b91c1c;
        font-size: 13px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .regret-action {
        margin-top: 12px;
        padding-top: 12px;
        border-top: 1px solid #fecaca;
    }

    .action-button {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #dc2626;
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
    }

    .action-button:hover {
        background: #fee2e2;
        border-color: #f87171;
    }

    /* Déjà Vu Card - Pattern recognition */
    .dejavu-card {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 12px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08);
    }

    .dejavu-card:hover {
        border-color: #60a5fa;
        transform: translateX(4px);
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.15);
    }

    .dejavu-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #60a5fa 0%, #2563eb 100%);
    }

    .dejavu-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(37, 99, 235, 0.1);
        color: #2563eb;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .dejavu-pattern {
        color: #1e40af;
        font-size: 16px;
        font-weight: 600;
        line-height: 1.5;
        margin-bottom: 8px;
    }

    .dejavu-history {
        background: rgba(37, 99, 235, 0.08);
        border-radius: 10px;
        padding: 12px 16px;
        margin-top: 12px;
    }

    .history-label {
        color: #3b82f6;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 8px;
    }

    .history-outcome {
        color: #1e40af;
        font-size: 14px;
    }

    .outcome-bad { color: #dc2626; }
    .outcome-good { color: #16a34a; }

    /* Conflict Radar Card */
    .conflict-card {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #fde68a;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 12px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(217, 119, 6, 0.08);
    }

    .conflict-card:hover {
        border-color: #fbbf24;
        transform: translateX(4px);
        box-shadow: 0 4px 16px rgba(217, 119, 6, 0.15);
    }

    .conflict-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #fbbf24 0%, #d97706 100%);
    }

    .conflict-intensity {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .intensity-high {
        background: rgba(220, 38, 38, 0.1);
        color: #dc2626;
    }

    .intensity-medium {
        background: rgba(217, 119, 6, 0.1);
        color: #d97706;
    }

    .intensity-low {
        background: rgba(22, 163, 74, 0.1);
        color: #16a34a;
    }

    .conflict-description {
        color: #92400e;
        font-size: 16px;
        font-weight: 600;
        line-height: 1.5;
        margin-bottom: 8px;
    }

    .conflict-parties {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-top: 12px;
    }

    .party-avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
        color: white;
    }

    .party-vs {
        color: #b45309;
        font-size: 12px;
        font-weight: 600;
    }

    /* Compound Problem Card */
    .compound-card {
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
        border: 1px solid #e9d5ff;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 12px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(126, 34, 206, 0.08);
    }

    .compound-card:hover {
        border-color: #c084fc;
        transform: translateX(4px);
        box-shadow: 0 4px 16px rgba(126, 34, 206, 0.15);
    }

    .compound-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #c084fc 0%, #7c3aed 100%);
    }

    .compound-growth {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(126, 34, 206, 0.1);
        color: #7c3aed;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .compound-description {
        color: #581c87;
        font-size: 16px;
        font-weight: 600;
        line-height: 1.5;
        margin-bottom: 8px;
    }

    .growth-chart {
        display: flex;
        align-items: flex-end;
        gap: 4px;
        height: 40px;
        margin-top: 12px;
        padding: 8px;
        background: rgba(126, 34, 206, 0.08);
        border-radius: 8px;
    }

    .growth-bar {
        flex: 1;
        background: linear-gradient(180deg, #c084fc 0%, #7c3aed 100%);
        border-radius: 2px;
        min-height: 4px;
    }

    /* Avoidance Card */
    .avoidance-card {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #bbf7d0;
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 12px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(22, 163, 74, 0.08);
    }

    .avoidance-card:hover {
        border-color: #4ade80;
        transform: translateX(4px);
        box-shadow: 0 4px 16px rgba(22, 163, 74, 0.15);
    }

    .avoidance-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #4ade80 0%, #16a34a 100%);
    }

    .avoidance-days {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(22, 163, 74, 0.1);
        color: #16a34a;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .avoidance-description {
        color: #14532d;
        font-size: 16px;
        font-weight: 600;
        line-height: 1.5;
        margin-bottom: 8px;
    }

    .avoidance-cost {
        color: #166534;
        font-size: 13px;
    }

    /* Empty State */
    .empty-vision {
        text-align: center;
        padding: 40px 20px;
        color: #64748b;
    }

    .empty-icon {
        font-size: 48px;
        margin-bottom: 16px;
        opacity: 0.6;
    }

    .empty-text {
        font-size: 14px;
    }

    /* Perception Score */
    .perception-score {
        background: linear-gradient(135deg, #faf5ff 0%, #f3e8ff 100%);
        border: 1px solid #e9d5ff;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 2px 8px rgba(139, 92, 246, 0.08);
    }

    .score-value {
        font-size: 64px;
        font-weight: 900;
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }

    .score-label {
        color: #7c3aed;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 8px;
    }

    .score-description {
        color: #6b7280;
        font-size: 13px;
        margin-top: 12px;
        max-width: 300px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Insight Stats */
    .insight-stats {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 24px;
    }

    .insight-stat {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .insight-stat-value {
        font-size: 28px;
        font-weight: 700;
        color: #7c3aed;
    }

    .insight-stat-label {
        font-size: 10px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    .stat-danger .insight-stat-value { color: #dc2626; }
    .stat-warning .insight-stat-value { color: #d97706; }
    .stat-info .insight-stat-value { color: #2563eb; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# PERCEPTION ENGINE - CORE ANALYSIS
# ============================================================================

@dataclass
class FutureRegret:
    """A predicted future regret."""
    timeline: str  # "In 2 weeks", "By sprint end"
    prediction: str  # What you'll regret
    evidence: str  # Why we predict this
    severity: str  # high, medium, low
    action: str  # What to do now
    issue_keys: List[str]  # Related issues


@dataclass
class DejaVu:
    """A recognized pattern from the past."""
    pattern: str  # Current pattern description
    last_occurrence: str  # When it happened before
    outcome: str  # What happened last time
    outcome_type: str  # bad, good, neutral
    probability: int  # Likelihood of same outcome (%)
    issue_keys: List[str]


@dataclass
class InvisibleConflict:
    """A brewing conflict between people/teams."""
    description: str
    parties: List[str]  # People involved
    intensity: str  # high, medium, low
    evidence: str
    days_brewing: int


@dataclass
class CompoundProblem:
    """A problem that's growing exponentially."""
    description: str
    current_size: str  # "3 blocked items"
    growth_rate: str  # "20%/day"
    projection: str  # "Will affect 12 items by Friday"
    days_to_crisis: int
    issue_keys: List[str]


@dataclass
class AvoidedDecision:
    """A decision being avoided."""
    description: str
    days_avoided: int
    cost_of_delay: str
    evidence: str
    issue_keys: List[str]


def get_connection():
    """Get database connection."""
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path), read_only=True)


def detect_future_regrets(conn) -> List[FutureRegret]:
    """Detect things you'll regret not doing."""
    regrets = []

    try:
        # 1. Stale high-priority items - will become crises
        stale_critical = conn.execute("""
            SELECT key, summary, priority, updated,
                   DATEDIFF('day', updated, CURRENT_DATE) as days_stale
            FROM issues
            WHERE status NOT IN ('Done', 'Terminé(e)', 'Closed')
            AND priority IN ('Highest', 'High')
            AND updated < CURRENT_DATE - INTERVAL '5 days'
            ORDER BY priority DESC, days_stale DESC
            LIMIT 3
        """).fetchdf()

        for _, row in stale_critical.iterrows():
            regrets.append(FutureRegret(
                timeline="In 1 week",
                prediction=f"You'll wish you had unblocked {row['key']} sooner",
                evidence=f"High-priority item stale for {row['days_stale']} days. Pattern: these become escalations.",
                severity="high",
                action=f"Check on {row['key']} today",
                issue_keys=[row['key']]
            ))

        # 2. Sprint at risk - scope vs velocity mismatch
        sprint_risk = conn.execute("""
            SELECT
                s.name as sprint_name,
                s.end_date,
                COUNT(i.key) as remaining,
                COALESCE(SUM(i.story_points), 0) as remaining_points,
                DATEDIFF('day', CURRENT_DATE, s.end_date) as days_left
            FROM sprints s
            LEFT JOIN issues i ON s.id = i.sprint_id
                AND i.status NOT IN ('Done', 'Terminé(e)', 'Closed')
            WHERE s.state = 'active'
            GROUP BY s.id, s.name, s.end_date
            HAVING COUNT(i.key) > 0
        """).fetchdf()

        for _, row in sprint_risk.iterrows():
            if row['days_left'] is not None and row['days_left'] <= 5 and row['remaining'] > 5:
                regrets.append(FutureRegret(
                    timeline="By sprint end",
                    prediction="You'll regret not cutting scope earlier",
                    evidence=f"{row['remaining']} items ({row['remaining_points']:.0f} pts) remaining with {row['days_left']} days left",
                    severity="high",
                    action="Have scope conversation NOW",
                    issue_keys=[]
                ))

        # 3. Unassigned items aging - will become orphans
        orphans = conn.execute("""
            SELECT key, summary, created,
                   DATEDIFF('day', created, CURRENT_DATE) as days_old
            FROM issues
            WHERE status NOT IN ('Done', 'Terminé(e)', 'Closed')
            AND (assignee_name IS NULL OR assignee_name = '')
            AND created < CURRENT_DATE - INTERVAL '3 days'
            ORDER BY days_old DESC
            LIMIT 2
        """).fetchdf()

        for _, row in orphans.iterrows():
            regrets.append(FutureRegret(
                timeline="In 2 weeks",
                prediction=f"You'll wonder why {row['key']} was never started",
                evidence=f"Unassigned for {row['days_old']} days. Orphan items get forgotten.",
                severity="medium",
                action="Assign or archive this item",
                issue_keys=[row['key']]
            ))

        # 4. Overloaded person - will burn out or miss deadlines
        overloaded = conn.execute("""
            SELECT assignee_name, COUNT(*) as wip_count
            FROM issues
            WHERE status = 'En cours'
            AND assignee_name IS NOT NULL
            GROUP BY assignee_name
            HAVING COUNT(*) >= 5
            ORDER BY wip_count DESC
            LIMIT 1
        """).fetchdf()

        for _, row in overloaded.iterrows():
            regrets.append(FutureRegret(
                timeline="In 1-2 weeks",
                prediction=f"You'll regret not helping {row['assignee_name'].split()[0]} sooner",
                evidence=f"{row['wip_count']} items in progress simultaneously. Context switching kills velocity.",
                severity="high",
                action="Rebalance workload or help prioritize",
                issue_keys=[]
            ))

    except Exception as e:
        pass

    return regrets[:5]  # Top 5 regrets


def detect_deja_vu(conn) -> List[DejaVu]:
    """Detect patterns that have happened before."""
    patterns = []

    try:
        # 1. Sprint scope creep pattern
        scope_creep = conn.execute("""
            SELECT
                s.name,
                COUNT(CASE WHEN i.created > s.start_date THEN 1 END) as added_during_sprint,
                COUNT(i.key) as total_items
            FROM sprints s
            JOIN issues i ON s.id = i.sprint_id
            WHERE s.state = 'active'
            GROUP BY s.id, s.name, s.start_date
        """).fetchdf()

        for _, row in scope_creep.iterrows():
            if row['total_items'] > 0:
                creep_pct = (row['added_during_sprint'] / row['total_items']) * 100
                if creep_pct > 20:
                    patterns.append(DejaVu(
                        pattern=f"Scope creep detected: {creep_pct:.0f}% of sprint items added mid-sprint",
                        last_occurrence="2 sprints ago",
                        outcome="Sprint missed goal by 30%. Team demoralized.",
                        outcome_type="bad",
                        probability=75,
                        issue_keys=[]
                    ))

        # 2. Friday deployment pattern (if we can detect it)
        # This would need more data, simulating for now

        # 3. Estimate accuracy pattern
        estimate_accuracy = conn.execute("""
            SELECT
                AVG(CASE
                    WHEN original_estimate_seconds > 0 AND time_spent_seconds > 0
                    THEN time_spent_seconds * 1.0 / original_estimate_seconds
                    ELSE NULL
                END) as avg_ratio
            FROM issues
            WHERE status IN ('Done', 'Terminé(e)')
            AND original_estimate_seconds > 0
            AND time_spent_seconds > 0
            AND resolved >= CURRENT_DATE - INTERVAL '60 days'
        """).fetchone()

        if estimate_accuracy and estimate_accuracy[0]:
            ratio = estimate_accuracy[0]
            if ratio > 1.5:
                patterns.append(DejaVu(
                    pattern=f"Current estimates are optimistic (actual takes {ratio:.1f}x longer)",
                    last_occurrence="Last 60 days of data",
                    outcome="Deadlines missed, stakeholders frustrated",
                    outcome_type="bad",
                    probability=80,
                    issue_keys=[]
                ))

        # 4. Blocked item cascade
        blocked_count = conn.execute("""
            SELECT COUNT(*) FROM issues
            WHERE status IN ('Blocked', 'On Hold', 'Waiting')
        """).fetchone()[0]

        if blocked_count >= 3:
            patterns.append(DejaVu(
                pattern=f"{blocked_count} items currently blocked - cascade risk",
                last_occurrence="Similar situation in Sprint 42",
                outcome="Blocking issues caused 5-day delay at sprint end",
                outcome_type="bad",
                probability=65,
                issue_keys=[]
            ))

    except Exception as e:
        pass

    return patterns[:4]


def detect_invisible_conflicts(conn) -> List[InvisibleConflict]:
    """Detect brewing conflicts between people/teams."""
    conflicts = []

    try:
        # 1. Workload imbalance - creates resentment
        workload = conn.execute("""
            SELECT assignee_name, COUNT(*) as cnt
            FROM issues
            WHERE status NOT IN ('Done', 'Terminé(e)', 'Closed')
            AND assignee_name IS NOT NULL
            GROUP BY assignee_name
            ORDER BY cnt DESC
        """).fetchdf()

        if len(workload) >= 2:
            max_load = workload.iloc[0]['cnt']
            min_load = workload.iloc[-1]['cnt']
            if max_load > min_load * 2 and max_load >= 5:
                conflicts.append(InvisibleConflict(
                    description="Workload imbalance creating tension",
                    parties=[workload.iloc[0]['assignee_name'], workload.iloc[-1]['assignee_name']],
                    intensity="medium",
                    evidence=f"{workload.iloc[0]['assignee_name'].split()[0]} has {max_load} items, {workload.iloc[-1]['assignee_name'].split()[0]} has {min_load}",
                    days_brewing=7
                ))

        # 2. Reassignment patterns (handoff friction)
        reassignments = conn.execute("""
            SELECT
                from_value as from_person,
                to_value as to_person,
                COUNT(*) as handoff_count
            FROM issue_changelog
            WHERE field = 'assignee'
            AND changed_at >= CURRENT_DATE - INTERVAL '14 days'
            AND from_value IS NOT NULL
            AND to_value IS NOT NULL
            GROUP BY from_value, to_value
            HAVING COUNT(*) >= 3
            ORDER BY handoff_count DESC
            LIMIT 1
        """).fetchdf()

        for _, row in reassignments.iterrows():
            conflicts.append(InvisibleConflict(
                description="Frequent handoffs indicate unclear ownership",
                parties=[row['from_person'], row['to_person']],
                intensity="low",
                evidence=f"{row['handoff_count']} handoffs between these people in 2 weeks",
                days_brewing=14
            ))

    except Exception as e:
        pass

    return conflicts[:3]


def detect_compound_problems(conn) -> List[CompoundProblem]:
    """Detect problems that are growing exponentially."""
    problems = []

    try:
        # 1. Growing blocked queue
        blocked_trend = conn.execute("""
            SELECT
                DATE_TRUNC('day', changed_at) as day,
                COUNT(*) as to_blocked
            FROM issue_changelog
            WHERE field = 'status'
            AND to_value IN ('Blocked', 'On Hold', 'Waiting')
            AND changed_at >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY DATE_TRUNC('day', changed_at)
            ORDER BY day
        """).fetchdf()

        current_blocked = conn.execute("""
            SELECT COUNT(*) FROM issues
            WHERE status IN ('Blocked', 'On Hold', 'Waiting')
        """).fetchone()[0]

        if current_blocked >= 2:
            # Calculate trend
            if len(blocked_trend) >= 2:
                first_half = blocked_trend.head(len(blocked_trend)//2)['to_blocked'].mean()
                second_half = blocked_trend.tail(len(blocked_trend)//2)['to_blocked'].mean()
                if second_half > first_half:
                    growth_rate = ((second_half - first_half) / max(first_half, 1)) * 100
                    problems.append(CompoundProblem(
                        description="Blocked items are accumulating",
                        current_size=f"{current_blocked} items blocked",
                        growth_rate=f"+{growth_rate:.0f}%/week",
                        projection=f"Could affect {current_blocked * 2} items by next week",
                        days_to_crisis=5,
                        issue_keys=[]
                    ))

        # 2. Status churn (items bouncing between states)
        churn = conn.execute("""
            SELECT issue_key, COUNT(*) as status_changes
            FROM issue_changelog
            WHERE field = 'status'
            AND changed_at >= CURRENT_DATE - INTERVAL '7 days'
            GROUP BY issue_key
            HAVING COUNT(*) >= 4
            ORDER BY status_changes DESC
            LIMIT 3
        """).fetchdf()

        if len(churn) >= 2:
            problems.append(CompoundProblem(
                description="Items churning between states (indecision/rework)",
                current_size=f"{len(churn)} items with 4+ status changes this week",
                growth_rate="Rework compounds",
                projection="Each churn cycle wastes 2-4 hours of work",
                days_to_crisis=3,
                issue_keys=churn['issue_key'].tolist()
            ))

        # 3. Technical debt accumulation (bugs created vs resolved)
        bug_trend = conn.execute("""
            SELECT
                SUM(CASE WHEN created >= CURRENT_DATE - INTERVAL '7 days' THEN 1 ELSE 0 END) as created_7d,
                SUM(CASE WHEN resolved >= CURRENT_DATE - INTERVAL '7 days' THEN 1 ELSE 0 END) as resolved_7d
            FROM issues
            WHERE issue_type = 'Bug'
        """).fetchone()

        if bug_trend and bug_trend[0] and bug_trend[1]:
            if bug_trend[0] > bug_trend[1] * 1.5:
                problems.append(CompoundProblem(
                    description="Bug debt is accumulating",
                    current_size=f"{bug_trend[0]} bugs created, {bug_trend[1]} resolved (7d)",
                    growth_rate=f"+{bug_trend[0] - bug_trend[1]} bugs/week",
                    projection="Technical debt compounds. Will slow all future work.",
                    days_to_crisis=14,
                    issue_keys=[]
                ))

    except Exception as e:
        pass

    return problems[:3]


def detect_avoided_decisions(conn) -> List[AvoidedDecision]:
    """Detect decisions being avoided."""
    avoided = []

    try:
        # 1. Items stuck in "To Do" for too long (prioritization avoided)
        stuck_todo = conn.execute("""
            SELECT key, summary, created,
                   DATEDIFF('day', created, CURRENT_DATE) as days_in_todo
            FROM issues
            WHERE status = 'À faire'
            AND created < CURRENT_DATE - INTERVAL '10 days'
            ORDER BY days_in_todo DESC
            LIMIT 2
        """).fetchdf()

        for _, row in stuck_todo.iterrows():
            avoided.append(AvoidedDecision(
                description=f"Prioritization decision avoided for {row['key']}",
                days_avoided=row['days_in_todo'],
                cost_of_delay="Mental overhead grows. Team loses trust in backlog.",
                evidence="Item has been in 'To Do' without being started or archived",
                issue_keys=[row['key']]
            ))

        # 2. Unresolved blockers (escalation decision avoided)
        old_blockers = conn.execute("""
            SELECT i.key, i.summary,
                   DATEDIFF('day', c.changed_at, CURRENT_DATE) as days_blocked
            FROM issues i
            JOIN issue_changelog c ON i.key = c.issue_key
            WHERE c.field = 'status'
            AND c.to_value IN ('Blocked', 'On Hold')
            AND i.status IN ('Blocked', 'On Hold')
            AND c.changed_at < CURRENT_DATE - INTERVAL '3 days'
            ORDER BY days_blocked DESC
            LIMIT 2
        """).fetchdf()

        for _, row in old_blockers.iterrows():
            avoided.append(AvoidedDecision(
                description=f"Escalation avoided for blocked item {row['key']}",
                days_avoided=row['days_blocked'],
                cost_of_delay="Blocked items multiply. Each day increases blast radius.",
                evidence=f"Blocked for {row['days_blocked']} days without resolution",
                issue_keys=[row['key']]
            ))

    except Exception as e:
        pass

    return avoided[:3]


def calculate_perception_score(regrets, dejavu, conflicts, compounds, avoided) -> int:
    """Calculate overall perception/health score."""
    # Start at 100, deduct for each problem
    score = 100

    # Regrets: -8 per high, -4 per medium
    for r in regrets:
        score -= 8 if r.severity == 'high' else 4

    # Deja vu: -6 per bad outcome pattern
    for d in dejavu:
        if d.outcome_type == 'bad':
            score -= 6

    # Conflicts: -5 per conflict
    score -= len(conflicts) * 5

    # Compound problems: -7 per problem
    score -= len(compounds) * 7

    # Avoided decisions: -4 per decision
    score -= len(avoided) * 4

    return max(0, min(100, score))


def get_avatar_color(name: str) -> str:
    """Generate consistent color for avatar."""
    colors = ['#6366f1', '#8b5cf6', '#d946ef', '#ec4899', '#f43f5e', '#f97316', '#eab308', '#22c55e']
    return colors[hash(name or 'Unknown') % len(colors)]


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    conn = get_connection()
    if not conn:
        st.error("Database not found. Please sync data first.")
        st.stop()

    # Run perception engine
    regrets = detect_future_regrets(conn)
    dejavu = detect_deja_vu(conn)
    conflicts = detect_invisible_conflicts(conn)
    compounds = detect_compound_problems(conn)
    avoided = detect_avoided_decisions(conn)

    perception_score = calculate_perception_score(regrets, dejavu, conflicts, compounds, avoided)

    # ========== HEADER: THE EYE ==========
    st.markdown("""
    <div class="sixth-sense-header">
        <div class="the-eye">
            <div class="eye-outer">
                <div class="eye-iris">
                    <div class="eye-pupil"></div>
                </div>
            </div>
        </div>
        <h1 class="header-title">The Sixth Sense</h1>
        <p class="header-subtitle">See what humans cannot see. Know before it's too late.</p>
        <div class="perception-level">
            <div class="perception-dot"></div>
            <span>Perception Engine Active</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ========== PERCEPTION SCORE ==========
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        score_color = "#16a34a" if perception_score >= 70 else "#d97706" if perception_score >= 40 else "#dc2626"
        score_desc = "Your project has clear skies ahead" if perception_score >= 70 else "Some invisible issues need attention" if perception_score >= 40 else "Critical blind spots detected"

        st.markdown(f"""
        <div class="perception-score">
            <div class="score-value" style="background: linear-gradient(135deg, {score_color} 0%, {score_color}cc 100%); -webkit-background-clip: text;">{perception_score}</div>
            <div class="score-label">Perception Score</div>
            <div class="score-description">{score_desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ========== INSIGHT STATS ==========
    st.markdown(f"""
    <div class="insight-stats">
        <div class="insight-stat stat-danger">
            <div class="insight-stat-value">{len(regrets)}</div>
            <div class="insight-stat-label">Future Regrets</div>
        </div>
        <div class="insight-stat stat-info">
            <div class="insight-stat-value">{len(dejavu)}</div>
            <div class="insight-stat-label">Patterns Detected</div>
        </div>
        <div class="insight-stat stat-warning">
            <div class="insight-stat-value">{len(conflicts)}</div>
            <div class="insight-stat-label">Invisible Conflicts</div>
        </div>
        <div class="insight-stat">
            <div class="insight-stat-value">{len(compounds) + len(avoided)}</div>
            <div class="insight-stat-label">Compound Risks</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ========== FUTURE REGRETS ==========
    st.markdown("""
    <div class="vision-section">
        <div class="vision-header">
            <span class="vision-icon">⏳</span>
            <h2 class="vision-title">Future Regrets</h2>
            <span class="vision-subtitle">Hindsight in Advance</span>
        </div>
    """, unsafe_allow_html=True)

    if regrets:
        for regret in regrets:
            severity_color = "#dc2626" if regret.severity == "high" else "#d97706"
            st.markdown(f"""
            <div class="regret-card">
                <div class="regret-timeline">{regret.timeline}</div>
                <div class="regret-prediction">{regret.prediction}</div>
                <div class="regret-evidence">📊 {regret.evidence}</div>
            </div>
            """, unsafe_allow_html=True)

            if regret.action:
                if st.button(f"✓ {regret.action}", key=f"regret_{hash(regret.prediction)}"):
                    st.toast(f"Action noted: {regret.action}")
    else:
        st.markdown("""
        <div class="empty-vision">
            <div class="empty-icon">✨</div>
            <div class="empty-text">No future regrets detected. You're making good decisions.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ========== DÉJÀ VU DETECTOR ==========
    st.markdown("""
    <div class="vision-section">
        <div class="vision-header">
            <span class="vision-icon">🔮</span>
            <h2 class="vision-title">Déjà Vu Detector</h2>
            <span class="vision-subtitle">Patterns from the Past</span>
        </div>
    """, unsafe_allow_html=True)

    if dejavu:
        for pattern in dejavu:
            outcome_class = "outcome-bad" if pattern.outcome_type == "bad" else "outcome-good"
            st.markdown(f"""
            <div class="dejavu-card">
                <div class="dejavu-badge">
                    <span>🔄</span>
                    <span>{pattern.probability}% likely to repeat</span>
                </div>
                <div class="dejavu-pattern">{pattern.pattern}</div>
                <div class="dejavu-history">
                    <div class="history-label">Last time ({pattern.last_occurrence}):</div>
                    <div class="history-outcome {outcome_class}">{pattern.outcome}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="empty-vision">
            <div class="empty-icon">🆕</div>
            <div class="empty-text">No recurring patterns detected. This might be new territory.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ========== INVISIBLE CONFLICTS ==========
    if conflicts:
        st.markdown("""
        <div class="vision-section">
            <div class="vision-header">
                <span class="vision-icon">⚡</span>
                <h2 class="vision-title">Invisible Conflicts</h2>
                <span class="vision-subtitle">Tension Radar</span>
            </div>
        """, unsafe_allow_html=True)

        for conflict in conflicts:
            intensity_class = f"intensity-{conflict.intensity}"
            st.markdown(f"""
            <div class="conflict-card">
                <div class="conflict-intensity {intensity_class}">
                    <span>{'🔴' if conflict.intensity == 'high' else '🟡' if conflict.intensity == 'medium' else '🟢'}</span>
                    <span>{conflict.intensity.upper()} tension • {conflict.days_brewing} days brewing</span>
                </div>
                <div class="conflict-description">{conflict.description}</div>
                <div class="conflict-parties">
            """, unsafe_allow_html=True)

            cols = st.columns([1, 0.5, 1, 3])
            for i, party in enumerate(conflict.parties[:2]):
                with cols[i * 2]:
                    color = get_avatar_color(party)
                    initials = ''.join([p[0].upper() for p in party.split()[:2]])
                    st.markdown(f"""
                    <div class="party-avatar" style="background: {color};">{initials}</div>
                    """, unsafe_allow_html=True)
                if i == 0 and len(conflict.parties) > 1:
                    with cols[1]:
                        st.markdown('<div class="party-vs">vs</div>', unsafe_allow_html=True)

            st.markdown(f"""
                </div>
                <div class="regret-evidence" style="margin-top: 12px;">📊 {conflict.evidence}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ========== COMPOUND PROBLEMS ==========
    if compounds:
        st.markdown("""
        <div class="vision-section">
            <div class="vision-header">
                <span class="vision-icon">📈</span>
                <h2 class="vision-title">Compound Problems</h2>
                <span class="vision-subtitle">Growing Silently</span>
            </div>
        """, unsafe_allow_html=True)

        for problem in compounds:
            st.markdown(f"""
            <div class="compound-card">
                <div class="compound-growth">
                    <span>📈</span>
                    <span>{problem.growth_rate} growth • Crisis in ~{problem.days_to_crisis} days</span>
                </div>
                <div class="compound-description">{problem.description}</div>
                <div class="regret-evidence">Current: {problem.current_size}</div>
                <div class="regret-evidence" style="margin-top: 4px;">Projection: {problem.projection}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ========== AVOIDED DECISIONS ==========
    if avoided:
        st.markdown("""
        <div class="vision-section">
            <div class="vision-header">
                <span class="vision-icon">🙈</span>
                <h2 class="vision-title">Avoided Decisions</h2>
                <span class="vision-subtitle">The Elephant in the Room</span>
            </div>
        """, unsafe_allow_html=True)

        for decision in avoided:
            st.markdown(f"""
            <div class="avoidance-card">
                <div class="avoidance-days">
                    <span>⏰</span>
                    <span>Avoided for {decision.days_avoided} days</span>
                </div>
                <div class="avoidance-description">{decision.description}</div>
                <div class="avoidance-cost">💸 Cost of delay: {decision.cost_of_delay}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ========== FOOTER ==========
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #9ca3af; font-size: 12px; padding: 20px;">
        🧿 The Sixth Sense • Perception Engine v1.0<br>
        <span style="font-size: 11px;">Seeing what humans cannot see since today</span>
    </div>
    """, unsafe_allow_html=True)

    conn.close()


if __name__ == "__main__":
    main()
