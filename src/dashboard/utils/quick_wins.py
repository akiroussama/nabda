"""
⚡ Quick Wins Component Library
Reusable high-value widgets for instant manager productivity.

Each widget answers ONE critical question that would take 15-30 minutes to figure out manually.
Designed to close deals - the "wow" factor that makes clients sign contracts.
"""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import hashlib


# =============================================================================
# PREMIUM CSS FOR ALL QUICK WIN WIDGETS
# =============================================================================

QUICK_WIN_CSS = """
<style>
    /* Base Quick Win Widget */
    .qw-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 24px 28px;
        margin-bottom: 24px;
        color: white;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);
        position: relative;
        overflow: hidden;
    }
    .qw-container::before {
        content: '';
        position: absolute;
        top: -50%; right: -50%;
        width: 100%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 60%);
        pointer-events: none;
    }
    .qw-container.success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 10px 40px rgba(16, 185, 129, 0.4);
    }
    .qw-container.warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        box-shadow: 0 10px 40px rgba(245, 158, 11, 0.4);
    }
    .qw-container.danger {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        box-shadow: 0 10px 40px rgba(239, 68, 68, 0.4);
    }
    .qw-container.info {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.4);
    }
    .qw-container.premium {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
    }

    /* Widget Header */
    .qw-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 16px;
    }
    .qw-badge {
        background: rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }
    .qw-badge .pulse {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #fff;
        animation: qw-pulse 2s infinite;
    }
    @keyframes qw-pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.3); }
    }

    /* Main Content */
    .qw-main {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 24px;
    }
    .qw-left {
        flex: 1;
    }
    .qw-title {
        font-size: 13px;
        opacity: 0.9;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .qw-value {
        font-size: 42px;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 8px;
    }
    .qw-value.small {
        font-size: 28px;
    }
    .qw-subtitle {
        font-size: 14px;
        opacity: 0.85;
    }

    /* Action Box */
    .qw-action-box {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 16px 20px;
        min-width: 200px;
        text-align: center;
    }
    .qw-action-label {
        font-size: 11px;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .qw-action-text {
        font-size: 16px;
        font-weight: 700;
        line-height: 1.3;
    }
    .qw-action-cta {
        margin-top: 12px;
        background: rgba(255,255,255,0.2);
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        cursor: pointer;
        transition: all 0.2s;
    }
    .qw-action-cta:hover {
        background: rgba(255,255,255,0.3);
        transform: translateY(-1px);
    }

    /* Copy-Paste Ready Text */
    .qw-copyable {
        background: rgba(0,0,0,0.2);
        border-radius: 12px;
        padding: 16px;
        margin-top: 16px;
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 13px;
        line-height: 1.5;
        position: relative;
    }
    .qw-copyable::after {
        content: '📋 Click to copy';
        position: absolute;
        top: 8px;
        right: 12px;
        font-size: 10px;
        opacity: 0.6;
        font-family: system-ui;
    }

    /* Stats Row */
    .qw-stats {
        display: flex;
        gap: 24px;
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid rgba(255,255,255,0.2);
    }
    .qw-stat {
        text-align: center;
    }
    .qw-stat-value {
        font-size: 24px;
        font-weight: 800;
    }
    .qw-stat-label {
        font-size: 11px;
        opacity: 0.7;
        text-transform: uppercase;
    }

    /* Confidence Indicator */
    .qw-confidence {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 8px;
    }
    .qw-confidence-bar {
        flex: 1;
        height: 6px;
        background: rgba(255,255,255,0.2);
        border-radius: 3px;
        overflow: hidden;
    }
    .qw-confidence-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease;
    }
    .qw-confidence-text {
        font-size: 12px;
        font-weight: 600;
        min-width: 45px;
    }

    /* Persona Avatar */
    .qw-avatar {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background: rgba(255,255,255,0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        flex-shrink: 0;
    }

    /* List Style */
    .qw-list {
        list-style: none;
        padding: 0;
        margin: 12px 0 0 0;
    }
    .qw-list li {
        padding: 8px 0;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 14px;
    }
    .qw-list li:last-child {
        border-bottom: none;
    }
    .qw-list-icon {
        font-size: 16px;
    }

    /* Time Saved Badge */
    .qw-time-saved {
        position: absolute;
        top: 16px;
        right: 16px;
        background: rgba(255,255,255,0.2);
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 10px;
        font-weight: 600;
    }
</style>
"""


def inject_quick_win_css():
    """Inject Quick Win CSS into the page. Call once at top of each page."""
    st.markdown(QUICK_WIN_CSS, unsafe_allow_html=True)


# =============================================================================
# WIDGET BUILDERS
# =============================================================================

def render_standup_script(
    team_status: str,
    blockers: List[str],
    wins: List[str],
    focus: str,
    time_saved_min: int = 15
) -> None:
    """
    📋 Standup Script Widget
    Page: Overview, Sprint Health
    Value: Ready-to-read standup update in 10 seconds
    """
    blocker_text = " • ".join(blockers[:3]) if blockers else "None"
    wins_text = " • ".join(wins[:2]) if wins else "Steady progress"

    script = f"Team is {team_status}. Blockers: {blocker_text}. Wins: {wins_text}. Focus: {focus}."

    st.markdown(f"""
    <div class="qw-container">
        <div class="qw-time-saved">⏱️ {time_saved_min} min saved</div>
        <div class="qw-header">
            <span class="qw-badge"><span class="pulse"></span> STANDUP READY</span>
        </div>
        <div class="qw-main">
            <div class="qw-left">
                <div class="qw-title">Your 30-Second Standup Script</div>
                <div class="qw-value small">"{team_status}"</div>
                <div class="qw-subtitle">Copy-paste ready for your meeting</div>
            </div>
            <div class="qw-action-box">
                <div class="qw-action-label">Today's Focus</div>
                <div class="qw-action-text">{focus}</div>
            </div>
        </div>
        <div class="qw-copyable">{script}</div>
    </div>
    """, unsafe_allow_html=True)


def render_meeting_talking_point(
    topic: str,
    insight: str,
    recommendation: str,
    confidence: int,
    meeting_type: str = "Status Update",
    time_saved_min: int = 20
) -> None:
    """
    💬 Meeting Talking Point Widget
    Page: Any page
    Value: Instant talking point for any meeting
    """
    conf_color = "#22c55e" if confidence >= 80 else "#f59e0b" if confidence >= 60 else "#ef4444"
    variant = "success" if confidence >= 80 else "warning" if confidence >= 60 else "danger"

    st.markdown(f"""
    <div class="qw-container {variant}">
        <div class="qw-time-saved">⏱️ {time_saved_min} min saved</div>
        <div class="qw-header">
            <span class="qw-badge"><span class="pulse"></span> {meeting_type.upper()}</span>
        </div>
        <div class="qw-main">
            <div class="qw-left">
                <div class="qw-title">Key Talking Point</div>
                <div class="qw-value small">{topic}</div>
                <div class="qw-subtitle">{insight}</div>
                <div class="qw-confidence">
                    <div class="qw-confidence-bar">
                        <div class="qw-confidence-fill" style="width: {confidence}%; background: {conf_color};"></div>
                    </div>
                    <span class="qw-confidence-text">{confidence}% conf</span>
                </div>
            </div>
            <div class="qw-action-box">
                <div class="qw-action-label">Recommendation</div>
                <div class="qw-action-text">{recommendation}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_risk_alert(
    risk_title: str,
    impact: str,
    action: str,
    severity: str = "warning",  # success, warning, danger
    metric_value: str = "",
    metric_label: str = "",
    time_saved_min: int = 30
) -> None:
    """
    🚨 Risk Alert Widget
    Page: Predictions, Burnout, Delivery Forecast
    Value: Early warning with clear action
    """
    icon = {"success": "✅", "warning": "⚠️", "danger": "🚨"}.get(severity, "ℹ️")

    st.markdown(f"""
    <div class="qw-container {severity}">
        <div class="qw-time-saved">⏱️ {time_saved_min} min saved</div>
        <div class="qw-header">
            <span class="qw-badge"><span class="pulse"></span> {icon} RISK ALERT</span>
        </div>
        <div class="qw-main">
            <div class="qw-left">
                <div class="qw-title">Requires Attention</div>
                <div class="qw-value small">{risk_title}</div>
                <div class="qw-subtitle">{impact}</div>
            </div>
            <div class="qw-action-box">
                <div class="qw-action-label">Recommended Action</div>
                <div class="qw-action-text">{action}</div>
                {f'<div style="margin-top: 12px; font-size: 24px; font-weight: 800;">{metric_value}</div><div style="font-size: 11px; opacity: 0.7;">{metric_label}</div>' if metric_value else ''}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_person_focus(
    person_name: str,
    reason: str,
    talking_points: List[str],
    urgency: str = "normal",  # low, normal, high
    avatar_emoji: str = "👤",
    time_saved_min: int = 25
) -> None:
    """
    👤 Person Focus Widget
    Page: Team Workload, Burnout Risk
    Value: Who needs your attention TODAY
    """
    variant = {"low": "info", "normal": "warning", "high": "danger"}.get(urgency, "")
    urgency_text = {"low": "Check In", "normal": "Follow Up", "high": "Urgent"}.get(urgency, "")

    points_html = "".join([f'<li><span class="qw-list-icon">→</span> {p}</li>' for p in talking_points[:3]])

    st.markdown(f"""
    <div class="qw-container {variant}">
        <div class="qw-time-saved">⏱️ {time_saved_min} min saved</div>
        <div class="qw-header">
            <span class="qw-badge"><span class="pulse"></span> {urgency_text.upper()}</span>
        </div>
        <div class="qw-main">
            <div class="qw-avatar">{avatar_emoji}</div>
            <div class="qw-left" style="margin-left: 16px;">
                <div class="qw-title">Talk To Today</div>
                <div class="qw-value small">{person_name}</div>
                <div class="qw-subtitle">{reason}</div>
                <ul class="qw-list">{points_html}</ul>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_slack_message(
    recipient: str,
    message: str,
    context: str,
    message_type: str = "update",  # update, ask, celebrate, alert
    time_saved_min: int = 10
) -> None:
    """
    💬 Slack Message Widget
    Page: Any page
    Value: Pre-written message ready to send
    """
    type_emoji = {"update": "📊", "ask": "❓", "celebrate": "🎉", "alert": "🚨"}.get(message_type, "💬")

    st.markdown(f"""
    <div class="qw-container premium">
        <div class="qw-time-saved">⏱️ {time_saved_min} min saved</div>
        <div class="qw-header">
            <span class="qw-badge">{type_emoji} SLACK READY</span>
        </div>
        <div class="qw-main">
            <div class="qw-left">
                <div class="qw-title">Send to {recipient}</div>
                <div class="qw-subtitle" style="margin-top: 8px;">{context}</div>
            </div>
        </div>
        <div class="qw-copyable">{message}</div>
    </div>
    """, unsafe_allow_html=True)


def render_decision_helper(
    question: str,
    recommendation: str,
    confidence: int,
    pros: List[str],
    cons: List[str],
    time_saved_min: int = 45
) -> None:
    """
    🤔 Decision Helper Widget
    Page: Strategic Gap, Predictions
    Value: Make a decision in 30 seconds
    """
    conf_color = "#22c55e" if confidence >= 70 else "#f59e0b"

    pros_html = "".join([f'<li><span class="qw-list-icon">✅</span> {p}</li>' for p in pros[:2]])
    cons_html = "".join([f'<li><span class="qw-list-icon">⚠️</span> {c}</li>' for c in cons[:2]])

    st.markdown(f"""
    <div class="qw-container">
        <div class="qw-time-saved">⏱️ {time_saved_min} min saved</div>
        <div class="qw-header">
            <span class="qw-badge"><span class="pulse"></span> DECISION HELPER</span>
        </div>
        <div class="qw-main">
            <div class="qw-left">
                <div class="qw-title">Question</div>
                <div class="qw-value small">{question}</div>
                <div class="qw-confidence">
                    <div class="qw-confidence-bar">
                        <div class="qw-confidence-fill" style="width: {confidence}%; background: {conf_color};"></div>
                    </div>
                    <span class="qw-confidence-text">{confidence}% conf</span>
                </div>
            </div>
            <div class="qw-action-box">
                <div class="qw-action-label">AI Recommendation</div>
                <div class="qw-action-text">{recommendation}</div>
            </div>
        </div>
        <div class="qw-stats">
            <div style="flex: 1;"><ul class="qw-list">{pros_html}</ul></div>
            <div style="flex: 1;"><ul class="qw-list">{cons_html}</ul></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_promise_date(
    date_str: str,
    confidence: int,
    buffer_days: int,
    context: str,
    time_saved_min: int = 60
) -> None:
    """
    📅 Promise Date Widget
    Page: Delivery Forecast, Sprint Health
    Value: The date to tell stakeholders
    """
    variant = "success" if confidence >= 85 else "warning" if confidence >= 70 else "danger"

    st.markdown(f"""
    <div class="qw-container {variant}">
        <div class="qw-time-saved">⏱️ {time_saved_min} min saved</div>
        <div class="qw-header">
            <span class="qw-badge"><span class="pulse"></span> STAKEHOLDER READY</span>
        </div>
        <div class="qw-main">
            <div class="qw-left">
                <div class="qw-title">Tell Stakeholders This Date</div>
                <div class="qw-value">{date_str}</div>
                <div class="qw-subtitle">{context}</div>
            </div>
            <div class="qw-action-box">
                <div class="qw-action-label">Confidence</div>
                <div style="font-size: 36px; font-weight: 800;">{confidence}%</div>
                <div style="font-size: 12px; opacity: 0.8; margin-top: 4px;">+{buffer_days} days buffer included</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_capacity_insight(
    available_points: int,
    recommended_load: int,
    overloaded_count: int,
    underloaded_count: int,
    action: str,
    time_saved_min: int = 20
) -> None:
    """
    ⚖️ Capacity Insight Widget
    Page: Team Workload
    Value: Instant capacity decision
    """
    variant = "success" if overloaded_count == 0 else "warning" if overloaded_count <= 2 else "danger"

    st.markdown(f"""
    <div class="qw-container {variant}">
        <div class="qw-time-saved">⏱️ {time_saved_min} min saved</div>
        <div class="qw-header">
            <span class="qw-badge"><span class="pulse"></span> CAPACITY CHECK</span>
        </div>
        <div class="qw-main">
            <div class="qw-left">
                <div class="qw-title">Sprint Capacity Status</div>
                <div class="qw-value">{available_points} pts</div>
                <div class="qw-subtitle">Available capacity this sprint</div>
            </div>
            <div class="qw-action-box">
                <div class="qw-action-label">Action Needed</div>
                <div class="qw-action-text">{action}</div>
            </div>
        </div>
        <div class="qw-stats">
            <div class="qw-stat">
                <div class="qw-stat-value">{overloaded_count}</div>
                <div class="qw-stat-label">Overloaded</div>
            </div>
            <div class="qw-stat">
                <div class="qw-stat-value">{underloaded_count}</div>
                <div class="qw-stat-label">Can Take More</div>
            </div>
            <div class="qw-stat">
                <div class="qw-stat-value">{recommended_load}</div>
                <div class="qw-stat-label">Ideal Load/Person</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_board_insight(
    bottleneck_column: str,
    bottleneck_count: int,
    oldest_card_days: int,
    wip_status: str,
    action: str,
    time_saved_min: int = 15
) -> None:
    """
    📋 Board Insight Widget
    Page: Board (Kanban)
    Value: Instant board health check
    """
    variant = "success" if bottleneck_count <= 3 else "warning" if bottleneck_count <= 6 else "danger"

    st.markdown(f"""
    <div class="qw-container {variant}">
        <div class="qw-time-saved">⏱️ {time_saved_min} min saved</div>
        <div class="qw-header">
            <span class="qw-badge"><span class="pulse"></span> BOARD HEALTH</span>
        </div>
        <div class="qw-main">
            <div class="qw-left">
                <div class="qw-title">Flow Bottleneck</div>
                <div class="qw-value small">{bottleneck_column}</div>
                <div class="qw-subtitle">{bottleneck_count} cards stuck • Oldest: {oldest_card_days} days</div>
            </div>
            <div class="qw-action-box">
                <div class="qw-action-label">WIP Status</div>
                <div class="qw-action-text">{wip_status}</div>
                <div class="qw-action-cta">→ {action}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sprint_rescue(
    at_risk_count: int,
    rescue_action: str,
    points_at_risk: int,
    completion_pct: int,
    days_left: int,
    time_saved_min: int = 30
) -> None:
    """
    🏃 Sprint Rescue Widget
    Page: Sprint Health
    Value: How to save the sprint
    """
    variant = "success" if at_risk_count == 0 else "warning" if at_risk_count <= 3 else "danger"

    st.markdown(f"""
    <div class="qw-container {variant}">
        <div class="qw-time-saved">⏱️ {time_saved_min} min saved</div>
        <div class="qw-header">
            <span class="qw-badge"><span class="pulse"></span> SPRINT STATUS</span>
        </div>
        <div class="qw-main">
            <div class="qw-left">
                <div class="qw-title">Sprint Rescue Plan</div>
                <div class="qw-value">{at_risk_count} at risk</div>
                <div class="qw-subtitle">{points_at_risk} story points may not complete</div>
            </div>
            <div class="qw-action-box">
                <div class="qw-action-label">To Complete Sprint</div>
                <div class="qw-action-text">{rescue_action}</div>
            </div>
        </div>
        <div class="qw-stats">
            <div class="qw-stat">
                <div class="qw-stat-value">{completion_pct}%</div>
                <div class="qw-stat-label">Complete</div>
            </div>
            <div class="qw-stat">
                <div class="qw-stat-value">{days_left}</div>
                <div class="qw-stat-label">Days Left</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_email_draft(
    subject: str,
    recipient: str,
    body: str,
    email_type: str = "update",  # update, escalation, celebration
    time_saved_min: int = 15
) -> None:
    """
    📧 Email Draft Widget
    Page: Reports, CEO Command Center
    Value: Pre-written email ready to send
    """
    type_config = {
        "update": ("📊", "STATUS UPDATE"),
        "escalation": ("🚨", "ESCALATION"),
        "celebration": ("🎉", "GOOD NEWS")
    }
    emoji, label = type_config.get(email_type, ("📧", "EMAIL"))

    st.markdown(f"""
    <div class="qw-container premium">
        <div class="qw-time-saved">⏱️ {time_saved_min} min saved</div>
        <div class="qw-header">
            <span class="qw-badge">{emoji} {label}</span>
        </div>
        <div class="qw-main">
            <div class="qw-left">
                <div class="qw-title">To: {recipient}</div>
                <div class="qw-value small">Subject: {subject}</div>
            </div>
        </div>
        <div class="qw-copyable">{body}</div>
    </div>
    """, unsafe_allow_html=True)


def render_ceo_oneliner(
    headline: str,
    metric: str,
    trend: str,
    recommendation: str,
    time_saved_min: int = 30
) -> None:
    """
    🎯 CEO One-Liner Widget
    Page: CEO Command Center, Executive Cockpit
    Value: What to tell the CEO in the elevator
    """
    trend_color = "#22c55e" if "up" in trend.lower() or "good" in trend.lower() else "#f59e0b" if "stable" in trend.lower() else "#ef4444"

    st.markdown(f"""
    <div class="qw-container premium">
        <div class="qw-time-saved">⏱️ {time_saved_min} min saved</div>
        <div class="qw-header">
            <span class="qw-badge"><span class="pulse"></span> CEO BRIEFING</span>
        </div>
        <div class="qw-main">
            <div class="qw-left">
                <div class="qw-title">30-Second Elevator Pitch</div>
                <div class="qw-value small">"{headline}"</div>
                <div class="qw-subtitle" style="color: {trend_color}; margin-top: 8px;">{trend}</div>
            </div>
            <div class="qw-action-box">
                <div class="qw-action-label">Key Metric</div>
                <div style="font-size: 32px; font-weight: 800;">{metric}</div>
                <div style="font-size: 12px; opacity: 0.8; margin-top: 8px;">{recommendation}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_morning_priority(
    priority_task: str,
    reason: str,
    blocked_by: Optional[str],
    impact: str,
    time_saved_min: int = 20
) -> None:
    """
    🌅 Morning Priority Widget
    Page: Good Morning
    Value: Your #1 priority for today
    """
    variant = "danger" if blocked_by else "success"

    st.markdown(f"""
    <div class="qw-container {variant}">
        <div class="qw-time-saved">⏱️ {time_saved_min} min saved</div>
        <div class="qw-header">
            <span class="qw-badge"><span class="pulse"></span> #1 PRIORITY TODAY</span>
        </div>
        <div class="qw-main">
            <div class="qw-left">
                <div class="qw-title">Focus On This First</div>
                <div class="qw-value small">{priority_task}</div>
                <div class="qw-subtitle">{reason}</div>
                {f'<div style="margin-top: 8px; color: #ff6b81;">⚠️ Blocked by: {blocked_by}</div>' if blocked_by else ''}
            </div>
            <div class="qw-action-box">
                <div class="qw-action-label">Impact</div>
                <div class="qw-action-text">{impact}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_ceremony_guide(
    ceremony_type: str,
    agenda_items: List[str],
    duration: int,
    key_question: str,
    time_saved_min: int = 20
) -> None:
    """
    🎖️ Ceremony Guide Widget
    Page: Scrum Master HQ
    Value: Run your ceremony in 60 seconds
    """
    ceremony_emoji = {
        "standup": "🧍",
        "retro": "🔄",
        "planning": "📋",
        "review": "🎯",
        "refinement": "✨"
    }.get(ceremony_type.lower(), "📅")

    agenda_html = "".join([f'<li><span class="qw-list-icon">→</span> {item}</li>' for item in agenda_items[:4]])

    st.markdown(f"""
    <div class="qw-container">
        <div class="qw-time-saved">⏱️ {time_saved_min} min saved</div>
        <div class="qw-header">
            <span class="qw-badge">{ceremony_emoji} {ceremony_type.upper()}</span>
        </div>
        <div class="qw-main">
            <div class="qw-left">
                <div class="qw-title">Today's Ceremony Guide</div>
                <div class="qw-value small">{duration} min • {len(agenda_items)} items</div>
                <ul class="qw-list">{agenda_html}</ul>
            </div>
            <div class="qw-action-box">
                <div class="qw-action-label">Key Question</div>
                <div class="qw-action-text">"{key_question}"</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_person_avatar_emoji(name: str) -> str:
    """Get consistent emoji avatar for a person."""
    avatars = ["👨‍💻", "👩‍💻", "🧑‍💻", "👨‍🔬", "👩‍🔬", "🧑‍🔬", "👨‍💼", "👩‍💼"]
    hash_val = int(hashlib.md5(name.encode()).hexdigest(), 16)
    return avatars[hash_val % len(avatars)]


def calculate_time_saved_total(widgets_used: int) -> int:
    """Calculate total time saved based on widgets used."""
    return widgets_used * 20  # Average 20 min per widget
