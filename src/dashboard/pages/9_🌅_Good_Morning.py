"""
Good Morning Dashboard - Your Personalized Morning Briefing.

The ultimate command center for starting your day with clarity.
Combines AI-powered insights with evidence-based data.
"""

import streamlit as st
import pandas as pd
import duckdb
import json
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Any

st.set_page_config(
    page_title="Good Morning Dashboard",
    page_icon="sunrise",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for the dashboard
st.markdown("""
<style>
    /* Main layout */
    .main > div {
        padding-top: 1rem;
    }

    /* Zone header */
    .zone-header {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    /* Morning brief card */
    .brief-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    .brief-card-morning {
        background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
    }

    .brief-card-afternoon {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
    }

    .brief-card-evening {
        background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
    }

    .brief-greeting {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .brief-time {
        font-size: 0.875rem;
        opacity: 0.9;
    }

    /* Decision queue items */
    .decision-item {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }

    .decision-item.critical {
        border-left: 4px solid #ef4444;
    }

    .decision-item.high {
        border-left: 4px solid #f97316;
    }

    .decision-item.medium {
        border-left: 4px solid #eab308;
    }

    .decision-item.low {
        border-left: 4px solid #10b981;
    }

    /* Vital tiles */
    .vital-tile {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }

    .vital-value {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
    }

    .vital-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
    }

    .vital-delta {
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }

    .vital-delta.positive { color: #10b981; }
    .vital-delta.negative { color: #ef4444; }
    .vital-delta.neutral { color: #6b7280; }

    /* Evidence badge */
    .evidence-badge {
        background: #f3f4f6;
        color: #374151;
        padding: 0.125rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-family: monospace;
    }

    /* Status indicators */
    .status-critical { color: #dc2626; }
    .status-high { color: #f97316; }
    .status-medium { color: #eab308; }
    .status-low { color: #10b981; }

    /* Good news styling */
    .good-news {
        background: linear-gradient(90deg, #ecfdf5 0%, #f0fdf4 100%);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }

    /* Concern styling */
    .concern {
        background: linear-gradient(90deg, #fefce8 0%, #fef9c3 100%);
        border-left: 4px solid #eab308;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }

    /* Risk styling */
    .risk {
        background: linear-gradient(90deg, #fef2f2 0%, #fee2e2 100%);
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def get_connection():
    """Get database connection."""
    db_path = Path("data/jira.duckdb")
    if not db_path.exists():
        return None
    return duckdb.connect(str(db_path), read_only=False)


def get_time_greeting() -> tuple[str, str, str]:
    """Get time-appropriate greeting and CSS class."""
    hour = datetime.now().hour
    if hour < 12:
        return "Good Morning", "brief-card-morning", "sunrise"
    elif hour < 17:
        return "Good Afternoon", "brief-card-afternoon", "sun"
    else:
        return "Good Evening", "brief-card-evening", "moon"


def format_delta(current: float, previous: float) -> tuple[str, str]:
    """Format a delta value with arrow and color class."""
    if previous == 0:
        return "---", "neutral"

    delta = ((current - previous) / previous) * 100

    if delta > 5:
        return f"+{delta:.0f}%", "positive"
    elif delta < -5:
        return f"{delta:.0f}%", "negative"
    else:
        return "stable", "neutral"


def severity_to_color(severity: str) -> str:
    """Convert severity to CSS color class."""
    return {
        "critical": "status-critical",
        "high": "status-high",
        "medium": "status-medium",
        "low": "status-low",
    }.get(severity.lower(), "")


def render_zone_a(project_key: str, timeframe: str):
    """Render Zone A: Context Controls."""
    greeting, css_class, icon = get_time_greeting()

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.markdown(f"""
        <div class="brief-card {css_class}">
            <div class="brief-greeting">{greeting}!</div>
            <div class="brief-time">
                {datetime.now().strftime('%A, %B %d, %Y')} at {datetime.now().strftime('%I:%M %p')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Timeframe selector
        new_timeframe = st.radio(
            "View",
            options=["daily", "weekly", "monthly"],
            index=["daily", "weekly", "monthly"].index(timeframe),
            horizontal=True,
            key="timeframe_selector",
            label_visibility="collapsed",
        )
        if new_timeframe != timeframe:
            st.session_state.timeframe = new_timeframe
            st.session_state.briefing_generated = False
            st.rerun()

    with col3:
        st.markdown(f"**Project:** `{project_key}`")


def render_zone_b(briefing: dict[str, Any]):
    """Render Zone B: Morning Brief Narrative."""
    st.markdown("### Your Briefing")

    narrative = briefing.get("narrative", "")

    # Parse the narrative into sections
    parsed = briefing.get("parsed", {})

    # Good News Section
    good_news = parsed.get("good_news", [])
    if good_news:
        st.markdown("#### The Good News")
        for item in good_news:
            st.markdown(f"""
            <div class="good-news">
                {item}
            </div>
            """, unsafe_allow_html=True)

    # Concerns Section
    concerns = parsed.get("concerns", [])
    if concerns:
        st.markdown("#### The Concern")
        for item in concerns:
            st.markdown(f"""
            <div class="concern">
                {item}
            </div>
            """, unsafe_allow_html=True)

    # Risks Section
    risks = parsed.get("risks", [])
    if risks:
        st.markdown("#### The Risk")
        for item in risks:
            st.markdown(f"""
            <div class="risk">
                {item}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="good-news">
            <strong>No critical risks today</strong>
        </div>
        """, unsafe_allow_html=True)

    # Top Recommendation
    recommendation = parsed.get("top_recommendation", "")
    if recommendation:
        st.markdown("#### My Top Recommendation")
        st.info(recommendation)

    # Show raw narrative in expander
    with st.expander("View Full Narrative"):
        st.markdown(narrative)


def render_zone_c(decision_queue: list[dict[str, Any]]):
    """Render Zone C: Decision Queue."""
    st.markdown("### Decision Queue")

    if not decision_queue:
        st.info("No items requiring immediate attention.")
        return

    st.markdown(f"_{len(decision_queue)} items need your attention (showing top 5)_")

    for item in decision_queue[:5]:
        severity = item.get("severity", "medium")
        color_class = severity_to_color(severity)

        with st.container():
            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"""
                <div class="decision-item {severity}">
                    <strong class="{color_class}">#{item.get('priority', '?')}</strong>
                    <strong>{item.get('ticket_key', 'N/A')}</strong>: {item.get('ticket_summary', 'No summary')[:80]}
                    <br>
                    <small style="color: #6b7280;">
                        {item.get('reason', 'Unknown reason')} - {item.get('evidence', {}).get('description', '')}
                    </small>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if item.get("draft_message"):
                    if st.button("Copy Draft", key=f"copy_{item.get('ticket_key', '')}"):
                        st.code(item.get("draft_message", ""), language=None)

        # Show suggested action
        if item.get("suggested_action"):
            st.caption(f"Suggested: {item.get('suggested_action')}")


def render_zone_d(delta: dict[str, Any], comparison: dict[str, Any]):
    """Render Zone D: Vital Signs."""
    st.markdown("### Quick Stats")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        completed = delta.get("tickets_completed", 0)
        prev_completed = comparison.get("previous_tickets_completed", 0)
        delta_str, delta_class = format_delta(completed, prev_completed)
        st.metric(
            label="Completed",
            value=completed,
            delta=delta_str if delta_str != "---" else None,
        )

    with col2:
        points = delta.get("points_completed", 0)
        st.metric(label="Points", value=f"{points:.0f}")

    with col3:
        blockers = delta.get("active_blockers", 0)
        st.metric(
            label="Blockers",
            value=blockers,
            delta=f"-{delta.get('resolved_blockers', 0)}" if delta.get('resolved_blockers', 0) > 0 else None,
            delta_color="normal",
        )

    with col4:
        created = delta.get("tickets_created", 0)
        st.metric(label="Created", value=created)

    with col5:
        transitions = delta.get("status_transitions", 0)
        regressions = delta.get("regressions", 0)
        st.metric(
            label="Transitions",
            value=transitions,
            delta=f"{regressions} regressions" if regressions > 0 else None,
            delta_color="inverse",
        )

    with col6:
        after_hours = delta.get("after_hours_events", 0)
        weekend = delta.get("weekend_events", 0)
        total_extra = after_hours + weekend
        st.metric(
            label="After-Hours",
            value=total_extra,
            delta="Watch burnout" if total_extra > 5 else None,
            delta_color="inverse" if total_extra > 5 else "off",
        )


def render_zone_e(
    attention_items: list[dict[str, Any]],
    delta: dict[str, Any],
):
    """Render Zone E: Deep Dive Intelligence."""
    st.markdown("### Deep Dive")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Progress",
        "Blockers",
        "Team Health",
        "All Attention Items",
    ])

    with tab1:
        st.markdown("#### Completed Tickets")
        completed = delta.get("completed_tickets", [])
        if completed:
            df = pd.DataFrame(completed)
            st.dataframe(
                df[["ticket_key", "summary", "points", "assignee"]].head(10),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No tickets completed in this period.")

        st.markdown("#### Created Tickets")
        created = delta.get("created_tickets", [])
        if created:
            df = pd.DataFrame(created)
            st.dataframe(
                df[["ticket_key", "summary", "type", "points"]].head(10),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No tickets created in this period.")

    with tab2:
        st.markdown("#### Active Blockers")
        blockers = delta.get("blocker_tickets", [])
        if blockers:
            df = pd.DataFrame(blockers)
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
            )
        else:
            st.success("No active blockers!")

    with tab3:
        st.markdown("#### After-Hours Activity")
        after_hours = delta.get("after_hours_events", 0)
        weekend = delta.get("weekend_events", 0)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("After-Hours Events", after_hours)
        with col2:
            st.metric("Weekend Events", weekend)

        if after_hours > 5 or weekend > 3:
            st.warning(
                "Elevated after-hours activity detected. "
                "Consider checking in with team members about workload."
            )
        else:
            st.success("Team activity patterns look healthy.")

    with tab4:
        st.markdown("#### All Items Needing Attention")
        if attention_items:
            for item in attention_items:
                severity = item.get("severity", "medium")
                color = severity_to_color(severity)

                with st.expander(
                    f"{item.get('ticket_key', 'N/A')}: {item.get('reason', 'Unknown')}",
                    expanded=item.get("severity") == "critical",
                ):
                    st.markdown(f"""
                    **Severity:** <span class="{color}">{severity.upper()}</span>

                    **Evidence:** {item.get('evidence', {}).get('description', 'N/A')}

                    **Suggested Action:** {item.get('suggested_action', 'N/A')}

                    **Attention Score:** {item.get('attention_score', 0):.1f}
                    """, unsafe_allow_html=True)

                    if item.get("draft_message"):
                        st.text_area(
                            "Draft Message",
                            value=item.get("draft_message", ""),
                            key=f"draft_{item.get('ticket_key', '')}_{item.get('reason', '')}",
                        )
        else:
            st.success("No items requiring attention!")


def generate_briefing_data(
    conn: duckdb.DuckDBPyConnection,
    project_key: str,
    timeframe: str,
) -> dict[str, Any]:
    """Generate the complete briefing data."""
    from src.features.delta_engine import DeltaEngine

    engine = DeltaEngine(conn)
    context = engine.get_timeframe_context(timeframe)

    # Compute delta
    delta = engine.compute_delta(project_key, context)

    # Get attention items
    attention_items = engine.detect_attention_items(project_key, context)

    # Get comparison metrics
    comparison = engine.get_comparison_metrics(project_key, context)

    # Get sprint context
    sprint_info = engine.get_sprint_context(project_key)

    # Generate briefing narrative (simplified without LLM for now)
    briefing = generate_simple_briefing(
        project_key=project_key,
        timeframe=timeframe,
        delta=delta,
        attention_items=attention_items,
        comparison=comparison,
    )

    # Generate decision queue
    decision_queue = [
        item.to_dict() | {"priority": idx + 1}
        for idx, item in enumerate(attention_items[:5])
    ]

    return {
        "briefing": briefing,
        "decision_queue": decision_queue,
        "delta": delta.to_dict(),
        "attention_items": [item.to_dict() for item in attention_items],
        "comparison": comparison.to_dict(),
        "sprint_info": sprint_info,
    }


def generate_simple_briefing(
    project_key: str,
    timeframe: str,
    delta,
    attention_items: list,
    comparison,
) -> dict[str, Any]:
    """Generate a simple briefing without LLM."""
    greeting, _, _ = get_time_greeting()

    completed = delta.tickets_completed
    points = delta.points_completed
    blockers = delta.active_blockers

    # Build good news
    good_news = []
    if completed > 0:
        ticket_keys = ", ".join([t["ticket_key"] for t in delta.completed_tickets[:3]])
        good_news.append(
            f"**{completed} ticket(s) completed** this period ({points:.0f} points) "
            f"[EVIDENCE: {ticket_keys}]"
        )

    if comparison.trend == "up" and comparison.velocity_change_percent > 10:
        good_news.append(
            f"Velocity is **up {comparison.velocity_change_percent:.0f}%** vs last period "
            "[DATA: velocity_trend]"
        )

    if not good_news:
        good_news.append("Team is making progress on the backlog")

    # Build concerns
    concerns = []
    if blockers > 0:
        blocker_keys = ", ".join([t["ticket_key"] for t in delta.blocker_tickets[:3]])
        concerns.append(
            f"**{blockers} active blocker(s)** in the project [EVIDENCE: {blocker_keys}]"
        )

    if delta.regressions > 0:
        concerns.append(
            f"**{delta.regressions} regression(s)** detected - tickets moved back from Done "
            "[DATA: regressions]"
        )

    if comparison.trend == "down" and comparison.velocity_change_percent < -10:
        concerns.append(
            f"Velocity is **down {abs(comparison.velocity_change_percent):.0f}%** vs last period "
            "[DATA: velocity_trend]"
        )

    # Build risks
    risks = []
    critical_items = [
        item for item in attention_items
        if item.severity.value in ("critical", "high")
    ]
    for item in critical_items[:2]:
        risks.append(
            f"**{item.ticket_key}**: {item.evidence.description} "
            f"[EVIDENCE: {item.ticket_key}]"
        )

    # Build recommendation
    if critical_items:
        top_item = critical_items[0]
        recommendation = (
            f"Focus on unblocking **{top_item.ticket_key}** today. "
            f"{top_item.suggested_action}"
        )
    elif blockers > 0:
        recommendation = "Review and prioritize unblocking the active blockers."
    elif delta.tickets_created > delta.tickets_completed:
        recommendation = (
            "More tickets created than completed this period. "
            "Consider reviewing incoming work to prevent backlog growth."
        )
    else:
        recommendation = "Keep up the good work! Monitor progress and stay proactive."

    # Build narrative
    narrative = f"""## {greeting}!

Here's your {timeframe} update for **{project_key}**.

---

### The Good News
{chr(10).join(['- ' + item for item in good_news])}

### The Concern
{chr(10).join(['- ' + item for item in concerns]) if concerns else '- No major concerns today'}

### The Risk
{chr(10).join(['- ' + item for item in risks]) if risks else '- No critical risks today'}

### My Top Recommendation
{recommendation}

---

### Quick Stats ({timeframe})
| Metric | Value | vs. Previous |
|--------|-------|--------------|
| Completed | {completed} | {comparison.velocity_change_percent:+.0f}% |
| Points | {points:.0f} | - |
| Blockers | {blockers} | - |
| Created | {delta.tickets_created} | - |

---

### Heads Up
- Monitor blocked items for resolution
- Check in with team members on stalled work
"""

    return {
        "narrative": narrative,
        "parsed": {
            "greeting": f"{greeting}!",
            "good_news": good_news,
            "concerns": concerns,
            "risks": risks,
            "top_recommendation": recommendation,
        },
        "evidence_citations": [],
    }


def main():
    """Main dashboard function."""
    st.title("Good Morning Dashboard")

    # Initialize session state
    if "timeframe" not in st.session_state:
        st.session_state.timeframe = "daily"
    if "briefing_generated" not in st.session_state:
        st.session_state.briefing_generated = False
    if "briefing_data" not in st.session_state:
        st.session_state.briefing_data = None

    # Get connection
    conn = get_connection()
    if not conn:
        st.error(
            "Database not found. Please sync data first using the CLI:\n\n"
            "`jira-copilot sync`"
        )
        st.stop()

    # Get available projects
    try:
        projects = conn.execute(
            "SELECT DISTINCT project_key FROM issues WHERE project_key IS NOT NULL"
        ).fetchall()
        project_keys = [p[0] for p in projects if p[0]]
    except Exception as e:
        st.error(f"Failed to fetch projects: {e}")
        st.stop()

    if not project_keys:
        st.warning("No projects found. Please sync data first.")
        st.stop()

    # Project selector in sidebar
    with st.sidebar:
        st.markdown("### Settings")
        project_key = st.selectbox(
            "Project",
            options=project_keys,
            index=0,
        )

        if st.button("Regenerate Briefing", type="primary"):
            st.session_state.briefing_generated = False

    # Render Zone A: Context Controls
    render_zone_a(project_key, st.session_state.timeframe)

    # Generate briefing if needed
    if not st.session_state.briefing_generated or st.session_state.briefing_data is None:
        with st.spinner("Generating your personalized briefing..."):
            try:
                st.session_state.briefing_data = generate_briefing_data(
                    conn=conn,
                    project_key=project_key,
                    timeframe=st.session_state.timeframe,
                )
                st.session_state.briefing_generated = True
            except Exception as e:
                st.error(f"Failed to generate briefing: {e}")
                st.stop()

    data = st.session_state.briefing_data

    # Layout: Two columns
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Zone B: Morning Brief
        render_zone_b(data.get("briefing", {}))

        # Zone D: Vital Signs
        render_zone_d(
            data.get("delta", {}),
            data.get("comparison", {}),
        )

    with col_right:
        # Zone C: Decision Queue
        render_zone_c(data.get("decision_queue", []))

    # Zone E: Deep Dive (full width)
    st.markdown("---")
    render_zone_e(
        data.get("attention_items", []),
        data.get("delta", {}),
    )

    # Footer
    st.markdown("---")
    st.caption(
        f"Generated at {datetime.now().strftime('%I:%M %p')} | "
        f"Timeframe: {st.session_state.timeframe} | "
        f"Project: {project_key}"
    )


if __name__ == "__main__":
    main()
