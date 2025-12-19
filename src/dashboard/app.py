"""Main Streamlit dashboard application."""

import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Jira AI Co-pilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-medium { color: #ffa726; font-weight: bold; }
    .risk-low { color: #66bb6a; font-weight: bold; }
    .stMetric > div { background-color: #f8f9fa; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


def get_connection():
    """Get database connection with caching."""
    from src.data.schema import get_connection as db_get_connection
    return db_get_connection()


def get_predictor():
    """Get unified predictor with caching."""
    from src.models.predictor import UnifiedPredictor
    return UnifiedPredictor(model_dir="models")


def get_intelligence():
    """Get intelligence orchestrator."""
    from src.intelligence.orchestrator import JiraIntelligence
    return JiraIntelligence()


def main():
    """Main dashboard entry point."""
    # Sidebar navigation
    st.sidebar.title("🤖 Jira AI Co-pilot")

    page = st.sidebar.radio(
        "Navigation",
        ["📊 Overview", "🏃 Sprint Health", "👥 Team Workload", "🎯 Predictions", "📋 Reports"],
        label_visibility="collapsed",
    )

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Stats")

    try:
        conn = get_connection()
        issue_count = conn.execute("SELECT COUNT(*) FROM issues").fetchone()[0]
        sprint_count = conn.execute("SELECT COUNT(*) FROM sprints").fetchone()[0]

        st.sidebar.metric("Total Issues", issue_count)
        st.sidebar.metric("Total Sprints", sprint_count)
    except Exception:
        st.sidebar.warning("Database not connected")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "Built with [Streamlit](https://streamlit.io) • "
        "[GitHub](https://github.com)"
    )

    # Page routing
    if page == "📊 Overview":
        show_overview()
    elif page == "🏃 Sprint Health":
        show_sprint_health()
    elif page == "👥 Team Workload":
        show_team_workload()
    elif page == "🎯 Predictions":
        show_predictions()
    elif page == "📋 Reports":
        show_reports()


def show_overview():
    """Show dashboard overview."""
    st.markdown('<h1 class="main-header">📊 Dashboard Overview</h1>', unsafe_allow_html=True)

    try:
        conn = get_connection()

        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)

        # Active sprint metrics
        active_sprint = conn.execute("""
            SELECT sprint_id, sprint_name, start_date, end_date
            FROM sprints WHERE state = 'active'
            ORDER BY start_date DESC LIMIT 1
        """).fetchone()

        if active_sprint:
            from src.features.sprint_features import SprintFeatureExtractor

            extractor = SprintFeatureExtractor(conn)
            features = extractor.extract_features(active_sprint[0])

            predictor = get_predictor()
            risk = predictor.predict_sprint_risk(features)

            with col1:
                st.metric(
                    "Active Sprint",
                    active_sprint[1][:20] if active_sprint[1] else "Unknown",
                    f"{features.get('days_remaining', 0)} days left"
                )

            with col2:
                completion = features.get("completion_rate", 0)
                st.metric(
                    "Completion",
                    f"{completion:.0f}%",
                    f"{features.get('completed_points', 0)}/{features.get('total_points', 0)} pts"
                )

            with col3:
                risk_score = risk.get("score", 0)
                risk_level = risk.get("level", "unknown")
                st.metric(
                    "Risk Score",
                    f"{risk_score:.0f}/100",
                    risk_level.upper(),
                    delta_color="inverse" if risk_score > 50 else "normal"
                )

            with col4:
                blocked = conn.execute("""
                    SELECT COUNT(*) FROM issues
                    WHERE sprint_id = ? AND is_blocked = true
                """, [active_sprint[0]]).fetchone()[0]
                st.metric("Blocked Issues", blocked)

            # Sprint progress visualization
            st.markdown("### Sprint Progress")

            progress = features.get("progress_percent", 0) / 100
            completion_rate = features.get("completion_rate", 0) / 100

            col1, col2 = st.columns(2)

            with col1:
                st.progress(progress, text=f"Time Progress: {progress*100:.0f}%")

            with col2:
                st.progress(min(completion_rate, 1.0), text=f"Work Completion: {completion_rate*100:.0f}%")

            # Recent activity
            st.markdown("### Recent Activity")

            recent_issues = conn.execute("""
                SELECT key, summary, status, updated_at
                FROM issues
                WHERE sprint_id = ?
                ORDER BY updated_at DESC
                LIMIT 10
            """, [active_sprint[0]]).fetchall()

            if recent_issues:
                import pandas as pd
                df = pd.DataFrame(
                    recent_issues,
                    columns=["Key", "Summary", "Status", "Updated"]
                )
                df["Summary"] = df["Summary"].str[:50] + "..."
                st.dataframe(df, use_container_width=True, hide_index=True)

        else:
            st.warning("No active sprint found. Run 'jira-copilot sync full' first.")

        # Team velocity chart
        st.markdown("### Team Velocity Trend")

        velocity_data = conn.execute("""
            SELECT sprint_name, committed_points, completed_points
            FROM sprints
            WHERE state = 'closed'
            ORDER BY end_date DESC
            LIMIT 6
        """).fetchall()

        if velocity_data:
            import pandas as pd

            df = pd.DataFrame(
                reversed(velocity_data),
                columns=["Sprint", "Committed", "Completed"]
            )

            st.bar_chart(df.set_index("Sprint")[["Committed", "Completed"]])

    except Exception as e:
        st.error(f"Error loading overview: {e}")


def show_sprint_health():
    """Show sprint health page."""
    st.markdown('<h1 class="main-header">🏃 Sprint Health</h1>', unsafe_allow_html=True)

    try:
        conn = get_connection()

        # Sprint selector
        sprints = conn.execute("""
            SELECT sprint_id, sprint_name, state
            FROM sprints
            ORDER BY start_date DESC
            LIMIT 20
        """).fetchall()

        if not sprints:
            st.warning("No sprints found")
            return

        sprint_options = {f"{s[1]} ({s[2]})": s[0] for s in sprints}
        selected = st.selectbox("Select Sprint", list(sprint_options.keys()))
        sprint_id = sprint_options[selected]

        # Get sprint data
        from src.features.sprint_features import SprintFeatureExtractor

        extractor = SprintFeatureExtractor(conn)
        features = extractor.extract_features(sprint_id)

        predictor = get_predictor()
        risk = predictor.predict_sprint_risk(features)

        # Risk gauge
        col1, col2 = st.columns([1, 2])

        with col1:
            risk_score = risk.get("score", 0)
            risk_level = risk.get("level", "unknown")

            st.markdown("### Risk Score")

            # Color based on risk
            color = {"low": "green", "medium": "orange", "high": "red"}.get(risk_level, "gray")
            st.markdown(
                f'<h1 style="color: {color}; text-align: center;">{risk_score:.0f}</h1>',
                unsafe_allow_html=True
            )
            st.markdown(f'<p style="text-align: center;">{risk_level.upper()}</p>', unsafe_allow_html=True)

        with col2:
            st.markdown("### Key Metrics")

            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

            with metrics_col1:
                st.metric("Total Points", features.get("total_points", 0))
                st.metric("Completed", features.get("completed_points", 0))

            with metrics_col2:
                st.metric("Days Elapsed", features.get("days_elapsed", 0))
                st.metric("Days Remaining", features.get("days_remaining", 0))

            with metrics_col3:
                st.metric("Completion Rate", f"{features.get('completion_rate', 0):.1f}%")
                st.metric("Blocked Items", features.get("blocked_count", 0))

        # Risk factors
        st.markdown("### Risk Factors")

        if risk.get("factors"):
            import pandas as pd

            factors_data = [
                {"Factor": k.replace("_", " ").title(), "Contribution": v.get("contribution", 0) * 100}
                for k, v in risk["factors"].items()
            ]
            df = pd.DataFrame(factors_data)
            df = df.sort_values("Contribution", ascending=True)

            st.bar_chart(df.set_index("Factor"))

        # AI explanation
        with st.expander("🤖 AI Risk Explanation", expanded=False):
            if st.button("Generate Explanation"):
                with st.spinner("Analyzing..."):
                    intel = get_intelligence()
                    explanation = intel.explain_sprint_risk(features, risk)

                    st.markdown(f"**Summary:** {explanation.risk_summary}")

                    if explanation.main_concerns:
                        st.markdown("**Main Concerns:**")
                        for concern in explanation.main_concerns:
                            st.markdown(f"- {concern}")

                    if explanation.recommended_actions:
                        st.markdown("**Recommendations:**")
                        for action in explanation.recommended_actions:
                            priority = action.get("priority", "medium")
                            emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
                            st.markdown(f"{emoji} **{action.get('action', '')}**")
                            st.markdown(f"   _{action.get('rationale', '')}_")

        # Sprint issues
        st.markdown("### Sprint Issues")

        issues = conn.execute("""
            SELECT key, summary, status, priority, story_points, assignee, is_blocked
            FROM issues WHERE sprint_id = ?
            ORDER BY
                CASE status
                    WHEN 'Blocked' THEN 1
                    WHEN 'In Progress' THEN 2
                    WHEN 'To Do' THEN 3
                    ELSE 4
                END
        """, [sprint_id]).fetchall()

        if issues:
            import pandas as pd

            df = pd.DataFrame(
                issues,
                columns=["Key", "Summary", "Status", "Priority", "Points", "Assignee", "Blocked"]
            )
            df["Summary"] = df["Summary"].str[:40]
            df["Blocked"] = df["Blocked"].apply(lambda x: "🚫" if x else "")

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Blocked": st.column_config.TextColumn("🚫", width="small"),
                }
            )

    except Exception as e:
        st.error(f"Error loading sprint health: {e}")


def show_team_workload():
    """Show team workload page."""
    st.markdown('<h1 class="main-header">👥 Team Workload</h1>', unsafe_allow_html=True)

    try:
        conn = get_connection()
        from src.features.developer_features import DeveloperFeatureExtractor
        from config.settings import get_settings

        settings = get_settings()
        project_key = settings.jira.project_key

        extractor = DeveloperFeatureExtractor(conn)
        predictor = get_predictor()

        developers = extractor.get_active_developers(project_key)

        if not developers:
            st.warning("No active developers found")
            return

        # Collect workload data
        workload_data = []
        for dev_id in developers:
            features = extractor.extract_features(dev_id, project_key)
            workload = predictor.assess_developer_workload(features)

            workload_data.append({
                "Developer": features.get("pseudonym", f"dev_{dev_id[:8]}"),
                "WIP Count": features.get("wip_count", 0),
                "WIP Points": features.get("wip_points", 0),
                "Blocked": features.get("blocked_count", 0),
                "7-day Velocity": features.get("completed_last_7_days", 0),
                "Status": workload.get("status", "optimal"),
                "Score": workload.get("score", 0),
            })

        import pandas as pd
        df = pd.DataFrame(workload_data)

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Team Size", len(developers))

        with col2:
            overloaded = len([d for d in workload_data if d["Status"] == "overloaded"])
            st.metric("Overloaded", overloaded, delta_color="inverse" if overloaded > 0 else "normal")

        with col3:
            total_wip = sum(d["WIP Points"] for d in workload_data)
            st.metric("Total WIP Points", total_wip)

        with col4:
            total_blocked = sum(d["Blocked"] for d in workload_data)
            st.metric("Total Blocked", total_blocked)

        # Workload chart
        st.markdown("### Workload Distribution")

        chart_df = df[["Developer", "WIP Points", "7-day Velocity"]].set_index("Developer")
        st.bar_chart(chart_df)

        # Team table
        st.markdown("### Team Status")

        def status_color(status):
            colors = {
                "overloaded": "🔴",
                "high": "🟡",
                "optimal": "🟢",
                "underloaded": "⚪"
            }
            return f"{colors.get(status, '⚪')} {status}"

        df["Status"] = df["Status"].apply(status_color)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Recommendations
        overloaded_devs = [d for d in workload_data if d["Status"].startswith("🔴")]
        if overloaded_devs:
            st.markdown("### ⚠️ Attention Required")
            for dev in overloaded_devs:
                st.warning(f"**{dev['Developer']}** is overloaded with {dev['WIP Points']} points in progress")

        # AI Assessment
        with st.expander("🤖 AI Workload Assessment", expanded=False):
            dev_select = st.selectbox(
                "Select Developer",
                [d["Developer"] for d in workload_data]
            )

            if st.button("Generate Assessment"):
                with st.spinner("Analyzing..."):
                    dev_data = next(d for d in workload_data if d["Developer"] == dev_select)
                    intel = get_intelligence()

                    # Prepare data for assessment
                    assessment_data = {
                        "assignee_id": dev_select,
                        "pseudonym": dev_select,
                        "wip_count": dev_data["WIP Count"],
                        "wip_points": dev_data["WIP Points"],
                        "blocked_count": dev_data["Blocked"],
                        "status": dev_data["Status"].split()[-1],
                        "score": dev_data["Score"],
                        "relative_to_team": 1.0,
                        "team_avg_wip_points": sum(d["WIP Points"] for d in workload_data) / len(workload_data),
                        "team_avg_completed_points": sum(d["7-day Velocity"] for d in workload_data) / len(workload_data),
                    }

                    assessment = intel.assess_developer_workload(assessment_data)

                    st.markdown(f"**Assessment:** {assessment.assessment}")
                    st.markdown(assessment.summary)

                    if assessment.recommendations:
                        st.markdown("**Recommendations:**")
                        for rec in assessment.recommendations:
                            st.markdown(f"- {rec}")

    except Exception as e:
        st.error(f"Error loading team workload: {e}")


def show_predictions():
    """Show predictions page."""
    st.markdown('<h1 class="main-header">🎯 Predictions</h1>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Ticket Duration", "Sprint Risk"])

    with tab1:
        st.markdown("### Predict Ticket Duration")

        try:
            conn = get_connection()

            # Get open issues
            issues = conn.execute("""
                SELECT key, summary FROM issues
                WHERE status NOT IN ('Done', 'Closed')
                ORDER BY updated_at DESC
                LIMIT 100
            """).fetchall()

            if not issues:
                st.warning("No open issues found")
            else:
                issue_options = {f"{i[0]}: {i[1][:50]}...": i[0] for i in issues}
                selected = st.selectbox("Select Issue", list(issue_options.keys()))
                issue_key = issue_options[selected]

                if st.button("Predict Duration", key="predict_ticket"):
                    with st.spinner("Predicting..."):
                        from src.features.ticket_features import TicketFeatureExtractor

                        extractor = TicketFeatureExtractor(conn)
                        features = extractor.extract_features(issue_key)

                        predictor = get_predictor()
                        prediction = predictor.predict_ticket(features)

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            hours = prediction.get("predicted_hours", 0)
                            st.metric("Predicted Duration", f"{hours:.1f} hours")

                        with col2:
                            days = prediction.get("predicted_days", 0)
                            st.metric("In Days", f"{days:.1f} days")

                        with col3:
                            if prediction.get("confidence_interval"):
                                ci = prediction["confidence_interval"]
                                st.metric(
                                    "Confidence Range",
                                    f"{ci.get('lower_hours', 0):.1f} - {ci.get('upper_hours', 0):.1f}h"
                                )

                        st.info(f"Model: {prediction.get('model_type', 'Unknown')}")

        except Exception as e:
            st.error(f"Error: {e}")

    with tab2:
        st.markdown("### Predict Sprint Risk")

        try:
            conn = get_connection()

            sprints = conn.execute("""
                SELECT sprint_id, sprint_name, state
                FROM sprints
                ORDER BY start_date DESC
                LIMIT 10
            """).fetchall()

            if not sprints:
                st.warning("No sprints found")
            else:
                sprint_options = {f"{s[1]} ({s[2]})": s[0] for s in sprints}
                selected = st.selectbox("Select Sprint", list(sprint_options.keys()), key="sprint_risk")
                sprint_id = sprint_options[selected]

                if st.button("Predict Risk", key="predict_risk"):
                    with st.spinner("Analyzing..."):
                        from src.features.sprint_features import SprintFeatureExtractor

                        extractor = SprintFeatureExtractor(conn)
                        features = extractor.extract_features(sprint_id)

                        predictor = get_predictor()
                        risk = predictor.predict_sprint_risk(features)

                        col1, col2 = st.columns(2)

                        with col1:
                            score = risk.get("score", 0)
                            level = risk.get("level", "unknown")
                            color = {"low": "green", "medium": "orange", "high": "red"}.get(level, "gray")

                            st.markdown(
                                f'<h1 style="color: {color}; text-align: center;">{score:.0f}/100</h1>',
                                unsafe_allow_html=True
                            )
                            st.markdown(f'<p style="text-align: center;"><b>{level.upper()}</b></p>', unsafe_allow_html=True)

                        with col2:
                            st.markdown("**Top Risk Factors:**")
                            if risk.get("factors"):
                                factors = sorted(
                                    risk["factors"].items(),
                                    key=lambda x: x[1].get("contribution", 0),
                                    reverse=True
                                )
                                for factor, details in factors[:5]:
                                    contrib = details.get("contribution", 0) * 100
                                    st.markdown(f"- {factor.replace('_', ' ').title()}: {contrib:.1f}%")

        except Exception as e:
            st.error(f"Error: {e}")


def show_reports():
    """Show reports page."""
    st.markdown('<h1 class="main-header">📋 Reports</h1>', unsafe_allow_html=True)

    report_type = st.selectbox(
        "Report Type",
        ["Sprint Health Report", "Team Workload Report", "Velocity Report"]
    )

    col1, col2 = st.columns(2)

    with col1:
        format_option = st.selectbox("Format", ["Markdown", "HTML"])

    with col2:
        if st.button("Generate Report"):
            with st.spinner("Generating..."):
                try:
                    conn = get_connection()

                    if report_type == "Sprint Health Report":
                        from src.features.sprint_features import SprintFeatureExtractor
                        from src.models.predictor import UnifiedPredictor
                        from src.intelligence.orchestrator import JiraIntelligence

                        sprint = conn.execute("""
                            SELECT sprint_id, sprint_name FROM sprints
                            WHERE state = 'active' LIMIT 1
                        """).fetchone()

                        if sprint:
                            extractor = SprintFeatureExtractor(conn)
                            features = extractor.extract_features(sprint[0])

                            predictor = UnifiedPredictor(model_dir="models")
                            risk = predictor.predict_sprint_risk(features)

                            intel = JiraIntelligence()
                            explanation = intel.explain_sprint_risk(features, risk)

                            report = f"""# Sprint Health Report

**Sprint:** {features.get('sprint_name', 'Unknown')}

## Summary
{explanation.risk_summary}

## Metrics
- Completion Rate: {features.get('completion_rate', 0):.1f}%
- Risk Score: {risk.get('score', 0):.0f}/100 ({risk.get('level', 'unknown')})
- Days Remaining: {features.get('days_remaining', 0)}

## Recommendations
"""
                            for action in explanation.recommended_actions:
                                report += f"- {action.get('action', '')}\n"

                            st.markdown(report)

                            if format_option == "HTML":
                                import markdown
                                html = markdown.markdown(report)
                                st.download_button(
                                    "Download HTML",
                                    html,
                                    file_name="sprint_report.html",
                                    mime="text/html"
                                )
                            else:
                                st.download_button(
                                    "Download Markdown",
                                    report,
                                    file_name="sprint_report.md",
                                    mime="text/markdown"
                                )
                        else:
                            st.warning("No active sprint found")

                    elif report_type == "Team Workload Report":
                        st.info("Use CLI: jira-copilot report workload")

                    elif report_type == "Velocity Report":
                        st.info("Use CLI: jira-copilot analyze velocity")

                except Exception as e:
                    st.error(f"Error generating report: {e}")


if __name__ == "__main__":
    main()
